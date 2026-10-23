import os
import sys
import threading
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from db.pool import connection
from db.tables.documents import cited_chunks
from db.tables.registry import EmbeddingModel
from db.vectors import get_vector_store
from llm.context_management import (
    Conversation,
    SlidingWindow,  # noqa: F401  -- the alternative for HISTORY below
    Summarize,
)

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

# The model that answers. Overridable per deployment, as CURRENT_USER_ID is.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Routing and summarising: one returns a bool, the other writes notes nobody
# reads directly, so neither needs the model that answers.
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gpt-4o-mini")

_llm: ChatOpenAI | None = None
_router: Runnable | None = None
_summarizer: ChatOpenAI | None = None
_llm_lock = threading.Lock()

# One per process, so a conversation survives between requests. Keyed by
# thread_id, which is per user -- changing a document's model continues the
# conversation rather than restarting it.
CHECKPOINTER = InMemorySaver()

ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Decide whether answering the user's latest message needs a "
                "search of their uploaded documents.\n"
                "Search when the answer depends on what a document says: "
                "facts, figures, names, or anything the documents would "
                "state.\n"
                "Search when asked what exists -- which documents, stories or "
                "topics are available, or what you have, know, hold or "
                "remember. You have no knowledge of your own, so anything you "
                "'have' is in their documents and finding out means "
                "searching.\n"
                "Only two things skip the search: greetings and small talk, "
                "and questions about this conversation itself -- what was "
                "said in it earlier, or redoing an answer already given.\n"
                "The latest message may lean on earlier turns, so read it in "
                "context before deciding."
            ),
        ),
        MessagesPlaceholder("messages"),
    ]
)

CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the question based on the provided context.\n"
                "Each passage is headed by the document it came from and its "
                "page, in brackets. Those headers say where the text is "
                "from; they are not part of it.\n\n"
                "{context}"
            ),
        ),
        MessagesPlaceholder("messages"),
    ]
)

PLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer from the conversation so far. No documents were "
                "searched for this question."
            ),
        ),
        MessagesPlaceholder("messages"),
    ]
)

TOP_K = 10
FETCH_K = 50

# Reciprocal rank fusion constant. Scores from two embedding models are not
# comparable -- 0.23 in one space does not mean what it means in the other --
# so lists are merged by position instead. 60 is the value from the original
# RRF paper; it damps the top of each list enough that a single model cannot
# monopolise the result.
RRF_K = 60


class Route(BaseModel):
    """Whether the question has to be answered from the documents."""

    needs_search: bool = Field(
        description="True if answering requires searching the user's documents."
    )


class ChatState(TypedDict):
    """What one conversation carries between turns.

    `messages` is the history, accumulated by the `add_messages` reducer.
    `needs_search` and `context` belong to the turn in flight: `decide` clears
    the context at the top of every turn, so a turn that skips retrieval
    cannot report the previous turn's chunks as its sources.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    needs_search: bool
    context: list[Document]
    summary: str | None
    summarized: int


def _chat_llm() -> ChatOpenAI:
    """One ChatOpenAI per process -- it owns an HTTP client worth reusing.

    `streaming=True` is stated rather than inherited: the answer is streamed
    token by token whether or not a caller happens to attach a handler that
    would have turned it on anyway.
    """
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)
        return _llm


def _route_llm():
    """The same model, asked for a bool instead of prose.

    `disable_streaming` because a streaming handler anywhere on the run makes
    every model call in the graph stream, this one included -- and streaming a
    one-field object back a token at a time buys nothing.
    """
    global _router
    with _llm_lock:
        if _router is None:
            _router = ChatOpenAI(
                model=UTILITY_MODEL, temperature=0, disable_streaming=True
            ).with_structured_output(Route)
        return _router


def _summary_llm() -> ChatOpenAI:
    """The model that writes the running summary.

    Not streamed: the summary goes into state, not to the browser, and a
    streaming handler on the run would otherwise stream it a token at a time
    for no reader.
    """
    global _summarizer
    with _llm_lock:
        if _summarizer is None:
            _summarizer = ChatOpenAI(
                model=UTILITY_MODEL, temperature=0, disable_streaming=True
            )
        return _summarizer


# How the conversation is kept from growing without bound. Swap the line:
# the window is free but forgets, the summary costs a call per compaction and
# remembers the gist. Both leave the stored transcript whole.
HISTORY = Summarize(
    llm=_summary_llm, budget_model=CHAT_MODEL, budget=3_000, keep_turns=4
)
# HISTORY = SlidingWindow(keep_turns=6)


def _retriever(
    user_id: int, model: EmbeddingModel, document_ids: list[str], *, mmr: bool
):
    """A retriever over one model's table, scoped to one set of documents.

    Both filters are applied by the search itself: HNSW picks its top-k before
    any caller could drop rows afterwards.

    MMR spreads the results out instead of returning ten near-duplicates, but
    it compares candidate vectors to each other, so it only means anything
    inside a single embedding space. A fan-out asks for plain similarity and
    lets the fusion below do the combining.
    """
    search_kwargs = {
        "k": TOP_K,
        "filter": {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"document_id": {"$in": document_ids}},
            ]
        },
    }
    if mmr:
        search_kwargs["fetch_k"] = FETCH_K

    return get_vector_store(model).as_retriever(
        search_type="mmr" if mmr else "similarity",
        search_kwargs=search_kwargs,
    )


def _fuse(ranked_lists: list[list[Document]]) -> list[Document]:
    """Reciprocal rank fusion over one result list per model.

    Each document scores sum(1 / (RRF_K + rank)) across the lists it appears
    in, so agreement between models outranks a strong showing in one. Chunks
    are identified by chunk_id: the same chunk embedded by two models is one
    result, not two copies of the same text in the prompt.
    """
    scores: dict[str, float] = {}
    best: dict[str, Document] = {}

    for documents in ranked_lists:
        for rank, document in enumerate(documents):
            key = str(document.metadata.get("chunk_id"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            best.setdefault(key, document)

    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [best[key] for key in order[:TOP_K]]


def _search(
    user_id: int, groups: dict[EmbeddingModel, list[str]], question: str
) -> list[Document]:
    """Retrieval across whichever models hold the searchable documents.

    One model is the common case: a single MMR search, one query embedding.
    Several means the question has to be embedded once per model -- a query
    vector is only comparable to vectors from the model that produced it --
    and the lists fused afterwards.
    """
    if len(groups) == 1:
        model, document_ids = next(iter(groups.items()))
        return _retriever(user_id, model, document_ids, mmr=True).invoke(question)

    retrievers = [
        _retriever(user_id, model, document_ids, mmr=False)
        for model, document_ids in groups.items()
    ]
    return _fuse([r.invoke(question) for r in retrievers])


def _label(user_id: int, documents: list[Document]) -> list[Document]:
    """Attach each chunk's document title and page.

    Retrieval knows ids and text, so the chunks arrive anonymous: the prompt
    cannot say which document a passage came from, and a question about which
    documents exist has nothing to answer from. Scoped to the user by the
    query, so a chunk belonging to someone else resolves to no title rather
    than theirs.
    """
    chunk_ids = [str(document.metadata.get("chunk_id")) for document in documents]
    if not chunk_ids:
        return documents

    with connection() as conn:
        rows = cited_chunks(conn, user_id, chunk_ids)

    found = {chunk_id: (title, page) for chunk_id, _, title, _, page in rows}
    for document in documents:
        title, page = found.get(str(document.metadata.get("chunk_id")), (None, None))
        document.metadata["title"] = title
        document.metadata["page"] = page
    return documents


def _format(documents: list[Document]) -> str:
    """The retrieved chunks as the prompt should see them.

    Each passage is headed by where it came from, so the model can attribute
    a claim and can answer which documents it has. Pages are shifted into the
    PDF's own numbering, which starts at 1.

    Interpolating the list itself instead hands the model a Python repr --
    row ids, chunk ids, the user id -- which it reads as content and reports
    back as though the ids were documents.
    """
    blocks = []
    for document in documents:
        title = document.metadata.get("title")
        page = document.metadata.get("page")
        if title and page is not None:
            blocks.append(f"[{title}, p. {page + 1}]\n{document.page_content}")
        elif title:
            blocks.append(f"[{title}]\n{document.page_content}")
        else:
            blocks.append(document.page_content)
    return "\n\n".join(blocks)


def build_graph(user_id: int, groups: dict[EmbeddingModel, list[str]]):
    """The chat graph for one request.

    Built per request rather than once at startup: which model searches which
    document is stored state the user can change, and a graph cached at
    startup would keep answering from whatever it was then. The expensive
    parts -- the vector stores, their engine, the LLM clients, the
    checkpointer -- are cached underneath, so this is object assembly and no
    I/O.
    """

    def _conversation(state: ChatState) -> Conversation:
        return Conversation(
            messages=state["messages"],
            summary=state.get("summary"),
            summarized=state.get("summarized", 0),
        )

    async def compact(state: ChatState) -> dict:
        return await HISTORY.compact(_conversation(state))

    async def decide(state: ChatState) -> dict:
        # Nothing searchable means there is nothing to route to, and asking
        # the model would spend a call to be told so.
        if not groups:
            return {"needs_search": False, "context": []}

        route = await _route_llm().ainvoke(
            ROUTE_PROMPT.invoke({"messages": HISTORY.view(_conversation(state))})
        )
        return {"needs_search": route.needs_search, "context": []}

    # Sync: the search is blocking work, so the graph runs it on a worker
    # thread instead of on the event loop.
    def retrieve(state: ChatState) -> dict:
        question = state["messages"][-1].text
        return {"context": _label(user_id, _search(user_id, groups, question))}

    async def answer(state: ChatState) -> dict:
        context = state["context"]
        messages = HISTORY.view(_conversation(state))
        prompt = (
            CONTEXT_PROMPT.invoke({"messages": messages, "context": _format(context)})
            if context
            else PLAIN_PROMPT.invoke({"messages": messages})
        )
        return {"messages": [await _chat_llm().ainvoke(prompt)]}

    builder = StateGraph(ChatState)
    builder.add_node("compact", compact)
    builder.add_node("decide", decide)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)

    builder.add_edge(START, "compact")
    builder.add_edge("compact", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda state: "retrieve" if state["needs_search"] else "answer",
        ["retrieve", "answer"],
    )
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile(checkpointer=CHECKPOINTER)
