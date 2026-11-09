import json
import os
import sys
import threading
from typing import Annotated, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from db import vectors
from db.pool import connection
from db.tables.chunk_embeddings import nearest
from db.tables.documents import cited_chunks
from db.tables.registry import EmbeddingModel
from llm.context_management import (
    Conversation,
    SlidingWindow,  # noqa: F401  -- the alternative for HISTORY below
    Summarize,
)

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

# The model that answers. Overridable per deployment.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Summarising writes notes nobody reads directly, so it does not need the
# model that answers.
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gpt-4o-mini")

# Seconds before an OpenAI call is abandoned; the SDK default is 600.
LLM_TIMEOUT = 60

# When a passage is shown as a source. Both must hold.
# Max cosine distance between the question and the passage.
SOURCE_MAX_DISTANCE = 0.75
# Min cosine similarity between the finished answer and the passage.
SOURCE_MIN_ANSWER_SIMILARITY = 0.51

_llm: ChatOpenAI | None = None
_summarizer: ChatOpenAI | None = None
_llm_lock = threading.Lock()

# No checkpointer. A conversation is stored in `conversations` and
# `chat_messages`, and every turn is handed the history it needs as its
# starting state -- so an answer does not depend on which process served the
# previous question, and history outlives a restart.

CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the question based on the provided context.\n"
                "Each passage is headed by the document it came from and its "
                "page. Those headers say where the text is from; they are not "
                "part of it.\n"
                "The passages are whatever the search returned, and may not "
                "cover the question. When they do not, say so plainly and "
                "then answer from your own knowledge.\n\n"
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
                "The user has no searchable documents, so there was "
                "nothing to retrieve. Answer from the conversation so far "
                "and your own knowledge."
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


class ChatState(TypedDict):
    """What one conversation carries between turns.

    `messages` is the history, accumulated by the `add_messages` reducer.
    `context` belongs to the turn in flight: `retrieve` replaces it on every
    turn, so a turn cannot report the previous turn's chunks as its sources.
    """

    messages: Annotated[list[AnyMessage], add_messages]
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
            _llm = ChatOpenAI(
                model=CHAT_MODEL,
                temperature=0,
                streaming=True,
                timeout=LLM_TIMEOUT,
                max_retries=2,
            )
        return _llm


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
                model=UTILITY_MODEL,
                temperature=0,
                disable_streaming=True,
                timeout=LLM_TIMEOUT,
                max_retries=2,
            )
        return _summarizer


# How the conversation is kept from growing without bound. Swap the line:
# the window is free but forgets, the summary costs a call per compaction and
# remembers the gist. Both leave the stored transcript whole.
HISTORY = Summarize(
    llm=_summary_llm, budget_model=CHAT_MODEL, budget=3_000, keep_turns=4
)
# HISTORY = SlidingWindow(keep_turns=6)


def _retrieve(
    model: EmbeddingModel, document_ids: list[str], question: str, *, mmr: bool
) -> list[Document]:
    """One model's closest chunks, scoped to one set of documents.

    The document filter is applied by the search itself: HNSW picks its top-k
    before any caller could drop rows afterwards. The documents are already the
    user's, so that filter is also what keeps other users' chunks out.

    MMR spreads the results out instead of returning ten near-duplicates, but
    it compares candidate vectors to each other, so it only means anything
    inside a single embedding space. A fan-out asks for plain similarity and
    lets the fusion below do the combining.
    """
    query = vectors.embeddings_for(model).embed_query(question)
    with connection() as conn:
        rows = nearest(conn, model, query, document_ids, FETCH_K if mmr else TOP_K)

    documents = [
        Document(
            page_content=content,
            metadata={
                "chunk_id": chunk_id,
                "distance": distance,
                "model": model,
                "vector": json.loads(vector),
            },
        )
        for chunk_id, content, vector, distance in rows
    ]
    if not mmr or not documents:
        return documents

    picked = maximal_marginal_relevance(
        np.array(query), [d.metadata["vector"] for d in documents], k=TOP_K
    )
    return [documents[i] for i in picked]


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


def _search(groups: dict[EmbeddingModel, list[str]], question: str) -> list[Document]:
    """Retrieval across whichever models hold the searchable documents.

    One model is the common case: a single MMR search, one query embedding.
    Several means the question has to be embedded once per model -- a query
    vector is only comparable to vectors from the model that produced it --
    and the lists fused afterwards.
    """
    if len(groups) == 1:
        model, document_ids = next(iter(groups.items()))
        return _retrieve(model, document_ids, question, mmr=True)

    return _fuse(
        [
            _retrieve(model, document_ids, question, mmr=False)
            for model, document_ids in groups.items()
        ]
    )


def _label(user_id: int | None, documents: list[Document]) -> list[Document]:
    """Attach each chunk's document id, title and page.

    Retrieval knows ids and text, so the chunks arrive anonymous: the prompt
    cannot say which document a passage came from, and a question about which
    documents exist has nothing to answer from. Scoped to what the user may
    read by the query, so a chunk in someone else's private document resolves
    to no title rather than theirs.
    """
    chunk_ids = [str(document.metadata.get("chunk_id")) for document in documents]
    if not chunk_ids:
        return documents

    with connection() as conn:
        rows = cited_chunks(conn, user_id, chunk_ids)

    found = {
        chunk_id: (document_id, title, page)
        for chunk_id, document_id, title, _, _, page, _ in rows
    }
    for document in documents:
        document_id, title, page = found.get(
            str(document.metadata.get("chunk_id")), (None, None, None)
        )
        document.metadata["document_id"] = document_id
        document.metadata["title"] = title
        document.metadata["page"] = page
    return documents


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector)
    return array / np.linalg.norm(array)


def drawn_on(documents: list[Document], answer: str) -> list[Document]:
    """The retrieved passages the answer came from.

    Two checks, both on embeddings: the passage is close to the question, and
    the answer is close to the passage. The first alone lets through a question
    that only sounds like the documents -- "tell me a joke" lands near story
    text -- and an answer saying the passages did not cover the question. The
    second catches both.

    The answer is embedded once per model, since a vector is only comparable to
    vectors from the model that produced it.
    """
    answers: dict[int, np.ndarray] = {}
    used = []
    for document in documents:
        if document.metadata["distance"] > SOURCE_MAX_DISTANCE:
            continue
        model = document.metadata["model"]
        if model.id not in answers:
            answers[model.id] = _unit(vectors.embeddings_for(model).embed_query(answer))
        similarity = float(answers[model.id] @ _unit(document.metadata["vector"]))
        if similarity >= SOURCE_MIN_ANSWER_SIMILARITY:
            used.append(document)
    return used


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
        header = f"{title}, p. {page + 1}" if title and page is not None else title
        blocks.append(
            f"{header}\n{document.page_content}" if header else document.page_content
        )
    return "\n\n".join(blocks)


def build_graph(user_id: int | None, groups: dict[EmbeddingModel, list[str]]):
    """The chat graph for one request.

    Built per request rather than once at startup: which model searches which
    document is stored state the user can change, and a graph cached at
    startup would keep answering from whatever it was then. The expensive
    parts -- the embedding services and the LLM clients -- are cached
    underneath, so this is object assembly and no I/O.

    The graph carries no history of its own. It is handed the conversation as
    its starting state and returns the turn it added, which the caller stores.
    """

    def _conversation(state: ChatState) -> Conversation:
        return Conversation(
            messages=state["messages"],
            summary=state.get("summary"),
            summarized=state.get("summarized", 0),
        )

    async def compact(state: ChatState) -> dict:
        return await HISTORY.compact(_conversation(state))

    # Sync: the search is blocking work, so the graph runs it on a worker
    # thread instead of on the event loop.
    def retrieve(state: ChatState) -> dict:
        # Every question is searched. Judging beforehand whether one needed
        # the documents cost a model call to get wrong: a question that looks
        # like general knowledge reads as not needing them, which is exactly
        # the question the documents were uploaded to answer. The passages go
        # to `answer`, which falls back to its own knowledge when they turn
        # out not to cover the question.
        if not groups:
            return {"context": []}

        question = state["messages"][-1].text
        return {"context": _label(user_id, _search(groups, question))}

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
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)

    builder.add_edge(START, "compact")
    builder.add_edge("compact", "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile()
