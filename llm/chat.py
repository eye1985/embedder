import os
import sys
import threading
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from db.registry import EmbeddingModel
from db.vectors import get_vector_store

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

session_store: dict[str, BaseChatMessageHistory] = {}

_llm: ChatOpenAI | None = None
_llm_lock = threading.Lock()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question based on the provided context:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

# Nothing retrieved, so the chain still runs and the model answers from the
# conversation alone. The alternative is an `$in ()` filter, which is not valid
# SQL and would turn "everything is switched off" into a 500.
NO_CONTEXT = RunnableLambda(lambda _: [])

TOP_K = 10
FETCH_K = 50

# Reciprocal rank fusion constant. Scores from two embedding models are not
# comparable -- 0.23 in one space does not mean what it means in the other --
# so lists are merged by position instead. 60 is the value from the original
# RRF paper; it damps the top of each list enough that a single model cannot
# monopolise the result.
RRF_K = 60


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def _chat_llm() -> ChatOpenAI:
    """One ChatOpenAI per process -- it owns an HTTP client worth reusing."""
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return _llm


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


def _context(user_id: int, groups: dict[EmbeddingModel, list[str]]) -> Runnable:
    """Where the prompt's context comes from.

    One model is the common case and stays exactly as it was: a single MMR
    search, one query embedding. Several means the question has to be embedded
    once per model -- a query vector is only comparable to vectors from the
    model that produced it -- and the lists fused afterwards.
    """
    if not groups:
        return NO_CONTEXT

    if len(groups) == 1:
        model, document_ids = next(iter(groups.items()))
        return itemgetter("question") | _retriever(
            user_id, model, document_ids, mmr=True
        )

    retrievers = [
        _retriever(user_id, model, document_ids, mmr=False)
        for model, document_ids in groups.items()
    ]

    def fan_out(inputs: dict) -> list[Document]:
        question = inputs["question"]
        return _fuse([r.invoke(question) for r in retrievers])

    return RunnableLambda(fan_out)


def _tap(sink: list[Document]) -> Runnable:
    """Records what retrieval returned on the way past.

    The chain's output is the answer text, so the chunks behind it would
    otherwise be unreachable -- and naming sources needs them. One sink per
    chain, which is one per request.
    """

    def record(documents: list[Document]) -> list[Document]:
        sink.clear()
        sink.extend(documents)
        return documents

    return RunnableLambda(record)


def build_chat(user_id: int, groups: dict[EmbeddingModel, list[str]]) -> dict:
    """The chat chain for one request.

    Built per request rather than once at startup: which model searches which
    document is stored state the user can change, and a chain cached at
    startup would keep answering from whatever it was then. The expensive
    parts -- the vector stores, their engine, the LLM client -- are cached
    underneath, so this is object assembly and no I/O.

    History is keyed by user, so changing a document's model continues the
    conversation rather than restarting it.

    `retrieved` is empty until the chain runs, and then holds the chunks the
    answer was built from. It stays out of the chain's output so history keeps
    storing the answer alone.
    """
    retrieved: list[Document] = []

    chain = (
        RunnablePassthrough.assign(context=_context(user_id, groups) | _tap(retrieved))
        | PROMPT
        | _chat_llm()
        | StrOutputParser()
    )

    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    return {
        "runnable_with_history": conversational_chain,
        "config": {"configurable": {"session_id": f"user-{user_id}"}},
        "retrieved": retrieved,
    }
