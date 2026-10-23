"""Answering questions from the user's searchable documents, as an event stream."""

import json
import logging
import re

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from api.config import CURRENT_USER_ID
from api.services.sources import source_path
from db.pool import connection
from db.tables.documents import cited_chunks
from db.tables.embeddings import search_groups
from db.tables.registry import get_model
from llm.chat import build_graph, cited_documents

logger = logging.getLogger("api")

router = APIRouter()


class Chat(BaseModel):
    prompt: str


def sse(event: str, data: dict) -> str:
    """One server-sent event.

    The payload is JSON rather than raw text because a newline inside `data:`
    ends the field and one leading space after the colon is framing -- model
    output has both.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# A citation the answer carries: the bracketed number of a document in the
# prompt's context. `[1][3]` is two of them, which is why this matches one at
# a time rather than a run.
CITATION = re.compile(r"\[(\d+)\]")


def sources(retrieved: list, answer: str) -> list[dict]:
    """The documents the answer cited, one entry each, best-ranked first.

    Retrieval cannot tell these apart: it returns the ten chunks closest to
    the question, which is most of the library on a broad question. Only the
    model knows which of them it leaned on, so it numbers its claims and this
    reads the numbers back out -- a document it never cited is left out.

    The markers themselves are plumbing and never reach the reader: they are
    stripped out of the answer in the browser. Pages are every retrieved page
    of the document, shifted into the PDF's own numbering, which starts at 1 --
    a marker is per document, so it does not say which passage carried a claim.

    `filename` is what the download actually arrives as, which is why it is
    sent alongside the title -- the two differ, the title being the stem.
    """
    cited = {int(number) for number in CITATION.findall(answer)}
    numbered = {
        document_id: number
        for number, document_id in enumerate(cited_documents(retrieved), 1)
        if number in cited
    }
    if not numbered:
        return []

    chunk_ids = [str(document.metadata.get("chunk_id")) for document in retrieved]
    with connection() as conn:
        rows = cited_chunks(conn, CURRENT_USER_ID, chunk_ids)

    found: dict[str, dict] = {}

    for _chunk_id, document_id, title, source_uri, page in rows:
        if document_id not in numbered:
            continue
        # Resolved against the disk, not just the column: a file deleted from
        # under its row would otherwise be offered as a link that 404s.
        stored = source_path(source_uri)
        source = found.setdefault(
            document_id,
            {
                "title": title,
                "filename": stored.name if stored else None,
                "pages": set(),
                "has_source": stored is not None,
            },
        )
        if page is not None:
            source["pages"].add(page + 1)

    return [
        {
            "document_id": document_id,
            "title": source["title"],
            "filename": source["filename"],
            "pages": sorted(source["pages"]),
            "has_source": source["has_source"],
        }
        for document_id, source in sorted(
            found.items(), key=lambda item: numbered[item[0]]
        )
    ]


# Sync `def`: reading which model searches which document is a blocking
# query, and the graph is assembled before a single token is streamed.
# FastAPI runs this in a worker thread and then streams on the event loop.
@router.post("/chat")
def chat(prompt: Chat):
    """Answer from the user's searchable documents.

    Each document is retrieved through the one model it was given, so this
    reads the grouping per request -- changing a document's model has to take
    effect on the next question, not the next restart.

    The body is an event stream: a `token` per piece of the answer, then
    `sources` naming the documents the answer cited, then `done`. Every
    question is searched, so an empty `sources` means the answer leaned on
    nothing retrieved -- the passages did not cover the question, or the user
    has nothing searchable.
    """
    with connection() as conn:
        groups = {
            get_model(conn, model_id=model_id): document_ids
            for model_id, document_ids in search_groups(conn, CURRENT_USER_ID).items()
        }

    graph = build_graph(CURRENT_USER_ID, groups)
    config = {"configurable": {"thread_id": f"user-{CURRENT_USER_ID}"}}

    async def stream():
        # A streaming response commits to 200 before the first token, so a
        # failure here cannot change the status. It goes out as an `error`
        # event instead -- the alternative is a silent truncation that reads
        # like a short answer.
        try:
            # `messages` mode emits every LLM call in the graph, and the
            # summariser is one of them -- its notes are not the answer.
            async for chunk, metadata in graph.astream(
                {"messages": [HumanMessage(prompt.prompt)]},
                config=config,
                stream_mode="messages",
            ):
                if metadata["langgraph_node"] == "answer" and chunk.text:
                    yield sse("token", {"t": chunk.text})

            # The chunks the answer was built from live in the graph's state,
            # which is only final once the stream is done. The answer is read
            # back from there too rather than accumulated here -- the same
            # text, and the citations have to be matched against all of it.
            state = await graph.aget_state(config)
            answer = state.values["messages"][-1].text
            # Off the loop: resolving chunks to documents is a blocking query,
            # and this generator runs on the event loop.
            resolved = await run_in_threadpool(
                sources, state.values.get("context", []), answer
            )
            yield sse("sources", {"sources": resolved})
        except Exception:
            logger.exception("chat stream failed")
            yield sse(
                "error",
                {"detail": "The answer could not be completed. See server logs."},
            )
        # Deliberately not in a `finally`: a client that disconnects throws
        # GeneratorExit in here, and yielding while that unwinds is its own
        # error.
        yield sse("done", {})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache"},
    )
