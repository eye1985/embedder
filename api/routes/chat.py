"""Answering questions from the user's searchable documents, as an event stream."""

import json
import logging
from operator import itemgetter

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from api.config import CURRENT_USER_ID
from api.services.sources import source_path
from db.pool import connection
from db.tables.documents import cited_chunks
from db.tables.embeddings import search_groups
from db.tables.registry import get_model
from llm.chat import build_chat

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


def sources(retrieved: list) -> list[dict]:
    """The documents behind an answer, one entry each, best-ranked first.

    These are the chunks retrieval returned, which is not the same as the ones
    the model leaned on -- it says what was searched, not what was quoted.
    Pages are shifted into the PDF's own numbering, which starts at 1.

    `filename` is what the download actually arrives as, which is why it is
    sent alongside the title -- the two differ, the title being the stem.
    """
    ranked = [str(document.metadata.get("chunk_id")) for document in retrieved]
    if not ranked:
        return []

    with connection() as conn:
        rows = cited_chunks(conn, CURRENT_USER_ID, ranked)

    rank = {chunk_id: index for index, chunk_id in enumerate(ranked)}
    last = len(ranked)
    found: dict[str, dict] = {}

    for chunk_id, document_id, title, source_uri, page in rows:
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
                "rank": last,
            },
        )
        source["rank"] = min(source["rank"], rank.get(chunk_id, last))
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
            found.items(), key=lambda item: item[1]["rank"]
        )
    ]


# Sync `def`: reading which model searches which document is a blocking
# query, and the chain is assembled before a single token is streamed.
# FastAPI runs this in a worker thread and then streams on the event loop.
@router.post("/chat")
def chat(prompt: Chat):
    """Answer from the user's searchable documents.

    Each document is retrieved through the one model it was given, so this
    reads the grouping per request -- changing a document's model has to take
    effect on the next question, not the next restart.

    The body is an event stream: a `token` per piece of the answer, then
    `sources` naming the documents it was searched against, then `done`.
    """
    with connection() as conn:
        groups = {
            get_model(conn, model_id=model_id): document_ids
            for model_id, document_ids in search_groups(conn, CURRENT_USER_ID).items()
        }

    run, config, retrieved = itemgetter("runnable_with_history", "config", "retrieved")(
        build_chat(CURRENT_USER_ID, groups)
    )

    async def stream():
        # A streaming response commits to 200 before the first token, so a
        # failure here cannot change the status. It goes out as an `error`
        # event instead -- the alternative is a silent truncation that reads
        # like a short answer.
        try:
            async for chunk in run.astream({"question": prompt.prompt}, config=config):
                if chunk:
                    yield sse("token", {"t": chunk})
            # Off the loop: resolving chunks to documents is a blocking query,
            # and this generator runs on the event loop.
            resolved = await run_in_threadpool(sources, retrieved)
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
