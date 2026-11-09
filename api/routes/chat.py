"""Answering questions from the user's searchable documents, as an event stream."""

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from api.auth.users import OptionalUser
from api.services.sources import resolve_sources
from db.pool import connection
from db.tables.chat import get_conversation, message_rows, store_turn
from db.tables.embeddings import search_groups
from db.tables.registry import get_model
from llm.chat import build_graph, drawn_on

logger = logging.getLogger("api")

router = APIRouter()


class Chat(BaseModel):
    prompt: str
    # Null starts a conversation. Either way the id comes back as a
    # `conversation` event, which is what the page sends the next turn into.
    conversation_id: uuid.UUID | None = None


def sse(event: str, data: dict) -> str:
    """One server-sent event.

    The payload is JSON rather than raw text because a newline inside `data:`
    ends the field and one leading space after the colon is framing -- model
    output has both.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def citations_from(retrieved: list[Document], answer: str) -> list[dict]:
    """The documents the answer drew on, in the shape `chat_messages.citations`
    stores.

    Retrieval always returns its closest chunks, however far off they are, so
    only those `drawn_on` keeps count -- a greeting, a question the documents
    do not cover, or an answer saying they did not cover it shows no sources.

    One entry per document, in retrieval order, each carrying the chunks of it
    the answer drew on: those are the pages the reader is shown.

    Ids only. Which of them still resolve to a document, and what that
    document can offer, is `resolve_sources` -- asked again every time the
    answer is rendered rather than frozen here.
    """
    chunks: dict[str, list[str]] = {}
    for document in drawn_on(retrieved, answer):
        document_id = document.metadata.get("document_id")
        if document_id is None:
            continue
        chunks.setdefault(document_id, []).append(
            str(document.metadata.get("chunk_id"))
        )

    return [
        {"document_id": document_id, "chunk_ids": chunk_ids}
        for document_id, chunk_ids in chunks.items()
    ]


def history_messages(rows: list[tuple]) -> list[AnyMessage]:
    """A stored transcript as the graph's message list."""
    return [
        HumanMessage(content) if role == "user" else AIMessage(content)
        for _seq, role, content, _citations in rows
    ]


def store(
    user_id: int | None,
    conversation_id: uuid.UUID | None,
    question: str,
    answer: str,
    retrieved: list[Document],
    summary: str | None,
    summarized: int,
) -> tuple[uuid.UUID | None, list[dict]]:
    """Write the turn, then resolve its sources.

    Both on one connection: the sources a reader is sent then describe rows
    that are committed, rather than rows that could still roll back.

    A signed-out user's turn is not stored, so no conversation id comes back.

    Working out the sources embeds the answer. If that fails the turn is
    stored without sources rather than lost -- the user has already read it.
    """
    try:
        citations = citations_from(retrieved, answer)
    except Exception:
        logger.exception("could not work out the answer's sources")
        citations = []
    with connection() as conn:
        if user_id is None:
            return None, resolve_sources(conn, None, citations)
        stored_id = store_turn(
            conn,
            user_id,
            conversation_id,
            question,
            answer,
            citations,
            summary=summary,
            summarized=summarized,
        )
        return stored_id, resolve_sources(conn, user_id, citations)


# Sync `def`: loading the history and reading which model searches which
# document are blocking queries, and the graph is assembled before a single
# token is streamed. FastAPI runs this in a worker thread and then streams on
# the event loop.
@router.post("/chat")
def chat(prompt: Chat, user_id: OptionalUser):
    """Answer from the user's searchable documents, and store the turn.

    The conversation is loaded from `chat_messages` rather than from a
    checkpointer: stored history is the source of truth, so an answer can be
    given against a conversation this process has never seen.

    Each document is retrieved through the one model it was given, so this
    reads the grouping per request -- changing a document's model has to take
    effect on the next question, not the next restart.

    The body is an event stream: a `token` per piece of the answer, then
    `conversation` naming the conversation the turn landed in, then `sources`
    naming the documents a close enough passage came from, then `done`. Every
    question is searched, so an empty `sources` means nothing retrieved was
    close to the question, or the user has nothing searchable.

    Signed out, only public documents are searched and nothing is stored: no
    `conversation` event is sent, and a `conversation_id` is reported missing.
    """
    with connection() as conn:
        summary: str | None = None
        summarized = 0
        history: list[AnyMessage] = []

        if prompt.conversation_id is not None:
            found = get_conversation(conn, user_id, prompt.conversation_id)
            if found is None:
                raise HTTPException(status_code=404, detail="No such conversation.")
            _id, _title, summary, summarized = found
            history = history_messages(message_rows(conn, prompt.conversation_id))

        groups = {
            get_model(conn, model_id=model_id): document_ids
            for model_id, document_ids in search_groups(conn, user_id).items()
        }

    graph = build_graph(user_id, groups)

    # The whole history goes in, because there is no checkpointer holding it
    # between turns. The compaction state comes with it: `summarized` counts
    # into this list, so the two have to arrive together.
    state = {
        "messages": [*history, HumanMessage(prompt.prompt)],
        "summary": summary,
        "summarized": summarized,
    }

    async def stream():
        # Accumulated only to have something to keep if the run fails partway:
        # the answer itself is read back off the final state, which is the same
        # text.
        streamed: list[str] = []
        final: dict = {}

        # A streaming response commits to 200 before the first token, so a
        # failure here cannot change the status. It goes out as an `error`
        # event instead -- the alternative is a silent truncation that reads
        # like a short answer.
        try:
            # `messages` carries the tokens and `values` the state after each
            # superstep, the last of which is the finished conversation --
            # which is how the answer and its chunks are read back without a
            # checkpointer to ask. `messages` emits every LLM call in the
            # graph, and the summariser is one of them: its notes are not the
            # answer.
            async for mode, payload in graph.astream(
                state, stream_mode=["messages", "values"]
            ):
                if mode == "values":
                    final = payload
                    continue
                chunk, metadata = payload
                if metadata["langgraph_node"] == "answer" and chunk.text:
                    streamed.append(chunk.text)
                    yield sse("token", {"t": chunk.text})

            # Off the loop: storing the turn and resolving its chunks to
            # documents are blocking queries, and this generator runs on the
            # event loop.
            stored_id, resolved = await run_in_threadpool(
                store,
                user_id,
                prompt.conversation_id,
                prompt.prompt,
                final["messages"][-1].text,
                final.get("context", []),
                final.get("summary"),
                final.get("summarized", 0),
            )
            if stored_id is not None:
                yield sse("conversation", {"conversation_id": str(stored_id)})
            yield sse("sources", {"sources": resolved})
        except Exception:
            logger.exception("chat stream failed")
            # Whatever arrived is kept, as the browser keeps it: a half
            # answered turn is still a turn the user asked, and dropping it
            # would leave the page showing an exchange the server denies.
            # The compaction state stored is the one loaded, not the one in
            # flight -- the run that would have produced a new one is what
            # just failed.
            partial = "".join(streamed)
            if partial and user_id is not None:
                try:
                    stored_id, _ = await run_in_threadpool(
                        store,
                        user_id,
                        prompt.conversation_id,
                        prompt.prompt,
                        partial,
                        # No sources for a broken-off answer, so no second
                        # call to the service that may be what just failed.
                        [],
                        summary,
                        summarized,
                    )
                    yield sse("conversation", {"conversation_id": str(stored_id)})
                except Exception:
                    logger.exception("could not store the partial turn")
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
