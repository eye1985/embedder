"""Reading stored conversations back out.

The writing half is on the chat path -- a turn is stored by the request that
answered it (`api/routes/chat.py`). This is what the chat page asks for when
it opens: the list to choose from, and one transcript to render.
"""

import uuid

import psycopg

from api.services.sources import resolve_each
from db.tables.chat import conversation_rows, get_conversation, message_rows


def list_conversations(conn: psycopg.Connection, user_id: int) -> list[dict]:
    """The user's conversations, most recently used first."""
    return [
        {
            "conversation_id": conversation_id,
            "title": title,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "messages": messages,
        }
        for conversation_id, title, created_at, updated_at, messages in (
            conversation_rows(conn, user_id)
        )
    ]


def read_conversation(
    conn: psycopg.Connection, user_id: int, conversation_id: uuid.UUID
) -> dict | None:
    """One conversation with its messages, or None if the user has no such row.

    Every answer's citations are resolved as the live path resolves them, so a
    reopened conversation says exactly what it said when it was new -- and
    says it about the documents as they stand now, not as they stood then. A
    document deleted since drops out; one deleted and restored from a JSON
    export comes back, as the import it now is.
    """
    found = get_conversation(conn, user_id, conversation_id)
    if found is None:
        return None
    _id, title, _summary, _summarized = found

    rows = message_rows(conn, conversation_id)
    sources = resolve_each(conn, user_id, [citations for *_, citations in rows])

    return {
        "conversation_id": str(conversation_id),
        "title": title,
        "messages": [
            {
                "seq": seq,
                "role": role,
                "content": content,
                "sources": resolved,
            }
            for (seq, role, content, _citations), resolved in zip(
                rows, sources, strict=True
            )
        ],
    }
