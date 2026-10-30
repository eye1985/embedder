"""The `conversations` and `chat_messages` tables.

Table access only -- no graph, no retrieval, no citation parsing. A turn
arrives here already answered: `api/routes/chat.py` resolves what the answer
cited and hands over the text and the ids.

Stored history is the source of truth for a conversation. The graph runs
without a checkpointer, so a turn loads its history from here and writes the
turn back once the answer is complete.
"""

import json
import uuid

import psycopg

# How much of the first question becomes the conversation's name. Long enough
# to tell two questions apart in a list, short enough not to wrap it.
MAX_TITLE = 60


def title_from(question: str) -> str:
    """A conversation's name, taken from the question that opened it.

    Whitespace is collapsed first: a question typed across several lines would
    otherwise carry its newlines into the list.
    """
    collapsed = " ".join(question.split())
    if len(collapsed) <= MAX_TITLE:
        return collapsed
    return collapsed[: MAX_TITLE - 1].rstrip() + "…"


def get_conversation(
    conn: psycopg.Connection, user_id: int, conversation_id: uuid.UUID
) -> tuple[uuid.UUID, str | None, str | None, int] | None:
    """One of the user's conversations: (id, title, summary, summarized).

    Scoped to the user, so another user's conversation reads as absent rather
    than as theirs.
    """
    return conn.execute(
        "select id, title, summary, summarized from conversations "
        "where id = %s and user_id = %s",
        (conversation_id, user_id),
    ).fetchone()


def conversation_rows(conn: psycopg.Connection, user_id: int) -> list[tuple]:
    """The user's conversations, most recently used first.

    Returns (id, title, created_at, updated_at, messages). The count comes
    from a lateral subquery rather than a group by, so a conversation whose
    turn failed to store still appears -- with zero messages, which is the
    truth about it.
    """
    return conn.execute(
        """
        select c.id::text, c.title, c.created_at, c.updated_at,
               (select count(*) from chat_messages m
                where m.conversation_id = c.id)
        from conversations c
        where c.user_id = %s
        order by c.updated_at desc
        """,
        (user_id,),
    ).fetchall()


def message_rows(
    conn: psycopg.Connection, conversation_id: uuid.UUID
) -> list[tuple[int, str, str, list]]:
    """A conversation's messages in reading order: (seq, role, content, citations).

    `citations` comes back as the stored list -- psycopg reads jsonb into
    Python objects, so no parsing happens here.
    """
    return conn.execute(
        "select seq, role, content, citations from chat_messages "
        "where conversation_id = %s order by seq",
        (conversation_id,),
    ).fetchall()


def store_turn(
    conn: psycopg.Connection,
    user_id: int,
    conversation_id: uuid.UUID | None,
    question: str,
    answer: str,
    citations: list[dict],
    *,
    summary: str | None,
    summarized: int,
) -> uuid.UUID:
    """Write one complete exchange, creating the conversation if it is new.

    A turn is stored whole or not at all, which is why this is one function
    and not four. The question and the answer land together, so history never
    holds a question with no reply -- the next turn would then send two user
    messages in a row, which some providers reject outright.

    The compaction state is written in the same transaction for a related
    reason: `summarized` is a count of messages, and committing it apart from
    the messages it counts would leave it pointing past the end of them.

    `conversation_id` of None starts a conversation, named after this
    question. Returns the id either way, which is how a new one reaches the
    caller.
    """
    if conversation_id is None:
        conversation_id = conn.execute(
            "insert into conversations (user_id, title) values (%s, %s) returning id",
            (user_id, title_from(question)),
        ).fetchone()[0]

    # One statement for both messages: the answer's seq follows the
    # question's, and reading the current maximum once keeps them adjacent.
    next_seq = conn.execute(
        "select coalesce(max(seq), 0) + 1 from chat_messages "
        "where conversation_id = %s",
        (conversation_id,),
    ).fetchone()[0]

    conn.execute(
        "insert into chat_messages "
        "(conversation_id, seq, role, content, citations) "
        "values (%s, %s, 'user', %s, '[]'::jsonb), "
        "       (%s, %s, 'assistant', %s, %s)",
        (
            conversation_id,
            next_seq,
            question,
            conversation_id,
            next_seq + 1,
            answer,
            json.dumps(citations),
        ),
    )

    conn.execute(
        "update conversations "
        "set summary = %s, summarized = %s, updated_at = now() where id = %s",
        (summary, summarized, conversation_id),
    )
    return conversation_id


def rename_conversation(
    conn: psycopg.Connection, user_id: int, conversation_id: uuid.UUID, title: str
) -> bool:
    """Rename one of the user's conversations. False if there was no such row.

    `updated_at` is left alone: it orders the list by when a conversation was
    last talked in, and renaming one is not talking in it.
    """
    return (
        conn.execute(
            "update conversations set title = %s where id = %s and user_id = %s",
            (title, conversation_id, user_id),
        ).rowcount
        > 0
    )


def delete_conversation(
    conn: psycopg.Connection, user_id: int, conversation_id: uuid.UUID
) -> bool:
    """Delete a conversation and its messages. False if there was no such row.

    The messages cascade from the row that owns them. Nothing else hangs off
    a conversation -- the citations inside it are plain ids, not references --
    so no document or chunk is touched.
    """
    return (
        conn.execute(
            "delete from conversations where id = %s and user_id = %s",
            (conversation_id, user_id),
        ).rowcount
        > 0
    )
