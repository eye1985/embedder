"""The `documents` and `document_chunks` tables.

Table access only -- no extraction, no chunking, no embedding. A document's
text is split before it gets here (`indexing/chunk.py`) and embedded after
(`indexing/embed.py`); this module just reads and writes rows.

Chunks are stored once, independent of any embedding model, which is what lets
the same document be embedded by OpenAI today and Qwen tomorrow without being
re-extracted.
"""

import json
import uuid
from collections.abc import Iterable

import psycopg


def find_document(
    conn: psycopg.Connection, user_id: int, source_uri: str
) -> uuid.UUID | None:
    """The user's existing document for this source, if it was already stored."""
    row = conn.execute(
        "select id from documents where user_id = %s and source_uri = %s "
        "order by created_at limit 1",
        (user_id, source_uri),
    ).fetchone()
    return row[0] if row else None


def count_chunks(conn: psycopg.Connection, document_id: uuid.UUID) -> int:
    return conn.execute(
        "select count(*) from document_chunks where document_id = %s", (document_id,)
    ).fetchone()[0]


def add_document(
    conn: psycopg.Connection,
    user_id: int,
    chunks: Iterable[tuple[str, dict]],
    *,
    title: str | None = None,
    source_uri: str | None = None,
) -> uuid.UUID:
    """Store a document and its chunks. No embedding happens here.

    `chunks` is (content, metadata) pairs in reading order -- what
    `indexing.chunk.split_documents` yields. They are numbered as they arrive,
    so `chunk_index` reflects that order.
    """
    document_id = conn.execute(
        "insert into documents (user_id, title, source_uri) "
        "values (%s, %s, %s) returning id",
        (user_id, title, source_uri),
    ).fetchone()[0]

    for index, (content, metadata) in enumerate(chunks):
        conn.execute(
            "insert into document_chunks "
            "(document_id, chunk_index, content, metadata) "
            "values (%s, %s, %s, %s)",
            (document_id, index, content, json.dumps(metadata or {})),
        )

    return document_id


def unembedded_chunks(
    conn: psycopg.Connection, document_id: uuid.UUID, table_name: str
) -> list[tuple[uuid.UUID, str, int]]:
    """A document's chunks that `table_name` holds no vector for yet.

    Returns (chunk_id, content, user_id). `user_id` comes along because it is a
    real column on every vector table -- a similarity search filters on columns
    of its own table, so the owner has to travel with the vector.

    Skipping what is already embedded is what makes re-running cheap, and makes
    adding a second model embed only what that model is missing.

    `table_name` is interpolated because an identifier cannot be a bound
    parameter. It comes from the registry, never from a request.
    """
    return conn.execute(
        f"""
        select c.id, c.content, d.user_id
        from document_chunks c
        join documents d on d.id = c.document_id
        where c.document_id = %s
          and not exists (
              select 1 from "{table_name}" v where v.chunk_id = c.id
          )
        order by c.chunk_index
        """,
        (document_id,),
    ).fetchall()


def get_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> tuple:
    """One of the user's documents: (id, user_id, title, source_uri, created_at).

    Raises LookupError if it does not exist *or* belongs to someone else -- the
    two are deliberately indistinguishable, so an endpoint cannot be used to
    test which document ids exist.
    """
    row = conn.execute(
        "select id::text, user_id, title, source_uri, created_at "
        "from documents where id = %s and user_id = %s",
        (document_id, user_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"No document {document_id} for this user.")
    return row


def document_rows(conn: psycopg.Connection, user_id: int) -> list[tuple]:
    """The user's documents with chunk counts, newest first.

    Returns (id, title, source_uri, created_at, searchable, chunks).

    `id` breaks ties on `created_at` so the order is stable: two documents
    imported in the same transaction share a timestamp, and an unstable sort
    would shuffle them between loads.
    """
    return conn.execute(
        """
        select d.id::text, d.title, d.source_uri, d.created_at, d.searchable,
               count(c.id)
        from documents d
        left join document_chunks c on c.document_id = d.id
        where d.user_id = %s
        group by d.id
        order by d.created_at desc, d.id
        """,
        (user_id,),
    ).fetchall()


def chunk_rows(conn: psycopg.Connection, document_id: uuid.UUID) -> list[tuple]:
    """A document's chunks in order: (id, chunk_index, content, metadata)."""
    return conn.execute(
        "select id::text, chunk_index, content, metadata from document_chunks "
        "where document_id = %s order by chunk_index",
        (document_id,),
    ).fetchall()


def vector_rows(
    conn: psycopg.Connection, document_id: uuid.UUID, table_name: str
) -> list[tuple]:
    """One model's vectors for a document: (chunk_id, vector) pairs.

    `embedding::text` is pgvector's own text form, "[0.1,-0.2]" -- already
    valid JSON, so the floats come back without registering a type adapter.

    `table_name` comes from the registry, never from a request.
    """
    return conn.execute(
        f"""
        select v.chunk_id::text, v.embedding::text
        from "{table_name}" v
        where v.document_id = %s
        """,
        (document_id,),
    ).fetchall()


def embedded_counts(
    conn: psycopg.Connection, user_id: int, table_names: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Per table, how many chunks of each of the user's documents it embedded.

    One query per model rather than a scalar subquery per model per document:
    there are only a handful of models, and this keeps each query readable.

    Table names come from the registry, never from a request.
    """
    counts: dict[str, dict[str, int]] = {}
    for table_name in table_names:
        rows = conn.execute(
            f"""
            select v.document_id::text, count(*)
            from "{table_name}" v
            where v.user_id = %s
            group by v.document_id
            """,
            (user_id,),
        ).fetchall()
        counts[table_name] = dict(rows)
    return counts


def delete_vectors(
    conn: psycopg.Connection, document_id: uuid.UUID, table_name: str
) -> int:
    """Drop one model's vectors for a document. Returns how many went.

    The chunks survive, so the text does not need re-extracting.
    """
    cursor = conn.execute(
        f'delete from "{table_name}" where document_id = %s', (document_id,)
    )
    return cursor.rowcount


def set_searchable(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID, searchable: bool
) -> None:
    """The document-level master switch, across every model.

    Per-model state in `document_embeddings` is untouched, so switching a
    document back on restores exactly the models it had. Raises LookupError if
    the document is not the user's.
    """
    updated = conn.execute(
        "update documents set searchable = %s where id = %s and user_id = %s",
        (searchable, document_id, user_id),
    ).rowcount
    if not updated:
        raise LookupError(f"No document {document_id} for this user.")


def delete_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> None:
    """Delete a document row, and with it everything that hangs off it.

    One statement does the whole database half: `document_chunks` cascades from
    `documents`, and every vector table cascades from `document_chunks`, so the
    vectors go with the row that owns them. The stored PDF is not touched here.
    """
    conn.execute(
        "delete from documents where id = %s and user_id = %s", (document_id, user_id)
    )


def has_any_vector(
    conn: psycopg.Connection, document_id: uuid.UUID, table_names: Iterable[str]
) -> bool:
    """Whether any of these models holds a vector for this document.

    Every model is checked, not just the default: an upload can name several,
    and one that got through means the import was not wasted.

    Table names come from the registry, never from a request.
    """
    for table_name in table_names:
        found = conn.execute(
            f'select 1 from "{table_name}" where document_id = %s limit 1',
            (document_id,),
        ).fetchone()
        if found is not None:
            return True
    return False
