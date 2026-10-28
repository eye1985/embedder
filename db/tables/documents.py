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
from datetime import datetime

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


def restore_document(
    conn: psycopg.Connection,
    user_id: int,
    document_id: uuid.UUID,
    chunks: Iterable[tuple[uuid.UUID, int, str, dict]],
    *,
    title: str | None = None,
    source_uri: str | None = None,
    created_at: datetime,
) -> None:
    """Store a document and its chunks under ids they already have.

    The counterpart to `add_document`, for rows being put back rather than
    created: an export carries its own document id, chunk ids, chunk order and
    timestamp, and restoring it keeps all four -- otherwise the file no longer
    describes what is in the database.

    `searchable` is left at the column default, which is where a fresh upload
    starts too. `origin` is not: the file carries chunks and vectors but no
    PDF, so this document has nothing to offer for download no matter what
    its `source_uri` says, and recording that is how a chat answer can say so
    rather than infer it from a file that happens to be missing.
    """
    conn.execute(
        "insert into documents "
        "(id, user_id, title, source_uri, origin, created_at) "
        "values (%s, %s, %s, %s, 'import', %s)",
        (document_id, user_id, title, source_uri, created_at),
    )

    for chunk_id, chunk_index, content, metadata in chunks:
        conn.execute(
            "insert into document_chunks "
            "(id, document_id, chunk_index, content, metadata) "
            "values (%s, %s, %s, %s, %s)",
            (chunk_id, document_id, chunk_index, content, json.dumps(metadata or {})),
        )


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


def cited_chunks(
    conn: psycopg.Connection, user_id: int, chunk_ids: list[str]
) -> list[tuple]:
    """Retrieved chunks resolved to their documents.

    Returns (chunk_id, document_id, title, source_uri, origin, page). `page`
    is left as the extractor stored it, which is 0-indexed, and `origin` says
    whether a PDF was ever stored for this document at all.

    Scoped to the user, so a chunk_id belonging to someone else returns no row
    rather than its document's title. Only the page is read out of `metadata`:
    the rest of it holds absolute local paths.
    """
    if not chunk_ids:
        return []
    return conn.execute(
        """
        select c.id::text, d.id::text, d.title, d.source_uri, d.origin,
               (c.metadata->>'page')::int
        from document_chunks c
        join documents d on d.id = c.document_id
        where c.id = any(%s::uuid[]) and d.user_id = %s
        """,
        (chunk_ids, user_id),
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


def insert_vectors(
    conn: psycopg.Connection,
    table_name: str,
    rows: Iterable[tuple[uuid.UUID, int, uuid.UUID, str, list[float]]],
) -> int:
    """Write vectors straight into a model's table. Returns how many landed.

    The one path into a vector table that does not go through PGVectorStore,
    because restoring an export has nothing to embed -- the vectors are in the
    file. `rows` is (chunk_id, user_id, document_id, content, vector).

    `content` is duplicated from the chunk and `langchain_metadata` is an empty
    object, which is exactly what the store writes for these columns.

    The vector is bound as pgvector's own text form and cast; a halfvec column
    takes the same literal through the implicit cast from vector.

    `table_name` comes from the registry, never from a request.
    """
    written = 0
    for chunk_id, user_id, document_id, content, vector in rows:
        conn.execute(
            f"""
            insert into "{table_name}"
                (content, embedding, chunk_id, user_id, document_id,
                 langchain_metadata)
            values (%s, %s::vector, %s, %s, %s, '{{}}'::jsonb)
            """,
            (
                content,
                json.dumps(vector, separators=(",", ":")),
                chunk_id,
                user_id,
                document_id,
            ),
        )
        written += 1
    return written


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
