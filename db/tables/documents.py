"""The `documents` and `document_chunks` tables.

Table access only -- no extraction, no chunking, no embedding. A document's
text is split before it gets here (`indexing/chunk.py`) and embedded after
(`indexing/embed.py`); this module just reads and writes rows.

Chunks are stored once, independent of any embedding model, which is what lets
the same document be embedded by one model today and another tomorrow without
being re-extracted. The vectors are `db/tables/chunk_embeddings.py`.
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


def document_id_taken(conn: psycopg.Connection, document_id: uuid.UUID) -> bool:
    """Whether any user's document already has this id."""
    return (
        conn.execute("select 1 from documents where id = %s", (document_id,)).fetchone()
        is not None
    )


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
    visibility: str = "private",
) -> uuid.UUID:
    """Store a document and its chunks. No embedding happens here.

    `chunks` is (content, metadata) pairs in reading order -- what
    `indexing.chunk.split_documents` yields. They are numbered as they arrive,
    so `chunk_index` reflects that order.
    """
    document_id = conn.execute(
        "insert into documents (user_id, title, source_uri, visibility) "
        "values (%s, %s, %s, %s) returning id",
        (user_id, title, source_uri, visibility),
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

    Returns (id, title, source_uri, created_at, searchable, visibility, chunks).

    `id` breaks ties on `created_at` so the order is stable: two documents
    imported in the same transaction share a timestamp, and an unstable sort
    would shuffle them between loads.
    """
    return conn.execute(
        """
        select d.id::text, d.title, d.source_uri, d.created_at, d.searchable,
               d.visibility, count(c.id)
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
    conn: psycopg.Connection, user_id: int | None, chunk_ids: list[str]
) -> list[tuple]:
    """Retrieved chunks resolved to their documents.

    Returns (chunk_id, document_id, title, source_uri, origin, page, owned).
    `page` is left as the extractor stored it, which is 0-indexed, `origin`
    says whether a PDF was ever stored for this document at all, and `owned`
    whether it is the user's own. Signed out (`user_id` None), only public
    documents resolve.

    Scoped to what the user may read -- their own documents and public ones --
    so a chunk_id in someone else's private document returns no row rather
    than its title. Only the page is read out of `metadata`: the rest of it
    holds absolute local paths.
    """
    if not chunk_ids:
        return []
    return conn.execute(
        """
        select c.id::text, d.id::text, d.title, d.source_uri, d.origin,
               (c.metadata->>'page')::int, coalesce(d.user_id = %(user)s, false)
        from document_chunks c
        join documents d on d.id = c.document_id
        where c.id = any(%(chunks)s::uuid[])
          and (d.user_id = %(user)s or d.visibility = 'public')
        """,
        {"chunks": chunk_ids, "user": user_id},
    ).fetchall()


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


def set_visibility(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID, visibility: str
) -> None:
    """Open a document to everyone's chat, or keep it to the owner's.

    Raises LookupError if the document is not the user's.
    """
    updated = conn.execute(
        "update documents set visibility = %s where id = %s and user_id = %s",
        (visibility, document_id, user_id),
    ).rowcount
    if not updated:
        raise LookupError(f"No document {document_id} for this user.")


def delete_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> None:
    """Delete a document row, and with it everything that hangs off it.

    One statement does the whole database half: `document_chunks` and
    `chunk_embeddings` cascade from `documents`, so the vectors go with the row
    that owns them. The stored PDF is not touched here.
    """
    conn.execute(
        "delete from documents where id = %s and user_id = %s", (document_id, user_id)
    )
