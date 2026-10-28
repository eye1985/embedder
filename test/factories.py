"""Builders for the rows a test needs, and readers for checking the result.

Each builder commits. That is not incidental: the code under test reads
through its own connections, so a row left uncommitted here would be invisible
to it and the test would be checking nothing.

The last two go through the API rather than the database, because what they
stand for is a user downloading a file and uploading it again.
"""

import json
import uuid

import psycopg

from api.config import CURRENT_USER_ID
from db.tables.documents import add_document


def a_document(
    conn: psycopg.Connection,
    *,
    title: str = "Report",
    chunks: int = 3,
    source_uri: str | None = None,
    metadata: dict | None = None,
    user_id: int = CURRENT_USER_ID,
) -> uuid.UUID:
    """A stored document with `chunks` chunks of distinct text, and no vectors.

    The text differs per chunk so the fake embedder produces distinct vectors,
    which is what lets a test tell "155 vectors" from "one vector 155 times".

    `metadata` is merged into every chunk's, for the tests that need what a
    real extractor leaves behind rather than just a page number.
    """
    document_id = add_document(
        conn,
        user_id,
        [
            (f"{title} chunk {i} text", {"page": i, **(metadata or {})})
            for i in range(chunks)
        ],
        title=title,
        source_uri=source_uri if source_uri is not None else f"uploads/{title}.pdf",
    )
    conn.commit()
    return document_id


def a_job(
    conn: psycopg.Connection,
    *,
    status: str = "running",
    kind: str = "upload",
    subject: str = "report.pdf",
    models: tuple[str, ...] = ("emb_openai_te3_small",),
    source_uri: str | None = None,
    document_id: uuid.UUID | None = None,
    user_id: int = CURRENT_USER_ID,
) -> int:
    """A job row in whatever state the test needs.

    A `running` one is how a test stands in for "a job is in flight", or for
    one a crashed process left behind -- the two are indistinguishable in the
    table, which is the problem the jobs page exists to solve.
    """
    finished = "null" if status == "running" else "now()"
    row = conn.execute(
        f"""
        insert into jobs
            (user_id, kind, subject, models, status, source_uri, document_id,
             finished_at)
        values (%s, %s, %s, %s, %s, %s, %s, {finished})
        returning id
        """,
        (
            user_id,
            kind,
            subject,
            json.dumps(list(models)),
            status,
            source_uri,
            document_id,
        ),
    ).fetchone()
    conn.commit()
    return row[0]


def vector_count(
    conn: psycopg.Connection, table_name: str, document_id: uuid.UUID | None = None
) -> int:
    """How many vectors a model holds, for one document or in total."""
    if document_id is None:
        return conn.execute(f'select count(*) from "{table_name}"').fetchone()[0]
    return conn.execute(
        f'select count(*) from "{table_name}" where document_id = %s', (document_id,)
    ).fetchone()[0]


def job_statuses(conn: psycopg.Connection) -> dict[int, str]:
    """Every job's status, by id. Committed state, read fresh."""
    conn.rollback()
    return dict(conn.execute("select id, status from jobs order by id").fetchall())


def enabled_models(conn: psycopg.Connection, document_id: uuid.UUID) -> set[str]:
    """Every model this document is retrieved through.

    A set, not one value, because the invariant worth asserting is that there
    is exactly ONE -- a test reading a single row could not tell "small is
    still the search model" from "small and large are both enabled now and
    this row came back first".

    This is the guarantee adding a model must not disturb: which model answers
    questions about a document is the user's choice, not a side effect of
    embedding it again.
    """
    conn.rollback()
    rows = conn.execute(
        """
        select m.table_name
        from document_embeddings de
        join embedding_models m on m.id = de.model_id
        where de.document_id = %s and de.enabled
        """,
        (document_id,),
    ).fetchall()
    return {table for (table,) in rows}


def embedded_models(conn: psycopg.Connection, document_id: uuid.UUID) -> set[str]:
    """Every model with a `document_embeddings` row for this document."""
    conn.rollback()
    rows = conn.execute(
        """
        select m.table_name
        from document_embeddings de
        join embedding_models m on m.id = de.model_id
        where de.document_id = %s
        """,
        (document_id,),
    ).fetchall()
    return {table for (table,) in rows}


def an_export(client, document_id: uuid.UUID) -> bytes:
    """The JSON a user downloads for one document."""
    response = client.get(f"/api/documents/{document_id}/embeddings")
    assert response.status_code == 200, response.text
    return response.content


def restore_from_json(
    client, data: bytes, *, overwrite: bool = False, name="export.json"
):
    """Upload an export back, the way the documents page does."""
    return client.post(
        "/api/documents/import",
        files={"file": (name, data, "application/json")},
        data={"overwrite": "true"} if overwrite else {},
    )
