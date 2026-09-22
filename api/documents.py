"""Reading, exporting and deleting a user's documents.

Vectors live in one table per registered model, so an export is a fan-out: a
chunk's text and metadata are read once from `document_chunks`, and each model's
table contributes its own vector for that chunk. The exported shape follows
that -- content stored once, `embeddings` keyed by table name -- rather than one
file per model, which would repeat the text for every model the user has.

Everything here is scoped to a single user_id. A document that is not theirs is
reported as missing rather than refused, so the endpoint cannot be used to test
which document ids exist.
"""

import io
import json
import re
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from add_docs import count_chunks
from api.ingest import PROJECT_ROOT, UPLOAD_DIR
from db import table_exists
from registry import EmbeddingModel, list_models

EXPORT_VERSION = 1

# Keeps archive entries readable without letting a long PDF title dominate the
# path. The document id in front is what actually makes the name unique.
MAX_TITLE_SLUG = 60


def _model_info(model: EmbeddingModel, *, registry_fields: bool = False) -> dict:
    """The model's identity as exported. `dimensions` is what a reader needs to
    interpret a vector; `distance` is what it was indexed for."""
    info = {
        "provider": model.provider,
        "model_name": model.model_name,
        "dimensions": model.dimensions,
        "distance": model.distance,
    }
    if registry_fields:
        # Only the manifest carries these -- they describe the registry row, not
        # the vectors, and are noise inside every document file.
        return {"id": model.id, **info, "is_active": model.is_active}
    return info


def _readable_models(conn: psycopg.Connection) -> list[EmbeddingModel]:
    """Registered models whose vector table actually exists.

    `list_models(active_only=False)` includes rows that were never set up --
    `setup_db.py` creates tables for active models only -- and querying a
    missing table raises, so those rows are dropped before any count or read.
    """
    return [
        m
        for m in list_models(conn, active_only=False)
        if table_exists(conn, m.table_name)
    ]


def _clean_metadata(metadata: dict | None) -> dict:
    """Rewrite extractor paths relative to the project root.

    PyMuPDF4LLM records `source` and `file_path` as absolute paths on the
    machine that ran the ingest, so exporting them verbatim would put the
    server's directory layout into a file the user downloads. Every other key
    is passed through untouched, so page numbers and titles survive a
    round-trip and an export can still be cited.
    """
    cleaned = dict(metadata or {})
    for key in ("source", "file_path"):
        value = cleaned.get(key)
        if not isinstance(value, str):
            continue
        try:
            cleaned[key] = str(Path(value).resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            # Ingested from outside the repo: keep the filename, drop the path.
            cleaned[key] = Path(value).name
    return cleaned


def _embedded_counts(
    conn: psycopg.Connection, user_id: int, models: list[EmbeddingModel]
) -> dict[str, dict[str, int]]:
    """Per model, how many chunks of each of the user's documents it embedded.

    One query per model rather than a scalar subquery per model per document:
    there are only a handful of models, and this keeps each query readable.
    """
    counts: dict[str, dict[str, int]] = {}
    for model in models:
        # The table name comes from the registry, not from the request.
        rows = conn.execute(
            f"""
            select c.document_id::text, count(*)
            from "{model.table_name}" v
            join document_chunks c on c.id = v.chunk_id
            join documents d on d.id = c.document_id
            where d.user_id = %s
            group by c.document_id
            """,
            (user_id,),
        ).fetchall()
        counts[model.table_name] = dict(rows)
    return counts


def list_documents(conn: psycopg.Connection, user_id: int) -> list[dict]:
    """The user's documents, newest first, with chunk and embedding counts.

    Every registered model appears in `embedded`, including ones with no table
    yet -- a zero there means "not embedded by this model", which is the
    question the caller is asking.

    `id` breaks ties on `created_at` so the order is stable: two documents
    ingested in the same transaction share a timestamp, and an unstable sort
    would shuffle them between loads.
    """
    models = list_models(conn, active_only=False)
    counts = _embedded_counts(conn, user_id, _readable_models(conn))

    rows = conn.execute(
        """
        select d.id::text, d.title, d.source_uri, d.created_at, count(c.id)
        from documents d
        left join document_chunks c on c.document_id = d.id
        where d.user_id = %s
        group by d.id
        order by d.created_at desc, d.id
        """,
        (user_id,),
    ).fetchall()

    return [
        {
            "document_id": document_id,
            "title": title,
            "source_uri": source_uri,
            "created_at": created_at.isoformat(),
            "chunks": chunks,
            "embedded": {
                m.table_name: counts.get(m.table_name, {}).get(document_id, 0)
                for m in models
            },
        }
        for document_id, title, source_uri, created_at, chunks in rows
    ]


def export_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> dict:
    """One document with its chunks and every vector held for them.

    Raises LookupError if the document does not exist or belongs to someone else.
    """
    row = conn.execute(
        "select id::text, user_id, title, source_uri, created_at "
        "from documents where id = %s and user_id = %s",
        (document_id, user_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"No document {document_id} for this user.")
    doc_id, owner_id, title, source_uri, created_at = row

    chunks = conn.execute(
        "select id::text, chunk_index, content, metadata from document_chunks "
        "where document_id = %s order by chunk_index",
        (document_id,),
    ).fetchall()

    # `embedding::text` is pgvector's own text form, "[0.1,-0.2]" -- already
    # valid JSON, so the floats come back without registering a type adapter.
    vectors: dict[str, dict[str, list[float]]] = {}
    models: dict[str, EmbeddingModel] = {}
    for model in _readable_models(conn):
        rows = conn.execute(
            f"""
            select v.chunk_id::text, v.embedding::text
            from "{model.table_name}" v
            join document_chunks c on c.id = v.chunk_id
            where c.document_id = %s
            """,
            (document_id,),
        ).fetchall()
        if not rows:
            # A model that never embedded this document is left out entirely,
            # rather than adding a column of nulls to every chunk.
            continue
        vectors[model.table_name] = {
            chunk_id: json.loads(vector) for chunk_id, vector in rows
        }
        models[model.table_name] = model

    return {
        "export_version": EXPORT_VERSION,
        "document": {
            "id": doc_id,
            "user_id": owner_id,
            "title": title,
            "source_uri": source_uri,
            "created_at": created_at.isoformat(),
        },
        "models": {name: _model_info(m) for name, m in models.items()},
        "chunks": [
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "content": content,
                "metadata": _clean_metadata(metadata),
                # null, not a missing key: a model that embedded this document
                # but not this chunk is a real state worth seeing.
                "embeddings": {
                    name: by_chunk.get(chunk_id) for name, by_chunk in vectors.items()
                },
            }
            for chunk_id, chunk_index, content, metadata in chunks
        ],
    }


def document_filename(document_id: str, title: str | None) -> str:
    """Archive path for one document.

    `documents.title` has no uniqueness constraint -- /ingest takes it from the
    PDF stem, so two files named report.pdf in different folders would collide.
    The id prefix is what keeps entries distinct; the slug is for humans.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title or "").strip("-.").lower()
    slug = slug[:MAX_TITLE_SLUG].strip("-.") or "untitled"
    return f"documents/{document_id}-{slug}.json"


def build_archive(conn: psycopg.Connection, user_id: int) -> tuple[bytes, str]:
    """Every document the user has, zipped, with a manifest. Returns (zip, filename).

    Built in memory: deflate does well on vector text, but the whole archive is
    still held at once. Documents sized in the thousands of chunks should move
    this to a temp file served by FileResponse.
    """
    user = conn.execute(
        "select id, email from users where id = %s", (user_id,)
    ).fetchone()
    if user is None:
        raise LookupError(f"No user with id {user_id}.")

    documents = list_documents(conn, user_id)
    models = list_models(conn, active_only=False)
    exported_at = datetime.now(UTC)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        entries = []
        for document in documents:
            payload = export_document(conn, user_id, uuid.UUID(document["document_id"]))
            name = document_filename(document["document_id"], document["title"])
            archive.writestr(name, json.dumps(payload, ensure_ascii=False))
            entries.append({**document, "file": name})

        manifest = {
            "export_version": EXPORT_VERSION,
            "exported_at": exported_at.isoformat(),
            "user": {"id": user[0], "email": user[1]},
            "models": {
                m.table_name: _model_info(m, registry_fields=True) for m in models
            },
            "documents": entries,
        }
        # Indented: the manifest is the file a person opens first.
        archive.writestr(
            "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False)
        )

    filename = f"embeddings-{user_id}-{exported_at:%Y%m%d}.zip"
    return buffer.getvalue(), filename


def _delete_stored_pdf(source_uri: str | None) -> bool:
    """Remove the stored PDF behind a document, if the user uploaded it.

    Only `uploads/` is cleaned. `sample/` holds the repo's own PDFs, which are
    not a user's to delete -- so deleting a document ingested from there
    removes the rows and leaves the file.
    """
    if not source_uri:
        return False
    path = (PROJECT_ROOT / source_uri).resolve()
    if not path.is_relative_to(UPLOAD_DIR) or not path.is_file():
        return False
    path.unlink()
    return True


def _owned_document(conn: psycopg.Connection, user_id: int, document_id: uuid.UUID):
    """The document's (title, source_uri), or LookupError if it is not theirs."""
    row = conn.execute(
        "select title, source_uri from documents where id = %s and user_id = %s",
        (document_id, user_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"No document {document_id} for this user.")
    return row


def delete_embeddings(
    conn: psycopg.Connection,
    user_id: int,
    document_id: uuid.UUID,
    table_name: str | None = None,
) -> dict:
    """Delete a document's vectors, keeping the document and its chunks.

    The chunks are what extraction produced, and they are model-independent, so
    the text survives and `embed_document` could fill the vectors back in from
    it. Note that no HTTP route does that today: /upload refuses a filename
    that already has a document, so a document emptied this way can only be
    re-embedded from Python until a re-embed route exists.
    """
    title, source_uri = _owned_document(conn, user_id, document_id)

    models = _readable_models(conn)
    if table_name is not None:
        models = [m for m in models if m.table_name == table_name]
        if not models:
            raise LookupError(f"No vector table named {table_name!r}.")

    removed = {}
    for model in models:
        # The table name comes from the registry, not from the request.
        cursor = conn.execute(
            f"""
            delete from "{model.table_name}" v
            using document_chunks c
            where v.chunk_id = c.id and c.document_id = %s
            """,
            (document_id,),
        )
        removed[model.table_name] = cursor.rowcount

    return {
        "document_id": str(document_id),
        "title": title,
        "source_uri": source_uri,
        "deleted_embeddings": removed,
        "chunks_kept": count_chunks(conn, document_id),
    }


def delete_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> dict:
    """Delete a document, its chunks, its vectors, and its uploaded PDF.

    One statement does the database half: `document_chunks` cascades from
    `documents`, and every vector table cascades from `document_chunks`, so the
    vectors go with the row that owns them.
    """
    title, source_uri = _owned_document(conn, user_id, document_id)

    # Counted before the delete -- afterwards there is nothing left to count.
    chunks = count_chunks(conn, document_id)
    counts = _embedded_counts(conn, user_id, _readable_models(conn))
    removed = {
        table: by_document.get(str(document_id), 0)
        for table, by_document in counts.items()
    }

    conn.execute(
        "delete from documents where id = %s and user_id = %s", (document_id, user_id)
    )

    return {
        "document_id": str(document_id),
        "title": title,
        "source_uri": source_uri,
        "deleted_chunks": chunks,
        "deleted_embeddings": removed,
        "deleted_file": _delete_stored_pdf(source_uri),
    }
