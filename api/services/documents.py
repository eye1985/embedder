"""The shapes the document endpoints return.

A thin layer: it reads rows through `db/tables/documents.py`, serializes through
`export.py`, and assembles the dicts FastAPI hands back. No SQL and no export
format live here.

Everything is scoped to a single user_id. A document that is not theirs is
reported as missing rather than refused, so the endpoint cannot be used to test
which document ids exist.
"""

import uuid

import psycopg

from api.services.uploads import UPLOAD_DIR
from db.tables.documents import (
    count_chunks,
    delete_vectors,
    document_rows,
    embedded_counts,
    get_document,
)
from db.tables.documents import delete_document as delete_document_rows
from db.tables.embeddings import models_by_document, unmark_embedded
from db.tables.registry import list_models, models_with_tables
from export import build_archive, export_document
from indexing.embed import embed_document
from paths import PROJECT_ROOT


def list_documents(conn: psycopg.Connection, user_id: int) -> list[dict]:
    """The user's documents, newest first, with chunk counts and search state.

    Every registered model appears in `embedded`, including ones with no table
    yet -- a zero there means "not embedded by this model", which is the
    question the caller is asking.

    `search_model` names the one model the document is retrieved through, or
    is null if it has none -- which is also what a document nothing has
    embedded looks like. `searchable` is the document-level switch that
    overrides it either way.
    """
    models = list_models(conn, active_only=False)
    readable = [m.table_name for m in models_with_tables(conn)]
    counts = embedded_counts(conn, user_id, readable)
    flags = models_by_document(conn, user_id)
    table_names = {m.id: m.table_name for m in models}

    return [
        {
            "document_id": document_id,
            "title": title,
            "source_uri": source_uri,
            "created_at": created_at.isoformat(),
            "searchable": searchable,
            "chunks": chunks,
            "embedded": {
                m.table_name: counts.get(m.table_name, {}).get(document_id, 0)
                for m in models
            },
            "search_model": next(
                (
                    table_names[model_id]
                    for model_id, searched in flags.get(document_id, {}).items()
                    if searched and model_id in table_names
                ),
                None,
            ),
        }
        for (
            document_id,
            title,
            source_uri,
            created_at,
            searchable,
            chunks,
        ) in document_rows(conn, user_id)
    ]


def export_archive(conn: psycopg.Connection, user_id: int) -> tuple[bytes, str]:
    """Zip of every document the user has. Returns (zip, filename).

    The listing is built here and handed to the archive so the manifest's
    per-document summary and the /documents page cannot disagree.
    """
    return build_archive(conn, user_id, list_documents(conn, user_id))


def _delete_stored_pdf(source_uri: str | None) -> bool:
    """Remove the stored PDF behind a document, if the user uploaded it.

    Only `uploads/` is cleaned. `sample/` holds the repo's own PDFs, which are
    not a user's to delete -- so deleting a document imported from there
    removes the rows and leaves the file.
    """
    if not source_uri:
        return False
    path = (PROJECT_ROOT / source_uri).resolve()
    if not path.is_relative_to(UPLOAD_DIR) or not path.is_file():
        return False
    path.unlink()
    return True


def delete_embeddings(
    conn: psycopg.Connection,
    user_id: int,
    document_id: uuid.UUID,
    table_name: str | None = None,
) -> dict:
    """Delete a document's vectors, keeping the document and its chunks.

    The chunks are what extraction produced, and they are model-independent, so
    the text survives and `embed_existing` can fill the vectors back in from it.

    Prefer the `enabled` toggle for taking a model out of search: that is a
    boolean, while this spends the embedding API again to undo.
    """
    _, _, title, source_uri, _ = get_document(conn, user_id, document_id)

    models = models_with_tables(conn)
    if table_name is not None:
        models = [m for m in models if m.table_name == table_name]
        if not models:
            raise LookupError(f"No vector table named {table_name!r}.")

    removed = {}
    for model in models:
        removed[model.table_name] = delete_vectors(conn, document_id, model.table_name)
        # The pairing claims vectors that are gone, so it goes with them --
        # back to "never embedded", which is the state that is now true.
        unmark_embedded(conn, document_id, model.id)

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
    """Delete a document, its chunks, its vectors, and its uploaded PDF."""
    _, _, title, source_uri, _ = get_document(conn, user_id, document_id)

    # Counted before the delete -- afterwards there is nothing left to count.
    chunks = count_chunks(conn, document_id)
    readable = [m.table_name for m in models_with_tables(conn)]
    counts = embedded_counts(conn, user_id, readable)
    removed = {
        table: by_document.get(str(document_id), 0)
        for table, by_document in counts.items()
    }

    delete_document_rows(conn, user_id, document_id)

    return {
        "document_id": str(document_id),
        "title": title,
        "source_uri": source_uri,
        "deleted_chunks": chunks,
        "deleted_embeddings": removed,
        "deleted_file": _delete_stored_pdf(source_uri),
    }


def embed_existing(
    conn: psycopg.Connection,
    user_id: int,
    document_id: uuid.UUID,
    models: list,
) -> dict:
    """Embed a stored document's chunks with each of `models`.

    The counterpart to delete_embeddings, and what makes adding a model to an
    existing document possible at all -- /upload refuses a filename that
    already has a document, so it cannot be the route for this.

    Chunks a model already holds are skipped, so this is safe to repeat.
    """
    _, _, title, source_uri, _ = get_document(conn, user_id, document_id)

    embedded = {m.table_name: embed_document(conn, document_id, m) for m in models}

    return {
        "document_id": str(document_id),
        "title": title,
        "source_uri": source_uri,
        "embedded": embedded,
        "chunks": count_chunks(conn, document_id),
    }


__all__ = [
    "delete_document",
    "delete_embeddings",
    "embed_existing",
    "export_archive",
    "export_document",
    "list_documents",
]
