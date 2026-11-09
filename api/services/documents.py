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

from api.services.export import build_archive, export_document
from api.services.uploads import user_upload_dir
from db.tables.chunk_embeddings import delete_vectors, embedded_counts
from db.tables.documents import count_chunks, document_rows, get_document
from db.tables.documents import delete_document as delete_document_rows
from db.tables.embeddings import models_by_document, unmark_embedded
from db.tables.registry import list_models
from indexing.embed import embed_document
from paths import PROJECT_ROOT


def list_documents(conn: psycopg.Connection, user_id: int) -> list[dict]:
    """The user's documents, newest first, with chunk counts and search state.

    Every registered model appears in `embedded`, keyed by its label -- a zero
    there means "not embedded by this model", which is the question the caller
    is asking.

    `search_model` names the one model the document is retrieved through, or
    is null if it has none -- which is also what a document nothing has
    embedded looks like. `searchable` is the document-level switch that
    overrides it either way.
    """
    models = list_models(conn, active_only=False)
    counts = embedded_counts(conn, user_id)
    flags = models_by_document(conn, user_id)
    labels = {m.id: m.label for m in models}

    return [
        {
            "document_id": document_id,
            "title": title,
            "source_uri": source_uri,
            "created_at": created_at.isoformat(),
            "searchable": searchable,
            "visibility": visibility,
            "chunks": chunks,
            "embedded": {
                m.label: counts.get(m.id, {}).get(document_id, 0) for m in models
            },
            "search_model": next(
                (
                    labels[model_id]
                    for model_id, searched in flags.get(document_id, {}).items()
                    if searched and model_id in labels
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
            visibility,
            chunks,
        ) in document_rows(conn, user_id)
    ]


def export_archive(conn: psycopg.Connection, user_id: int) -> tuple[bytes, str]:
    """Zip of every document the user has. Returns (zip, filename).

    The listing is built here and handed to the archive so the manifest's
    per-document summary and the /documents page cannot disagree.
    """
    return build_archive(conn, user_id, list_documents(conn, user_id))


def _delete_stored_pdf(source_uri: str | None, user_id: int) -> bool:
    """Remove the stored PDF behind a document, if the user uploaded it.

    Only the user's own upload folder is cleaned. `sample/` holds the repo's
    own PDFs, and another user's folder holds theirs -- a restored export can
    name either -- so deleting such a document removes the rows and leaves
    the file.
    """
    if not source_uri:
        return False
    path = (PROJECT_ROOT / source_uri).resolve()
    if not path.is_relative_to(user_upload_dir(user_id)) or not path.is_file():
        return False
    path.unlink()
    return True


def delete_embeddings(
    conn: psycopg.Connection,
    user_id: int,
    document_id: uuid.UUID,
    model_id: int | None = None,
) -> dict:
    """Delete a document's vectors, keeping the document and its chunks.

    The chunks are what extraction produced, and they are model-independent, so
    the text survives and `embed_existing` can fill the vectors back in from it.

    Prefer the `enabled` toggle for taking a model out of search: that is a
    boolean, while this spends the embedding API again to undo.
    """
    _, _, title, source_uri, _ = get_document(conn, user_id, document_id)

    models = list_models(conn, active_only=False)
    if model_id is not None:
        models = [m for m in models if m.id == model_id]
        if not models:
            raise LookupError(f"No embedding model {model_id}.")

    removed = {}
    for model in models:
        removed[model.label] = delete_vectors(conn, document_id, model.id)
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
    """Delete a document, its chunks, its vectors, and its uploaded PDF.

    Commits.
    """
    _, _, title, source_uri, _ = get_document(conn, user_id, document_id)

    # Counted before the delete -- afterwards there is nothing left to count.
    chunks = count_chunks(conn, document_id)
    counts = embedded_counts(conn, user_id)
    removed = {
        m.label: counts.get(m.id, {}).get(str(document_id), 0)
        for m in list_models(conn, active_only=False)
    }

    delete_document_rows(conn, user_id, document_id)
    # Committed before the file goes, so a failed commit cannot leave rows
    # pointing at a deleted PDF.
    conn.commit()

    return {
        "document_id": str(document_id),
        "title": title,
        "source_uri": source_uri,
        "deleted_chunks": chunks,
        "deleted_embeddings": removed,
        "deleted_file": _delete_stored_pdf(source_uri, user_id),
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

    embedded = {m.label: embed_document(conn, document_id, m) for m in models}

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
