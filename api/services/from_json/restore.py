"""Writing a downloaded export back into the database.

The file is parsed and checked by `parse.py`, which knows nothing about
Postgres. What is left is the half that does: matching the file's models
against this installation's registry, noticing that the document is already
here, and writing the rows.

Everything lands on one connection, so document, chunks and vectors commit
together or not at all. That is possible here and not on the upload path
because nothing has to be embedded -- the vectors arrive in the file, so
PGVectorStore and its separate pool are not involved.

Scoped to a single user_id, like every other document service. The `user_id`
in the file is ignored: a restore puts the document in the library of whoever
uploads it.
"""

import uuid

import psycopg

from api.services.from_json.parse import ExportedDocument, ExportedModel
from db.tables.documents import (
    count_chunks,
    find_document,
    get_document,
    insert_vectors,
    restore_document,
)
from db.tables.documents import delete_document as delete_document_rows
from db.tables.embeddings import mark_embedded, set_document_model
from db.tables.registry import EmbeddingModel, get_model, table_exists


class CannotRestore(Exception):
    """The file is readable, but this installation cannot hold it as it is."""


def _registered(
    conn: psycopg.Connection, exported: list[ExportedModel]
) -> list[EmbeddingModel]:
    """The registry rows the file's models name, in the file's order.

    Checked before anything is written, because the alternative is vectors in
    a table that cannot interpret them. A model the registry does not have
    raises LookupError; one that has been redefined since the export, or has
    no vector table yet, raises CannotRestore.
    """
    models = []
    for entry in exported:
        # LookupError on an unknown table name, which the endpoint reports as
        # a 404 the same way an unknown model elsewhere is.
        model = get_model(conn, table_name=entry.table_name)

        if not table_exists(conn, model.table_name):
            raise CannotRestore(
                f"No vector table for {model.table_name}. "
                f"Create it with: uv run python -m db.setup"
            )
        if (model.dimensions, model.distance) != (entry.dimensions, entry.distance):
            raise CannotRestore(
                f"{model.table_name} holds {model.dimensions} dimensions at "
                f"{model.distance} distance, but the file's vectors are "
                f"{entry.dimensions} at {entry.distance}."
            )
        models.append(model)
    return models


def conflicting_document(
    conn: psycopg.Connection, user_id: int, exported: ExportedDocument
) -> dict | None:
    """The user's document this file would collide with, or None.

    Two ways to already have it. The same id means the file came from here.
    The same `source_uri` means a different id over the same PDF -- that is
    the key uploads dedupe on, so letting both exist would leave two documents
    claiming one file.
    """
    candidates = [(exported.document_id, "id")]
    if exported.source_uri:
        same_source = find_document(conn, user_id, exported.source_uri)
        if same_source is not None:
            candidates.append((same_source, "source_uri"))

    for document_id, matched in candidates:
        try:
            found, _, title, source_uri, _ = get_document(conn, user_id, document_id)
        except LookupError:
            continue
        return {
            "document_id": found,
            "title": title,
            "source_uri": source_uri,
            "matched": matched,
        }
    return None


def restore_export(
    conn: psycopg.Connection,
    user_id: int,
    exported: ExportedDocument,
    *,
    replace: str | None = None,
) -> dict:
    """Write the file's document, chunks and vectors.

    `replace` is the id of a document to drop first -- what overwriting means.
    Its chunks and vectors cascade with it; the PDF it was imported from is
    left on disk, because the file being restored has no replacement for it.

    Which model the document is searched with is the first the file lists,
    which is the lowest-numbered model that embedded it. A file carrying no
    vectors at all restores its chunks and is searched by nothing, which is
    the same state deleting a document's embeddings leaves behind.
    """
    models = _registered(conn, exported.models)

    if replace is not None:
        delete_document_rows(conn, user_id, uuid.UUID(replace))

    restore_document(
        conn,
        user_id,
        exported.document_id,
        [(c.chunk_id, c.chunk_index, c.content, c.metadata) for c in exported.chunks],
        title=exported.title,
        source_uri=exported.source_uri,
        created_at=exported.created_at,
    )

    embedded = {}
    for model in models:
        rows = [
            (
                chunk.chunk_id,
                user_id,
                exported.document_id,
                chunk.content,
                chunk.embeddings[model.table_name],
            )
            for chunk in exported.chunks
            if model.table_name in chunk.embeddings
        ]
        embedded[model.table_name] = insert_vectors(conn, model.table_name, rows)
        if rows:
            mark_embedded(conn, exported.document_id, model.id)

    searched = next((m for m in models if embedded[m.table_name]), None)
    if searched is not None:
        set_document_model(conn, user_id, exported.document_id, searched.id)

    return {
        "document_id": str(exported.document_id),
        "title": exported.title,
        "source_uri": exported.source_uri,
        "chunks": count_chunks(conn, exported.document_id),
        "embedded": embedded,
        "search_model": searched.table_name if searched else None,
        "replaced_existing": replace is not None,
    }
