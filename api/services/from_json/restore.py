"""Writing a downloaded export back into the database.

The file is parsed and checked by `parse.py`, which knows nothing about
Postgres. What is left is the half that does: matching the file's models
against this installation's registry, noticing that the document is already
here, and writing the rows.

Everything lands on one connection, so document, chunks and vectors commit
together or not at all. Nothing has to be embedded -- the vectors arrive in
the file.

Scoped to a single user_id, like every other document service: a restore puts
the document in the library of whoever uploads it.
"""

import dataclasses
import uuid

import psycopg

from api.services.from_json.parse import ExportedDocument, ExportedModel
from db.tables.chunk_embeddings import insert_vectors
from db.tables.documents import (
    count_chunks,
    document_id_taken,
    find_document,
    get_document,
    restore_document,
)
from db.tables.documents import delete_document as delete_document_rows
from db.tables.embeddings import mark_embedded, set_document_model
from db.tables.registry import EmbeddingModel, get_model


class CannotRestore(Exception):
    """The file is readable, but this installation cannot hold it as it is."""


def _registered(
    conn: psycopg.Connection, exported: list[ExportedModel]
) -> list[tuple[str, EmbeddingModel]]:
    """The registry rows the file's models name, with the file's label for
    each, in the file's order.

    Checked before anything is written, because the alternative is vectors
    nothing can interpret. A model the registry does not have raises
    LookupError; one indexed for a different distance raises CannotRestore.
    """
    models = []
    for entry in exported:
        # LookupError on an unknown model, which the endpoint reports as a 404
        # the same way an unknown model elsewhere is.
        model = get_model(
            conn,
            provider=entry.provider,
            model_name=entry.model_name,
            dimensions=entry.dimensions,
        )
        if model.distance != entry.distance:
            raise CannotRestore(
                f"{model.label} is indexed for {model.distance} distance, but "
                f"the file's vectors are for {entry.distance}."
            )
        models.append((entry.label, model))
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


def _with_fresh_ids(exported: ExportedDocument) -> ExportedDocument:
    """The same document under new document and chunk ids. Vectors travel
    with their chunk, so they follow."""
    return dataclasses.replace(
        exported,
        document_id=uuid.uuid4(),
        chunks=[
            dataclasses.replace(chunk, chunk_id=uuid.uuid4())
            for chunk in exported.chunks
        ],
    )


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

    The file's ids are kept unless another user's document already has them --
    an export handed from one user to another while the original still
    exists. Then the copy gets fresh ids.
    """
    models = _registered(conn, exported.models)

    if replace is not None:
        delete_document_rows(conn, user_id, uuid.UUID(replace))

    # The user's own document with this id was reported as a conflict or
    # deleted above, so one still here is someone else's.
    if document_id_taken(conn, exported.document_id):
        exported = _with_fresh_ids(exported)

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
    for label, model in models:
        rows = [
            (chunk.chunk_id, chunk.embeddings[label])
            for chunk in exported.chunks
            if label in chunk.embeddings
        ]
        embedded[model.label] = insert_vectors(conn, model, exported.document_id, rows)
        if rows:
            mark_embedded(conn, exported.document_id, model.id)

    searched = next((m for _, m in models if embedded[m.label]), None)
    if searched is not None:
        set_document_model(conn, user_id, exported.document_id, searched.id)

    return {
        "document_id": str(exported.document_id),
        "title": exported.title,
        "source_uri": exported.source_uri,
        "chunks": count_chunks(conn, exported.document_id),
        "embedded": embedded,
        "search_model": searched.label if searched else None,
        "replaced_existing": replace is not None,
    }
