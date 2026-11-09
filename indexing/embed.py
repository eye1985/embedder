"""Embedding a document's chunks with one model."""

import uuid

import psycopg

from db import vectors
from db.tables.chunk_embeddings import insert_vectors, unembedded_chunks
from db.tables.embeddings import mark_embedded
from db.tables.registry import EmbeddingModel


def embed_document(
    conn: psycopg.Connection,
    document_id: uuid.UUID,
    model: EmbeddingModel,
) -> int:
    """Embed a document's chunks with `model`. Returns how many were embedded.

    Chunks already embedded by this model are skipped, so re-running is cheap
    and adding a second model only embeds what that model is missing.

    The embedding API is called outside any transaction, so a slow call does
    not leave one open. Whatever the caller had pending is committed first.
    The vectors and the pairing then commit together.
    """
    rows = unembedded_chunks(conn, document_id, model.id)
    conn.commit()
    if not rows:
        return 0

    chunk_ids = [chunk_id for chunk_id, _ in rows]
    embedded = vectors.embeddings_for(model).embed_documents(
        [content for _, content in rows]
    )

    with conn.transaction():
        insert_vectors(conn, model, document_id, zip(chunk_ids, embedded))
        mark_embedded(conn, document_id, model.id)
    return len(rows)
