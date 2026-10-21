"""Writing a document's chunks into one model's vector table."""

import uuid

import psycopg
from langchain_core.documents import Document

from db.documents import unembedded_chunks
from db.registry import EmbeddingModel
from db.vectors import get_vector_store


def embed_document(
    conn: psycopg.Connection,
    document_id: uuid.UUID,
    model: EmbeddingModel,
) -> int:
    """Embed a document's chunks into `model`'s vector table.

    Chunks already embedded by this model are skipped, so re-running is cheap
    and adding a second model only embeds what that model is missing. Returns
    how many chunks were embedded.
    """
    rows = unembedded_chunks(conn, document_id, model.table_name)
    if not rows:
        return 0

    # Only chunk_id and user_id travel into the vector table -- both are real
    # columns there. Everything else stays on document_chunks.
    docs = [
        Document(
            page_content=content,
            metadata={"chunk_id": str(chunk_id), "user_id": user_id},
        )
        for chunk_id, content, user_id in rows
    ]

    store = get_vector_store(model)
    store.add_documents(docs)
    return len(docs)
