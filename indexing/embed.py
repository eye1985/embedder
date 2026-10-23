"""Writing a document's chunks into one model's vector table."""

import uuid

import psycopg
from langchain_core.documents import Document

from db.tables.documents import unembedded_chunks
from db.tables.embeddings import mark_embedded
from db.tables.registry import EmbeddingModel
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

    # Only the three real columns travel into the vector table. Everything
    # else stays on document_chunks.
    docs = [
        Document(
            page_content=content,
            metadata={
                "chunk_id": str(chunk_id),
                "user_id": user_id,
                "document_id": str(document_id),
            },
        )
        for chunk_id, content, user_id in rows
    ]

    store = get_vector_store(model)
    store.add_documents(docs)

    # After the write, so a failed embed leaves no pairing claiming vectors
    # that are not there. The vectors go over a second pool, so this is not
    # one transaction -- the other direction would be the lie.
    mark_embedded(conn, document_id, model.id)
    return len(docs)
