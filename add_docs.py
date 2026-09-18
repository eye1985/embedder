"""Ingestion: chunk once, embed many times.

Text is split and stored in `document_chunks` exactly once, independent of any
embedding model. `embed_document` then writes vectors for one model at a time,
so the same document can be embedded by OpenAI today and Qwen tomorrow without
being re-extracted or re-chunked.
"""

import json
import uuid

import psycopg
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pg_vector import get_vector_store
from registry import EmbeddingModel

CHUNK_SIZE = 500
CHUNK_OVERLAP = 70


def make_splitter() -> RecursiveCharacterTextSplitter:
    """The one splitter used for every model.

    Chunks are shared across embedding models, so this deliberately does not
    vary by provider -- the tiktoken encoder is just a stable way to measure
    length, not a statement about which model will embed the result.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


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


def count_chunks(conn: psycopg.Connection, document_id: uuid.UUID) -> int:
    return conn.execute(
        "select count(*) from document_chunks where document_id = %s", (document_id,)
    ).fetchone()[0]


def add_document(
    conn: psycopg.Connection,
    user_id: int,
    texts: list[Document],
    *,
    title: str | None = None,
    source_uri: str | None = None,
) -> uuid.UUID:
    """Store a document and its chunks. No embedding happens here."""
    document_id = conn.execute(
        "insert into documents (user_id, title, source_uri) "
        "values (%s, %s, %s) returning id",
        (user_id, title, source_uri),
    ).fetchone()[0]

    splitter = make_splitter()
    index = 0
    for text in texts:
        for chunk in splitter.split_text(text.page_content):
            conn.execute(
                "insert into document_chunks "
                "(document_id, chunk_index, content, metadata) "
                "values (%s, %s, %s, %s)",
                (document_id, index, chunk, json.dumps(text.metadata or {})),
            )
            index += 1

    return document_id


def embed_document(
    conn: psycopg.Connection,
    document_id: uuid.UUID,
    model: EmbeddingModel,
) -> int:
    """Embed a document's chunks into `model`'s vector table.

    Chunks already embedded by this model are skipped, so re-running is cheap
    and adding a second model only embeds what that model is missing.
    """
    rows = conn.execute(
        f"""
        select c.id, c.content, d.user_id
        from document_chunks c
        join documents d on d.id = c.document_id
        where c.document_id = %s
          and not exists (
              select 1 from "{model.table_name}" v where v.chunk_id = c.id
          )
        order by c.chunk_index
        """,
        (document_id,),
    ).fetchall()

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


def ingest(
    conn: psycopg.Connection,
    user_id: int,
    texts: list[Document],
    models: list[EmbeddingModel],
    *,
    title: str | None = None,
    source_uri: str | None = None,
) -> tuple[uuid.UUID, dict[str, int]]:
    """Store a document, then embed it with each model given."""
    document_id = add_document(conn, user_id, texts, title=title, source_uri=source_uri)
    conn.commit()

    counts = {}
    for model in models:
        counts[model.table_name] = embed_document(conn, document_id, model)
    conn.commit()

    return document_id, counts
