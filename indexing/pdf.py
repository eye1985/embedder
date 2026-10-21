"""The PDF-to-embeddings pipeline: extract, chunk, store, embed.

This is the whole flow in one function, and it has no HTTP in it. `/upload` is
a wrapper over `import_pdf` that adds a browser's constraints -- an untrusted
filename, a size ceiling, one upload at a time -- but the pipeline itself is
callable directly, which is how a script or a backfill uses it.
"""

import uuid
from pathlib import Path

import psycopg

from db.documents import add_document, count_chunks, find_document
from db.registry import EmbeddingModel
from indexing.chunk import split_documents
from indexing.embed import embed_document
from paths import PROJECT_ROOT


class ExtractionFailed(Exception):
    """No text could be read from the PDF. Nothing was stored.

    A domain error, not an HTTP one: this module has no status codes in it, so
    the caller decides what a failed extraction means. `/upload` maps it to 422.
    """


def import_pdf(
    conn: psycopg.Connection,
    user_id: int,
    pdf: Path,
    models: list[EmbeddingModel],
) -> dict:
    """Store a PDF's chunks if new, then embed them with each model.

    `pdf` must be an absolute path inside the project. `source_uri` is derived
    from it relative to `PROJECT_ROOT`, and that string is the key uploads
    dedupe on -- so a direct call and an upload of the same file agree on one
    document rather than storing it twice.

    Extraction and chunking happen once and are independent of the models. A
    document already stored for this path is reused, so calling again with a
    different model embeds the existing chunks instead of re-reading the PDF.
    """
    source_uri = str(pdf.relative_to(PROJECT_ROOT))
    document_id = find_document(conn, user_id, source_uri)
    reused = document_id is not None

    if document_id is None:
        document_id = _store(conn, user_id, pdf, source_uri)

    embedded = {m.table_name: embed_document(conn, document_id, m) for m in models}

    return {
        "document_id": str(document_id),
        "source_uri": source_uri,
        "reused_existing_document": reused,
        "chunks": count_chunks(conn, document_id),
        "embedded": embedded,
    }


def _store(
    conn: psycopg.Connection, user_id: int, pdf: Path, source_uri: str
) -> uuid.UUID:
    """Extract, chunk and store a PDF that has no document row yet.

    Committed before returning because embedding is what happens next: the
    vector tables are written over a second pool, and their foreign keys cannot
    see an uncommitted chunk.
    """
    # Imported here rather than at module scope: extract pulls in docling,
    # which is slow to import and not needed to serve chat.
    from indexing.extract import simple_extractor

    texts = simple_extractor(str(pdf))
    if not texts:
        raise ExtractionFailed(f"Extracted no text from {source_uri}")

    document_id = add_document(
        conn,
        user_id,
        split_documents(texts),
        title=pdf.stem,
        source_uri=source_uri,
    )
    conn.commit()
    return document_id
