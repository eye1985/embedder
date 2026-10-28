"""The export format: one document as JSON, or a whole library as a zip.

Vectors live in one table per registered model, so an export is a fan-out: a
chunk's text and metadata are read once from `document_chunks`, and each
model's table contributes its own vector for that chunk. The exported shape
follows that -- content stored once, `embeddings` keyed by table name --
rather than one file per model, which would repeat the text for every model.

`EXPORT_VERSION` makes this a contract: anything that changes the shape of a
payload below should bump it. Nothing here reads the database or knows about
HTTP; it takes rows and returns bytes.
"""

import io
import json
import re
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from db.tables.documents import chunk_rows, get_document, vector_rows
from db.tables.registry import EmbeddingModel, list_models, models_with_tables
from db.tables.users import get_user
from paths import PROJECT_ROOT

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


def _clean_metadata(metadata: dict | None) -> dict:
    """Rewrite extractor paths relative to the project root.

    PyMuPDF4LLM records `source` and `file_path` as absolute paths on the
    machine that ran the import, so exporting them verbatim would put the
    server's directory layout into a file the user downloads. Every other key
    is passed through untouched, so page numbers and titles survive a
    round-trip and an export can still be cited.

    Cleaning is idempotent, which `restore_from_json.py` depends on: a restored
    document exported again has to produce the same file.
    """
    cleaned = dict(metadata or {})
    for key in ("source", "file_path"):
        value = cleaned.get(key)
        if not isinstance(value, str):
            continue
        # A relative value is already one of ours -- a restored export brings
        # its metadata back cleaned. Resolved against the project rather than
        # the working directory, so running this twice is running it once,
        # whatever the server was started from.
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            cleaned[key] = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            # Imported from outside the repo: keep the filename, drop the path.
            cleaned[key] = Path(value).name
    return cleaned


def document_filename(document_id: str, title: str | None) -> str:
    """Archive path for one document.

    `documents.title` has no uniqueness constraint -- /upload takes it from the
    PDF stem, so two files named report.pdf in different folders would collide.
    The id prefix is what keeps entries distinct; the slug is for humans.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title or "").strip("-.").lower()
    slug = slug[:MAX_TITLE_SLUG].strip("-.") or "untitled"
    return f"documents/{document_id}-{slug}.json"


def export_document(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID
) -> dict:
    """One document with its chunks and every vector held for them.

    Raises LookupError if the document does not exist or belongs to someone else.
    """
    doc_id, owner_id, title, source_uri, created_at = get_document(
        conn, user_id, document_id
    )
    chunks = chunk_rows(conn, document_id)

    vectors: dict[str, dict[str, list[float]]] = {}
    models: dict[str, EmbeddingModel] = {}
    for model in models_with_tables(conn):
        rows = vector_rows(conn, document_id, model.table_name)
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


def build_archive(
    conn: psycopg.Connection, user_id: int, documents: list[dict]
) -> tuple[bytes, str]:
    """Every document the user has, zipped, with a manifest. Returns (zip, filename).

    `documents` is the listing from `api.services.documents.list_documents` -- passed in
    rather than read here, so the manifest's per-document summary and the
    /documents page cannot disagree.

    Built in memory: deflate does well on vector text, but the whole archive is
    still held at once. Documents sized in the thousands of chunks should move
    this to a temp file served by FileResponse.
    """
    user_row = get_user(conn, user_id)
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
            "user": {"id": user_row[0], "email": user_row[1]},
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
