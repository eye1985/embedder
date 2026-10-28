"""Reading an export back in: the reverse of `export.py`.

An export already holds everything a document is made of -- its id, its
chunks, and every model's vectors for them -- so restoring one is a write, not
an import pipeline. No PDF is read, nothing is chunked, and no embedding API
is called: the vectors in the file are the vectors that land.

pgvector's text form is what `export.py` writes out, and parsing it back
produces the identical float4 or fp16 value, so a document that goes out and
comes back is the same document down to the bit.

Nothing here reads the database or knows about HTTP. It takes the bytes of a
downloaded file and returns the rows `api/services/restore_from_json.py`
writes, or raises InvalidExport saying what is wrong with them.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from export import EXPORT_VERSION


class InvalidExport(Exception):
    """The file is not an export this version can read. Nothing was stored."""


@dataclass(frozen=True)
class ExportedModel:
    """One entry of the file's `models`, identifying a vector table.

    Carried so the registry row it names can be checked against it before any
    vector is written -- a table whose width has changed would otherwise take
    the rows and leave them unusable.
    """

    table_name: str
    provider: str
    model_name: str
    dimensions: int
    distance: str


@dataclass(frozen=True)
class ExportedChunk:
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    metadata: dict
    # Keyed by vector table name. A model that embedded the document but not
    # this chunk exports null, and is left out here.
    embeddings: dict[str, list[float]]


@dataclass(frozen=True)
class ExportedDocument:
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    created_at: datetime
    # In file order, which is registry order. The first is the one the
    # document is searched with once restored.
    models: list[ExportedModel]
    chunks: list[ExportedChunk]


def _reject_constant(name: str):
    raise InvalidExport(f"{name} is not a number a vector can hold.")


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidExport(f"{where} should be an object.")
    return value


def _require(payload: dict, key: str, kind: type, where: str):
    if key not in payload:
        raise InvalidExport(f"{where} is missing {key!r}.")
    value = payload[key]
    # bool passes isinstance(_, int), and `true` where a number belongs is a
    # broken file rather than the number 1.
    wrong_type = isinstance(value, bool) and kind is not bool
    if wrong_type or not isinstance(value, kind):
        raise InvalidExport(
            f"{where}.{key} should be {kind.__name__}, not {type(value).__name__}."
        )
    return value


def _optional_text(payload: dict, key: str, where: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise InvalidExport(f"{where}.{key} should be text or null.")
    return value


def _a_source_uri(document: dict) -> str | None:
    """The file the document was imported from, as the export recorded it.

    Project-relative and nothing else. This value comes out of an uploaded
    file and is stored on the document, where the rest of the app resolves it
    against PROJECT_ROOT and reads what it finds -- `replay_plan` does exactly
    that when a job is re-run. An absolute path, or one climbing out with
    `..`, would make an import the way to point that at any file on the
    machine, so it is refused here rather than sanitised: no export this app
    writes contains one.
    """
    value = _optional_text(document, "source_uri", "document")
    if value is None:
        return None
    if value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise InvalidExport(
            f"document.source_uri must stay inside the project: {value!r}"
        )
    return value


def _an_id(value, where: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidExport(f"{where} is not a uuid: {value!r}") from exc


def _models(payload: dict) -> list[ExportedModel]:
    """The file's `models`, in the order it lists them.

    An empty object is valid: a document whose vectors were deleted exports
    its chunks and no models, and restores to that same state.
    """
    models = []
    for table_name, info in _mapping(payload.get("models", {}), "models").items():
        where = f"models.{table_name}"
        info = _mapping(info, where)
        models.append(
            ExportedModel(
                table_name=table_name,
                provider=_require(info, "provider", str, where),
                model_name=_require(info, "model_name", str, where),
                dimensions=_require(info, "dimensions", int, where),
                distance=_require(info, "distance", str, where),
            )
        )
    return models


def _vector(value, model: ExportedModel, where: str) -> list[float]:
    if not isinstance(value, list):
        raise InvalidExport(f"{where} should be a list of numbers.")
    if len(value) != model.dimensions:
        raise InvalidExport(
            f"{where} has {len(value)} dimensions, but {model.table_name} "
            f"holds {model.dimensions}."
        )
    for number in value:
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise InvalidExport(f"{where} contains something that is not a number.")
    return [float(number) for number in value]


def _chunks(payload: dict, models: list[ExportedModel]) -> list[ExportedChunk]:
    rows = payload.get("chunks")
    if not isinstance(rows, list):
        raise InvalidExport("chunks should be a list.")

    by_name = {m.table_name: m for m in models}
    chunks = []
    seen_ids: set[uuid.UUID] = set()
    seen_indexes: set[int] = set()

    for position, row in enumerate(rows):
        where = f"chunks[{position}]"
        row = _mapping(row, where)
        chunk_id = _an_id(_require(row, "chunk_id", str, where), f"{where}.chunk_id")
        chunk_index = _require(row, "chunk_index", int, where)

        if chunk_id in seen_ids:
            raise InvalidExport(f"chunk_id {chunk_id} appears more than once.")
        if chunk_index in seen_indexes:
            raise InvalidExport(f"chunk_index {chunk_index} appears more than once.")
        seen_ids.add(chunk_id)
        seen_indexes.add(chunk_index)

        vectors = {}
        for table_name, value in _mapping(
            row.get("embeddings", {}), f"{where}.embeddings"
        ).items():
            model = by_name.get(table_name)
            if model is None:
                raise InvalidExport(
                    f"{where}.embeddings names {table_name!r}, which the file's "
                    f"`models` does not describe."
                )
            if value is None:
                continue
            vectors[table_name] = _vector(
                value, model, f"{where}.embeddings.{table_name}"
            )

        chunks.append(
            ExportedChunk(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                content=_require(row, "content", str, where),
                metadata=_mapping(row.get("metadata") or {}, f"{where}.metadata"),
                embeddings=vectors,
            )
        )

    return chunks


def parse_export(data: bytes) -> ExportedDocument:
    """One downloaded document file, checked and taken apart.

    Everything that can be known without the database is settled here, so a
    file that would fail halfway is refused before a single row is written.
    The exported `user_id` is deliberately not read: a file restores into the
    library of whoever uploads it.
    """
    try:
        payload = json.loads(data, parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise InvalidExport("That file is not text, so it is not an export.") from exc
    except json.JSONDecodeError as exc:
        raise InvalidExport(f"Not valid JSON: {exc}") from exc

    payload = _mapping(payload, "The file")

    version = _require(payload, "export_version", int, "The file")
    if version > EXPORT_VERSION:
        raise InvalidExport(
            f"This file is export version {version}; this app reads up to "
            f"{EXPORT_VERSION}."
        )

    document = _mapping(_require(payload, "document", dict, "The file"), "document")
    created_at = _require(document, "created_at", str, "document")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise InvalidExport(
            f"document.created_at is not a timestamp: {created_at!r}"
        ) from exc

    models = _models(payload)

    return ExportedDocument(
        document_id=_an_id(_require(document, "id", str, "document"), "document.id"),
        title=_optional_text(document, "title", "document"),
        source_uri=_a_source_uri(document),
        created_at=created,
        models=models,
        chunks=_chunks(payload, models),
    )
