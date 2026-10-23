"""The response shapes, declared so they reach the OpenAPI schema.

The endpoint functions assemble plain dicts. These models are what FastAPI
validates those dicts against, and what `openapi-typescript` turns into the
frontend's types -- without them every response is an untyped object in the
schema and the generated client sees `unknown`.

`document_id` is declared as a UUID even where the dict already holds a
string: pydantic accepts both and serializes either as a string, so the two
spellings in the codebase agree here.

Three endpoints are deliberately absent. `/chat` streams, `/embeddings`
returns a zip, and a single document's export is a file download -- none are
read as JSON by the frontend, which reaches them through plain links.
"""

import uuid

from pydantic import BaseModel


class ModelOut(BaseModel):
    table_name: str
    provider: str
    model_name: str
    dimensions: int
    is_active: bool
    available: bool
    documents: int
    is_upload_default: bool


class DocumentOut(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    created_at: str
    searchable: bool
    chunks: int
    embedded: dict[str, int]
    search_model: str | None


class ImportResult(BaseModel):
    document_id: uuid.UUID
    source_uri: str
    reused_existing_document: bool
    chunks: int
    embedded: dict[str, int]
    search_model: str


class EmbedResult(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    embedded: dict[str, int]
    chunks: int


class DeleteEmbeddingsResult(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    deleted_embeddings: dict[str, int]
    chunks_kept: int


class DeleteDocumentResult(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    deleted_chunks: int
    deleted_embeddings: dict[str, int]
    deleted_file: bool


class LastUpload(BaseModel):
    filename: str | None
    result: ImportResult | None
    error: str | None
    finished_seconds_ago: float


class UploadStatus(BaseModel):
    active: bool
    filename: str | None
    models: list[str] | None
    elapsed_seconds: float | None
    last: LastUpload | None


class SearchableResult(BaseModel):
    document_id: uuid.UUID
    searchable: bool


class SearchModelResult(BaseModel):
    document_id: uuid.UUID
    search_model: str
