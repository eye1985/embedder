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
read as JSON by the frontend, which reaches them through plain links. The
reverse of that download does have a model here: restoring an export returns
JSON the documents page reads.

`ChatSource` is the exception to that. It is what the `/chat` stream's
`sources` event carries as well as what a reopened conversation returns, so
it is declared once here and the frontend types both from it.
"""

import uuid
from typing import Literal

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


class RestoreResult(BaseModel):
    document_id: uuid.UUID
    title: str | None
    source_uri: str | None
    chunks: int
    embedded: dict[str, int]
    # Null when the file carried no vectors, which is a document whose
    # embeddings were deleted before it was exported.
    search_model: str | None
    replaced_existing: bool


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


# A job's result is whichever shape its work returned: an upload reports an
# ImportResult, a re-embed an EmbedResult, a restored export a RestoreResult.
# They overlap on what the pages read, so the order matters -- each shape here
# requires a field the ones after it do not, which is what makes a stored
# result validate as the type that produced it.
JobResult = RestoreResult | ImportResult | EmbedResult


class LastJob(BaseModel):
    kind: str
    subject: str
    result: JobResult | None
    error: str | None
    finished_seconds_ago: float


class JobStatus(BaseModel):
    active: bool
    kind: str | None
    subject: str | None
    models: list[str] | None
    elapsed_seconds: float | None
    last: LastJob | None


class JobOut(BaseModel):
    job_id: int
    kind: str
    subject: str
    models: list[str]
    status: str
    source_uri: str | None
    document_id: uuid.UUID | None
    result: JobResult | None
    error: str | None
    started_at: str
    finished_at: str | None
    elapsed_seconds: float
    can_rerun: bool


class RerunResult(BaseModel):
    job_id: int
    document_id: uuid.UUID
    source_uri: str | None
    embedded: dict[str, int]
    chunks: int


class DeleteJobResult(BaseModel):
    job_id: int
    deleted: bool


class SearchableResult(BaseModel):
    document_id: uuid.UUID
    searchable: bool


class SearchModelResult(BaseModel):
    document_id: uuid.UUID
    search_model: str


class ChatSource(BaseModel):
    """One document a chat answer cited.

    Also the shape the `/chat` stream's `sources` event carries. That endpoint
    streams, so it is not in the schema itself -- declaring the shape here is
    what lets the frontend type both the stream and a reopened conversation
    from one definition instead of hand-writing one of them.
    """

    document_id: uuid.UUID
    title: str | None
    filename: str | None
    pages: list[int]
    has_source: bool
    # Declared as the two values `documents.origin` allows, so the generated
    # type is the union the sources panel switches on rather than `string`.
    origin: Literal["upload", "import"]


class ConversationOut(BaseModel):
    conversation_id: uuid.UUID
    title: str | None
    created_at: str
    updated_at: str
    messages: int


class MessageOut(BaseModel):
    seq: int
    role: Literal["user", "assistant"]
    # As stored: an answer keeps the citation markers the model wrote, which
    # the browser strips for the reader.
    content: str
    sources: list[ChatSource]


class ConversationDetail(BaseModel):
    conversation_id: uuid.UUID
    title: str | None
    messages: list[MessageOut]


class RenameResult(BaseModel):
    conversation_id: uuid.UUID
    title: str


class DeleteConversationResult(BaseModel):
    conversation_id: uuid.UUID
    deleted: bool
