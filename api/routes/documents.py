"""The document endpoints: listing, downloads, search state, and deletion.

The work itself is `api/services/documents.py`. These are the requests
around it -- every one scoped to the current user, so another user's
document is reported missing rather than refused.
"""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse, Response

from api.config import CURRENT_USER_ID
from api.routes.jobs import slot_conflict
from api.routes.models import models_for
from api.schemas import (
    DeleteDocumentResult,
    DeleteEmbeddingsResult,
    DocumentOut,
    EmbedResult,
    SearchableResult,
    SearchModelResult,
)
from api.services.documents import (
    delete_document,
    delete_embeddings,
    embed_existing,
    export_archive,
    export_document,
    list_documents,
)
from api.services.jobs import begin_job, finish_job
from api.services.sources import source_path
from db.pool import connection
from db.tables.documents import get_document, set_searchable
from db.tables.embeddings import set_document_model
from db.tables.registry import get_model
from export import document_filename

router = APIRouter()


class Searchable(BaseModel):
    searchable: bool


class ReEmbed(BaseModel):
    models: list[str] | None = None


def attachment(filename: str) -> dict[str, str]:
    """Headers that make a response save to disk under `filename`."""
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


# Sync `def` for the same reason as POST /upload: psycopg blocks, so FastAPI
# runs these in a worker thread rather than stalling the event loop while a
# large document is serialized.
@router.get("/documents", response_model=list[DocumentOut])
def list_documents_endpoint():
    """The user's documents, with chunk counts and per-model embedding counts."""
    with connection() as conn:
        return list_documents(conn, CURRENT_USER_ID)


@router.get("/documents/{document_id}/embeddings")
def download_document_embeddings(document_id: uuid.UUID):
    """One document's chunks and vectors, as a JSON file.

    Every model holding vectors for the document is included -- a chunk's
    `embeddings` is keyed by vector table name, so content and metadata are
    stored once no matter how many models embedded it.
    """
    with connection() as conn:
        try:
            payload = export_document(conn, CURRENT_USER_ID, document_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    title = payload["document"]["title"]
    # Reuses the archive's naming so a single download and the same document
    # inside a full export arrive under the same name.
    filename = document_filename(str(document_id), title).split("/")[-1]
    return JSONResponse(payload, headers=attachment(filename))


@router.get("/documents/{document_id}/source")
def download_document_source(document_id: uuid.UUID):
    """The file a document was imported from, as a download.

    What a chat answer's sources link to. Scoped like every other document
    endpoint, so someone else's document is reported missing rather than
    refused.
    """
    with connection() as conn:
        try:
            _, _, _, source_uri, _ = get_document(conn, CURRENT_USER_ID, document_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    path = source_path(source_uri)
    if path is None:
        raise HTTPException(404, f"No stored file for document {document_id}.")

    return FileResponse(
        path, media_type="application/pdf", headers=attachment(path.name)
    )


@router.get("/embeddings")
def download_all_embeddings():
    """Every document the user has, as a zip of JSON files plus a manifest."""
    with connection() as conn:
        try:
            archive, filename = export_archive(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    return Response(archive, media_type="application/zip", headers=attachment(filename))


@router.patch("/documents/{document_id}", response_model=SearchableResult)
def set_document_searchable(document_id: uuid.UUID, body: Searchable):
    """Take a document in or out of search entirely, for every model.

    The vectors stay where they are, and so does the per-model state in
    `document_embeddings` -- switching the document back on restores exactly
    the models it had.
    """
    with connection() as conn:
        try:
            set_searchable(conn, CURRENT_USER_ID, document_id, body.searchable)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"document_id": str(document_id), "searchable": body.searchable}


@router.put(
    "/documents/{document_id}/search-model/{table_name}",
    response_model=SearchModelResult,
)
def set_document_search_model(document_id: uuid.UUID, table_name: str):
    """Choose which model this document is retrieved through.

    Exactly one at a time. The models hold the same chunks, so this picks the
    ranking questions about this document get -- every other model keeps its
    vectors and simply stops being searched, which costs nothing to undo.
    """
    with connection() as conn:
        try:
            model = get_model(conn, table_name=table_name)
            set_document_model(conn, CURRENT_USER_ID, document_id, model.id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"document_id": str(document_id), "search_model": table_name}


# Sync `def` for the same reason as POST /upload: this calls the embedding API.
@router.post("/documents/{document_id}/embeddings", response_model=EmbedResult)
def embed_document_endpoint(document_id: uuid.UUID, body: ReEmbed):
    """Embed an existing document with one or more models.

    How a document gains a model it was not uploaded with, and how one emptied
    by DELETE gets its vectors back. The chunks are already stored and are
    model-independent, so nothing is extracted again; chunks a model already
    holds are skipped, so repeating this costs nothing.

    Which model the document is *searched* with does not move: `mark_embedded`
    enables a new pairing only when the document has none, so adding a model
    never changes answers under the user. The documents page is where that
    choice is made.

    This holds the same one-job-per-user slot as /upload, so a re-embed and an
    upload turn each other away rather than both spending the embedding API.
    """
    with connection() as conn:
        try:
            _, _, title, source_uri, _ = get_document(
                conn, CURRENT_USER_ID, document_id
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        models = models_for(conn, body.models)

    job_id, running = begin_job(
        CURRENT_USER_ID,
        "embed",
        title or source_uri or str(document_id),
        [m.table_name for m in models],
        source_uri=source_uri,
        document_id=document_id,
    )
    if running is not None:
        raise slot_conflict(running)

    try:
        with connection() as conn:
            try:
                result = embed_existing(conn, CURRENT_USER_ID, document_id, models)
            except LookupError as exc:
                raise HTTPException(404, str(exc)) from exc
        finish_job(job_id, result=result)
        return result
    except HTTPException as exc:
        # Recorded, not just raised: the client may already be gone, and the
        # jobs page reads the outcome from here.
        finish_job(job_id, error=str(exc.detail))
        raise
    except Exception as exc:
        finish_job(job_id, error=f"Embedding failed ({type(exc).__name__}).")
        raise


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResult)
def delete_document_endpoint(document_id: uuid.UUID):
    """Delete a document, its chunks, its vectors, and its uploaded PDF.

    A document imported from sample/ keeps its file -- those are the repo's,
    not the user's.
    """
    with connection() as conn:
        try:
            return delete_document(conn, CURRENT_USER_ID, document_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc


@router.delete(
    "/documents/{document_id}/embeddings", response_model=DeleteEmbeddingsResult
)
def delete_document_embeddings(document_id: uuid.UUID, model: str | None = None):
    """Delete a document's vectors, keeping the document and its chunks.

    `model` names one vector table; omitted, every model's vectors go. The
    chunks survive either way, so POST on this path can embed them again
    without re-extracting the PDF.

    To stop a model searching a document without losing the work, PATCH its
    `enabled` flag instead.
    """
    with connection() as conn:
        try:
            return delete_embeddings(conn, CURRENT_USER_ID, document_id, model)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
