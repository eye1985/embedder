import logging
import os
import uuid
from contextlib import asynccontextmanager
from operator import itemgetter
from typing import Annotated

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from psycopg_pool import PoolTimeout
from pydantic import BaseModel
from starlette.responses import (
    JSONResponse,
    Response,
    StreamingResponse,
)

from api.assets import AssetsNotBuilt, assets
from api.documents import (
    delete_document,
    delete_embeddings,
    embed_existing,
    export_archive,
    export_document,
    list_documents,
)
from api.schemas import (
    DeleteDocumentResult,
    DeleteEmbeddingsResult,
    DocumentOut,
    EmbedResult,
    ImportResult,
    ModelOut,
    SearchableResult,
    SearchModelResult,
    UploadStatus,
)
from api.uploads import (
    begin_upload,
    finish_upload,
    save_upload,
    sweep_running_uploads,
    upload_filename,
    upload_source_uri,
    upload_status,
)
from db.documents import find_document, set_searchable
from db.embeddings import document_counts, search_groups, set_document_model
from db.pool import close_pool, connection, get_pool, table_exists
from db.registry import get_model, list_models, upload_model_for_user
from db.vectors import close_engines
from export import document_filename
from indexing.pdf import ExtractionFailed, import_pdf
from llm.chat import build_chat
from paths import PROJECT_ROOT

logger = logging.getLogger("api")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Vite's build output. Absent until `npm run build` has run, which is why
# the mount below does not insist on it existing.
STATIC_DIR = PROJECT_ROOT / "static"

# Placeholder until there is real auth -- every request is treated as this
# user, and retrieval is scoped to their embeddings only.
CURRENT_USER_ID = int(os.getenv("CURRENT_USER_ID", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the pool here rather than on the first request, so an unreachable
    # database fails at startup instead of under traffic.
    get_pool()

    # Nothing can still be uploading at startup, so a job still marked running
    # belongs to a process that is gone. Left alone it would hold the
    # one-upload-per-user index forever and lock that user out.
    with connection() as conn:
        swept = sweep_running_uploads(conn)
    if swept:
        logger.warning(
            "cleared %d upload(s) interrupted by a restart: %s",
            len(swept),
            ", ".join(swept),
        )

    try:
        yield
    finally:
        close_engines()
        close_pool()


app = FastAPI(lifespan=lifespan)


class NoCacheStaticFiles(StaticFiles):
    """Static files the browser has to revalidate before reusing.

    Starlette sends `etag` and `last-modified` but no `Cache-Control`, which
    leaves the browser free to reuse a cached copy without asking. An edited
    page or script then keeps running the old version until a hard reload --
    and `--reload` restarting the server does nothing about it.

    `no-cache` means "revalidate", not "do not store": an unchanged file still
    comes back 304 with no body. The header is set after the parent has
    already decided between 200 and 304, so it lands on both.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


app.mount(
    "/static",
    NoCacheStaticFiles(directory=STATIC_DIR, check_dir=False),
    name="static",
)

# `frontend/` is searched too, so a template can inline a source file that
# has to run before anything else -- the theme preload in base.html.
templates = Jinja2Templates(directory=[TEMPLATES_DIR, FRONTEND_DIR])

# The JSON API. Prefixed so it cannot collide with a page: the documents
# page and the documents list would otherwise both want GET "/documents".
api = APIRouter(prefix="/api")


@app.exception_handler(PoolTimeout)
async def pool_timeout_handler(request: Request, exc: PoolTimeout):
    """Postgres unreachable or the pool exhausted -- the service is down, not
    the request malformed, so 503 rather than 500."""
    logger.error("database unavailable on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": "Database unavailable. Is Postgres running?"}, status_code=503
    )


@app.exception_handler(AssetsNotBuilt)
async def assets_not_built_handler(request: Request, exc: AssetsNotBuilt):
    """The frontend has not been built, so no page can be rendered."""
    logger.error("frontend not built: %s", exc)
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Anything not already mapped to a status.

    The exception text is deliberately NOT returned: a psycopg or SQLAlchemy
    error can carry the connection string, and these endpoints have no auth.
    The type is enough to recognise, and the traceback goes to the log.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": f"Internal error ({type(exc).__name__}). See server logs."},
        status_code=500,
    )


def _page(request: Request, name: str, title: str, heading: str, current: str):
    """One of the page shells.

    `current` is the nav href to mark as the page you are on; the nav itself
    lives in `templates/base.html` and is the only copy of it. The page's
    script entry follows from the template name, and its built URL is resolved
    through the Vite manifest, so the markup never names a hashed file.

    `no-cache` is set for the same reason the static mount sets it -- an edited
    template would otherwise keep being served from the browser's cache. The
    built assets are hashed, so they are safe to cache hard.
    """
    entry = f"pages/{name.removesuffix('.html')}.ts"
    bundle = assets(STATIC_DIR, "main.css", entry)
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={
            "title": title,
            "heading": heading,
            "current": current,
            "styles": bundle["styles"],
            "scripts": bundle["scripts"],
            "preloads": bundle["preloads"],
        },
        headers={"cache-control": "no-cache"},
    )


@app.get("/", include_in_schema=False)
def chat_page(request: Request):
    return _page(request, "chat.html", "Chat", "Document chat", "/")


@app.get("/upload", include_in_schema=False)
def upload_page(request: Request):
    return _page(request, "upload.html", "Upload", "Upload", "/upload")


@app.get("/documents", include_in_schema=False)
def documents_page(request: Request):
    return _page(request, "documents.html", "Documents", "Documents", "/documents")


class Chat(BaseModel):
    prompt: str


# Sync `def`: reading which model searches which document is a blocking
# query, and the chain is assembled before a single token is streamed.
# FastAPI runs this in a worker thread and then streams on the event loop.
@api.post("/chat")
def chat(prompt: Chat):
    """Answer from the user's searchable documents.

    Each document is retrieved through the one model it was given, so this
    reads the grouping per request -- changing a document's model has to take
    effect on the next question, not the next restart.
    """
    with connection() as conn:
        groups = {
            get_model(conn, model_id=model_id): document_ids
            for model_id, document_ids in search_groups(conn, CURRENT_USER_ID).items()
        }

    run, config = itemgetter("runnable_with_history", "config")(
        build_chat(CURRENT_USER_ID, groups)
    )

    async def stream():
        # A streaming response commits to 200 before the first token, so a
        # failure here cannot change the status. The error is written into the
        # body instead -- the alternative is a silent truncation that reads
        # like a short answer.
        try:
            async for chunk in run.astream({"question": prompt.prompt}, config=config):
                yield chunk
        except Exception:
            logger.exception("chat stream failed")
            yield "\n\n[error] The answer could not be completed. See server logs."

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


def _models_for(conn, names: list[str] | None):
    """The models to embed with: those named, or the user's upload model.

    A registered model with no vector table is rejected here rather than deep
    inside PGVectorStore, where the same mistake surfaces as a 500.
    """
    try:
        if names:
            models = [get_model(conn, table_name=name) for name in names]
        else:
            models = [upload_model_for_user(conn, CURRENT_USER_ID)]
    except LookupError as exc:
        # Unknown table name, or no user row -- a client error, not a crash.
        raise HTTPException(404, str(exc)) from exc

    missing = [m.table_name for m in models if not table_exists(conn, m.table_name)]
    if missing:
        raise HTTPException(
            400,
            f"No vector table for {', '.join(missing)}. "
            f"Create it with: uv run python -m db.setup",
        )
    return models


def _search_model_for(chosen: list, table_name: str | None):
    """Which of `chosen` the document is retrieved through.

    One model embedded means no choice to make. Several and the caller has to
    say, rather than have one picked for them -- the answers a question gets
    depend on it, so a silent default would be a decision made invisibly.
    """
    if table_name is None:
        if len(chosen) == 1:
            return chosen[0]
        raise HTTPException(
            400,
            "Embedding with more than one model needs `search_model` naming "
            "which of them this document is searched with.",
        )

    for model in chosen:
        if model.table_name == table_name:
            return model
    raise HTTPException(
        400,
        f"search_model {table_name!r} is not one of the models being embedded "
        f"({', '.join(m.table_name for m in chosen)}).",
    )


@api.get("/models", response_model=list[ModelOut])
def list_models_endpoint():
    """Every registered embedding model, for the upload form's pickers.

    `available` is what the form gates on: `db/setup.py` only creates tables
    for active models, so an inactive row has nothing to embed into.
    `documents` says how many of the user's documents each has embedded.
    """
    with connection() as conn:
        try:
            upload = upload_model_for_user(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

        models = list_models(conn, active_only=False)
        counts = document_counts(conn, CURRENT_USER_ID, [m.id for m in models])

        return [
            {
                "table_name": m.table_name,
                "provider": m.provider,
                "model_name": m.model_name,
                "dimensions": m.dimensions,
                "is_active": m.is_active,
                "available": table_exists(conn, m.table_name),
                "documents": counts.get(m.id, 0),
                "is_upload_default": m.id == upload.id,
            }
            for m in models
        ]


# Defined with `def`, not `async def`: extraction and the embedding API call
# both block, so FastAPI runs this in a worker thread instead of stalling the
# event loop.
def _discard_upload(conn, pdf):
    """Undo a half-finished upload.

    `import_pdf` commits the document and its chunks before embedding starts --
    it has to, because the vector tables are written over a second pool and
    their foreign keys cannot see an uncommitted chunk. So a failure during
    embedding leaves a committed document that a rollback cannot reach, and it
    has to be deleted explicitly. Without this the user is stuck: the row makes
    the filename taken, so re-uploading returns 409.
    """
    document_id = find_document(
        conn, CURRENT_USER_ID, str(pdf.relative_to(PROJECT_ROOT))
    )
    if document_id is None:
        pdf.unlink(missing_ok=True)
        return
    try:
        delete_document(conn, CURRENT_USER_ID, document_id)
        # Committed here because the caller is about to raise, and the
        # surrounding context manager would otherwise roll this back too.
        conn.commit()
    except Exception:
        logger.exception("could not clean up failed upload %s", pdf.name)


@api.post("/upload", response_model=ImportResult)
def upload_endpoint(
    file: UploadFile,
    models: Annotated[list[str] | None, Form()] = None,
    search_model: Annotated[str | None, Form()] = None,
):
    """Accept a PDF, store it under uploads/, then chunk and embed it.

    `models` is a repeated form field of vector table names. Omitted, the
    user's upload model is used. Naming several embeds the same chunks once
    per model -- the text is extracted and stored only once either way.

    `search_model` is the one of those the document is retrieved through, and
    is required when more than one model is named. The models hold the same
    text, so this decides which ranking questions about this document get,
    not which of it can be reached. It can be changed later from the
    documents page.

    A name already taken by one of the user's documents is refused with 409,
    and refused *before* the file is written. Overwriting would destroy the
    stored PDF while `find_document` matched the old row on source_uri, so the
    new content would never be extracted -- a 200 that imported nothing.

    The whole pipeline runs inside the request: a large PDF holds the
    connection for as long as extraction and the embedding API take. Fine for
    a demo, and the point at which this needs a job queue instead.
    """
    name = upload_filename(file)

    job_id, running = begin_upload(CURRENT_USER_ID, name, models or [])
    if running is not None:
        # Claimed before the file is read, so a second upload is turned away
        # without spending time on its body.
        raise HTTPException(
            409,
            f"An upload is already in progress ({running['filename']}). "
            f"Wait for it to finish before starting another.",
        )

    try:
        with connection() as conn:
            if (
                find_document(conn, CURRENT_USER_ID, upload_source_uri(name))
                is not None
            ):
                raise HTTPException(
                    409,
                    f"A document named {name!r} already exists. Rename the file, or "
                    f"delete the existing document before uploading a new version.",
                )

            chosen = _models_for(conn, models)
            searched = _search_model_for(chosen, search_model)
            pdf = save_upload(file, name)

            try:
                result = import_pdf(conn, CURRENT_USER_ID, pdf, chosen, searched)
            except ExtractionFailed as exc:
                # Refused before anything was stored, so there is no document
                # row to clean up -- only the file that was just written.
                pdf.unlink(missing_ok=True)
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                _discard_upload(conn, pdf)
                # Distinguish "this provider is not usable here" from "the call
                # to it failed": the first is configuration, the second upstream.
                if isinstance(exc, RuntimeError | ValueError):
                    logger.error("embedding unavailable for %s: %s", name, exc)
                    raise HTTPException(503, str(exc)) from exc
                logger.exception("embedding failed for %s", name)
                raise HTTPException(
                    502,
                    f"Embedding failed ({type(exc).__name__}). Nothing was stored.",
                ) from exc

        finish_upload(job_id, result=result)
        return result
    except HTTPException as exc:
        # Recorded, not just raised: the client that started this may already
        # be gone, and a reloaded page reads the outcome from here.
        finish_upload(job_id, error=str(exc.detail))
        raise
    except Exception as exc:
        finish_upload(job_id, error=f"Internal error ({type(exc).__name__}).")
        raise


def _attachment(filename: str) -> dict[str, str]:
    """Headers that make a response save to disk under `filename`."""
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


# Sync `def` for the same reason as /upload: psycopg blocks, so FastAPI runs
# these in a worker thread rather than stalling the event loop while a large
# document is serialized.
@api.get("/upload/status", response_model=UploadStatus)
def upload_status_endpoint():
    """Whether an upload is running, and the outcome of the last one.

    The upload request itself can outlive the page that started it -- a sync
    endpoint runs in a thread that no client disconnect can cancel -- so this
    is how a reloaded page finds out it is still waiting, and what happened.
    """
    return upload_status(CURRENT_USER_ID)


@api.get("/documents", response_model=list[DocumentOut])
def list_documents_endpoint():
    """The user's documents, with chunk counts and per-model embedding counts."""
    with connection() as conn:
        return list_documents(conn, CURRENT_USER_ID)


@api.get("/documents/{document_id}/embeddings")
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
    return JSONResponse(payload, headers=_attachment(filename))


@api.get("/embeddings")
def download_all_embeddings():
    """Every document the user has, as a zip of JSON files plus a manifest."""
    with connection() as conn:
        try:
            archive, filename = export_archive(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    return Response(
        archive, media_type="application/zip", headers=_attachment(filename)
    )


class Searchable(BaseModel):
    searchable: bool


class ReEmbed(BaseModel):
    models: list[str] | None = None


@api.patch("/documents/{document_id}", response_model=SearchableResult)
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


@api.put(
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


# Sync `def` for the same reason as /upload: this calls the embedding API.
@api.post("/documents/{document_id}/embeddings", response_model=EmbedResult)
def embed_document_endpoint(document_id: uuid.UUID, body: ReEmbed):
    """Embed an existing document with one or more models.

    How a document gains a model it was not uploaded with, and how one emptied
    by DELETE gets its vectors back. Chunks a model already holds are skipped,
    so repeating this costs nothing.
    """
    with connection() as conn:
        models = _models_for(conn, body.models)
        try:
            return embed_existing(conn, CURRENT_USER_ID, document_id, models)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc


@api.delete("/documents/{document_id}", response_model=DeleteDocumentResult)
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


@api.delete(
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


app.include_router(api)
