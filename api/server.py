import logging
import os
import uuid
from contextlib import asynccontextmanager
from operator import itemgetter
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from psycopg_pool import PoolTimeout
from pydantic import BaseModel
from starlette.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from add_docs import find_document
from api.documents import (
    build_archive,
    delete_document,
    delete_embeddings,
    document_filename,
    export_document,
    list_documents,
)
from api.ingest import (
    PROJECT_ROOT,
    begin_upload,
    finish_upload,
    ingest_pdf,
    save_upload,
    sweep_running_uploads,
    upload_filename,
    upload_source_uri,
    upload_status,
)
from db import close_pool, connection, get_pool, table_exists
from llm.init import init_llm
from pg_vector import close_engines
from registry import default_model_for_user, get_model, list_models

logger = logging.getLogger("api")

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

    app.state.chat = init_llm(user_id=CURRENT_USER_ID)
    try:
        yield
    finally:
        close_engines()
        close_pool()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(PoolTimeout)
async def pool_timeout_handler(request: Request, exc: PoolTimeout):
    """Postgres unreachable or the pool exhausted -- the service is down, not
    the request malformed, so 503 rather than 500."""
    logger.error("database unavailable on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": "Database unavailable. Is Postgres running?"}, status_code=503
    )


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


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


class Chat(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(prompt: Chat, request: Request):
    run, config = itemgetter("runnable_with_history", "config")(request.app.state.chat)

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
    """The models to embed with: those named, or the user's default.

    A registered model with no vector table is rejected here rather than deep
    inside PGVectorStore, where the same mistake surfaces as a 500.
    """
    try:
        if names:
            models = [get_model(conn, table_name=name) for name in names]
        else:
            models = [default_model_for_user(conn, CURRENT_USER_ID)]
    except LookupError as exc:
        # Unknown table name, or no user row -- a client error, not a crash.
        raise HTTPException(404, str(exc)) from exc

    missing = [m.table_name for m in models if not table_exists(conn, m.table_name)]
    if missing:
        raise HTTPException(
            400,
            f"No vector table for {', '.join(missing)}. "
            f"Create it with: uv run python setup_db.py",
        )
    return models


@app.get("/models")
def list_models_endpoint():
    """Every registered embedding model, for the upload form's picker.

    `available` is what the form gates on: `setup_db.py` only creates tables
    for active models, so an inactive row has nothing to embed into.
    """
    with connection() as conn:
        try:
            default = default_model_for_user(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

        return [
            {
                "table_name": m.table_name,
                "provider": m.provider,
                "model_name": m.model_name,
                "dimensions": m.dimensions,
                "is_active": m.is_active,
                "available": table_exists(conn, m.table_name),
                "is_default": m.table_name == default.table_name,
            }
            for m in list_models(conn, active_only=False)
        ]


# Defined with `def`, not `async def`: extraction and the embedding API call
# both block, so FastAPI runs this in a worker thread instead of stalling the
# event loop.
def _discard_upload(conn, pdf):
    """Undo a half-finished upload.

    `ingest_pdf` commits the document and its chunks before embedding starts --
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


@app.post("/upload")
def upload_endpoint(
    file: UploadFile,
    models: Annotated[list[str] | None, Form()] = None,
):
    """Accept a PDF, store it under uploads/, then ingest and embed it.

    `models` is a repeated form field of vector table names. Omitted, the
    user's default model is used. Naming several embeds the same chunks once
    per model -- the text is extracted and stored only once either way.

    A name already taken by one of the user's documents is refused with 409,
    and refused *before* the file is written. Overwriting would destroy the
    stored PDF while `find_document` matched the old row on source_uri, so the
    new content would never be extracted -- a 200 that ingested nothing.

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
            pdf = save_upload(file, name)

            try:
                result = ingest_pdf(conn, CURRENT_USER_ID, pdf, chosen)
            except HTTPException:
                # Extraction refused the file before anything was stored.
                pdf.unlink(missing_ok=True)
                raise
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
@app.get("/upload/status")
def upload_status_endpoint():
    """Whether an upload is running, and the outcome of the last one.

    The upload request itself can outlive the page that started it -- a sync
    endpoint runs in a thread that no client disconnect can cancel -- so this
    is how a reloaded page finds out it is still waiting, and what happened.
    """
    return upload_status(CURRENT_USER_ID)


@app.get("/documents")
def list_documents_endpoint():
    """The user's documents, with chunk counts and per-model embedding counts."""
    with connection() as conn:
        return list_documents(conn, CURRENT_USER_ID)


@app.get("/documents/{document_id}/embeddings")
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


@app.get("/embeddings")
def download_all_embeddings():
    """Every document the user has, as a zip of JSON files plus a manifest."""
    with connection() as conn:
        try:
            archive, filename = build_archive(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    return Response(
        archive, media_type="application/zip", headers=_attachment(filename)
    )


@app.delete("/documents/{document_id}")
def delete_document_endpoint(document_id: uuid.UUID):
    """Delete a document, its chunks, its vectors, and its uploaded PDF.

    A document ingested from sample/ keeps its file -- those are the repo's,
    not the user's.
    """
    with connection() as conn:
        try:
            return delete_document(conn, CURRENT_USER_ID, document_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc


@app.delete("/documents/{document_id}/embeddings")
def delete_document_embeddings(document_id: uuid.UUID, model: str | None = None):
    """Delete a document's vectors, keeping the document and its chunks.

    `model` names one vector table; omitted, every model's vectors go. The
    chunks survive either way, so the text does not need re-extracting -- but
    nothing over HTTP re-embeds them yet, so the document stays searchable by
    whichever models it has left, and by none if this removed them all.
    """
    with connection() as conn:
        try:
            return delete_embeddings(conn, CURRENT_USER_ID, document_id, model)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
