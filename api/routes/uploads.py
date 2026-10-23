"""Accepting a PDF and running it through the import pipeline.

The job bookkeeping is `api/services/uploads.py`; the pipeline is `indexing/pdf.py`.
What is here is the request around them, and the cleanup when it fails.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile

from api.config import CURRENT_USER_ID
from api.routes.models import models_for, search_model_for
from api.schemas import ImportResult, UploadStatus
from api.services.documents import delete_document
from api.services.uploads import (
    begin_upload,
    finish_upload,
    save_upload,
    upload_filename,
    upload_source_uri,
    upload_status,
)
from db.pool import connection
from db.tables.documents import find_document
from indexing.pdf import ExtractionFailed, import_pdf
from paths import PROJECT_ROOT

logger = logging.getLogger("api")

router = APIRouter()


def discard_upload(conn, pdf):
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


# Defined with `def`, not `async def`: extraction and the embedding API call
# both block, so FastAPI runs this in a worker thread instead of stalling the
# event loop.
@router.post("/upload", response_model=ImportResult)
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

            chosen = models_for(conn, models)
            searched = search_model_for(chosen, search_model)
            pdf = save_upload(file, name)

            try:
                result = import_pdf(conn, CURRENT_USER_ID, pdf, chosen, searched)
            except ExtractionFailed as exc:
                # Refused before anything was stored, so there is no document
                # row to clean up -- only the file that was just written.
                pdf.unlink(missing_ok=True)
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                discard_upload(conn, pdf)
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


# Sync `def` for the same reason as the endpoint above: psycopg blocks, so
# FastAPI runs this in a worker thread rather than stalling the event loop.
@router.get("/upload/status", response_model=UploadStatus)
def upload_status_endpoint():
    """Whether an upload is running, and the outcome of the last one.

    The upload request itself can outlive the page that started it -- a sync
    endpoint runs in a thread that no client disconnect can cancel -- so this
    is how a reloaded page finds out it is still waiting, and what happened.
    """
    return upload_status(CURRENT_USER_ID)
