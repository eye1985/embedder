"""The job endpoints: what is running, what has run, and what to do about it.

One job at a time per user, across uploads and re-embeds both. The slot is a
row in `jobs` and the claim is `api/services/jobs.begin_job`, so the 409 here
and the one /upload returns are the same rule.

A crash leaves a row marked running and nothing alive to correct it. Startup
fixes the status (`sweep_running_jobs`), and these endpoints are how the user
fixes the rest: re-run what was interrupted, or delete a row that is holding
the slot for work that is gone.
"""

import uuid

from fastapi import APIRouter, HTTPException

from api.config import CURRENT_USER_ID
from api.routes.models import models_for
from api.schemas import DeleteJobResult, JobOut, JobStatus, RerunResult
from api.services.documents import embed_existing
from api.services.jobs import (
    begin_job,
    current_job,
    delete_job,
    finish_job,
    get_job,
    list_jobs,
    replay_plan,
)
from db.pool import connection
from indexing.pdf import ExtractionFailed, import_pdf
from paths import PROJECT_ROOT

router = APIRouter()


def slot_conflict(running: dict) -> HTTPException:
    """The 409 for a job that cannot start because one is already running.

    Shared so an upload turned away by a re-embed, and a re-embed turned away
    by an upload, both say which kind of work is in the way.
    """
    doing = "An upload" if running["kind"] == "upload" else "An embedding job"
    return HTTPException(
        409,
        f"{doing} is already in progress ({running['subject']}). "
        f"Wait for it to finish, or clear it from the jobs page.",
    )


@router.get("/jobs/current", response_model=JobStatus)
def current_job_endpoint():
    """Whether a job is running, and the outcome of the last one.

    A job outlives the page that started it -- a sync endpoint runs in a thread
    no client disconnect can cancel -- so this is how a reloaded page finds out
    it is still waiting, and what happened.
    """
    return current_job(CURRENT_USER_ID)


@router.get("/jobs", response_model=list[JobOut])
def list_jobs_endpoint():
    """The user's jobs, newest first."""
    with connection() as conn:
        return list_jobs(conn, CURRENT_USER_ID)


# Sync `def` for the same reason as POST /upload: this calls the embedding API.
@router.post("/jobs/{job_id}/rerun", response_model=RerunResult)
def rerun_job_endpoint(job_id: int):
    """Run a finished or interrupted job again.

    This is the repair path for a crash. Embedding skips chunks a model already
    holds, so re-running costs only the work that did not land -- and re-running
    a job that succeeded embeds nothing at all.

    A new row is written rather than the old one reused: what happened stays in
    the history, and the slot is claimed the same way any other job claims it.
    """
    with connection() as conn:
        try:
            kind, subject, model_names, status, source_uri, document_id = get_job(
                conn, CURRENT_USER_ID, job_id
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

        if status == "running":
            raise HTTPException(
                409,
                "That job is still marked running. Delete it first if its "
                "process is gone.",
            )

        plan, target = replay_plan(
            conn, CURRENT_USER_ID, source_uri, str(document_id) if document_id else None
        )
        if plan == "none":
            raise HTTPException(
                404,
                "Nothing left to re-run: this job's document and its source "
                "file are both gone.",
            )
        models = models_for(conn, model_names)

    new_id, running = begin_job(
        CURRENT_USER_ID,
        kind,
        subject,
        model_names,
        source_uri=source_uri,
        document_id=target,
    )
    if running is not None:
        raise slot_conflict(running)

    try:
        with connection() as conn:
            try:
                if plan == "embed":
                    result = embed_existing(
                        conn, CURRENT_USER_ID, uuid.UUID(target), models
                    )
                else:
                    result = import_pdf(
                        conn, CURRENT_USER_ID, PROJECT_ROOT / source_uri, models
                    )
            except ExtractionFailed as exc:
                raise HTTPException(422, str(exc)) from exc
            except LookupError as exc:
                raise HTTPException(404, str(exc)) from exc

        finish_job(new_id, result=result)
        return {
            "job_id": new_id,
            "document_id": result["document_id"],
            "source_uri": result.get("source_uri"),
            "embedded": result["embedded"],
            "chunks": result["chunks"],
        }
    except HTTPException as exc:
        finish_job(new_id, error=str(exc.detail))
        raise
    except Exception as exc:
        finish_job(new_id, error=f"Embedding failed ({type(exc).__name__}).")
        raise


@router.delete("/jobs/{job_id}", response_model=DeleteJobResult)
def delete_job_endpoint(job_id: int):
    """Forget a job row.

    Allowed on a running one, because that is the only way back from a row left
    behind by a process that died: the index would otherwise hold the slot
    forever. It does not stop work that is genuinely still going -- nothing
    running reads this row -- so the page warns before asking.
    """
    with connection() as conn:
        try:
            delete_job(conn, CURRENT_USER_ID, job_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
    return {"job_id": job_id, "deleted": True}
