"""The `jobs` table: one row per job that writes embeddings.

Two kinds share it -- an upload and a re-embed of a stored document -- because
they share one rule: one at a time per user. `jobs_one_running_per_user` is
what enforces that, and it does not care which kind the row describes, so an
upload and a re-embed exclude each other for free.

Every function here takes its own short-lived connection rather than the one
the job is holding. That is deliberate: the job's connection stays in a
transaction for the whole run, so a row written on it would be invisible to
everyone else until the work finished -- which is exactly when the row stops
being interesting. Committing separately is what lets a reloaded page, or a
second request, see that a job is running.
"""

import json
import uuid

import psycopg

from db.pool import connection
from db.tables.documents import find_document
from paths import PROJECT_ROOT

# How long a finished job stays reportable on the upload page. A page that
# reloads just as the work ends still has something to show; one opened an
# hour later does not. The jobs page lists history regardless of this.
LAST_JOB_TTL_SECONDS = 120


def begin_job(
    user_id: int,
    kind: str,
    subject: str,
    models: list[str],
    *,
    source_uri: str | None = None,
    document_id: uuid.UUID | str | None = None,
) -> tuple[int | None, dict | None]:
    """Claim the user's job slot.

    Returns (job_id, None) on success, or (None, running_job) if one is already
    in flight. The claim is the insert itself: `jobs_one_running_per_user`
    makes a second running row impossible, so two simultaneous requests cannot
    both win no matter which process they land in.
    """
    with connection() as conn:
        try:
            row = conn.execute(
                "insert into jobs "
                "(user_id, kind, subject, models, status, source_uri, document_id) "
                "values (%s, %s, %s, %s, 'running', %s, %s) returning id",
                (
                    user_id,
                    kind,
                    subject,
                    json.dumps(models),
                    source_uri,
                    document_id,
                ),
            ).fetchone()
            return row[0], None
        except psycopg.errors.UniqueViolation:
            # Someone else holds the slot. The transaction is now aborted, so
            # the running row has to be read on a fresh connection.
            pass

    with connection() as conn:
        running = conn.execute(
            "select kind, subject, models, started_at, "
            "extract(epoch from (now() - started_at)) "
            "from jobs where user_id = %s and status = 'running'",
            (user_id,),
        ).fetchone()

    if running is None:
        # It finished between the failed insert and this read. Rare, and not
        # worth a retry loop -- the caller gets a 409 and can try again.
        return None, {"kind": "job", "subject": "another job", "elapsed_seconds": 0.0}

    return None, {
        "kind": running[0],
        "subject": running[1],
        "models": running[2],
        "started_at": running[3].isoformat(),
        "elapsed_seconds": round(float(running[4]), 1),
    }


def finish_job(
    job_id: int | None,
    *,
    result: dict | None = None,
    error: str | None = None,
    source_uri: str | None = None,
) -> None:
    """Release the slot and record the outcome.

    `document_id` is taken from the result when the job produced one, so an
    upload's row gains the link it could not have at the start. A re-embed
    already has it from `begin_job`, and coalesce keeps that.
    """
    if job_id is None:
        return
    document_id = (result or {}).get("document_id")
    with connection() as conn:
        conn.execute(
            "update jobs set status = %s, finished_at = now(), "
            "result = %s, error = %s, "
            "source_uri = coalesce(%s, source_uri), "
            "document_id = coalesce(%s, document_id) "
            "where id = %s",
            (
                "succeeded" if error is None else "failed",
                json.dumps(result) if result else None,
                error,
                source_uri or (result or {}).get("source_uri"),
                document_id,
                job_id,
            ),
        )


def current_job(user_id: int) -> dict:
    """Whether a job is running, and the outcome of the last one."""
    with connection() as conn:
        running = conn.execute(
            "select kind, subject, models, "
            "extract(epoch from (now() - started_at)) "
            "from jobs where user_id = %s and status = 'running'",
            (user_id,),
        ).fetchone()

        if running is not None:
            return {
                "active": True,
                "kind": running[0],
                "subject": running[1],
                "models": running[2],
                "elapsed_seconds": round(float(running[3]), 1),
                "last": None,
            }

        last = conn.execute(
            "select kind, subject, result, error, "
            "extract(epoch from (now() - finished_at)) "
            "from jobs "
            "where user_id = %s and status <> 'running' "
            "  and finished_at > now() - make_interval(secs => %s) "
            "order by finished_at desc limit 1",
            (user_id, LAST_JOB_TTL_SECONDS),
        ).fetchone()

    report = None
    if last is not None:
        report = {
            "kind": last[0],
            "subject": last[1],
            "result": last[2],
            "error": last[3],
            "finished_seconds_ago": round(float(last[4]), 1),
        }

    return {
        "active": False,
        "kind": None,
        "subject": None,
        "models": None,
        "elapsed_seconds": None,
        "last": report,
    }


def replay_plan(
    conn: psycopg.Connection,
    user_id: int,
    source_uri: str | None,
    document_id: str | None,
) -> tuple[str, str | None]:
    """How this job would be re-run: ("embed", document_id), ("import", None)
    or ("none", None).

    One resolver for both the jobs page and the rerun endpoint, so the button
    and what it does cannot disagree.

    "embed" is the usual answer, including for an interrupted upload: that
    commits its document and chunks before embedding starts, so the text is
    already stored and only the vectors are missing. An interrupted job never
    reached `finish_job` and so has no `document_id` on its row -- the
    document is found by `source_uri` instead, which is the key uploads dedupe
    on anyway.

    "import" is only for an upload that died before storing anything, where
    the PDF is still on disk and has to be extracted again.
    """
    if document_id is not None:
        row = conn.execute(
            "select 1 from documents where id = %s and user_id = %s",
            (document_id, user_id),
        ).fetchone()
        if row is not None:
            return "embed", document_id

    if source_uri is None:
        return "none", None

    existing = find_document(conn, user_id, source_uri)
    if existing is not None:
        return "embed", str(existing)

    if (PROJECT_ROOT / source_uri).is_file():
        return "import", None
    return "none", None


def list_jobs(conn: psycopg.Connection, user_id: int) -> list[dict]:
    """The user's jobs, newest first.

    `can_rerun` says whether the inputs to replay this job still exist -- its
    document, or the PDF it came from. Without it the page would offer a
    button that 404s.
    """
    rows = conn.execute(
        """
        select id, kind, subject, models, status, source_uri, document_id::text,
               result, error, started_at, finished_at,
               extract(epoch from (coalesce(finished_at, now()) - started_at))
        from jobs
        where user_id = %s
        order by started_at desc, id desc
        """,
        (user_id,),
    ).fetchall()

    jobs = []
    for (
        job_id,
        kind,
        subject,
        models,
        status,
        source_uri,
        document_id,
        result,
        error,
        started_at,
        finished_at,
        elapsed,
    ) in rows:
        plan, target = replay_plan(conn, user_id, source_uri, document_id)
        jobs.append(
            {
                "job_id": job_id,
                "kind": kind,
                "subject": subject,
                "models": models,
                "status": status,
                "source_uri": source_uri,
                # The row's own link, or the document a rerun resolved to --
                # an interrupted upload has the second and not the first.
                "document_id": document_id or target,
                "result": result,
                "error": error,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat() if finished_at else None,
                "elapsed_seconds": round(float(elapsed), 1),
                "can_rerun": (status != "running" and bool(models) and plan != "none"),
            }
        )
    return jobs


def get_job(conn: psycopg.Connection, user_id: int, job_id: int) -> tuple:
    """One of the user's jobs: (kind, subject, models, status, source_uri,
    document_id).

    Raises LookupError if it is not theirs, so the endpoint cannot be used to
    probe which job ids exist.
    """
    row = conn.execute(
        "select kind, subject, models, status, source_uri, document_id "
        "from jobs where id = %s and user_id = %s",
        (job_id, user_id),
    ).fetchone()
    if row is None:
        raise LookupError(f"No job {job_id} for this user.")
    return row


def delete_job(conn: psycopg.Connection, user_id: int, job_id: int) -> None:
    """Forget a job row.

    Allowed on a running one too: a job whose process died leaves a row that
    holds the slot forever, and this is how the user gets it back. The work
    itself cannot be cancelled -- a running job's thread does not read this --
    so deleting a genuinely live row releases the slot while the work carries
    on. The jobs page says so before it asks.
    """
    deleted = conn.execute(
        "delete from jobs where id = %s and user_id = %s", (job_id, user_id)
    ).rowcount
    if not deleted:
        raise LookupError(f"No job {job_id} for this user.")


def sweep_running_jobs(conn: psycopg.Connection) -> list[str]:
    """Mark jobs left 'running' by a process that is gone. Call at startup.

    Durable state has a failure mode the in-memory version did not: a crash
    mid-job leaves a row claiming to be running, and the partial unique index
    would then lock that user out of both uploading and re-embedding. Nothing
    can still be in flight at startup, so anything still marked running was
    interrupted.

    Only the status is corrected. What the job half-wrote is left alone:
    chunks are model-independent and embedding skips what is already there, so
    an interrupted job is repaired by re-running it from the jobs page, not by
    deleting what it managed to store.

    Assumes one process. Run several workers and each would sweep the others'
    live jobs at startup; that is the point to swap this for a heartbeat.
    """
    rows = conn.execute(
        "update jobs set status = 'interrupted', finished_at = now(), "
        "error = 'The server restarted while this job was running.' "
        "where status = 'running' returning subject"
    ).fetchall()
    return [subject for (subject,) in rows]
