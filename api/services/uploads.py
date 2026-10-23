"""Upload handling: a browser's bytes onto disk, and the job row tracking it.

No pipeline here -- that is `indexing/pdf.py`. What is left is everything the
web adds to a plain import: an untrusted filename, a size ceiling, content
sniffing, and one upload at a time per user.

PDFs reach the app one way: posted to /upload, which writes them into
`uploads/`. The filename comes from the browser, so it is untrusted input --
reduced to a basename before anything opens or creates a path from it.
"""

import json
from pathlib import Path

import psycopg
from fastapi import HTTPException, UploadFile

from db.pool import connection
from db.tables.documents import delete_document, has_any_vector
from db.tables.registry import models_with_tables
from paths import PROJECT_ROOT

# Where uploaded PDFs land. `sample/` holds the repo's own PDFs and is not
# this -- deleting a document only removes a file from here.
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# A demo-sized ceiling, enforced while streaming to disk so an oversized file
# is never held in memory or left behind on disk.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def upload_filename(upload: UploadFile) -> str:
    """The basename an upload will be stored under.

    Validation only, no I/O: the caller needs the name before deciding whether
    the upload is allowed to land at all. A name like "../../db.py" would
    otherwise be a path out of the directory.
    """
    name = Path(upload.filename or "").name
    if not name or name.startswith("."):
        raise HTTPException(400, "Upload needs a filename.")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files can be imported.")
    return name


def upload_source_uri(name: str) -> str:
    """The `source_uri` an upload of `name` would be stored under."""
    return str((UPLOAD_DIR / name).relative_to(PROJECT_ROOT))


def save_upload(upload: UploadFile, name: str) -> Path:
    """Write an uploaded PDF into uploads/ under `name` and return its path.

    Content is sniffed rather than trusted: an extension says nothing about
    what the bytes are, and the extractor is what would otherwise find out.

    This overwrites whatever sits at that path. Callers must reject a name that
    already has a document row first -- /upload does -- because a document's
    chunks are extracted once and would not be re-read for new bytes.
    """
    header = upload.file.read(5)
    upload.file.seek(0)
    if not header.startswith(b"%PDF"):
        raise HTTPException(415, f"{name} is not a PDF.")

    UPLOAD_DIR.mkdir(exist_ok=True)
    destination = UPLOAD_DIR / name

    size = 0
    try:
        with open(destination, "wb") as out:
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Files are limited to {MAX_UPLOAD_BYTES // 1024 // 1024}MB.",
                    )
                out.write(chunk)
    except Exception:
        # Never leave a partial or rejected file behind.
        destination.unlink(missing_ok=True)
        raise

    return destination


# How long a finished upload stays reportable. A page that reloads just as the
# work ends still has something to show; one opened an hour later does not.
LAST_UPLOAD_TTL_SECONDS = 120

# --- Upload jobs -------------------------------------------------------------
#
# Every function here takes its own short-lived connection rather than the one
# the upload is holding. That is deliberate: the upload's connection stays in a
# transaction for the whole import, so a job row written on it would be
# invisible to everyone else until the work finished -- which is exactly when
# the row stops being interesting. Committing separately is what lets a
# reloaded page, or a second request, see that an upload is running.


def begin_upload(
    user_id: int, filename: str, models: list[str]
) -> tuple[int | None, dict | None]:
    """Claim the user's upload slot.

    Returns (job_id, None) on success, or (None, running_job) if one is already
    in flight. The claim is the insert itself: `upload_jobs_one_running_per_user`
    makes a second running row impossible, so two simultaneous requests cannot
    both win no matter which process they land in.
    """
    with connection() as conn:
        try:
            row = conn.execute(
                "insert into upload_jobs (user_id, filename, models, status) "
                "values (%s, %s, %s, 'running') returning id",
                (user_id, filename, json.dumps(models)),
            ).fetchone()
            return row[0], None
        except psycopg.errors.UniqueViolation:
            # Someone else holds the slot. The transaction is now aborted, so
            # the running row has to be read on a fresh connection.
            pass

    with connection() as conn:
        running = conn.execute(
            "select filename, models, started_at, "
            "extract(epoch from (now() - started_at)) "
            "from upload_jobs where user_id = %s and status = 'running'",
            (user_id,),
        ).fetchone()

    if running is None:
        # It finished between the failed insert and this read. Rare, and not
        # worth a retry loop -- the caller gets a 409 and can try again.
        return None, {"filename": "another upload", "elapsed_seconds": 0.0}

    return None, {
        "filename": running[0],
        "models": running[1],
        "started_at": running[2].isoformat(),
        "elapsed_seconds": round(float(running[3]), 1),
    }


def finish_upload(
    job_id: int | None,
    *,
    result: dict | None = None,
    error: str | None = None,
    source_uri: str | None = None,
) -> None:
    """Release the slot and record the outcome."""
    if job_id is None:
        return
    document_id = (result or {}).get("document_id")
    with connection() as conn:
        conn.execute(
            "update upload_jobs set status = %s, finished_at = now(), "
            "result = %s, error = %s, source_uri = %s, document_id = %s "
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


def upload_status(user_id: int) -> dict:
    """Whether an upload is running, and the outcome of the last one."""
    with connection() as conn:
        running = conn.execute(
            "select filename, models, extract(epoch from (now() - started_at)) "
            "from upload_jobs where user_id = %s and status = 'running'",
            (user_id,),
        ).fetchone()

        if running is not None:
            return {
                "active": True,
                "filename": running[0],
                "models": running[1],
                "elapsed_seconds": round(float(running[2]), 1),
                "last": None,
            }

        last = conn.execute(
            "select filename, result, error, "
            "extract(epoch from (now() - finished_at)) "
            "from upload_jobs "
            "where user_id = %s and status <> 'running' "
            "  and finished_at > now() - make_interval(secs => %s) "
            "order by finished_at desc limit 1",
            (user_id, LAST_UPLOAD_TTL_SECONDS),
        ).fetchone()

    report = None
    if last is not None:
        report = {
            "filename": last[0],
            "result": last[1],
            "error": last[2],
            "finished_seconds_ago": round(float(last[3]), 1),
        }

    return {
        "active": False,
        "filename": None,
        "models": None,
        "elapsed_seconds": None,
        "last": report,
    }


def sweep_running_uploads(conn: psycopg.Connection) -> list[str]:
    """Clear jobs left 'running' by a process that is gone. Call at startup.

    Durable state has a failure mode the in-memory version did not: a crash
    mid-upload leaves a row claiming to be running, and the partial unique
    index would then lock that user out of uploading forever. Nothing can still
    be in flight at startup, so anything still marked running was interrupted.

    A half-finished import is cleaned up with it. `indexing.pdf` commits the
    document before embedding starts, so a crash in between leaves a document
    with chunks and no vectors -- and with /upload refusing a filename that is
    taken, the user could not retry it. Only documents with no vectors at all
    are removed, so a crash after embedding succeeded keeps its work.

    Assumes one process. Run several workers and each would sweep the others'
    live jobs at startup; that is the point to swap this for a heartbeat.
    """
    rows = conn.execute(
        "update upload_jobs set status = 'interrupted', finished_at = now(), "
        "error = 'The server restarted while this upload was running.' "
        "where status = 'running' returning filename, document_id, user_id"
    ).fetchall()

    readable = [m.table_name for m in models_with_tables(conn)]

    swept = []
    for filename, document_id, user_id in rows:
        swept.append(filename)
        if document_id is None:
            continue
        if not has_any_vector(conn, document_id, readable):
            delete_document(conn, user_id, document_id)
    return swept
