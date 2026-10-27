"""A job row left saying `running` by a process that died.

`finish_job` only runs inside the request, so a kill or a restart never
reaches it and the index then locks the user out of uploading AND
re-embedding. Startup corrects the status and keeps everything the job wrote.
"""

from api.services.jobs import sweep_running_jobs
from db.pool import connection
from test.factories import a_document, a_job, job_statuses, vector_count


def test_a_stuck_job_is_marked_interrupted(conn):
    """The status is corrected, with a message saying why."""
    a_job(conn, kind="upload", subject="stuck.pdf", status="running")

    with connection() as fresh:
        swept = sweep_running_jobs(fresh)
        fresh.commit()

    assert swept == ["stuck.pdf"]
    assert job_statuses(conn) == {1: "interrupted"}

    conn.rollback()
    error, finished_at = conn.execute(
        "select error, finished_at from jobs where id = 1"
    ).fetchone()
    assert "restarted" in error
    assert finished_at is not None, (
        "the row must satisfy the check constraint that a finished job has an end time"
    )


def test_an_interrupted_reembed_keeps_its_document(conn):
    """The regression this exists to prevent.

    An earlier version deleted the document behind an interrupted job whenever
    that document had no vectors, on the assumption that no vectors meant a
    failed import with nothing worth keeping.

    That is wrong for a re-embed, whose document existed before the job: the
    user had cleared its vectors and asked for them back, so "no vectors" is
    the expected state, not evidence of a dead import. Deleting it destroyed a
    document they only wanted embedded again -- chunks and uploaded PDF
    included.
    """
    document_id = a_document(conn, title="Nutrition", chunks=4)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="running",
        document_id=document_id,
        source_uri="uploads/Nutrition.pdf",
    )

    with connection() as fresh:
        sweep_running_jobs(fresh)
        fresh.commit()

    conn.rollback()
    assert (
        conn.execute("select 1 from documents where id = %s", (document_id,)).fetchone()
        is not None
    ), "the document must survive its interrupted job"

    chunks = conn.execute(
        "select count(*) from document_chunks where document_id = %s", (document_id,)
    ).fetchone()[0]
    assert chunks == 4, (
        "the extracted text must survive too -- it is what a re-run reuses"
    )


def test_recovering_frees_the_slot(conn, client, small_model):
    """After the sweep the user can work again, which is the whole point."""
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(conn, kind="upload", subject="stuck.pdf", status="running")

    # Before: locked out.
    blocked = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    assert blocked.status_code == 409

    with connection() as fresh:
        sweep_running_jobs(fresh)
        fresh.commit()

    # After: through.
    allowed = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    assert allowed.status_code == 200
    assert vector_count(conn, small_model.table_name, document_id) == 2


def test_startup_runs_the_sweep(conn):
    """The sweep is wired into the app's lifespan, not merely available.

    Entering the client runs startup, which is the only thing that calls it in
    production.
    """
    from fastapi.testclient import TestClient

    from api.server import app

    a_job(conn, kind="upload", subject="stuck.pdf", status="running")

    with TestClient(app):
        pass

    assert job_statuses(conn) == {1: "interrupted"}
