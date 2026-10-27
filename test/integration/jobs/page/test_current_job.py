"""What `GET /api/jobs/current` reports.

Polled by the upload page to know the slot is taken and by what, and by the
documents page to disable embedding while something runs.
"""

from test.factories import a_job


def test_the_current_job_says_what_is_running(conn, client):
    """What the upload page polls to know the slot is taken, and by what."""
    a_job(conn, kind="embed", subject="Nutrition", status="running")

    state = client.get("/api/jobs/current").json()

    assert state["active"] is True
    assert state["kind"] == "embed"
    assert state["subject"] == "Nutrition"


def test_a_job_that_finished_long_ago_is_not_reported_as_the_last_one(conn, client):
    """The outcome panel is for a page that just missed the end, not history.

    Past `LAST_JOB_TTL_SECONDS` the job belongs on the jobs page instead, so a
    page opened an hour later does not reopen a stale result.
    """
    a_job(conn, subject="old.pdf", status="succeeded")
    conn.execute("update jobs set finished_at = now() - interval '10 minutes'")
    conn.commit()

    state = client.get("/api/jobs/current").json()

    assert state["active"] is False
    assert state["last"] is None
