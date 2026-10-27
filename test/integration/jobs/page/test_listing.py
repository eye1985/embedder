"""What `GET /api/jobs` shows, and whether Re-run is offered.

`can_rerun` has to agree with `replay_plan`, or the page shows a button that
404s when pressed.
"""

from test.factories import a_document, a_job


def test_jobs_are_listed_newest_first(conn, client):
    a_job(conn, subject="first.pdf", status="succeeded")
    a_job(conn, subject="second.pdf", status="succeeded")

    subjects = [job["subject"] for job in client.get("/api/jobs").json()]

    assert subjects == ["second.pdf", "first.pdf"]


def test_can_rerun_is_false_when_there_is_nothing_to_replay(conn, client):
    """The page must not offer a button the endpoint would refuse.

    Both answers come from `replay_plan`, and this is the pairing that proves
    the listing uses it: same job shape, one with its document and one without.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="interrupted",
        document_id=document_id,
    )
    a_job(
        conn,
        kind="embed",
        subject="Gone",
        status="interrupted",
        source_uri="uploads/deleted-long-ago.pdf",
    )

    by_subject = {job["subject"]: job for job in client.get("/api/jobs").json()}

    assert by_subject["Nutrition"]["can_rerun"] is True
    assert by_subject["Gone"]["can_rerun"] is False


def test_a_running_job_cannot_be_rerun(conn, client):
    """Nothing to repair yet, so the page offers no button."""
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="running",
        document_id=document_id,
    )

    assert client.get("/api/jobs").json()[0]["can_rerun"] is False


def test_an_interrupted_upload_can_be_rerun_despite_having_no_document_id(conn, client):
    """The crash case has to be repairable from the page, not just the API.

    `finish_job` never ran, so the row has no `document_id` -- the listing has
    to resolve the document by `source_uri` to know the job is replayable.
    """
    a_document(conn, title="Nutrition", chunks=3, source_uri="uploads/Nutrition.pdf")
    a_job(
        conn,
        kind="upload",
        subject="Nutrition.pdf",
        status="interrupted",
        source_uri="uploads/Nutrition.pdf",
        document_id=None,
    )

    job = client.get("/api/jobs").json()[0]

    assert job["can_rerun"] is True
    assert job["document_id"] is not None, (
        "the page needs the resolved document to link to it"
    )
