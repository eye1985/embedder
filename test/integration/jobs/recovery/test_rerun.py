"""Re-running a job: the repair path for one that did not finish.

A re-run is an ordinary job -- it writes its own row and claims the slot like
anything else -- so the history keeps what happened and nothing races.
"""

from test.factories import a_document, a_job, job_statuses, vector_count


def test_rerunning_an_interrupted_job_finishes_the_work(conn, client, small_model):
    """The repair path, end to end: interrupted job in, vectors out."""
    document_id = a_document(conn, title="Nutrition", chunks=4)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="interrupted",
        models=(small_model.table_name,),
        document_id=document_id,
        source_uri="uploads/Nutrition.pdf",
    )

    response = client.post("/api/jobs/1/rerun")

    assert response.status_code == 200
    assert response.json()["embedded"] == {small_model.table_name: 4}
    assert vector_count(conn, small_model.table_name, document_id) == 4


def test_rerunning_writes_a_new_row_and_leaves_the_old_one(conn, client, small_model):
    """What happened stays in the history; the re-run is its own job."""
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="interrupted",
        models=(small_model.table_name,),
        document_id=document_id,
    )

    client.post("/api/jobs/1/rerun")

    assert job_statuses(conn) == {1: "interrupted", 2: "succeeded"}


def test_rerunning_a_job_that_already_succeeded_embeds_nothing(
    conn, client, small_model
):
    """Harmless, because embedding skips what is already there."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    response = client.post("/api/jobs/1/rerun")

    assert response.status_code == 200
    assert response.json()["embedded"] == {small_model.table_name: 0}
    assert vector_count(conn, small_model.table_name, document_id) == 3


def test_a_rerun_needs_the_slot_like_any_other_job(conn, client, small_model):
    """A re-run is a job, so it waits its turn too."""
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="interrupted",
        models=(small_model.table_name,),
        document_id=document_id,
    )
    a_job(conn, kind="upload", subject="other.pdf", status="running")

    response = client.post("/api/jobs/1/rerun")

    assert response.status_code == 409


def test_a_job_still_marked_running_is_not_rerun(conn, client, small_model):
    """Refused with the fix in the message, rather than racing the first run."""
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="running",
        models=(small_model.table_name,),
        document_id=document_id,
    )

    response = client.post("/api/jobs/1/rerun")

    assert response.status_code == 409
    assert "Delete it first" in response.json()["detail"]


def test_a_job_with_nothing_left_is_refused_with_a_reason(conn, client, small_model):
    """The document and the file are both gone, so the button should not have
    been there -- and the endpoint says so rather than failing obscurely."""
    a_job(
        conn,
        kind="embed",
        subject="Nutrition",
        status="interrupted",
        models=(small_model.table_name,),
        source_uri="uploads/deleted-long-ago.pdf",
    )

    response = client.post("/api/jobs/1/rerun")

    assert response.status_code == 404
    assert "both gone" in response.json()["detail"]


def test_rerunning_a_job_that_does_not_exist_is_a_404(client):
    assert client.post("/api/jobs/999/rerun").status_code == 404
