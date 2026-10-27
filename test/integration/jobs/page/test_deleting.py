"""Deleting a job row: history pruning, or freeing a stuck slot.

A finished row is only history, and nothing references `jobs`, so deleting one
reaches no document. A running row is the other case -- deleting that frees
the slot without stopping work, which is why the page warns first.
"""

from test.factories import a_document, a_job, vector_count


def test_deleting_a_finished_job_removes_only_the_row(conn, client, small_model):
    """History pruning: no document, no chunk and no vector goes with it.

    Nothing references `jobs`, and its own foreign keys point outward, so there
    is no path from deleting a job to deleting data.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    response = client.delete("/api/jobs/1")

    assert response.status_code == 200
    assert client.get("/api/jobs").json() == []

    conn.rollback()
    assert (
        conn.execute("select 1 from documents where id = %s", (document_id,)).fetchone()
        is not None
    )
    assert vector_count(conn, small_model.table_name, document_id) == 3


def test_deleting_a_stuck_running_job_frees_the_slot(conn, client, small_model):
    """The way back from a row a dead process left behind.

    It does not stop work that is genuinely in flight -- nothing running reads
    this row -- which is why the page warns before asking.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    a_job(conn, kind="upload", subject="stuck.pdf", status="running")

    assert (
        client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [small_model.table_name]},
        ).status_code
        == 409
    )

    client.delete("/api/jobs/1")

    assert (
        client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [small_model.table_name]},
        ).status_code
        == 200
    )


def test_deleting_a_job_that_does_not_exist_is_a_404(client):
    assert client.delete("/api/jobs/999").status_code == 404
