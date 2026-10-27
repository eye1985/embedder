"""An upload and a re-embed cannot run at the same time.

Neither endpoint checks a flag. Both insert into `jobs`, and the partial
unique index `jobs_one_running_per_user` refuses the second -- which is why
these go through a real Postgres. Drop that index and every test here fails.
"""

from test.factories import a_document, a_job, vector_count


def test_a_running_upload_turns_a_reembed_away(conn, client, large_model):
    """An upload holds the slot, so the documents page cannot start embedding."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    a_job(conn, kind="upload", subject="other.pdf", status="running")

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )

    assert response.status_code == 409
    assert "An upload is already in progress (other.pdf)" in response.json()["detail"]
    assert vector_count(conn, large_model.table_name, document_id) == 0, (
        "the refusal must happen before any embedding is paid for"
    )


def test_a_running_reembed_turns_an_upload_away(conn, client):
    """And the other direction: embedding holds the slot against an upload.

    The posted bytes are not a real PDF, which is the point -- the slot is
    claimed before the file is read, so the upload is refused without the
    content ever being looked at.
    """
    a_job(conn, kind="embed", subject="Nutrition", status="running")

    response = client.post(
        "/api/upload",
        files={"file": ("new.pdf", b"not a real pdf", "application/pdf")},
        data={"models": ["emb_openai_te3_small"]},
    )

    assert response.status_code == 409
    assert (
        "An embedding job is already in progress (Nutrition)"
        in response.json()["detail"]
    )


def test_two_reembeds_cannot_run_at_once(conn, client, small_model, large_model):
    """Two documents, one slot: the second request is refused, not queued."""
    first = a_document(conn, title="First", chunks=2)
    second = a_document(conn, title="Second", chunks=2)
    a_job(conn, kind="embed", subject="First", status="running", document_id=first)

    response = client.post(
        f"/api/documents/{second}/embeddings",
        json={"models": [small_model.table_name]},
    )

    assert response.status_code == 409
    assert vector_count(conn, small_model.table_name, second) == 0
