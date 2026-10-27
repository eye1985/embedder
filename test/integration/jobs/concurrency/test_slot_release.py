"""The slot is handed back when a job ends, however it ends.

`finish_job` sits in the success path and the failure path both, so a provider
outage must not leave the user locked out.
"""

import pytest

from test.factories import a_document, job_statuses


def test_the_slot_is_free_again_once_the_work_succeeds(
    conn, client, small_model, large_model
):
    """A finished job releases the slot, so the next one can start."""
    document_id = a_document(conn, title="Nutrition", chunks=3)

    first = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )
    assert second.status_code == 200, "the first job should no longer hold the slot"

    assert job_statuses(conn) == {1: "succeeded", 2: "succeeded"}


def test_the_slot_is_free_again_even_when_the_work_fails(
    conn, client, small_model, monkeypatch
):
    """A crash inside the work must not leave the slot held.

    `finish_job` sits in the failure path as well as the success one, so the
    row is closed either way and the user is not locked out by a provider
    outage.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)

    def embedding_provider_is_down(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "api.routes.documents.embed_existing", embedding_provider_is_down
    )

    with pytest.raises(RuntimeError):
        client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [small_model.table_name]},
        )

    assert job_statuses(conn) == {1: "failed"}
    assert client.get("/api/jobs/current").json()["active"] is False
