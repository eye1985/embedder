"""An import holds the job slot, so it has to hand it back.

Nothing here is paid for -- the vectors arrive in the file -- but an import
writes the rows an upload writes, so it claims the same one-per-user slot. A
job that takes that slot and never releases it locks the user out of uploading
and re-embedding too, which is why the failing path matters as much as the
succeeding one.

Read through /api/jobs rather than the table: what the jobs page is shown is
the thing worth pinning, and `can_rerun` is decided there rather than stored.
"""

import json

from test.factories import a_document, an_export, restore_from_json


def import_jobs(client) -> list[dict]:
    """Only the import rows -- embedding a document to export it leaves its
    own job behind, and that one is not what these tests are about."""
    return [job for job in client.get("/api/jobs").json() if job["kind"] == "import"]


def undescribed_model(payload: dict) -> bytes:
    """The file, with its model renamed to one no registry has."""
    (table,) = payload["models"]
    payload["models"]["emb_imaginary"] = payload["models"].pop(table)
    for chunk in payload["chunks"]:
        chunk["embeddings"]["emb_imaginary"] = chunk["embeddings"].pop(table)
    return json.dumps(payload).encode()


def test_an_import_is_recorded_as_its_own_kind_of_job(client, an_exported_file):
    """And is never offered a re-run.

    The input was a file that was parsed and thrown away, so there is nothing
    to replay -- the button would re-embed the document it restored and spend
    against the API for work this job never did.
    """
    assert restore_from_json(client, an_exported_file).status_code == 200

    (job,) = import_jobs(client)

    assert job["status"] == "succeeded"
    assert job["can_rerun"] is False
    assert job["result"]["replaced_existing"] is False
    assert job["models"] == list(json.loads(an_exported_file)["models"])


def test_a_failed_import_hands_the_slot_back(client, an_exported_file):
    """The refusal is recorded and the slot is released.

    Without this the next upload, re-embed or import would be turned away
    until the server restarted -- one refused import would cost the user every
    other job as well.
    """
    payload = json.loads(an_exported_file)

    assert restore_from_json(client, undescribed_model(payload)).status_code == 404

    (failed,) = import_jobs(client)
    assert failed["status"] == "failed"
    assert failed["error"]

    # The real proof: the slot is free, so the corrected file goes straight in.
    assert restore_from_json(client, an_exported_file).status_code == 200
    assert [job["status"] for job in import_jobs(client)] == ["succeeded", "failed"]


def test_an_import_refused_before_it_starts_leaves_no_job(conn, client, small_model):
    """A document already stored is turned away before the slot is claimed.

    Skipping is not a job that failed -- nothing ran -- so the jobs page has
    nothing to show and there is no row to clean up.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    assert restore_from_json(client, an_export(client, document_id)).status_code == 409

    assert import_jobs(client) == []
