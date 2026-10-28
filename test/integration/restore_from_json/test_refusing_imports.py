"""What an import turns away, and why nothing is written when it does.

The headline refusal is the one the user asked for: a document already here
is reported rather than duplicated or silently replaced. The rest are the
file being wrong, which is settled before a single row is written.
"""

import json

import pytest

from test.factories import a_document, a_job, an_export, restore_from_json


def reason(response) -> str | None:
    """The machine-readable code, for the refusals that carry one."""
    detail = response.json()["detail"]
    return detail["reason"] if isinstance(detail, dict) else None


def test_importing_the_same_file_twice_is_refused(conn, client, small_model):
    """The second import says so and changes nothing.

    `document_exists` is what the documents page keys on to offer the
    overwrite -- the other 409 this endpoint returns is a busy job slot, which
    is not an offer.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)

    response = restore_from_json(client, original)

    assert response.status_code == 409
    assert reason(response) == "document_exists"
    assert an_export(client, document_id) == original


def test_a_second_export_of_the_same_pdf_is_refused(conn, client):
    """A different document id over a `source_uri` that is taken.

    That string is the key uploads dedupe on, so letting both in would leave
    two documents claiming one PDF and an upload unable to say which it meant.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    original = json.loads(an_export(client, document_id))
    original["document"]["id"] = "11111111-1111-1111-1111-111111111111"

    response = restore_from_json(client, json.dumps(original).encode())

    assert response.status_code == 409
    assert reason(response) == "document_exists"
    assert "uploads/Nutrition.pdf" in response.json()["detail"]["message"]


@pytest.mark.parametrize(
    "data",
    [b"not json at all", b"", b"[]", b"{}", b'{"hello": "world"}'],
    ids=["not-json", "empty", "a-list", "an-empty-object", "an-unrelated-object"],
)
def test_a_file_that_is_not_an_export_is_refused(client, data):
    assert restore_from_json(client, data).status_code == 422


def _from_the_future(payload):
    payload["export_version"] += 1


def _repeated_chunk_id(payload):
    payload["chunks"][1]["chunk_id"] = payload["chunks"][0]["chunk_id"]


def _repeated_chunk_index(payload):
    payload["chunks"][1]["chunk_index"] = payload["chunks"][0]["chunk_index"]


def _an_undescribed_model(payload):
    payload["chunks"][0]["embeddings"]["emb_nobody_declared"] = [0.1]


def _the_wrong_width(payload):
    payload["chunks"][0]["embeddings"][_table(payload)] = [0.1, 0.2]


def _not_a_number(payload):
    payload["chunks"][0]["embeddings"][_table(payload)][0] = "quite large"


def _not_finite(payload):
    payload["chunks"][0]["embeddings"][_table(payload)][0] = float("nan")


def _no_timestamp(payload):
    payload["document"]["created_at"] = "the other day"


def _no_document_id(payload):
    payload["document"]["id"] = "not-a-uuid"


def _table(payload) -> str:
    (table,) = payload["models"]
    return table


# Everything the parser settles before a row is written. They are one test
# because they are one guarantee: a file that is wrong anywhere is refused
# whole, so there is never half a document to clean up.
@pytest.mark.parametrize(
    "break_it",
    [
        _from_the_future,
        _repeated_chunk_id,
        _repeated_chunk_index,
        _an_undescribed_model,
        _the_wrong_width,
        _not_a_number,
        _not_finite,
        _no_timestamp,
        _no_document_id,
    ],
    ids=lambda fn: fn.__name__.lstrip("_").replace("_", " "),
)
def test_a_readable_file_that_is_not_a_valid_export_is_refused(
    client, an_exported_file, break_it
):
    payload = json.loads(an_exported_file)
    break_it(payload)

    response = restore_from_json(client, json.dumps(payload).encode())

    assert response.status_code == 422, response.text
    # The file is parsed before anything is looked up or written, so the
    # library the fixture emptied is still empty.
    assert client.get("/api/documents").json() == []


@pytest.mark.parametrize(
    "source_uri",
    [
        "../../../../etc/hosts",
        "uploads/../../../etc/hosts",
        "/etc/hosts",
        "/Users/someone/private.pdf",
    ],
    ids=["climbs-out", "climbs-out-mid-path", "absolute", "absolute-elsewhere"],
)
def test_a_source_uri_that_leaves_the_project_is_refused(
    client, an_exported_file, source_uri
):
    """The file names where its document came from, and that string is stored.

    Everything downstream resolves it against the project and reads what it
    finds -- re-running a job re-imports it -- so an import is not allowed to
    put a path there that points anywhere else. No export this app writes
    contains one.
    """
    payload = json.loads(an_exported_file)
    payload["document"]["source_uri"] = source_uri

    response = restore_from_json(client, json.dumps(payload).encode())

    assert response.status_code == 422
    assert "inside the project" in response.json()["detail"]
    assert client.get("/api/documents").json() == []


def test_a_model_whose_width_has_changed_since_the_export_is_refused(
    client, an_exported_file
):
    """The registry row the file names is no longer the one that made it.

    The vectors are the width the file declares, so the parser is satisfied --
    it is the table they would land in that cannot hold them, which only the
    registry can say.
    """
    payload = json.loads(an_exported_file)
    (table,) = payload["models"]
    payload["models"][table]["dimensions"] = 8
    for chunk in payload["chunks"]:
        chunk["embeddings"][table] = [0.1] * 8

    response = restore_from_json(client, json.dumps(payload).encode())

    assert response.status_code == 400
    assert "dimensions" in response.json()["detail"]


def test_a_model_this_installation_does_not_have_is_refused(conn, client, small_model):
    """Vectors are only interpretable by the table they were made for.

    Nothing partial is written: the registry is checked before the document
    row goes in, so a file naming one unknown model imports none of itself.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    payload = json.loads(an_export(client, document_id))
    client.delete(f"/api/documents/{document_id}")

    payload["models"]["emb_imaginary"] = payload["models"].pop(small_model.table_name)
    for chunk in payload["chunks"]:
        chunk["embeddings"]["emb_imaginary"] = chunk["embeddings"].pop(
            small_model.table_name
        )

    response = restore_from_json(client, json.dumps(payload).encode())

    assert response.status_code == 404
    assert client.get(f"/api/documents/{document_id}/embeddings").status_code == 404


def test_an_import_waits_for_the_job_slot(conn, client):
    """The same rule an upload and a re-embed hold each other to.

    An import pays for no embedding, but it writes the rows an upload writes,
    so the two cannot be in flight at once.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    payload = json.loads(an_export(client, document_id))
    payload["document"]["id"] = "11111111-1111-1111-1111-111111111111"
    payload["document"]["source_uri"] = "uploads/Other.pdf"
    client.delete(f"/api/documents/{document_id}")

    a_job(conn, status="running", kind="upload", subject="busy.pdf")

    response = restore_from_json(client, json.dumps(payload).encode())

    assert response.status_code == 409
    assert reason(response) is None
    assert "already in progress" in response.json()["detail"]
