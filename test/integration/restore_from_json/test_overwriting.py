"""Replacing a stored document with the file, when the user asks for that.

Overwrite is the escape hatch from the refusal, and it is a replacement: the
stored document, its chunks and every model's vectors go, and the file's take
their place. The PDF behind it is the one thing that stays -- the file being
imported has nothing to put there.
"""

import json

from test.factories import (
    a_document,
    an_export,
    embedded_models,
    restore_from_json,
    vector_count,
)


def test_overwriting_replaces_the_stored_document(conn, client, small_model):
    """The file wins, down to the chunks the stored document no longer has."""
    document_id = a_document(conn, title="Nutrition", chunks=4)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    payload = json.loads(an_export(client, document_id))
    payload["chunks"] = payload["chunks"][:2]

    response = restore_from_json(client, json.dumps(payload).encode(), overwrite=True)

    assert response.status_code == 200, response.text
    assert response.json()["replaced_existing"] is True
    assert response.json()["chunks"] == 2
    assert vector_count(conn, small_model.table_name, document_id) == 2
    # Parsed, not byte-compared: this file was rebuilt here, so its spacing is
    # this test's rather than the server's. The round-trip tests are the ones
    # that compare bytes, and both sides of those come from the server.
    assert json.loads(an_export(client, document_id)) == payload


def test_overwriting_keeps_the_pdf_the_document_came_from(
    conn, client, small_model, stored_pdf
):
    """Deleting a document removes its PDF. Replacing one must not.

    The import has only chunks and vectors in it, so removing the file would
    lose the source for good and leave the restored document unable to show
    what it was made from.
    """
    path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=2, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)

    assert restore_from_json(client, original, overwrite=True).status_code == 200

    assert path.is_file()
    assert client.get(f"/api/documents/{document_id}/source").status_code == 200


def test_overwriting_nothing_is_just_an_import(conn, client, small_model):
    """The flag says what to do about a collision, not that one has to exist."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)
    client.delete(f"/api/documents/{document_id}")

    response = restore_from_json(client, original, overwrite=True)

    assert response.status_code == 200, response.text
    assert response.json()["replaced_existing"] is False
    assert an_export(client, document_id) == original


def test_a_refused_overwrite_leaves_the_stored_document_alone(
    conn, client, small_model
):
    """Nothing is deleted until the whole file is known to be writable.

    The document, the chunks and the vectors are one transaction, and the
    models are resolved before it opens -- so a file this installation cannot
    hold cannot take the stored document down with it.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)

    payload = json.loads(original)
    payload["models"]["emb_imaginary"] = payload["models"].pop(small_model.table_name)
    for chunk in payload["chunks"]:
        chunk["embeddings"]["emb_imaginary"] = chunk["embeddings"].pop(
            small_model.table_name
        )

    response = restore_from_json(client, json.dumps(payload).encode(), overwrite=True)

    assert response.status_code == 404
    assert an_export(client, document_id) == original
    assert vector_count(conn, small_model.table_name, document_id) == 3
    assert embedded_models(conn, document_id) == {small_model.table_name}


def test_overwriting_a_different_id_over_the_same_pdf(conn, client, small_model):
    """The collision was on `source_uri`, so the id that goes is not the id
    that lands.

    One PDF, one document, either way -- which is the rule that made this a
    refusal in the first place.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    payload = json.loads(an_export(client, document_id))
    payload["document"]["id"] = "11111111-1111-1111-1111-111111111111"

    response = restore_from_json(client, json.dumps(payload).encode(), overwrite=True)

    assert response.status_code == 200, response.text
    assert response.json()["document_id"] == payload["document"]["id"]
    assert client.get(f"/api/documents/{document_id}/embeddings").status_code == 404

    conn.rollback()
    stored = conn.execute(
        "select id::text from documents where source_uri = %s",
        (payload["document"]["source_uri"],),
    ).fetchall()
    assert stored == [(payload["document"]["id"],)]
