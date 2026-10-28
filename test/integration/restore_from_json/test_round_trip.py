"""A download and an upload of it leave the database where it started.

This is the whole promise of the feature, and the assertion that carries it is
comparing the two files byte for byte: the document keeps its id, its
timestamp, its chunk ids, its chunk order and its vectors, or the second
export would differ somewhere.

Nothing is embedded on the way back in. The vectors are in the file, and
pgvector's text form parses back to the identical value, so a restore
reproduces them exactly rather than approximately.
"""

import json

from test.factories import (
    a_document,
    an_export,
    embedded_models,
    enabled_models,
    restore_from_json,
    vector_count,
)


def test_a_document_survives_a_download_and_an_upload(conn, client, small_model):
    """Export, delete everything, upload the file: the same export comes back."""
    document_id = a_document(conn, title="Nutrition", chunks=4)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)

    client.delete(f"/api/documents/{document_id}")
    response = restore_from_json(client, original)

    assert response.status_code == 200, response.text
    assert an_export(client, document_id) == original


def test_every_model_in_the_file_comes_back(conn, client, small_model, large_model):
    """A document two models embedded restores both, each at its own width.

    The widths differ on purpose: small is a `vector` column and large a
    `halfvec` one, so this covers both of the element types `db/setup.py`
    creates.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    for model in (small_model, large_model):
        client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [model.table_name]},
        )
    original = an_export(client, document_id)

    client.delete(f"/api/documents/{document_id}")
    response = restore_from_json(client, original)

    assert response.json()["embedded"] == {
        small_model.table_name: 3,
        large_model.table_name: 3,
    }
    assert vector_count(conn, small_model.table_name, document_id) == 3
    assert vector_count(conn, large_model.table_name, document_id) == 3
    assert embedded_models(conn, document_id) == {
        small_model.table_name,
        large_model.table_name,
    }
    assert an_export(client, document_id) == original


def test_a_restored_document_is_searchable_through_the_first_model_listed(
    conn, client, small_model, large_model
):
    """The two pieces of state the file does not carry, and what they default to.

    `searchable` and the search model are not in the export, so a restore sets
    them the way an upload would: on, and retrieved through the first model the
    file lists -- which is the lowest-numbered model that embedded it,
    whichever one was actually being searched when the file was written.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    for model in (small_model, large_model):
        client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [model.table_name]},
        )
    client.put(f"/api/documents/{document_id}/search-model/{large_model.table_name}")
    client.patch(f"/api/documents/{document_id}", json={"searchable": False})

    original = an_export(client, document_id)
    client.delete(f"/api/documents/{document_id}")
    response = restore_from_json(client, original)

    assert response.json()["search_model"] == small_model.table_name
    assert enabled_models(conn, document_id) == {small_model.table_name}

    conn.rollback()
    searchable = conn.execute(
        "select searchable from documents where id = %s", (document_id,)
    ).fetchone()[0]
    assert searchable is True


def test_the_export_does_not_depend_on_where_the_server_runs(
    conn, client, small_model, monkeypatch, tmp_path
):
    """Real chunks carry the extractor's own `source` and `file_path`.

    The export rewrites those relative to the project so a download does not
    carry the server's directory layout. A restore brings them back already
    rewritten, so exporting again has to produce the same string -- including
    when the server has since been started from somewhere else, which is what
    the directory change here stands for.
    """
    document_id = a_document(
        conn,
        title="Nutrition",
        chunks=3,
        metadata={
            "source": "uploads/Nutrition.pdf",
            "file_path": "uploads/Nutrition.pdf",
        },
    )
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    original = an_export(client, document_id)
    assert json.loads(original)["chunks"][0]["metadata"] == {
        "page": 0,
        "source": "uploads/Nutrition.pdf",
        "file_path": "uploads/Nutrition.pdf",
    }

    client.delete(f"/api/documents/{document_id}")
    assert restore_from_json(client, original).status_code == 200

    monkeypatch.chdir(tmp_path)
    assert an_export(client, document_id) == original


def test_a_document_with_no_vectors_restores_its_chunks(conn, client, small_model):
    """An export written after the embeddings were deleted is still an export.

    It restores to the state it describes: chunks stored, nothing embedded,
    searched by nothing -- which is what POST on the embeddings path is for.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    client.delete(f"/api/documents/{document_id}/embeddings")

    original = an_export(client, document_id)
    client.delete(f"/api/documents/{document_id}")
    response = restore_from_json(client, original)

    assert response.status_code == 200, response.text
    assert response.json()["search_model"] is None
    assert response.json()["chunks"] == 3
    assert embedded_models(conn, document_id) == set()
    assert an_export(client, document_id) == original


def test_a_restored_document_says_where_its_rows_came_from(conn, client, small_model):
    """A restore brings back chunks and vectors, never the PDF.

    So a restored document has nothing to download no matter what its
    `source_uri` says, and that is not the same thing as an upload whose file
    went missing. `origin` is what tells a chat answer's sources which of the
    two it is looking at, rather than inferring it from the absence of a file
    and getting it wrong for a PDF someone deleted by hand.

    It is deliberately not in the export: the act of restoring is what makes a
    document fileless, so it is set here rather than carried in the file.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    conn.rollback()
    before = conn.execute(
        "select origin from documents where id = %s", (document_id,)
    ).fetchone()[0]
    assert before == "upload"

    original = an_export(client, document_id)
    assert "origin" not in json.loads(original)["document"]

    client.delete(f"/api/documents/{document_id}")
    assert restore_from_json(client, original).status_code == 200

    conn.rollback()
    after = conn.execute(
        "select origin from documents where id = %s", (document_id,)
    ).fetchone()[0]
    assert after == "import"

    # The round trip still holds: `origin` is not part of the file.
    assert an_export(client, document_id) == original
