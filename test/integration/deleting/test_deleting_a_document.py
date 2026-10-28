"""Delete means gone: the rows, the vectors, and the file on disk.

One statement does the database half -- `document_chunks` cascades from
`documents` and every vector table cascades from `document_chunks` -- so what
needs asserting is that all three actually went, in one place, rather than
each being assumed by the tests that happen to delete something on their way
to checking something else.

The file is the half nothing enforces. `_delete_stored_pdf` resolves the path
and unlinks it, and the only way to know that happened is to ask the disk.
`deleted_file` in the response says what the API claims; `path.is_file()`
says what is true. A delete that reported success without unlinking would
satisfy the first and fail the second, and that is the bug worth catching:
the response looks right, the PDF is still there, and the next upload of the
same name is refused by a document row that no longer exists.
"""

from db.tables.documents import count_chunks
from test.factories import a_document, vector_count


def test_deleting_removes_the_pdf_and_everything_in_the_database(
    conn, client, small_model, stored_pdf
):
    path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    assert path.is_file(), "arranged wrong if the file is not there to begin with"

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_file"] is True
    assert not path.is_file(), "reported deleted, but the file is still on disk"
    assert body["deleted_chunks"] == 3
    # Keyed by every model with a table, so the zeros belong to models that
    # never embedded this document rather than to the delete.
    assert body["deleted_embeddings"][small_model.table_name] == 3

    conn.rollback()
    assert count_chunks(conn, document_id) == 0
    assert vector_count(conn, small_model.table_name, document_id) == 0
    assert client.get(f"/api/documents/{document_id}/embeddings").status_code == 404


def test_deleting_a_document_with_no_stored_file_still_deletes_the_rows(
    conn, client, small_model
):
    """`source_uri` pointing at nothing is the ordinary state of a restored
    document, and deleting one must not depend on a file being there."""
    document_id = a_document(
        conn, title="Nutrition", chunks=2, source_uri="uploads/never-existed.pdf"
    )
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["deleted_file"] is False
    assert response.json()["deleted_chunks"] == 2

    conn.rollback()
    assert count_chunks(conn, document_id) == 0
    assert vector_count(conn, small_model.table_name, document_id) == 0
