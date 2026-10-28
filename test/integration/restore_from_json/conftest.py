"""A valid export to take apart. The stored PDF lives in test/conftest.py,
because the chat-sources tests need one too."""

import pytest

from test.factories import a_document, an_export


@pytest.fixture
def an_exported_file(conn, client, small_model):
    """A real export whose document has been deleted again.

    What the tests about malformed files start from: the file is valid, so
    whatever one of them breaks is the only thing wrong with it, and the
    document is gone so a refusal cannot be the "already here" one.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    data = an_export(client, document_id)
    client.delete(f"/api/documents/{document_id}")
    return data
