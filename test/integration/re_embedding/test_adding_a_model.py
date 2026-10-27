"""A stored document gains a model it was not uploaded with.

The feature itself: chunks are stored once and are independent of any model,
so a second model embeds the same text without the document being re-uploaded.
"""

from test.factories import (
    a_document,
    embedded_models,
    enabled_models,
    vector_count,
)


def test_a_document_gains_a_model_it_was_not_uploaded_with(
    conn, client, small_model, large_model
):
    """The headline case: uploaded with small, later also embedded with large."""
    document_id = a_document(conn, title="Nutrition", chunks=5)

    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )

    assert response.status_code == 200
    assert response.json()["embedded"] == {large_model.table_name: 5}
    assert vector_count(conn, small_model.table_name, document_id) == 5
    assert vector_count(conn, large_model.table_name, document_id) == 5
    assert embedded_models(conn, document_id) == {
        small_model.table_name,
        large_model.table_name,
    }


def test_adding_a_model_does_not_change_which_one_answers(
    conn, client, small_model, large_model
):
    """Adding a model must never silently change the answers a user gets.

    `mark_embedded` enables a new pairing only when the document has none, so
    the first model in stays the one searched and a later one is merely
    available. Choosing between them is a separate, explicit action on the
    documents page.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)

    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    assert enabled_models(conn, document_id) == {small_model.table_name}

    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )

    assert enabled_models(conn, document_id) == {small_model.table_name}, (
        "the added model must be available but NOT searched -- exactly one "
        "model answers for a document, and it stays the user's choice"
    )
