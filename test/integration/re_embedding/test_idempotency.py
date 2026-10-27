"""Embedding the same model twice is free and writes nothing new.

`unembedded_chunks` selects only chunks the target table has no vector for,
and `chunk_id ... unique` would refuse a duplicate even if it tried. This is
what makes a re-run safe to press twice.
"""

from test.factories import a_document, vector_count


def test_embedding_the_same_model_twice_writes_nothing_the_second_time(
    conn, client, small_model
):
    """Idempotent, so a repeat -- or a re-run -- costs nothing.

    `unembedded_chunks` selects only chunks the target table has no vector
    for, so the second call has nothing to do.
    """
    document_id = a_document(conn, title="Nutrition", chunks=4)

    first = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    second = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    assert first.json()["embedded"] == {small_model.table_name: 4}
    assert second.json()["embedded"] == {small_model.table_name: 0}
    assert vector_count(conn, small_model.table_name, document_id) == 4, (
        "a repeat must not duplicate vectors"
    )
