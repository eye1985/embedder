"""What the endpoint turns away, and with which status.

These two are what the documents page depends on to stay honest: it only
offers models with a vector table, and it only ever shows one user's
documents.
"""

from test.factories import a_document


def test_a_model_without_a_vector_table_is_refused(conn, client):
    """The guard the documents page relies on to only offer usable models.

    `emb_qwen3_8b` is registered but inactive, so `db/setup.py` never created
    its table. Refused here with a 400 that says what to do, rather than
    surfacing from inside PGVectorStore as a 500.
    """
    document_id = a_document(conn, title="Nutrition", chunks=2)

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": ["emb_qwen3_8b"]},
    )

    assert response.status_code == 400
    assert "db.setup" in response.json()["detail"]


def test_a_document_that_is_not_yours_is_reported_missing(conn, client, small_model):
    """Scoped like every other document endpoint, so ids cannot be probed."""
    # `users` is seeded by the migrations and so is not truncated between
    # tests -- hence on conflict, which keeps the suite re-runnable.
    conn.execute(
        "insert into users (email, upload_model_id) values "
        "('other@example.com', %s) on conflict (email) do nothing",
        (small_model.id,),
    )
    someone_elses = conn.execute(
        "select id from users where email = 'other@example.com'"
    ).fetchone()[0]
    conn.commit()
    document_id = a_document(conn, title="Theirs", chunks=2, user_id=someone_elses)

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )

    assert response.status_code == 404
