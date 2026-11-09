"""What the endpoint turns away, and with which status.

These two are what the documents page depends on to stay honest: it only
offers registered models, and it only ever shows one user's documents.
"""

from test.factories import a_document


def test_an_unregistered_model_is_refused(conn, client):
    """Reported as missing rather than surfacing as a 500 mid-embedding."""
    document_id = a_document(conn, title="Nutrition", chunks=2)

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [999]},
    )

    assert response.status_code == 404


def test_a_document_that_is_not_yours_is_reported_missing(conn, client, small_model):
    """Scoped like every other document endpoint, so ids cannot be probed."""
    # `users` is not truncated between tests -- hence on conflict, which keeps
    # the suite re-runnable.
    conn.execute(
        "insert into users (auth0_sub, email, upload_model_id) values "
        "('test|other', 'other@example.com', %s) on conflict (auth0_sub) do nothing",
        (small_model.id,),
    )
    someone_elses = conn.execute(
        "select id from users where auth0_sub = 'test|other'"
    ).fetchone()[0]
    conn.commit()
    document_id = a_document(conn, title="Theirs", chunks=2, user_id=someone_elses)

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )

    assert response.status_code == 404
