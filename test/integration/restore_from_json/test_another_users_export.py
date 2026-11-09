"""Restoring an export that came from another user.

Document and chunk ids are unique across every user, so an export handed over
while its original still exists cannot keep them. The copy gets fresh ids and
the original is left exactly as it was.
"""

import uuid

from api.auth.users import viewer
from api.server import app
from db.tables.documents import chunk_rows
from test.factories import a_document, an_export, restore_from_json, vector_count


def test_another_users_export_is_restored_as_a_copy(conn, client, small_model):
    original = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{original}/embeddings",
        json={"models": [small_model.id]},
    )
    exported = an_export(client, original)

    conn.execute(
        "insert into users (auth0_sub, email, upload_model_id) "
        "values ('test|bob', 'bob@example.com', %s)",
        (small_model.id,),
    )
    (bob,) = conn.execute(
        "select id from users where auth0_sub = 'test|bob'"
    ).fetchone()
    conn.commit()
    app.dependency_overrides[viewer] = lambda: (bob, True)

    response = restore_from_json(client, exported)

    assert response.status_code == 200, response.text
    copy = uuid.UUID(response.json()["document_id"])
    assert copy != original

    conn.rollback()
    owners = dict(
        conn.execute(
            "select id, user_id from documents where id = any(%s)", ([original, copy],)
        ).fetchall()
    )
    assert owners[copy] == bob
    assert owners[original] != bob

    original_chunks = {row[0] for row in chunk_rows(conn, original)}
    copy_chunks = {row[0] for row in chunk_rows(conn, copy)}
    assert len(original_chunks) == len(copy_chunks) == 3
    assert not original_chunks & copy_chunks
    assert vector_count(conn, small_model, original) == 3
    assert vector_count(conn, small_model, copy) == 3
