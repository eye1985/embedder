"""Whose documents a question can reach.

Chat searches the asker's own documents and everyone's public ones. Another
user's private document must not be searched, nor resolve to a title if one of
its chunk ids turns up.
"""

from db.tables.documents import chunk_rows, cited_chunks
from db.tables.embeddings import mark_embedded, search_groups
from test.factories import TEST_USER_ID, a_document, another_user


def searchable(conn, document_id, model_id=1):
    mark_embedded(conn, document_id, model_id)
    conn.commit()


def searched_ids(conn, user_id):
    return {d for ids in search_groups(conn, user_id).values() for d in ids}


def test_another_users_private_document_is_not_searched(conn):
    bob = another_user(conn)
    private = a_document(conn, title="Secret", user_id=bob)
    searchable(conn, private)

    assert str(private) not in searched_ids(conn, TEST_USER_ID)
    assert str(private) not in searched_ids(conn, None)
    assert str(private) in searched_ids(conn, bob)


def test_a_public_document_is_searched_by_everyone(conn):
    bob = another_user(conn)
    public = a_document(conn, title="Handbook", user_id=bob, visibility="public")
    searchable(conn, public)

    assert str(public) in searched_ids(conn, TEST_USER_ID)
    assert str(public) in searched_ids(conn, None)


def test_a_private_chunk_resolves_to_nothing_for_anyone_else(conn):
    bob = another_user(conn)
    private = a_document(conn, title="Secret", user_id=bob)
    chunk_ids = [row[0] for row in chunk_rows(conn, private)]

    assert cited_chunks(conn, TEST_USER_ID, chunk_ids) == []
    assert cited_chunks(conn, None, chunk_ids) == []
    assert len(cited_chunks(conn, bob, chunk_ids)) == len(chunk_ids)
