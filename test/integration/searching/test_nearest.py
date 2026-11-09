"""A search filtered to a few documents still fills its limit.

HNSW returns at most `ef_search` rows before the document filter. The planner
is forced onto the index, since at this size it would pick an exact scan.
"""

from db import vectors
from db.tables.chunk_embeddings import nearest
from test.factories import a_document


def test_a_filtered_search_returns_the_full_limit(conn, client, small_model):
    documents = [a_document(conn, title=f"Doc {i}", chunks=20) for i in range(6)]
    for document_id in documents:
        response = client.post(
            f"/api/documents/{document_id}/embeddings",
            json={"models": [small_model.id]},
        )
        assert response.status_code == 200, response.text
    allowed = [str(documents[0])]
    query = vectors.embeddings_for(small_model).embed_query("question")

    conn.rollback()
    conn.execute("set local enable_seqscan = off")
    conn.execute("set local enable_sort = off")
    rows = nearest(conn, small_model, query, allowed, 20)

    assert len(rows) == 20
