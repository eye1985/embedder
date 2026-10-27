"""The chunks in Postgres are the input, never the PDF.

Re-embedding reads `document_chunks`, so no extraction runs and the source
file does not even have to exist. What lands is one distinct vector per chunk,
at the model's declared width.
"""

from test.factories import a_document


def test_the_source_pdf_is_never_read_again(conn, client, large_model):
    """Nothing is re-extracted, which is what makes this cheap.

    The document's `source_uri` points at a file that does not exist. If
    re-embedding touched the PDF at all this would fail; it reads chunks out of
    Postgres instead.
    """
    document_id = a_document(
        conn, title="Nutrition", chunks=3, source_uri="uploads/deleted-long-ago.pdf"
    )

    response = client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )

    assert response.status_code == 200
    assert response.json()["embedded"] == {large_model.table_name: 3}


def test_each_chunk_gets_its_own_vector(conn, client, large_model):
    """One vector per chunk, all distinct, at the model's declared width.

    Guards against the failure that looks like success: rows written, but every
    one of them the same vector, or the wrong number of dimensions.
    """
    document_id = a_document(conn, title="Nutrition", chunks=4)

    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [large_model.table_name]},
    )

    conn.rollback()
    rows, chunks, distinct, dims = conn.execute(
        f"""
        select count(*), count(distinct chunk_id), count(distinct embedding::text),
               max(vector_dims(embedding))
        from "{large_model.table_name}"
        where document_id = %s
        """,
        (document_id,),
    ).fetchone()

    assert rows == 4
    assert chunks == 4, "no chunk embedded twice"
    assert distinct == 4, "every chunk got a different vector"
    assert dims == large_model.dimensions == 3072
