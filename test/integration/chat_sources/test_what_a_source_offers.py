"""What a chat answer says about where each cited passage came from.

The sources panel has one job beyond naming documents: telling the reader
whether there is a file behind a passage, and if not, why not. Three states,
and the pair of fields that distinguishes them:

  uploaded, file on disk    has_source=True   origin='upload'   downloadable
  restored from an export   has_source=False  origin='import'   never had one
  uploaded, file gone       has_source=False  origin='upload'   should be there

`has_source` alone cannot separate the last two, which is the whole reason
`documents.origin` exists -- an imported document is not a broken one, and
saying so is the difference between "your own export" and "something is
missing".

The chain under test is the live one in two steps: which retrieved passages
the answer drew on decides which documents are sources (`citations_from`, whose
output is also what gets stored), and the ids are then resolved against the
database and the disk (`resolve_sources`). No OpenAI is reached: the distances
are set by hand, the embeddings are the deterministic fake, and the chunks come
out of Postgres. An answer repeating a passage's text is as close to it as an
answer can be; an unrelated sentence is nowhere near.
"""

from langchain_core.documents import Document

from api.routes.chat import citations_from
from api.services.sources import resolve_sources
from db import vectors
from db.pool import connection
from db.tables.documents import chunk_rows
from db.tables.registry import get_model
from llm.chat import SOURCE_MAX_DISTANCE, _label
from test.factories import TEST_USER_ID, a_document, an_export, restore_from_json


def sources(retrieved, answer=None):
    """The two halves composed as the chat route composes them. Without an
    answer, one that repeats the first passage -- drawing on it."""
    answer = retrieved[0].page_content if answer is None else answer
    with connection() as conn:
        return resolve_sources(conn, TEST_USER_ID, citations_from(retrieved, answer))


def retrieved_from(conn, document_id, distance=SOURCE_MAX_DISTANCE / 2):
    """A document's chunks, shaped the way retrieval hands them to `sources`.

    Put through `_label`, which is what attaches the document id and page the
    panel is built from -- so this covers the real chain rather than a
    hand-made approximation of it. Close to the question unless told otherwise.
    """
    conn.rollback()
    model = get_model(conn, model_id=1)
    embeddings = vectors.embeddings_for(model)
    documents = [
        Document(
            page_content=content,
            metadata={
                "chunk_id": chunk_id,
                "distance": distance,
                "model": model,
                "vector": embeddings.embed_query(content),
            },
        )
        for chunk_id, _index, content, _metadata in chunk_rows(conn, document_id)
    ]
    return _label(TEST_USER_ID, documents)


def test_has_source_follows_a_document_through_a_delete_and_a_reimport(
    conn, client, small_model, stored_pdf
):
    """The whole lifecycle, because the middle step is what decides the end.

    Deleting a document takes its PDF with it, and a JSON export never
    carried that file -- so restoring brings back every row and the one thing
    the chip wants to offer is gone for good. Asserting the two end states
    separately would not show that the delete is what causes the second one.
    """
    path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    exported = an_export(client, document_id)

    (before,) = sources(retrieved_from(conn, document_id))
    assert before["has_source"] is True
    assert before["filename"] == path.name
    assert before["origin"] == "upload"

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.json()["deleted_file"] is True
    assert not path.is_file(), "the delete has to actually take the file"

    assert restore_from_json(client, exported).status_code == 200

    (after,) = sources(retrieved_from(conn, document_id))
    assert after["has_source"] is False
    assert after["filename"] is None
    assert after["origin"] == "import"


def test_a_pdf_deleted_outside_the_app_is_not_reported_as_an_import(
    conn, client, small_model, stored_pdf
):
    """The case that makes `origin` necessary rather than nice to have.

    Deleting the PDF by hand leaves `has_source` false exactly as a restore
    does. Inferring provenance from that would label this document "from
    JSON", which is untrue and would hide a file that really is missing.
    """
    path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    path.unlink()

    (source,) = sources(retrieved_from(conn, document_id))

    assert source["has_source"] is False
    assert source["filename"] is None
    assert source["origin"] == "upload", (
        "a deleted PDF is a missing file, not a document restored from an export"
    )


def test_a_restored_document_whose_pdf_is_still_there_offers_it(
    conn, client, small_model, stored_pdf
):
    """`origin` says where the rows came from, `has_source` what is on disk.

    They are independent: overwriting a document from an export keeps the PDF
    it was imported from, so a restored document can still have one to give.
    The download is the useful thing, and it is offered.
    """
    path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    exported = an_export(client, document_id)
    assert restore_from_json(client, exported, overwrite=True).status_code == 200

    (source,) = sources(retrieved_from(conn, document_id))

    assert path.is_file(), "overwriting must not take the PDF with it"
    assert source["has_source"] is True
    assert source["filename"] == path.name
    assert source["origin"] == "import"


def test_passages_too_far_from_the_question_are_not_sources(conn, client, small_model):
    """Retrieval always returns its closest chunks, however far off. Only
    those close enough to the question are offered as sources."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    far = SOURCE_MAX_DISTANCE + 0.1

    assert sources(retrieved_from(conn, document_id, distance=far)) == []


def test_an_answer_that_did_not_use_the_passages_has_no_sources(
    conn, client, small_model
):
    """Close passages are not enough: an answer saying they did not cover the
    question is nowhere near them, and offers none of them."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    refusal = "The provided passages do not contain that."

    assert sources(retrieved_from(conn, document_id), refusal) == []


def test_only_the_passages_the_answer_drew_on_give_pages(conn, client, small_model):
    """The pages shown are the passages the answer is close to, not every
    passage of the document that was retrieved."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.id]},
    )
    retrieved = retrieved_from(conn, document_id)

    (source,) = sources(retrieved, retrieved[1].page_content)

    assert source["pages"] == [retrieved[1].metadata["page"] + 1]
