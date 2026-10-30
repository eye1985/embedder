"""What a stored answer still says about its sources, later on.

A conversation outlives the documents it was about, so a citation cannot be a
foreign key: cascading would erase it, and setting it null would break the
link for good. It is a plain document and chunk id, re-resolved on every read.

That choice is only correct because of what a JSON export carries. Restoring
one puts the document back under the id it had, and its chunks under theirs
(`restore_document`), so the ids a stored answer kept are the ids that come
back -- which is what lets an answer written months ago find its sources again
after the document behind them was deleted and reimported.

These tests go through `store_turn` and `read_conversation` rather than
`/chat`, so no OpenAI is reached: the answer is a literal string and the
citations are built from stored chunks.
"""

import uuid

import pytest
from langchain_core.documents import Document

from api.config import CURRENT_USER_ID
from api.routes.chat import citations_from
from api.services.conversations import read_conversation
from db.tables.chat import message_rows, store_turn
from db.tables.documents import chunk_rows
from llm.chat import _label
from test.factories import a_document, an_export, restore_from_json

A_QUESTION = "What does it say about nutrition?"
AN_ANSWER = "It covers the basics [1]."


def a_stored_turn(conn, document_id, *, question=A_QUESTION, answer=AN_ANSWER):
    """One exchange stored the way the chat route stores it.

    The citations go through `citations_from` over labelled chunks, which is
    the real live path -- a hand-written citation list would prove the storage
    works and not that what the route stores is what it resolves.
    """
    conn.rollback()
    documents = _label(
        CURRENT_USER_ID,
        [
            Document(page_content=content, metadata={"chunk_id": chunk_id})
            for chunk_id, _index, content, _metadata in chunk_rows(conn, document_id)
        ],
    )
    conversation_id = store_turn(
        conn,
        CURRENT_USER_ID,
        None,
        question,
        answer,
        citations_from(documents, answer),
        summary=None,
        summarized=0,
    )
    conn.commit()
    return conversation_id


def answer_sources(conn, conversation_id):
    """The sources on the one answer in a stored conversation."""
    conn.rollback()
    found = read_conversation(conn, CURRENT_USER_ID, conversation_id)
    (answer,) = [m for m in found["messages"] if m["role"] == "assistant"]
    return answer["sources"]


def test_a_citation_survives_a_delete_and_a_reimport(
    conn, client, small_model, stored_pdf
):
    """The lifecycle the no-foreign-key choice exists for.

    Deleting the document takes its chunks with it, so the stored ids resolve
    to nothing and the answer offers no sources -- while keeping them, which
    is what makes the third step possible. Restoring the export brings the
    same ids back, and the source returns as the import it now is.
    """
    _path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    client.post(
        f"/api/documents/{document_id}/embeddings",
        json={"models": [small_model.table_name]},
    )
    exported = an_export(client, document_id)
    conversation_id = a_stored_turn(conn, document_id)

    (before,) = answer_sources(conn, conversation_id)
    assert before["document_id"] == str(document_id)
    assert before["has_source"] is True

    assert client.delete(f"/api/documents/{document_id}").status_code == 200

    assert answer_sources(conn, conversation_id) == [], (
        "a deleted document resolves to nothing, so the answer offers no source"
    )

    assert restore_from_json(client, exported).status_code == 200

    (after,) = answer_sources(conn, conversation_id)
    assert after["document_id"] == str(document_id), (
        "the export restores the document under its own id, which is the id "
        "the stored citation kept"
    )
    assert after["origin"] == "import"
    assert after["has_source"] is False


def test_deleting_a_document_leaves_the_citation_in_place(
    conn, client, small_model, stored_pdf
):
    """The row behind the previous test's third step.

    The chip disappearing has to be a resolution that found nothing, not a
    citation that was deleted -- a cascade here would read the same way until
    the document came back, and then stay empty forever.
    """
    _path, source_uri = stored_pdf
    document_id = a_document(conn, title="Nutrition", chunks=3, source_uri=source_uri)
    conversation_id = a_stored_turn(conn, document_id)

    assert client.delete(f"/api/documents/{document_id}").status_code == 200

    conn.rollback()
    stored = {
        role: citations
        for _seq, role, _content, citations in message_rows(conn, conversation_id)
    }
    assert stored["assistant"] == [
        {
            "document_id": str(document_id),
            "chunk_ids": stored["assistant"][0]["chunk_ids"],
        }
    ]
    assert len(stored["assistant"][0]["chunk_ids"]) == 3


def test_a_question_is_stored_with_its_answer_and_nothing_else(
    conn, client, small_model
):
    """A turn is one exchange: the question, then the reply, in that order.

    Citations belong to the answer -- a question cites nothing, because it was
    written before anything was retrieved.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    conversation_id = a_stored_turn(conn, document_id)

    conn.rollback()
    rows = message_rows(conn, conversation_id)

    assert [(seq, role) for seq, role, _content, _citations in rows] == [
        (1, "user"),
        (2, "assistant"),
    ]
    assert rows[0][2] == A_QUESTION
    assert rows[0][3] == []
    assert rows[1][2] == AN_ANSWER, (
        "the answer is stored as the model wrote it, markers and all"
    )


def test_a_conversation_is_named_after_the_question_that_opened_it(conn, client):
    """The list has to be readable before anyone has renamed anything."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    conversation_id = a_stored_turn(conn, document_id)

    conn.rollback()
    found = read_conversation(conn, CURRENT_USER_ID, conversation_id)
    assert found["title"] == A_QUESTION


def test_a_second_turn_continues_the_same_conversation(conn, client):
    """`seq` numbers the whole conversation, not each turn.

    The history a later question is answered against is read back in this
    order, so a turn that restarted the numbering would interleave with the
    one before it.
    """
    document_id = a_document(conn, title="Nutrition", chunks=3)
    conversation_id = a_stored_turn(conn, document_id)

    store_turn(
        conn,
        CURRENT_USER_ID,
        conversation_id,
        "And the second thing?",
        "Also covered [1].",
        [],
        summary=None,
        summarized=0,
    )
    conn.commit()

    conn.rollback()
    rows = message_rows(conn, conversation_id)
    assert [seq for seq, *_ in rows] == [1, 2, 3, 4]
    assert rows[2][2] == "And the second thing?"


@pytest.mark.parametrize("missing", ["conversation", "user"])
def test_another_users_conversation_is_absent_rather_than_theirs(conn, client, missing):
    """Reading is scoped to the user, like every other document request."""
    document_id = a_document(conn, title="Nutrition", chunks=3)
    conversation_id = a_stored_turn(conn, document_id)

    conn.rollback()
    if missing == "conversation":
        assert read_conversation(conn, CURRENT_USER_ID, uuid.uuid4()) is None
    else:
        assert read_conversation(conn, CURRENT_USER_ID + 1, conversation_id) is None
