"""A failed upload leaves nothing behind, even when the failure aborted the
transaction the cleanup runs on."""

import psycopg
import pytest

from api.routes.uploads import discard_upload
from api.services.uploads import user_upload_dir
from db.tables.documents import find_document
from paths import PROJECT_ROOT
from test.factories import TEST_USER_ID, a_document


def test_cleanup_survives_an_aborted_transaction(conn):
    folder = user_upload_dir(TEST_USER_ID)
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "half-done.pdf"
    pdf.write_bytes(b"%PDF-1.4 stand-in")
    source_uri = str(pdf.relative_to(PROJECT_ROOT))
    a_document(conn, title="half-done", source_uri=source_uri)

    with pytest.raises(psycopg.errors.DivisionByZero):
        conn.execute("select 1 / 0")

    assert discard_upload(conn, TEST_USER_ID, pdf) is True
    assert not pdf.exists()
    assert find_document(conn, TEST_USER_ID, source_uri) is None
