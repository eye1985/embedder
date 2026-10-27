"""Working out what re-running a job would even mean.

One resolver behind both the jobs page's Re-run button and the endpoint behind
it, so the button cannot offer something the endpoint then refuses. Four
possible answers, one test each.
"""

from api.services.jobs import replay_plan
from test.factories import a_document


def test_a_job_with_its_document_still_there_replays_as_an_embed(conn):
    """The ordinary case: the document is known, so only vectors are missing."""
    document_id = a_document(conn, title="Nutrition", chunks=3)

    plan, target = replay_plan(conn, 1, "uploads/Nutrition.pdf", str(document_id))

    assert (plan, target) == ("embed", str(document_id))


def test_an_interrupted_upload_is_found_by_its_source_file(conn):
    """The crash case, and the reason `source_uri` is on the row at all.

    An interrupted job never reached `finish_job`, so it never got a
    `document_id` -- but `import_pdf` commits the document before embedding
    starts, so the document does exist. It is found by `source_uri`, which is
    the key uploads already dedupe on.
    """
    document_id = a_document(
        conn, title="Nutrition", chunks=3, source_uri="uploads/Nutrition.pdf"
    )

    plan, target = replay_plan(conn, 1, "uploads/Nutrition.pdf", None)

    assert (plan, target) == ("embed", str(document_id)), (
        "a crashed upload must be repairable even with no document_id on its row"
    )


def test_a_job_that_never_stored_anything_replays_as_an_import(conn):
    """No document, but the PDF is still on disk, so the text must be read again.

    `sample/storybook.pdf` is a file the repo actually ships, which is all
    `replay_plan` checks for.
    """
    plan, target = replay_plan(conn, 1, "sample/storybook.pdf", None)

    assert (plan, target) == ("import", None)


def test_a_job_with_nothing_left_cannot_be_replayed(conn):
    """Document gone and file gone: there is nothing to re-run."""
    plan, target = replay_plan(conn, 1, "uploads/deleted-long-ago.pdf", None)

    assert (plan, target) == ("none", None)
