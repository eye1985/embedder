"""What an answer says about where its passages came from.

Two halves. `source_path` resolves a document's stored `source_uri` to a file
on disk, and the resolvers turn stored citations into the chips a reader sees.
Nothing here knows about LangChain or HTTP: which documents an answer cited is
worked out on the live path (`api/routes/chat.py`) and stored, and this reads
ids back.

That is what keeps a replayed answer honest. A stored citation is a plain
document and chunk id, so it is re-resolved against the database and the disk
every time it is rendered rather than describing how things stood when the
answer was written.
"""

from pathlib import Path

import psycopg

from api.services.uploads import UPLOAD_DIR
from db.tables.documents import cited_chunks
from paths import PROJECT_ROOT

# Where a document's file may be: uploads/ for anything imported through the
# web, sample/ for the PDFs that ship with the repo.
SOURCE_DIRS = (UPLOAD_DIR, PROJECT_ROOT / "sample")


def source_path(source_uri: str | None) -> Path | None:
    """The file behind a document, if it is still there.

    Checked against the directories the app imports from, so a `source_uri`
    edited to point elsewhere on disk resolves to nothing.
    """
    if not source_uri:
        return None
    path = (PROJECT_ROOT / source_uri).resolve()
    if not any(path.is_relative_to(directory) for directory in SOURCE_DIRS):
        return None
    return path if path.is_file() else None


def _chunk_ids(citations: list[dict]) -> list[str]:
    return [chunk_id for entry in citations for chunk_id in entry["chunk_ids"]]


def _by_document(rows: list[tuple]) -> dict[str, dict]:
    """Resolved chunk rows collapsed to one entry per document.

    `filename` is what the download actually arrives as, which is why it is
    kept alongside the title -- the two differ, the title being the stem.
    Pages are every retrieved page of the document, shifted into the PDF's own
    numbering, which starts at 1; a citation is per document, so it does not
    say which passage carried a claim.

    `origin` tells the two fileless cases apart. A restored export never had a
    PDF here, so its passages came out of the user's own stored chunks and are
    no less theirs; an upload whose file went missing is a file that should be
    there and is not. Both are unclickable, and the chip says which.

    The disk is consulted once per document rather than once per row: a
    document contributes a row per chunk, and they all resolve to the same
    file.
    """
    found: dict[str, dict] = {}
    for _chunk_id, document_id, title, source_uri, origin, page in rows:
        source = found.get(document_id)
        if source is None:
            # Resolved against the disk, not just the column: a file deleted
            # from under its row would otherwise be offered as a link that
            # 404s.
            stored = source_path(source_uri)
            source = found[document_id] = {
                "title": title,
                "filename": stored.name if stored else None,
                "pages": set(),
                "has_source": stored is not None,
                "origin": origin,
            }
        if page is not None:
            source["pages"].add(page + 1)
    return found


def _chips(found: dict[str, dict], citations: list[dict]) -> list[dict]:
    """One citation list as the documents a reader can be offered, best first.

    Rank is the position in `citations`, so the order out follows the order
    in. A document that no longer resolves is left out rather than reported as
    empty -- that is the deleted case: the ids outlive the rows, so they are
    still here to try, and nothing is offered until they resolve again, which
    they do under the same ids if the document is restored from a JSON export.
    """
    return [
        {
            "document_id": document_id,
            "title": found[document_id]["title"],
            "filename": found[document_id]["filename"],
            "pages": sorted(found[document_id]["pages"]),
            "has_source": found[document_id]["has_source"],
            "origin": found[document_id]["origin"],
        }
        for document_id in (entry["document_id"] for entry in citations)
        if document_id in found
    ]


def resolve_sources(
    conn: psycopg.Connection, user_id: int, citations: list[dict]
) -> list[dict]:
    """One answer's stored citations, resolved.

    `citations` is what `chat_messages.citations` holds: one entry per cited
    document in rank order, each naming the retrieved chunks that document
    contributed.
    """
    chunk_ids = _chunk_ids(citations)
    if not chunk_ids:
        return []
    # Scoped to the user by the query, so a chunk id belonging to someone else
    # resolves to no row rather than to their document.
    return _chips(_by_document(cited_chunks(conn, user_id, chunk_ids)), citations)


def resolve_each(
    conn: psycopg.Connection, user_id: int, transcript: list[list[dict]]
) -> list[list[dict]]:
    """A whole transcript's citations, one list per message, in one query.

    Resolving each message on its own would be a query per answer, so a
    conversation would cost a round trip per turn to open. The chunk ids are
    asked for together and the rows shared out afterwards.
    """
    chunk_ids = [
        chunk_id for citations in transcript for chunk_id in _chunk_ids(citations)
    ]
    if not chunk_ids:
        return [[] for _ in transcript]

    found = _by_document(cited_chunks(conn, user_id, chunk_ids))
    return [_chips(found, citations) for citations in transcript]
