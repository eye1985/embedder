"""The `document_embeddings` table: which models embedded which documents.

A junction table between `documents` and `embedding_models`. The vectors stay
in the per-model tables, one row per chunk; this is the same relationship at
document granularity, carrying the flag that has nowhere else to live.

A row exists only where vectors exist, so "never embedded by this model" stays
distinguishable from "embedded, not the one being searched".

At most one row per document has `enabled`: a document is retrieved through
exactly one model, chosen at upload and changeable afterwards. Two models hold
the same chunks, so the choice decides which ranking a question gets, not
which text it can reach. Nothing in Postgres enforces the one-per-document
rule -- `set_document_model` is what keeps it true, by clearing the others in
the same statement pair.
"""

import uuid
from collections.abc import Iterable

import psycopg


def mark_embedded(
    conn: psycopg.Connection, document_id: uuid.UUID, model_id: int
) -> None:
    """Record that `model_id` now holds vectors for `document_id`.

    Enabled only if the document has no searched model yet, so the first model
    in is the one used and a later one is merely available. Adding a model to
    a document never moves the choice out from under the user; `/upload` and
    the documents page call `set_document_model` for that.
    """
    conn.execute(
        """
        insert into document_embeddings (document_id, model_id, enabled)
        select %s, %s, not exists (
            select 1 from document_embeddings
            where document_id = %s and enabled
        )
        on conflict do nothing
        """,
        (document_id, model_id, document_id),
    )


def unmark_embedded(
    conn: psycopg.Connection, document_id: uuid.UUID, model_id: int
) -> None:
    """Forget the pairing. Call wherever its vectors are deleted.

    If that was the searched model, the document is left with none and drops
    out of search until one is chosen -- better than silently promoting
    another, which would change answers without anyone asking.
    """
    conn.execute(
        "delete from document_embeddings where document_id = %s and model_id = %s",
        (document_id, model_id),
    )


def set_document_model(
    conn: psycopg.Connection, user_id: int, document_id: uuid.UUID, model_id: int
) -> None:
    """Make `model_id` the one this document is retrieved through.

    Every other model keeps its vectors and stops being searched. Raises
    LookupError if the user has no such pairing -- which includes a document
    that is not theirs, so this cannot be used to probe ids.
    """
    row = conn.execute(
        """
        update document_embeddings de set enabled = (de.model_id = %s)
        from documents d
        where d.id = de.document_id and d.user_id = %s and de.document_id = %s
        returning de.model_id
        """,
        (model_id, user_id, document_id),
    ).fetchall()

    if model_id not in {r[0] for r in row}:
        raise LookupError(
            f"Document {document_id} has no embeddings from model {model_id}."
        )


def search_groups(conn: psycopg.Connection, user_id: int) -> dict[int, list[str]]:
    """The documents to retrieve from, grouped by the model that searches them.

    One entry per model actually in use. A single entry is the common case and
    costs one query embedding; several means the question has to be embedded
    once per model and the results fused.

    Document ids come back as uuid strings -- they go straight into a vector
    store filter.
    """
    rows = conn.execute(
        """
        select de.model_id, de.document_id::text
        from document_embeddings de
        join documents d on d.id = de.document_id
        where d.user_id = %s and d.searchable and de.enabled
        order by de.model_id
        """,
        (user_id,),
    ).fetchall()

    groups: dict[int, list[str]] = {}
    for model_id, document_id in rows:
        groups.setdefault(model_id, []).append(document_id)
    return groups


def models_by_document(
    conn: psycopg.Connection, user_id: int
) -> dict[str, dict[int, bool]]:
    """Per document id, every model that embedded it and whether it is the one
    being searched. Models that never embedded a document are absent."""
    rows = conn.execute(
        """
        select de.document_id::text, de.model_id, de.enabled
        from document_embeddings de
        join documents d on d.id = de.document_id
        where d.user_id = %s
        """,
        (user_id,),
    ).fetchall()

    flags: dict[str, dict[int, bool]] = {}
    for document_id, model_id, enabled in rows:
        flags.setdefault(document_id, {})[model_id] = enabled
    return flags


def document_counts(
    conn: psycopg.Connection, user_id: int, model_ids: Iterable[int]
) -> dict[int, int]:
    """Per model, how many of the user's documents it has embedded."""
    rows = conn.execute(
        """
        select de.model_id, count(*)
        from document_embeddings de
        join documents d on d.id = de.document_id
        where d.user_id = %s and de.model_id = any(%s)
        group by de.model_id
        """,
        (user_id, list(model_ids)),
    ).fetchall()
    return dict(rows)
