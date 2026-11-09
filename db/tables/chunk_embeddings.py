"""The `chunk_embeddings` table: every model's vectors, one row per chunk.

Each model has its own partial HNSW index (`sql/009`), on its column cast to
its width. A search only uses that index when it repeats the same cast and
names the model id as a literal -- a bound parameter can hide the id from the
planner once psycopg prepares the statement -- so `nearest` composes both
from the registry row rather than binding them.
"""

import json
import uuid
from collections.abc import Iterable

import psycopg
from psycopg import sql

from db.tables.registry import EmbeddingModel

OPERATORS = {"cosine": "<=>", "inner_product": "<#>", "l2": "<->"}


def _literal(vector: list[float]) -> str:
    """pgvector's text form, "[0.1,-0.2]", which casts to vector or halfvec."""
    return json.dumps(vector, separators=(",", ":"))


def unembedded_chunks(
    conn: psycopg.Connection, document_id: uuid.UUID, model_id: int
) -> list[tuple[uuid.UUID, str]]:
    """A document's chunks this model holds no vector for yet: (chunk_id, content).

    Skipping what is already embedded is what makes re-running cheap, and makes
    adding a second model embed only what that model is missing.
    """
    return conn.execute(
        """
        select c.id, c.content
        from document_chunks c
        where c.document_id = %s
          and not exists (
              select 1 from chunk_embeddings e
              where e.chunk_id = c.id and e.model_id = %s
          )
        order by c.chunk_index
        """,
        (document_id, model_id),
    ).fetchall()


def insert_vectors(
    conn: psycopg.Connection,
    model: EmbeddingModel,
    document_id: uuid.UUID,
    rows: Iterable[tuple[uuid.UUID, list[float]]],
) -> int:
    """Write one model's vectors for a document. Returns how many landed.

    `rows` is (chunk_id, vector). A vector of the wrong width is refused by the
    model's index, whose cast to a fixed width runs on every insert.
    """
    query = sql.SQL(
        "insert into chunk_embeddings (chunk_id, model_id, document_id, {column}) "
        "values (%s, %s, %s, %s::{element})"
    ).format(column=sql.Identifier(model.column), element=sql.SQL(model.element))
    params = [
        (chunk_id, model.id, document_id, _literal(vector)) for chunk_id, vector in rows
    ]
    with conn.cursor() as cursor:
        cursor.executemany(query, params)
    return len(params)


def vector_rows(
    conn: psycopg.Connection, document_id: uuid.UUID, model: EmbeddingModel
) -> list[tuple[str, str]]:
    """One model's vectors for a document: (chunk_id, vector) pairs.

    The vector comes back in pgvector's text form, "[0.1,-0.2]" -- already
    valid JSON, so the floats come back without registering a type adapter.
    """
    query = sql.SQL(
        "select chunk_id::text, {column}::text from chunk_embeddings "
        "where document_id = %s and model_id = %s"
    ).format(column=sql.Identifier(model.column))
    return conn.execute(query, (document_id, model.id)).fetchall()


def embedded_counts(
    conn: psycopg.Connection, user_id: int
) -> dict[int, dict[str, int]]:
    """Per model id, how many chunks of each of the user's documents it embedded."""
    rows = conn.execute(
        """
        select e.model_id, e.document_id::text, count(*)
        from chunk_embeddings e
        join documents d on d.id = e.document_id
        where d.user_id = %s
        group by e.model_id, e.document_id
        """,
        (user_id,),
    ).fetchall()

    counts: dict[int, dict[str, int]] = {}
    for model_id, document_id, n in rows:
        counts.setdefault(model_id, {})[document_id] = n
    return counts


def delete_vectors(
    conn: psycopg.Connection, document_id: uuid.UUID, model_id: int
) -> int:
    """Drop one model's vectors for a document. Returns how many went.

    The chunks survive, so the text does not need re-extracting.
    """
    return conn.execute(
        "delete from chunk_embeddings where document_id = %s and model_id = %s",
        (document_id, model_id),
    ).rowcount


def nearest(
    conn: psycopg.Connection,
    model: EmbeddingModel,
    query_vector: list[float],
    document_ids: list[str],
    limit: int,
) -> list[tuple[str, str, str, float]]:
    """The chunks closest to `query_vector`, best first:
    (chunk_id, content, vector, distance).

    Restricted to `document_ids`. HNSW returns at most `ef_search` rows before
    that filter, so iterative scan keeps walking the index until `limit` match.

    The vector comes back as text for the caller that re-ranks with it (MMR).
    """
    cast = sql.SQL("{element}({dimensions})").format(
        element=sql.SQL(model.element), dimensions=sql.Literal(model.dimensions)
    )
    query = sql.SQL(
        """
        select n.chunk_id::text, c.content, n.vector, n.distance
        from (
            select chunk_id, {column}::text as vector,
                   {column}::{cast} {operator} %s::{cast} as distance
            from chunk_embeddings
            where model_id = {model_id} and document_id = any(%s::uuid[])
            order by distance
            limit %s
        ) n
        join document_chunks c on c.id = n.chunk_id
        order by n.distance
        """
    ).format(
        column=sql.Identifier(model.column),
        cast=cast,
        operator=sql.SQL(OPERATORS[model.distance]),
        model_id=sql.Literal(model.id),
    )
    conn.execute("set local hnsw.iterative_scan = relaxed_order")
    conn.execute(
        sql.SQL("set local hnsw.ef_search = {}").format(sql.Literal(max(limit, 100)))
    )
    return conn.execute(query, (_literal(query_vector), document_ids, limit)).fetchall()
