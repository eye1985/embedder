"""Reads the `embedding_models` table.

Every embedding type in the system is one row here, and each row owns exactly
one physical vector table named by `table_name`. Postgres cannot enforce that
link, so `db/setup.py` is what keeps the two in sync.
"""

from dataclasses import dataclass

import psycopg

_COLUMNS = "id, provider, model_name, dimensions, table_name, distance, is_active"


def table_exists(conn: psycopg.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from information_schema.tables "
        "where table_schema = 'public' and table_name = %s",
        (table_name,),
    ).fetchone()
    return row is not None


@dataclass(frozen=True)
class EmbeddingModel:
    id: int
    provider: str
    model_name: str
    dimensions: int
    table_name: str
    distance: str
    is_active: bool


def _row(row) -> EmbeddingModel:
    return EmbeddingModel(*row)


def list_models(
    conn: psycopg.Connection, *, active_only: bool = True
) -> list[EmbeddingModel]:
    sql = f"select {_COLUMNS} from embedding_models"
    if active_only:
        sql += " where is_active"
    sql += " order by id"
    return [_row(r) for r in conn.execute(sql).fetchall()]


def get_model(
    conn: psycopg.Connection,
    *,
    model_id: int | None = None,
    table_name: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
) -> EmbeddingModel:
    """Look a model up by id, by table name, or by provider + model name."""
    if model_id is not None:
        where, params = "id = %s", (model_id,)
    elif table_name is not None:
        where, params = "table_name = %s", (table_name,)
    elif provider is not None and model_name is not None:
        where, params = "provider = %s and model_name = %s", (provider, model_name)
    else:
        raise ValueError("Pass model_id, table_name, or both provider and model_name.")

    row = conn.execute(
        f"select {_COLUMNS} from embedding_models where {where}", params
    ).fetchone()
    if row is None:
        raise LookupError(f"No embedding_models row matching {params!r}.")
    return _row(row)


def _user_model(conn: psycopg.Connection, user_id: int, column: str) -> EmbeddingModel:
    """The model named by one of the user's model columns.

    `column` is a literal from this module, never from a request.
    """
    row = conn.execute(
        f"select {column} from users where id = %s", (user_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No user with id {user_id}.")
    return get_model(conn, model_id=row[0])


def upload_model_for_user(conn: psycopg.Connection, user_id: int) -> EmbeddingModel:
    """The model the upload form pre-selects, and what /upload falls back to."""
    return _user_model(conn, user_id, "upload_model_id")


def models_with_tables(conn: psycopg.Connection) -> list[EmbeddingModel]:
    """Registered models whose vector table actually exists.

    `list_models(active_only=False)` includes rows that were never set up --
    `db/setup.py` creates tables for active models only -- and querying a
    missing table raises, so those rows are dropped before any count or read.
    """
    return [
        m
        for m in list_models(conn, active_only=False)
        if table_exists(conn, m.table_name)
    ]
