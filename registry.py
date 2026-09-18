"""Reads the `embedding_models` table.

Every embedding type in the system is one row here, and each row owns exactly
one physical vector table named by `table_name`. Postgres cannot enforce that
link, so `setup_db.py` is what keeps the two in sync.
"""

from dataclasses import dataclass

import psycopg

_COLUMNS = "id, provider, model_name, dimensions, table_name, distance, is_active"


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


def default_model_for_user(conn: psycopg.Connection, user_id: int) -> EmbeddingModel:
    """The user's chosen model, falling back to the lowest-id active one."""
    row = conn.execute(
        "select default_embedding_model_id from users where id = %s", (user_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No user with id {user_id}.")

    if row[0] is not None:
        return get_model(conn, model_id=row[0])

    models = list_models(conn)
    if not models:
        raise LookupError("No active embedding models registered.")
    return models[0]
