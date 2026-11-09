"""Reads the `embedding_models` table.

Every embedding model in the system is one row here. Its vectors live in
`chunk_embeddings`, in the column its width decides.
"""

from dataclasses import dataclass

import psycopg

_COLUMNS = "id, provider, model_name, dimensions, distance, is_active"

# The widest column each element type can index with HNSW. `vector` holds 4
# bytes per dimension, `halfvec` 2, which is why halfvec reaches twice as far.
MAX_VECTOR_DIMENSIONS = 2000


@dataclass(frozen=True)
class EmbeddingModel:
    id: int
    provider: str
    model_name: str
    dimensions: int
    distance: str
    is_active: bool

    @property
    def label(self) -> str:
        """How the model is named outside the database: in the API, the
        export, and job history."""
        return f"{self.provider}/{self.model_name}@{self.dimensions}"

    @property
    def element(self) -> str:
        """`vector` up to 2000 dimensions, `halfvec` above, where `vector`
        can no longer be indexed. halfvec rounds to fp16, which moves a cosine
        score by ~1e-6 and leaves the top-k ordering intact."""
        return "vector" if self.dimensions <= MAX_VECTOR_DIMENSIONS else "halfvec"

    @property
    def column(self) -> str:
        """The `chunk_embeddings` column this model's vectors are stored in."""
        return "embedding" if self.element == "vector" else "embedding_half"


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
    provider: str | None = None,
    model_name: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingModel:
    """Look a model up by id, or by provider + model name + dimensions."""
    if model_id is not None:
        where, params = "id = %s", (model_id,)
    elif provider is not None and model_name is not None and dimensions is not None:
        where = "provider = %s and model_name = %s and dimensions = %s"
        params = (provider, model_name, dimensions)
    else:
        raise ValueError(
            "Pass model_id, or provider, model_name and dimensions together."
        )

    row = conn.execute(
        f"select {_COLUMNS} from embedding_models where {where}", params
    ).fetchone()
    if row is None:
        raise LookupError(f"No embedding_models row matching {params!r}.")
    return _row(row)


def upload_model_for_user(conn: psycopg.Connection, user_id: int) -> EmbeddingModel:
    """The model the upload form pre-selects, and what /upload falls back to."""
    row = conn.execute(
        "select upload_model_id from users where id = %s", (user_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No user with id {user_id}.")
    return get_model(conn, model_id=row[0])
