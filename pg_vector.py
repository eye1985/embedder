"""Builds a PGVectorStore for any registered embedding model.

The embedding service is no longer hardcoded -- it is chosen from the
`embedding_models` row, so one user can hold OpenAI vectors and Qwen vectors
side by side, each in its own table.

Engines and stores are cached per process. A PGEngine wraps its own SQLAlchemy
async pool, so building one per call would open a fresh set of connections
every time and never release them.
"""

import logging
import os
import threading

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy

from db import SQLALCHEMY_URL
from registry import EmbeddingModel

# Columns every vector table carries as real columns rather than JSON. A search
# filters on columns of its own table, so user_id has to live here -- HNSW
# returns its top-k before a join to `documents` could scope the results.
METADATA_COLUMNS = ["chunk_id", "user_id"]

DISTANCE_STRATEGIES = {
    "cosine": DistanceStrategy.COSINE_DISTANCE,
    "inner_product": DistanceStrategy.INNER_PRODUCT,
    "l2": DistanceStrategy.EUCLIDEAN,
}

# SQLAlchemy's pool, separate from the psycopg pool in db.py though both run on
# psycopg 3. Read the connection budget note in db.py before raising these.
ENGINE_POOL_SIZE = int(os.getenv("DB_ENGINE_POOL_SIZE", "5"))
ENGINE_MAX_OVERFLOW = int(os.getenv("DB_ENGINE_MAX_OVERFLOW", "10"))

logger = logging.getLogger(__name__)

_engines: dict[str, PGEngine] = {}
_stores: dict[tuple[str, str], PGVectorStore] = {}
_lock = threading.Lock()


def get_engine(connection_string: str = SQLALCHEMY_URL) -> PGEngine:
    """The cached PGEngine for a connection string, built on first use."""
    with _lock:
        engine = _engines.get(connection_string)
        if engine is None:
            engine = PGEngine.from_connection_string(
                url=connection_string,
                pool_size=ENGINE_POOL_SIZE,
                max_overflow=ENGINE_MAX_OVERFLOW,
                # Discard connections the server has already dropped.
                pool_pre_ping=True,
            )
            _engines[connection_string] = engine
        return engine


def get_embeddings(model: EmbeddingModel) -> Embeddings:
    """Resolve a registry row to the embedding service that produced it."""
    if model.provider == "openai":
        return OpenAIEmbeddings(model=model.model_name, dimensions=model.dimensions)

    if model.provider in ("qwen", "huggingface"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                f"Provider {model.provider!r} needs extra packages: "
                f"uv add langchain-huggingface sentence-transformers"
            ) from exc
        return HuggingFaceEmbeddings(model_name=model.model_name)

    raise ValueError(
        f"Unknown provider {model.provider!r}. Add it to get_embeddings()."
    )


def get_vector_store(
    model: EmbeddingModel, connection_string: str = SQLALCHEMY_URL
) -> PGVectorStore:
    """The cached store for `model`'s vector table.

    The table is not created here -- run `setup_db.py` once to create it.
    """
    key = (connection_string, model.table_name)
    with _lock:
        store = _stores.get(key)
        if store is not None:
            return store

    # Built outside the lock: constructing a store talks to the database, and
    # an embedding service may load a local model.
    try:
        store = PGVectorStore.create_sync(
            engine=get_engine(connection_string),
            table_name=model.table_name,
            embedding_service=get_embeddings(model),
            metadata_columns=METADATA_COLUMNS,
            distance_strategy=DISTANCE_STRATEGIES[model.distance],
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Table {model.table_name!r} is missing or has the wrong columns. "
            f"Create it with: uv run python setup_db.py"
        ) from exc

    with _lock:
        # Another thread may have won the race; keep whichever landed first so
        # callers never hold two stores for one table.
        return _stores.setdefault(key, store)


def close_engines() -> None:
    """Dispose every cached engine. Call on process shutdown.

    PGEngine exposes only an async `close`. `_run_as_sync` is the same bridge
    its own sync API uses -- it hands the coroutine to the background loop, so
    this works whether or not the caller is already inside an event loop.
    `asyncio.run` does not: it raises inside FastAPI's async lifespan.
    """
    with _lock:
        for url, engine in _engines.items():
            try:
                engine._run_as_sync(engine.close())
            except Exception:
                # Shutdown must not raise -- a leaked pool dies with the process.
                logger.warning("Failed to dispose engine for %s", url, exc_info=True)
        _engines.clear()
        _stores.clear()
