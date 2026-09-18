"""Database connection settings and the psycopg connection pool.

Two URL forms are needed: psycopg3 for the plain SQL this project runs itself,
and the SQLAlchemy form for PGEngine.

Connections are pooled. Nothing here opens a connection per call -- `connection()`
borrows one and hands it back, so a request burst reuses a bounded set of
sockets instead of negotiating a new one each time.
"""

import atexit
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:admin@localhost:5432/postgres"
)
SQLALCHEMY_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")

# --- Connection budget -------------------------------------------------------
#
# One driver, two pools. This pool and the one inside PGEngine both reach
# Postgres through psycopg 3: SQLAlchemy's `postgresql+psycopg://` dialect is
# PGDialectAsync_psycopg, whose DBAPI module is psycopg itself. What differs is
# the layer above -- sync psycopg here, async SQLAlchemy there -- so the
# duplication is in pooling, not in drivers.
#
# They cannot be merged while PGVectorStore is in use. Its only sharing hook,
# PGEngine.from_engine(), takes an AsyncEngine and passes thread=None, which
# leaves _run_as_sync raising "Engine was initialized without a background
# loop" -- breaking create_sync, add_documents and similarity_search. Getting
# to one pool means dropping PGVectorStore and issuing the vector SQL here.
#
# Worst case per process is DB_POOL_MAX_SIZE (10) + DB_ENGINE_POOL_SIZE (5) +
# DB_ENGINE_MAX_OVERFLOW (10) = 25 connections. Postgres defaults to
# max_connections = 100 with 3 reserved, so four workers at these numbers
# exhaust the server. Size them down before running more than two.
POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "30"))

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """The process-wide pool, opened on first use.

    Safe to call from several threads -- FastAPI runs sync endpoints in a
    worker thread, so this is reached concurrently.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = ConnectionPool(
                    DATABASE_URL,
                    name="embeddings",
                    min_size=POOL_MIN_SIZE,
                    max_size=POOL_MAX_SIZE,
                    timeout=POOL_TIMEOUT,
                    # Validate a connection before lending it out, so a database
                    # restart costs one reconnect rather than a failed request.
                    check=ConnectionPool.check_connection,
                    open=False,
                )
                # wait=True so a bad DATABASE_URL fails at startup, not on the
                # first query.
                pool.open(wait=True, timeout=POOL_TIMEOUT)
                _pool = pool
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection.

    Commits when the block exits cleanly, rolls back if it raises, and returns
    the connection to the pool either way.
    """
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    """Close every pooled connection. Call on process shutdown."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


# Scripts that use connection() without a shutdown hook would otherwise let the
# pool's worker threads outlive the interpreter, which surfaces as a
# PythonFinalizationError traceback after the work has already succeeded.
atexit.register(close_pool)


def table_exists(conn: psycopg.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from information_schema.tables "
        "where table_schema = 'public' and table_name = %s",
        (table_name,),
    ).fetchone()
    return row is not None
