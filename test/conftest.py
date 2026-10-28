"""Fixtures for the integration tests.

These tests run against a real Postgres, because what they check lives in
Postgres: the one-job-at-a-time rule is a partial unique index, idempotent
embedding is a `NOT EXISTS` plus a unique constraint, and restart recovery is
an UPDATE. Mocking psycopg would assert that we called `insert` -- which would
still pass with the index dropped, and so would prove nothing.

Two things make that safe and free:

  * Its own database (`embeddings_test`), created and migrated here, so a run
    never touches development data.
  * A fake embedding service, so no test spends against the OpenAI key. The
    real PGVectorStore still does the writing, so the foreign keys, the unique
    constraint on chunk_id and the halfvec column are all genuinely exercised.
"""

import os

# Set before importing anything from the project: `db/pool.py` reads
# DATABASE_URL at import time into module constants, so a later override would
# arrive after the pool had already resolved the development database.
_DEV_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:admin@localhost:5432/postgres"
)
TEST_DB_NAME = os.getenv("TEST_DATABASE_NAME", "embeddings_test")
ADMIN_URL = _DEV_URL
TEST_URL = _DEV_URL.rsplit("/", 1)[0] + "/" + TEST_DB_NAME
os.environ["DATABASE_URL"] = TEST_URL

import psycopg
import pytest

from api.services.uploads import UPLOAD_DIR
from db import vectors
from db.pool import close_pool, connection
from db.setup import apply_migrations, create_vector_table
from db.tables.registry import get_model, list_models
from paths import PROJECT_ROOT
from test.fakes import FakeEmbeddings

# Everything a test may write. Truncated between tests, newest-dependency
# first; `cascade` reaches the vector tables, which hang off document_chunks.
#
# `users`, `embedding_models` and `schema_migrations` are deliberately absent:
# they are what the migrations seed, and the tests read them as fixed.
WRITABLE_TABLES = ["jobs", "documents"]


def _create_test_database() -> None:
    """Create the test database if it is not there yet.

    Outside any transaction -- `create database` cannot run inside one.
    """
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        exists = conn.execute(
            "select 1 from pg_database where datname = %s", (TEST_DB_NAME,)
        ).fetchone()
        if exists is None:
            conn.execute(f'create database "{TEST_DB_NAME}"')


@pytest.fixture(scope="session", autouse=True)
def database():
    """A migrated test database, with a vector table per active model.

    Session-scoped: migrating once per run is enough, and each test starts from
    empty tables rather than from a fresh schema.
    """
    _create_test_database()

    with connection() as conn:
        apply_migrations(conn)
        for model in list_models(conn):
            create_vector_table(conn, model)
        conn.commit()

    yield

    vectors.close_engines()
    close_pool()


@pytest.fixture(scope="session", autouse=True)
def no_real_embedding_calls(database):
    """Replace the embedding service with a deterministic fake, for every test.

    Session-wide and autouse rather than opt-in: a test that reached OpenAI
    would spend real money, so the safe state is the default and there is no
    way to forget it.

    `get_vector_store` looks `get_embeddings` up on this module when it builds
    a store, so replacing the attribute is enough -- the store itself, and
    everything it writes, stays real.
    """
    real = vectors.get_embeddings
    vectors.get_embeddings = lambda model: FakeEmbeddings(model.dimensions)
    yield
    vectors.get_embeddings = real
    # Stores built with the fake must not outlive it.
    vectors.close_engines()


@pytest.fixture(autouse=True)
def clean_tables(database):
    """Empty every table a test may write, before each test.

    Truncation rather than a rolled-back transaction, and that is forced by the
    design being tested: `begin_job` claims the slot on its OWN connection and
    commits, so that a running job is visible to other requests. A test holding
    everything open in one transaction could not see those rows at all.

    `restart identity` resets the job id sequence, so each test can refer to
    job 1 and mean its own.
    """
    with connection() as conn:
        conn.execute(f"truncate {', '.join(WRITABLE_TABLES)} restart identity cascade")
        conn.commit()
    yield


@pytest.fixture
def conn():
    """A connection for a test to arrange and inspect rows with.

    Committed on the way in to each use by the helpers in `test/factories.py`,
    because the code under test reads through its own connections.
    """
    with connection() as conn:
        yield conn


@pytest.fixture
def small_model(conn):
    """text-embedding-3-small: 1536 dimensions, a `vector` column."""
    return get_model(conn, table_name="emb_openai_te3_small")


@pytest.fixture
def large_model(conn):
    """text-embedding-3-large: 3072 dimensions, a `halfvec` column."""
    return get_model(conn, table_name="emb_openai_te3_large")


@pytest.fixture
def stored_pdf():
    """The PDF an upload would have left behind, and its `source_uri`.

    Written for real rather than faked: whether a document can offer its file
    is decided by resolving the path under uploads/ and asking the disk, so a
    file that is not there would make a test pass for the wrong reason.
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / "stored-source.pdf"
    path.write_bytes(b"%PDF-1.4 stand-in")
    yield path, str(path.relative_to(PROJECT_ROOT))
    path.unlink(missing_ok=True)


@pytest.fixture
def client():
    """The API, with the app's lifespan deliberately not run.

    Lifespan runs the restart sweep, which would clear a `running` row the test
    had just arranged. The one test that is about startup enters a client as a
    context manager itself, so that wiring is still covered.
    """
    from fastapi.testclient import TestClient

    from api.server import app

    return TestClient(app)
