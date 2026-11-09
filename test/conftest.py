"""Fixtures for the integration tests.

These tests run against a real Postgres, because what they check lives in
Postgres: the one-job-at-a-time rule is a partial unique index, idempotent
embedding is a `NOT EXISTS` plus a unique constraint, and restart recovery is
an UPDATE. Mocking psycopg would assert that we called `insert` -- which would
still pass with the index dropped, and so would prove nothing.

Three things make that safe and free:

  * Its own database (`embeddings_test`), created and migrated at the start of
    every run and dropped at the end, so a run never touches development data
    and never inherits an old schema.
  * Its own upload folder (`.test-uploads/`), created and removed the same
    way, so a test never writes or deletes a real user's PDFs.
  * A fake embedding service, so no test spends against the OpenAI key. The
    real SQL still does the writing and the searching, so the foreign keys,
    the partial indexes and the halfvec column are all genuinely exercised.
"""

import os
import shutil

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

from api.services import uploads
from api.services.uploads import user_upload_dir
from db import vectors
from db.pool import close_pool, connection
from db.setup import apply_migrations
from db.tables.registry import get_model
from paths import PROJECT_ROOT
from test.factories import TEST_USER_ID
from test.fakes import FakeEmbeddings

# Everything a test may write. Truncated between tests, newest-dependency
# first; `cascade` reaches chunk_embeddings, which hangs off documents.
#
# `users`, `embedding_models` and `schema_migrations` are deliberately absent:
# the migrations seed the models, `database` adds the test user, and the tests
# read them as fixed.
WRITABLE_TABLES = ["jobs", "conversations", "documents"]


def _drop_test_database() -> None:
    """Drop the test database if it exists.

    `force` ends any connection still open on it. Outside any transaction --
    `create database` and `drop database` cannot run inside one.
    """
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'drop database if exists "{TEST_DB_NAME}" with (force)')


@pytest.fixture(scope="session", autouse=True)
def database():
    """A test database created and migrated for this run, and dropped after.

    Dropped first too, in case an interrupted run left one behind, so the
    schema always comes from the migrations as they are now.

    Session-scoped: migrating once per run is enough, and each test starts from
    empty tables rather than from a fresh schema.
    """
    _drop_test_database()
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'create database "{TEST_DB_NAME}"')

    with connection() as conn:
        apply_migrations(conn)
        # A fixed id so factories can default to it. The sequence is moved past
        # it, or the next user a test inserts would collide.
        conn.execute(
            "insert into users (id, auth0_sub, email, upload_model_id) "
            "values (%s, 'test|user', 'test@example.com', 1)",
            (TEST_USER_ID,),
        )
        conn.execute(
            "select setval(pg_get_serial_sequence('users', 'id'), max(id)) from users"
        )
        conn.commit()

    yield

    close_pool()
    _drop_test_database()


@pytest.fixture(scope="session", autouse=True)
def test_uploads():
    """Point uploaded PDFs at a folder of the run's own.

    Inside the project, because a `source_uri` is stored relative to it.
    Removed first too, in case an interrupted run left one behind.
    """
    folder = PROJECT_ROOT / ".test-uploads"
    shutil.rmtree(folder, ignore_errors=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(uploads, "UPLOAD_DIR", folder)
        yield
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def no_real_embedding_calls(database):
    """Replace the embedding service with a deterministic fake, for every test.

    Session-wide and autouse rather than opt-in: a test that reached OpenAI
    would spend real money, so the safe state is the default and there is no
    way to forget it.

    `embeddings_for` looks `get_embeddings` up on this module when it builds a
    service, so replacing the attribute is enough -- everything written with
    the vectors stays real.
    """
    real = vectors.get_embeddings
    vectors.clear_embeddings()
    vectors.get_embeddings = lambda model: FakeEmbeddings(model.dimensions)
    yield
    vectors.get_embeddings = real
    # Services built with the fake must not outlive it.
    vectors.clear_embeddings()


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
    """text-embedding-3-small: 1536 dimensions, the `vector` column."""
    return get_model(conn, model_id=1)


@pytest.fixture
def large_model(conn):
    """text-embedding-3-large: 3072 dimensions, the `halfvec` column."""
    return get_model(conn, model_id=2)


@pytest.fixture
def stored_pdf():
    """The PDF an upload would have left behind, and its `source_uri`.

    Written for real rather than faked: whether a document can offer its file
    is decided by resolving the path under the user's upload folder and asking
    the disk, so a file that is not there would make a test pass for the wrong
    reason.
    """
    folder = user_upload_dir(TEST_USER_ID)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "stored-source.pdf"
    path.write_bytes(b"%PDF-1.4 stand-in")
    yield path, str(path.relative_to(PROJECT_ROOT))
    path.unlink(missing_ok=True)


@pytest.fixture
def client():
    """The API, with the app's lifespan deliberately not run.

    Lifespan runs the restart sweep, which would clear a `running` row the test
    had just arranged. The one test that is about startup enters a client as a
    context manager itself, so that wiring is still covered.

    Signed in as the test user: the session lookup is replaced, so no Auth0
    cookie is needed.
    """
    from fastapi.testclient import TestClient

    from api.auth.users import current_user, optional_user
    from api.server import app

    app.dependency_overrides[current_user] = lambda: TEST_USER_ID
    app.dependency_overrides[optional_user] = lambda: TEST_USER_ID
    yield TestClient(app)
    app.dependency_overrides.clear()
