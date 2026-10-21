"""Schema setup.

Applies the numbered migrations in `sql/`, then creates one vector table per
active row in `embedding_models`.

Run it as `uv run python -m db.setup`: once before starting the server, and
again after adding a row to `embedding_models` -- new rows get their table on
the next run.
"""

import argparse
from pathlib import Path

import psycopg
from psycopg import sql

from db.pool import close_pool, connection, table_exists
from db.registry import EmbeddingModel, get_model, list_models
from paths import MIGRATIONS_DIR

# How wide a column each element type can be and still take an HNSW index.
# `vector` holds 4 bytes per dimension, `halfvec` 2 -- which is why halfvec
# reaches twice as far before hitting the same index limit.
MAX_HNSW_DIMENSIONS = {"vector": 2000, "halfvec": 4000}

# Opclass per element type and distance. The distance keys are the three values
# `embedding_models.distance` allows.
INDEX_OPS = {
    "vector": {
        "cosine": "vector_cosine_ops",
        "inner_product": "vector_ip_ops",
        "l2": "vector_l2_ops",
    },
    "halfvec": {
        "cosine": "halfvec_cosine_ops",
        "inner_product": "halfvec_ip_ops",
        "l2": "halfvec_l2_ops",
    },
}


def element_type(dimensions: int) -> str:
    """`vector` or `halfvec`, whichever can index a column this wide.

    A model past 2000 dimensions cannot have an indexed `vector` column, but
    `halfvec` indexes to 4000, so storing it as halfvec is what makes an index
    possible at all -- and it halves the column besides. The cost is fp16
    precision: components round to about three decimal places, which moves a
    cosine score by ~1e-6 and leaves the top-k ordering intact.

    Nothing is gained by using halfvec below the vector limit, where the
    full-precision column is already indexable, so that stays `vector`.

    `vector -> halfvec` is an implicit cast, so a store writing vectors into a
    halfvec column needs no changes; langchain binds its query vector untyped
    and the operator resolves to halfvec, keeping the index in play.
    """
    return "vector" if dimensions <= MAX_HNSW_DIMENSIONS["vector"] else "halfvec"


def indexable(dimensions: int) -> bool:
    """Whether a column this wide can take an HNSW index at all.

    Above 4000 even halfvec is out, so the table is created without a vector
    index and every search over it is an exact scan.
    """
    return dimensions <= MAX_HNSW_DIMENSIONS[element_type(dimensions)]


# The DDL for one model's vector table. It is built here rather than kept in
# `sql/` because the table name and the vector width come from the
# registry, and a file that has to be string-formatted before it will parse is
# not SQL any more. Composed with `psycopg.sql` so the identifiers are quoted
# by the driver.
#
# These tables are written by hand rather than by PGEngine.init_vectorstore_table
# so the foreign keys can be declared inline. PGVectorStore.create_sync does not
# create a table -- it reads information_schema.columns and checks only that the
# id, content, embedding and declared metadata columns exist, with content being
# text/char and the embedding column being 'USER-DEFINED' or 'vector'. Every
# extension type reports as USER-DEFINED, so halfvec satisfies it as readily as
# vector. Constraints and indexes are not inspected, so the additions below are
# invisible to it.
#
# Vector tables have no relationship to one another. Dropping one leaves every
# other table untouched.
#
# Three foreign keys are carried:
#   chunk_id    -- the real relationship. Deleting a document cascades to its
#                  chunks, which cascades to the vectors here. UNIQUE because a
#                  chunk is embedded at most once per model.
#   user_id     -- denormalized from documents.user_id. Required, not
#                  convenience: a similarity search filters on columns of THIS
#                  table, and HNSW returns its top-k before a join to
#                  `documents` could scope the results to one user.
#   document_id -- denormalized from document_chunks.document_id, for the same
#                  reason: it is what lets a search be restricted to the
#                  documents `document_embeddings` says are searchable.
#
# Both denormalized columns are identity, never state. They are written once
# and never updated, so there is nothing to drift. The enable/disable flags
# stay in `document_embeddings`, where they exist exactly once.
#
# Note the FK on user_id guarantees only that the user exists, NOT that it
# matches the chunk's owner. Keeping the two in agreement is the writer's job.
TABLE_DDL = sql.SQL("""
    create table {table} (
        langchain_id uuid primary key default gen_random_uuid(),
        content      text not null,
        embedding    {element_type}({dimensions}) not null,
        chunk_id     uuid not null unique
                     references document_chunks(id) on delete cascade,
        user_id      int not null
                     references users(id) on delete cascade,
        document_id  uuid not null
                     references documents(id) on delete cascade,
        langchain_metadata jsonb
    )
""")

COLUMN_INDEX_DDL = sql.SQL("create index on {table} ({column})")

VECTOR_INDEX_DDL = sql.SQL("create index on {table} using hnsw ({column} {ops})")


def _migration_files() -> list[Path]:
    """The numbered migrations, in order."""
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Run every migration not yet recorded in `schema_migrations`."""
    conn.execute(
        "create table if not exists schema_migrations ("
        "  filename   text primary key,"
        "  applied_at timestamptz not null default now())"
    )
    applied = {
        r[0] for r in conn.execute("select filename from schema_migrations").fetchall()
    }

    ran = []
    for path in _migration_files():
        name = path.name
        if name in applied:
            continue
        with open(path) as fh:
            conn.execute(fh.read())
        conn.execute("insert into schema_migrations (filename) values (%s)", (name,))
        ran.append(name)
    return ran


def create_vector_table(conn: psycopg.Connection, model: EmbeddingModel) -> bool | None:
    """Create `model`'s vector table.

    Returns True if it was created with an HNSW index, False if it was created
    without one (the model is wider than pgvector will index), or None if the
    table was already there.
    """
    if table_exists(conn, model.table_name):
        return None

    table = sql.Identifier(model.table_name)
    element = element_type(model.dimensions)
    conn.execute(
        TABLE_DDL.format(
            table=table,
            element_type=sql.SQL(element),
            dimensions=sql.Literal(model.dimensions),
        )
    )
    conn.execute(COLUMN_INDEX_DDL.format(table=table, column=sql.Identifier("user_id")))
    conn.execute(
        COLUMN_INDEX_DDL.format(table=table, column=sql.Identifier("document_id"))
    )

    if not indexable(model.dimensions):
        return False
    conn.execute(
        VECTOR_INDEX_DDL.format(
            table=table,
            column=sql.Identifier("embedding"),
            ops=sql.Identifier(INDEX_OPS[element][model.distance]),
        )
    )
    return True


def drop_vector_table(conn: psycopg.Connection, model: EmbeddingModel) -> None:
    conn.execute(
        sql.SQL("drop table if exists {table}").format(
            table=sql.Identifier(model.table_name)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        metavar="TABLE_NAME",
        help="Only set up this model's vector table (default: every active model).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the vector tables first. Destroys all stored embeddings.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Show registered models and exit."
    )
    args = parser.parse_args()

    with connection() as conn:
        if args.list:
            if not table_exists(conn, "embedding_models"):
                print("Registry not created yet. Run without --list first.")
                return
            for m in list_models(conn, active_only=False):
                mark = "ok     " if table_exists(conn, m.table_name) else "MISSING"
                flag = "" if m.is_active else "  (inactive)"
                print(
                    f"{mark}  {m.table_name:<24} {m.provider}/{m.model_name} "
                    f"dim={m.dimensions}{flag}"
                )
            return

        ran = apply_migrations(conn)
        for name in ran:
            print(f"Applied {name}")
        if not ran:
            print("Migrations already applied.")

        models = (
            [get_model(conn, table_name=args.model)]
            if args.model
            else list_models(conn)
        )

        if args.recreate:
            names = ", ".join(m.table_name for m in models)
            answer = input(f"Drop {names} and every embedding in them? [y/N] ")
            if answer.strip().lower() != "y":
                print("Aborted.")
                return
            for m in models:
                drop_vector_table(conn, m)
                print(f"Dropped {m.table_name!r}.")

        for m in models:
            indexed = create_vector_table(conn, m)
            if indexed is None:
                print(f"Table {m.table_name!r} already exists -- nothing to do.")
                continue
            element = element_type(m.dimensions)
            print(f"Created {m.table_name!r} with {element}({m.dimensions}).")
            if not indexed:
                print(
                    f"  WARNING: {m.dimensions} dimensions is above pgvector's "
                    f"HNSW limit of {MAX_HNSW_DIMENSIONS['halfvec']} for "
                    f"{element}, so this table has no vector index and every "
                    f"search scans it in full. Consider registering the model "
                    f"at a truncated dimension instead."
                )

        conn.commit()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()
