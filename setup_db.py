"""Schema setup.

Applies the numbered migrations in `database/`, then creates one vector table
per active row in `embedding_models` by rendering
`database/005_vector_table_template.sql`.

Run it once before `main.py`, and again after adding a row to
`embedding_models` -- new rows get their table on the next run.
"""

import argparse
import glob
import os

import psycopg

from db import MIGRATIONS_DIR, close_pool, connection, table_exists
from registry import EmbeddingModel, get_model, list_models

TEMPLATE = os.path.join(MIGRATIONS_DIR, "005_vector_table_template.sql")

# pgvector refuses an HNSW index above this many dimensions.
MAX_HNSW_DIMENSIONS = 2000

INDEX_OPS = {
    "cosine": "vector_cosine_ops",
    "inner_product": "vector_ip_ops",
    "l2": "vector_l2_ops",
}


def index_ddl(model: EmbeddingModel) -> str:
    """The vector index for `model`, or a comment saying why there is none."""
    if model.dimensions > MAX_HNSW_DIMENSIONS:
        return (
            f"-- No vector index: {model.dimensions} dimensions exceeds pgvector's "
            f"HNSW limit of {MAX_HNSW_DIMENSIONS}. Searches do an exact scan."
        )
    return (
        f"create index on {model.table_name} "
        f"using hnsw (embedding {INDEX_OPS[model.distance]});"
    )


def _migration_files() -> list[str]:
    """The numbered migrations, in order. The template is not one of them."""
    paths = sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "[0-9][0-9][0-9]_*.sql")))
    return [p for p in paths if "template" not in os.path.basename(p)]


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
        name = os.path.basename(path)
        if name in applied:
            continue
        with open(path) as fh:
            conn.execute(fh.read())
        conn.execute("insert into schema_migrations (filename) values (%s)", (name,))
        ran.append(name)
    return ran


def create_vector_table(conn: psycopg.Connection, model: EmbeddingModel) -> bool:
    """Render the template for `model`. Returns False if the table was there."""
    if table_exists(conn, model.table_name):
        return False
    with open(TEMPLATE) as fh:
        ddl = fh.read()
    conn.execute(
        ddl.format(
            table_name=model.table_name,
            dimensions=model.dimensions,
            index_ddl=index_ddl(model),
        )
    )
    return True


def drop_vector_table(conn: psycopg.Connection, model: EmbeddingModel) -> None:
    # The identifier comes from the registry, not from user input.
    conn.execute(f'drop table if exists "{model.table_name}"')


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
            if create_vector_table(conn, m):
                print(f"Created {m.table_name!r} with vector({m.dimensions}).")
                if m.dimensions > MAX_HNSW_DIMENSIONS:
                    print(
                        f"  WARNING: {m.dimensions} dimensions is above pgvector's "
                        f"HNSW limit of {MAX_HNSW_DIMENSIONS}, so this table has no "
                        f"vector index and every search scans it in full. Consider "
                        f"registering the model at a truncated dimension instead."
                    )
            else:
                print(f"Table {m.table_name!r} already exists -- nothing to do.")

        conn.commit()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()
