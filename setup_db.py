"""One-off schema setup for the vector store.

Run this once before `main.py`, and again whenever the table shape changes
(for example when switching embedding model, which changes VECTOR_SIZE).
"""

import argparse

from psycopg.errors import DuplicateTable
from sqlalchemy.exc import ProgrammingError

from pg_vector import CONNECTION_STRING, TABLE_NAME, VECTOR_SIZE, create_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing table first. Destroys all stored embeddings.",
    )
    args = parser.parse_args()

    if args.recreate:
        answer = input(
            f"Drop table {TABLE_NAME!r} and every embedding in it? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    engine = create_engine(CONNECTION_STRING)

    try:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE,
            overwrite_existing=args.recreate,
        )
    except ProgrammingError as exc:
        if isinstance(exc.orig, DuplicateTable):
            print(
                f"Table {TABLE_NAME!r} already exists -- nothing to do. "
                f"Use --recreate to rebuild it."
            )
            return
        raise

    print(f"Created table {TABLE_NAME!r} with vector({VECTOR_SIZE}).")


if __name__ == "__main__":
    main()
