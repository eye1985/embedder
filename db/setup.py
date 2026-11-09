"""Schema setup: applies the numbered migrations in `sql/`.

Run it as `uv run python -m db.setup` once before starting the server.
"""

from pathlib import Path

import psycopg

from db.pool import close_pool, connection
from paths import PROJECT_ROOT

# Plain SQL migrations, applied in filename order.
MIGRATIONS_DIR = PROJECT_ROOT / "sql"


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


def main():
    with connection() as conn:
        ran = apply_migrations(conn)
        for name in ran:
            print(f"Applied {name}")
        if not ran:
            print("Migrations already applied.")
        conn.commit()


if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()
