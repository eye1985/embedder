"""The `users` table.

One row per user, naming the embedding model their uploads and searches
default to. Auth does not exist yet -- `api/server.py` treats every request as
`CURRENT_USER_ID`.
"""

import psycopg


def get_user(conn: psycopg.Connection, user_id: int) -> tuple[int, str]:
    """A user's (id, email). Raises LookupError if there is no such user."""
    row = conn.execute(
        "select id, email from users where id = %s", (user_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No user with id {user_id}.")
    return row
