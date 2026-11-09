"""The `users` table.

One row per Auth0 account, naming the embedding model their uploads default
to. Created on first sign-in by `user_for_login`.
"""

import psycopg


def get_user(conn: psycopg.Connection, user_id: int) -> tuple[int, str | None]:
    """A user's (id, email). Raises LookupError if there is no such user."""
    row = conn.execute(
        "select id, email from users where id = %s", (user_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No user with id {user_id}.")
    return row


def user_for_login(conn: psycopg.Connection, auth0_sub: str, email: str | None) -> int:
    """The user id for an Auth0 account, creating the row on first sign-in.

    A new user's upload model is the first active one in the registry.
    """
    row = conn.execute(
        "select id from users where auth0_sub = %s", (auth0_sub,)
    ).fetchone()
    if row is not None:
        return row[0]

    # `on conflict` covers two first requests from the same account racing.
    conn.execute(
        """
        insert into users (auth0_sub, email, upload_model_id)
        select %s, %s, id from embedding_models
        where is_active order by id limit 1
        on conflict (auth0_sub) do nothing
        """,
        (auth0_sub, email),
    )
    return conn.execute(
        "select id from users where auth0_sub = %s", (auth0_sub,)
    ).fetchone()[0]
