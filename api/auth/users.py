"""Who is asking: the Auth0 session resolved to a row in `users`.

`Viewer` is the (user id, is admin) pair, or None when signed out; the others
derive from it. `CurrentUser` is for endpoints that need an account and
answers 401 without one. `AdminUser` also answers 403 for a non-admin.
`OptionalUser` is for chat, which anyone may use.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from db.pool import connection
from db.tables.users import user_for_login


async def session_of(request: Request, response: Response) -> dict | None:
    """The Auth0 session, or None when signed out."""
    return await request.app.state.auth_client.client.get_session(
        store_options={"request": request, "response": response}
    )


def _login(session: dict) -> tuple[int, bool]:
    with connection() as conn:
        user = session["user"]
        return user_for_login(conn, user["sub"], user.get("email"))


async def viewer(request: Request, response: Response) -> tuple[int, bool] | None:
    session = await session_of(request, response)
    if session is None:
        return None
    return await run_in_threadpool(_login, session)


Viewer = Annotated[tuple[int, bool] | None, Depends(viewer)]


async def optional_user(signed_in: Viewer) -> int | None:
    return signed_in[0] if signed_in else None


async def current_user(signed_in: Viewer) -> int:
    if signed_in is None:
        raise HTTPException(401, "Please log in.")
    return signed_in[0]


async def admin_user(signed_in: Viewer) -> int:
    user_id = await current_user(signed_in)
    if not signed_in[1]:
        raise HTTPException(403, "Only admins can do that.")
    return user_id


CurrentUser = Annotated[int, Depends(current_user)]
AdminUser = Annotated[int, Depends(admin_user)]
OptionalUser = Annotated[int | None, Depends(optional_user)]
