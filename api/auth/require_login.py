from fastapi import Request, Response


async def require_login(request: Request, response: Response) -> dict:
    return await request.app.state.auth_client.require_session(request, response)
