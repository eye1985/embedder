"""The HTML page shells.

One route per page, each rendering the same shell with a different template
and nav entry. No data: every page fetches what it needs from the JSON API
once its script runs.

Chat is open to everyone. Upload, documents and jobs send a signed-out visitor
to /login, which comes back to the page they asked for.
"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from api.assets import assets
from api.auth.require_login import require_login
from api.auth.users import session_of
from api.config import FRONTEND_DIR, STATIC_DIR, TEMPLATES_DIR

router = APIRouter(include_in_schema=False)

# `frontend/` is searched too, so a template can inline a source file that
# has to run before anything else -- the theme preload in base.html.
templates = Jinja2Templates(directory=[TEMPLATES_DIR, FRONTEND_DIR])


def render_page(
    request: Request,
    name: str,
    title: str,
    heading: str,
    current: str,
    signed_in: bool,
    **context,
):
    """One of the page shells.

    `current` is the nav href to mark as the page you are on; the nav itself
    lives in `templates/base.html` and is the only copy of it. The page's
    script entry follows from the template name, and its built URL is resolved
    through the Vite manifest, so the markup never names a hashed file.

    `no-cache` is set for the same reason the static mount sets it -- an edited
    template would otherwise keep being served from the browser's cache. The
    built assets are hashed, so they are safe to cache hard.
    """
    entry = f"pages/{name.removesuffix('.html')}.ts"
    bundle = assets(STATIC_DIR, "main.css", entry)
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={
            "title": title,
            "heading": heading,
            "current": current,
            "signed_in": signed_in,
            "styles": bundle["styles"],
            "scripts": bundle["scripts"],
            "preloads": bundle["preloads"],
            **context,
        },
        headers={"cache-control": "no-cache"},
    )


async def private_page(
    request: Request, response: Response, name: str, title: str, current: str
):
    """A page that needs an account. Signed out, it is a detour via /login."""
    if await session_of(request, response) is None:
        return RedirectResponse(f"/login?returnTo={quote(current)}")
    return render_page(request, name, title, title, current, signed_in=True)


def local_path(target: str | None) -> str:
    """`target` if it is a path on this site, otherwise the chat page.

    Keeps `returnTo` from sending someone to another host after signing in.
    """
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return "/"


@router.get("/")
async def chat_page(request: Request, response: Response):
    signed_in = await session_of(request, response) is not None
    return render_page(
        request, "chat.html", "Chat", "Document chat", "/", signed_in=signed_in
    )


@router.get("/login")
async def login_page(request: Request, response: Response):
    return_to = local_path(request.query_params.get("returnTo"))
    if await session_of(request, response) is not None:
        return RedirectResponse(return_to)
    return render_page(
        request,
        "login.html",
        "Sign in",
        "Sign in",
        "/login",
        signed_in=False,
        return_to=return_to,
    )


# testing this now
@router.get("/private")
async def private(session: dict = Depends(require_login)):
    return {
        "message": "Hello from a private endpoint! You need to be authenticated to see this.",
        "user_id": session["user"]["sub"],
        "email": session["user"]["email"],
    }


@router.get("/upload")
async def upload_page(request: Request, response: Response):
    return await private_page(request, response, "upload.html", "Upload", "/upload")


@router.get("/documents")
async def documents_page(request: Request, response: Response):
    return await private_page(
        request, response, "documents.html", "Documents", "/documents"
    )


@router.get("/jobs")
async def jobs_page(request: Request, response: Response):
    return await private_page(request, response, "jobs.html", "Jobs", "/jobs")
