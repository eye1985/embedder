"""The HTML page shells.

One route per page, each rendering the same shell with a different template
and nav entry. No data: every page fetches what it needs from the JSON API
once its script runs.

Chat is open to everyone. Upload, documents and jobs are for admins: a
signed-out visitor goes to Auth0, which comes back to the page they asked
for, and a signed-in non-admin goes to chat.
"""

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from api.assets import assets
from api.auth.users import Viewer
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
    admin: bool,
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
            "admin": admin,
            "styles": bundle["styles"],
            "scripts": bundle["scripts"],
            "preloads": bundle["preloads"],
            **context,
        },
        headers={"cache-control": "no-cache"},
    )


def admin_page(
    request: Request,
    signed_in: tuple[int, bool] | None,
    name: str,
    title: str,
    current: str,
):
    """A page for admins. Signed out, it is a detour via Auth0."""
    if signed_in is None:
        return RedirectResponse(f"/auth/login?returnTo={quote(current)}")
    if not signed_in[1]:
        return RedirectResponse("/")
    return render_page(request, name, title, title, current, signed_in=True, admin=True)


@router.get("/")
async def chat_page(request: Request, signed_in: Viewer):
    return render_page(
        request,
        "chat.html",
        "Chat",
        "Document chat",
        "/",
        signed_in=signed_in is not None,
        admin=signed_in is not None and signed_in[1],
    )


@router.get("/upload")
async def upload_page(request: Request, signed_in: Viewer):
    return admin_page(request, signed_in, "upload.html", "Upload", "/upload")


@router.get("/documents")
async def documents_page(request: Request, signed_in: Viewer):
    return admin_page(request, signed_in, "documents.html", "Documents", "/documents")


@router.get("/jobs")
async def jobs_page(request: Request, signed_in: Viewer):
    return admin_page(request, signed_in, "jobs.html", "Jobs", "/jobs")
