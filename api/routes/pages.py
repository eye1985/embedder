"""The HTML page shells.

Three routes, one per page, each rendering the same shell with a different
template and nav entry. No data: every page fetches what it needs from the
JSON API once its script runs.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from api.assets import assets
from api.config import FRONTEND_DIR, STATIC_DIR, TEMPLATES_DIR

router = APIRouter(include_in_schema=False)

# `frontend/` is searched too, so a template can inline a source file that
# has to run before anything else -- the theme preload in base.html.
templates = Jinja2Templates(directory=[TEMPLATES_DIR, FRONTEND_DIR])


def render_page(request: Request, name: str, title: str, heading: str, current: str):
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
            "styles": bundle["styles"],
            "scripts": bundle["scripts"],
            "preloads": bundle["preloads"],
        },
        headers={"cache-control": "no-cache"},
    )


@router.get("/")
def chat_page(request: Request):
    return render_page(request, "chat.html", "Chat", "Document chat", "/")


@router.get("/upload")
def upload_page(request: Request):
    return render_page(request, "upload.html", "Upload", "Upload", "/upload")


@router.get("/documents")
def documents_page(request: Request):
    return render_page(
        request, "documents.html", "Documents", "Documents", "/documents"
    )
