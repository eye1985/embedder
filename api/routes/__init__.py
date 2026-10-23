"""The routers, composed into the two the app mounts.

`api_router` is prefixed so a JSON endpoint cannot collide with a page: the
documents page and the documents list would otherwise both want GET
"/documents". `pages_router` is the unprefixed HTML.

Included in the order the endpoints were defined in, which is the order they
appear in the OpenAPI schema.
"""

from fastapi import APIRouter

from api.routes import chat, documents, models, pages, uploads

api_router = APIRouter(prefix="/api")
api_router.include_router(chat.router)
api_router.include_router(models.router)
api_router.include_router(uploads.router)
api_router.include_router(documents.router)

pages_router = pages.router
