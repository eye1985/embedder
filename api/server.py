"""The FastAPI app: lifespan, static files, error handlers, and the routers.

No endpoints live here. They are in `api/routes/`, included at the bottom.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from psycopg_pool import PoolTimeout
from starlette.responses import JSONResponse, Response

from api.assets import AssetsNotBuilt
from api.config import STATIC_DIR
from api.routes import api_router, pages_router
from api.services.uploads import sweep_running_uploads
from db.pool import close_pool, connection, get_pool
from db.vectors import close_engines

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the pool here rather than on the first request, so an unreachable
    # database fails at startup instead of under traffic.
    get_pool()

    # Nothing can still be uploading at startup, so a job still marked running
    # belongs to a process that is gone. Left alone it would hold the
    # one-upload-per-user index forever and lock that user out.
    with connection() as conn:
        swept = sweep_running_uploads(conn)
    if swept:
        logger.warning(
            "cleared %d upload(s) interrupted by a restart: %s",
            len(swept),
            ", ".join(swept),
        )

    try:
        yield
    finally:
        close_engines()
        close_pool()


app = FastAPI(lifespan=lifespan)


class NoCacheStaticFiles(StaticFiles):
    """Static files the browser has to revalidate before reusing.

    Starlette sends `etag` and `last-modified` but no `Cache-Control`, which
    leaves the browser free to reuse a cached copy without asking. An edited
    page or script then keeps running the old version until a hard reload --
    and `--reload` restarting the server does nothing about it.

    `no-cache` means "revalidate", not "do not store": an unchanged file still
    comes back 304 with no body. The header is set after the parent has
    already decided between 200 and 304, so it lands on both.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


app.mount(
    "/static",
    NoCacheStaticFiles(directory=STATIC_DIR, check_dir=False),
    name="static",
)


@app.exception_handler(PoolTimeout)
async def pool_timeout_handler(request: Request, exc: PoolTimeout):
    """Postgres unreachable or the pool exhausted -- the service is down, not
    the request malformed, so 503 rather than 500."""
    logger.error("database unavailable on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": "Database unavailable. Is Postgres running?"}, status_code=503
    )


@app.exception_handler(AssetsNotBuilt)
async def assets_not_built_handler(request: Request, exc: AssetsNotBuilt):
    """The frontend has not been built, so no page can be rendered."""
    logger.error("frontend not built: %s", exc)
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Anything not already mapped to a status.

    The exception text is deliberately NOT returned: a psycopg or SQLAlchemy
    error can carry the connection string, and these endpoints have no auth.
    The type is enough to recognise, and the traceback goes to the log.
    """
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": f"Internal error ({type(exc).__name__}). See server logs."},
        status_code=500,
    )


app.include_router(pages_router)
app.include_router(api_router)
