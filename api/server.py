import os
from contextlib import asynccontextmanager
from operator import itemgetter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from add_docs import add_document, count_chunks, embed_document, find_document
from db import close_pool, connection, get_pool
from llm.init import init_llm
from pg_vector import close_engines
from registry import default_model_for_user, get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
SAMPLE_DIR = PROJECT_ROOT / "sample"

DEFAULT_INGEST_PATH = "sample/storybook.pdf"

# Placeholder until there is real auth -- every request is treated as this
# user, and retrieval is scoped to their embeddings only.
CURRENT_USER_ID = int(os.getenv("CURRENT_USER_ID", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the pool here rather than on the first request, so an unreachable
    # database fails at startup instead of under traffic.
    get_pool()
    app.state.chat = init_llm(user_id=CURRENT_USER_ID)
    try:
        yield
    finally:
        close_engines()
        close_pool()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


class Chat(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(prompt: Chat, request: Request):
    run, config = itemgetter("runnable_with_history", "config")(request.app.state.chat)

    async def stream():
        async for chunk in run.astream({"question": prompt.prompt}, config=config):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


class Ingest(BaseModel):
    path: str = DEFAULT_INGEST_PATH
    # Vector tables to embed into. Defaults to the user's default model.
    models: list[str] | None = None


def _resolve_pdf(path: str) -> Path:
    """Resolve `path` inside sample/, refusing anything that escapes it."""
    resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(SAMPLE_DIR):
        raise HTTPException(400, f"path must be inside {SAMPLE_DIR.name}/")
    if not resolved.is_file():
        raise HTTPException(404, f"No such file: {path}")
    return resolved


# Defined with `def`, not `async def`: extraction and the embedding API call
# both block, so FastAPI runs this in a worker thread instead of stalling the
# event loop.
@app.post("/ingest")
def ingest_endpoint(req: Ingest | None = None):
    """Extract a PDF, store its chunks, and embed them.

    Safe to call twice. The document is stored once per source path, and each
    model skips chunks it has already embedded -- so a repeat call re-reads the
    PDF but does not pay for the same vectors again.
    """
    # The body is optional -- a bare POST ingests the default sample PDF.
    req = req or Ingest()

    pdf = _resolve_pdf(req.path)

    with connection() as conn:
        try:
            if req.models:
                models = [get_model(conn, table_name=name) for name in req.models]
            else:
                models = [default_model_for_user(conn, CURRENT_USER_ID)]
        except LookupError as exc:
            # Unknown table name, or no user row -- a client error, not a crash.
            raise HTTPException(404, str(exc)) from exc

        source_uri = str(pdf.relative_to(PROJECT_ROOT))
        document_id = find_document(conn, CURRENT_USER_ID, source_uri)
        reused = document_id is not None

        if not reused:
            # Imported here rather than at module scope: pdf_extractors pulls in
            # docling, which is slow to import and not needed to serve chat.
            from pdf_extractors import simple_extractor

            texts = simple_extractor(str(pdf))
            if not texts:
                raise HTTPException(422, f"Extracted no text from {req.path}")
            document_id = add_document(
                conn,
                CURRENT_USER_ID,
                texts,
                title=pdf.stem,
                source_uri=source_uri,
            )
            conn.commit()

        embedded = {m.table_name: embed_document(conn, document_id, m) for m in models}

        return {
            "document_id": str(document_id),
            "source_uri": source_uri,
            "reused_existing_document": reused,
            "chunks": count_chunks(conn, document_id),
            "embedded": embedded,
        }
