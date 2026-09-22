"""Request model and path handling for the /ingest endpoint.

Ingestion only ever reads PDFs that already sit in the repo's sample/ folder,
so the path a client sends is untrusted input that has to be pinned back
inside that directory before anything opens it.
"""

from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample"

DEFAULT_INGEST_PATH = "sample/storybook.pdf"


class Ingest(BaseModel):
    path: str = DEFAULT_INGEST_PATH
    # Vector tables to embed into. Defaults to the user's default model.
    models: list[str] | None = None


def resolve_pdf(path: str) -> Path:
    """Resolve `path` inside sample/, refusing anything that escapes it."""
    resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(SAMPLE_DIR):
        raise HTTPException(400, f"path must be inside {SAMPLE_DIR.name}/")
    if not resolved.is_file():
        raise HTTPException(404, f"No such file: {path}")
    return resolved
