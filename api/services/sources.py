"""Resolving a document's stored `source_uri` to a file on disk."""

from pathlib import Path

from api.services.uploads import UPLOAD_DIR
from paths import PROJECT_ROOT

# Where a document's file may be: uploads/ for anything imported through the
# web, sample/ for the PDFs that ship with the repo.
SOURCE_DIRS = (UPLOAD_DIR, PROJECT_ROOT / "sample")


def source_path(source_uri: str | None) -> Path | None:
    """The file behind a document, if it is still there.

    Checked against the directories the app imports from, so a `source_uri`
    edited to point elsewhere on disk resolves to nothing.
    """
    if not source_uri:
        return None
    path = (PROJECT_ROOT / source_uri).resolve()
    if not any(path.is_relative_to(directory) for directory in SOURCE_DIRS):
        return None
    return path if path.is_file() else None
