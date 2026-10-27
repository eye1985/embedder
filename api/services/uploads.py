"""Upload handling: a browser's bytes onto disk.

No pipeline here -- that is `indexing/pdf.py`. What is left is everything the
web adds to a plain import: an untrusted filename, a size ceiling and content
sniffing. The job row tracking an upload is `api/services/jobs.py`.

PDFs reach the app one way: posted to /upload, which writes them into
`uploads/`. The filename comes from the browser, so it is untrusted input --
reduced to a basename before anything opens or creates a path from it.
"""

from pathlib import Path

from fastapi import HTTPException, UploadFile

from paths import PROJECT_ROOT

# Where uploaded PDFs land. `sample/` holds the repo's own PDFs and is not
# this -- deleting a document only removes a file from here.
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# A demo-sized ceiling, enforced while streaming to disk so an oversized file
# is never held in memory or left behind on disk.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def upload_filename(upload: UploadFile) -> str:
    """The basename an upload will be stored under.

    Validation only, no I/O: the caller needs the name before deciding whether
    the upload is allowed to land at all. A name like "../../db.py" would
    otherwise be a path out of the directory.
    """
    name = Path(upload.filename or "").name
    if not name or name.startswith("."):
        raise HTTPException(400, "Upload needs a filename.")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files can be imported.")
    return name


def upload_source_uri(name: str) -> str:
    """The `source_uri` an upload of `name` would be stored under."""
    return str((UPLOAD_DIR / name).relative_to(PROJECT_ROOT))


def save_upload(upload: UploadFile, name: str) -> Path:
    """Write an uploaded PDF into uploads/ under `name` and return its path.

    Content is sniffed rather than trusted: an extension says nothing about
    what the bytes are, and the extractor is what would otherwise find out.

    This overwrites whatever sits at that path. Callers must reject a name that
    already has a document row first -- /upload does -- because a document's
    chunks are extracted once and would not be re-read for new bytes.
    """
    header = upload.file.read(5)
    upload.file.seek(0)
    if not header.startswith(b"%PDF"):
        raise HTTPException(415, f"{name} is not a PDF.")

    UPLOAD_DIR.mkdir(exist_ok=True)
    destination = UPLOAD_DIR / name

    size = 0
    try:
        with open(destination, "wb") as out:
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Files are limited to {MAX_UPLOAD_BYTES // 1024 // 1024}MB.",
                    )
                out.write(chunk)
    except Exception:
        # Never leave a partial or rejected file behind.
        destination.unlink(missing_ok=True)
        raise

    return destination
