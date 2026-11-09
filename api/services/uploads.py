"""Upload handling: a browser's bytes onto disk.

No pipeline here -- that is `indexing/pdf.py`. What is left is everything the
web adds to a plain import: an untrusted filename, a size ceiling and content
sniffing. The job row tracking an upload is `api/services/jobs.py`.

PDFs reach the app one way: posted to /upload, which writes them into the
user's own folder, `uploads/<user_id>/`, so two users' files of the same name
never meet. The filename comes from the browser, so it is untrusted input --
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


def user_upload_dir(user_id: int) -> Path:
    """Where one user's uploaded PDFs land."""
    return UPLOAD_DIR / str(user_id)


def upload_source_uri(user_id: int, name: str) -> str:
    """The `source_uri` an upload of `name` would be stored under."""
    return str((user_upload_dir(user_id) / name).relative_to(PROJECT_ROOT))


def read_upload(upload: UploadFile) -> bytes:
    """An upload read into memory instead of onto disk, under the same ceiling.

    What restoring an export reads. The file is parsed and thrown away rather
    than stored, so it never lands in uploads/ and needs no filename -- but it
    is still a browser's bytes, and a JSON export is mostly vectors, so the
    size limit still applies.
    """
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Files are limited to {MAX_UPLOAD_BYTES // 1024 // 1024}MB."
        )
    return data


def save_upload(upload: UploadFile, user_id: int, name: str) -> Path:
    """Write an uploaded PDF into the user's folder under `name` and return
    its path.

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

    folder = user_upload_dir(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / name

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
