"""Where the project's directories are.

Resolved from this file's own location, so they hold regardless of the working
directory a process was started from. They live at the root rather than inside
any one package because they belong to the project: `db/` finding `sql/` by
walking out to its own parent would make a package's layout depend on where it
happens to sit.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Plain SQL migrations, applied in filename order by `db/setup.py`.
MIGRATIONS_DIR = PROJECT_ROOT / "sql"

# Where uploaded PDFs land. `sample/` holds the repo's own PDFs and is not this
# -- deleting a document only removes a file from here.
UPLOAD_DIR = PROJECT_ROOT / "uploads"
