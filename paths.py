"""Where the project root is.

Resolved from this file's own location, so it holds regardless of the working
directory a process was started from. It lives at the root rather than inside
any one package because it belongs to the project: `db/` finding `sql/` by
walking out to its own parent would make a package's layout depend on where it
happens to sit.

It is also the base every stored `source_uri` is relative to, so it is one
constant rather than something each caller recomputes -- those paths are in
the database and have to resolve the same way everywhere.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
