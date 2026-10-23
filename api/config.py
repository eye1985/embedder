"""Settings the API reads at import time.

The directories are where Vite's build and the Jinja templates are, resolved
through `PROJECT_ROOT` so they hold whatever directory the server was started
from.
"""

import os

from paths import PROJECT_ROOT

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Vite's build output. Absent until `npm run build` has run, which is why the
# static mount does not insist on it existing.
STATIC_DIR = PROJECT_ROOT / "static"

# Placeholder until there is real auth -- every request is treated as this
# user, and retrieval is scoped to their embeddings only.
CURRENT_USER_ID = int(os.getenv("CURRENT_USER_ID", "1"))
