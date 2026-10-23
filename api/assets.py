"""Resolving a frontend entry to the files the browser should load.

Vite builds `frontend/` into `static/` under hashed filenames and writes a
manifest mapping each source entry to what it produced. Templates ask for an
entry by its source path, so no markup ever names a hashed file.

Set `VITE_DEV_SERVER` (normally `http://localhost:5173`, alongside
`npm run dev`) and the URLs point at Vite instead, which serves the modules
itself and adds hot reloading. The manifest is not read at all in that mode;
it is the only difference between the two.
"""

import json
import os
from pathlib import Path

# Must match `base` in vite.config.ts, which is where the static mount is.
BASE = "/static/"

MANIFEST = Path(".vite") / "manifest.json"


class AssetsNotBuilt(RuntimeError):
    """The manifest is missing or does not contain an entry a page asked for."""


# Keyed by modification time so `vite build --watch` is picked up without a
# server restart, while a page render does not re-read the file every time.
_cache: dict[int, dict] = {}


def _manifest(static_dir: Path) -> dict:
    path = static_dir / MANIFEST
    try:
        stamp = path.stat().st_mtime_ns
    except OSError as exc:
        raise AssetsNotBuilt(
            f"No frontend build at {path}. Run `npm run build` "
            f"(or `npm run dev` and set VITE_DEV_SERVER)."
        ) from exc

    if stamp not in _cache:
        _cache.clear()
        _cache[stamp] = json.loads(path.read_text())
    return _cache[stamp]


def _walk(manifest: dict, name: str, seen: set[str]) -> list[dict]:
    """A chunk and everything it imports, depth first, each visited once."""
    if name in seen:
        return []
    seen.add(name)

    chunk = manifest.get(name)
    if chunk is None:
        raise AssetsNotBuilt(
            f"{name!r} is not in the Vite manifest. Add it to "
            f"`rollupOptions.input` in vite.config.ts and rebuild."
        )

    out = [chunk]
    for imported in chunk.get("imports", ()):
        out.extend(_walk(manifest, imported, seen))
    return out


def assets(static_dir: Path, *entries: str) -> dict[str, list[str]]:
    """The stylesheets, scripts and preloads for `entries`, in the order given.

    Returns `{"styles": [...], "scripts": [...], "preloads": [...]}`.

    Only the entries themselves become script tags. The shared chunks they
    import are reached through the module graph, so they are announced as
    `modulepreload` instead -- otherwise the browser does not learn about them
    until it has parsed the entry, which costs a round trip. Vite's own HTML
    plugin does this; rendering the markup ourselves means doing it here.
    """
    origin = os.getenv("VITE_DEV_SERVER")
    if origin:
        root = f"{origin.rstrip('/')}{BASE}"
        # Vite serves CSS as a module in dev, so everything is a script and
        # the client has to come first for hot reloading to attach.
        return {
            "styles": [],
            "scripts": [f"{root}@vite/client", *(f"{root}{e}" for e in entries)],
            "preloads": [],
        }

    manifest = _manifest(static_dir)
    styles: list[str] = []
    scripts: list[str] = []
    preloads: list[str] = []
    seen: set[str] = set()

    for name in entries:
        for index, chunk in enumerate(_walk(manifest, name, seen)):
            url = f"{BASE}{chunk['file']}"
            if chunk["file"].endswith(".css"):
                styles.append(url)
            elif index == 0:
                scripts.append(url)
            else:
                preloads.append(url)
            styles.extend(f"{BASE}{css}" for css in chunk.get("css", ()))

    return {"styles": styles, "scripts": scripts, "preloads": preloads}
