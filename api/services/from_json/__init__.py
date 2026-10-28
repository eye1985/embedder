"""Restoring a downloaded export: the read half of `api/services/export.py`.

Split where the database starts. `parse.py` takes the bytes of an uploaded
file and either returns what it describes or says what is wrong with it,
knowing nothing about Postgres. `restore.py` is the half that does: matching
the file's models against this installation's registry, noticing the document
is already here, and writing the rows.

A file that would fail halfway is therefore refused before a single row is
written.
"""

from api.services.from_json.parse import InvalidExport, parse_export
from api.services.from_json.restore import (
    CannotRestore,
    conflicting_document,
    restore_export,
)

__all__ = [
    "CannotRestore",
    "InvalidExport",
    "conflicting_document",
    "parse_export",
    "restore_export",
]
