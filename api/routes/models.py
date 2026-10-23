"""The embedding model registry, and choosing which models an import uses.

`models_for` and `search_model_for` are here rather than beside either caller
because both /upload and the re-embed endpoint need them.
"""

from fastapi import APIRouter, HTTPException

from api.config import CURRENT_USER_ID
from api.schemas import ModelOut
from db.pool import connection
from db.tables.embeddings import document_counts
from db.tables.registry import (
    get_model,
    list_models,
    table_exists,
    upload_model_for_user,
)

router = APIRouter()


def models_for(conn, names: list[str] | None):
    """The models to embed with: those named, or the user's upload model.

    A registered model with no vector table is rejected here rather than deep
    inside PGVectorStore, where the same mistake surfaces as a 500.
    """
    try:
        if names:
            models = [get_model(conn, table_name=name) for name in names]
        else:
            models = [upload_model_for_user(conn, CURRENT_USER_ID)]
    except LookupError as exc:
        # Unknown table name, or no user row -- a client error, not a crash.
        raise HTTPException(404, str(exc)) from exc

    missing = [m.table_name for m in models if not table_exists(conn, m.table_name)]
    if missing:
        raise HTTPException(
            400,
            f"No vector table for {', '.join(missing)}. "
            f"Create it with: uv run python -m db.setup",
        )
    return models


def search_model_for(chosen: list, table_name: str | None):
    """Which of `chosen` the document is retrieved through.

    One model embedded means no choice to make. Several and the caller has to
    say, rather than have one picked for them -- the answers a question gets
    depend on it, so a silent default would be a decision made invisibly.
    """
    if table_name is None:
        if len(chosen) == 1:
            return chosen[0]
        raise HTTPException(
            400,
            "Embedding with more than one model needs `search_model` naming "
            "which of them this document is searched with.",
        )

    for model in chosen:
        if model.table_name == table_name:
            return model
    raise HTTPException(
        400,
        f"search_model {table_name!r} is not one of the models being embedded "
        f"({', '.join(m.table_name for m in chosen)}).",
    )


@router.get("/models", response_model=list[ModelOut])
def list_models_endpoint():
    """Every registered embedding model, for the upload form's pickers.

    `available` is what the form gates on: `db/setup.py` only creates tables
    for active models, so an inactive row has nothing to embed into.
    `documents` says how many of the user's documents each has embedded.
    """
    with connection() as conn:
        try:
            upload = upload_model_for_user(conn, CURRENT_USER_ID)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

        models = list_models(conn, active_only=False)
        counts = document_counts(conn, CURRENT_USER_ID, [m.id for m in models])

        return [
            {
                "table_name": m.table_name,
                "provider": m.provider,
                "model_name": m.model_name,
                "dimensions": m.dimensions,
                "is_active": m.is_active,
                "available": table_exists(conn, m.table_name),
                "documents": counts.get(m.id, 0),
                "is_upload_default": m.id == upload.id,
            }
            for m in models
        ]
