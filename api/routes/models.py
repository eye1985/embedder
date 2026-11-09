"""The embedding model registry, and choosing which models an import uses.

`models_for` and `search_model_for` are here rather than beside either caller
because both /upload and the re-embed endpoint need them.
"""

from fastapi import APIRouter, HTTPException

from api.auth.users import AdminUser
from api.schemas import ModelOut
from db.pool import connection
from db.tables.embeddings import document_counts
from db.tables.registry import (
    EmbeddingModel,
    get_model,
    list_models,
    upload_model_for_user,
)

router = APIRouter()


def models_for(conn, user_id: int, model_ids: list[int] | None) -> list[EmbeddingModel]:
    """The models to embed with: those named, or the user's upload model."""
    try:
        if model_ids:
            return [get_model(conn, model_id=model_id) for model_id in model_ids]
        return [upload_model_for_user(conn, user_id)]
    except LookupError as exc:
        # Unknown model id, or no user row -- a client error, not a crash.
        raise HTTPException(404, str(exc)) from exc


def models_by_label(conn, labels: list[str]) -> list[EmbeddingModel]:
    """The models a job recorded, which it names by label."""
    known = {m.label: m for m in list_models(conn, active_only=False)}
    missing = [label for label in labels if label not in known]
    if missing:
        raise HTTPException(404, f"No longer registered: {', '.join(missing)}.")
    return [known[label] for label in labels]


def search_model_for(
    chosen: list[EmbeddingModel], model_id: int | None
) -> EmbeddingModel:
    """Which of `chosen` the document is retrieved through.

    One model embedded means no choice to make. Several and the caller has to
    say, rather than have one picked for them -- the answers a question gets
    depend on it, so a silent default would be a decision made invisibly.
    """
    if model_id is None:
        if len(chosen) == 1:
            return chosen[0]
        raise HTTPException(
            400,
            "Embedding with more than one model needs `search_model` naming "
            "which of them this document is searched with.",
        )

    for model in chosen:
        if model.id == model_id:
            return model
    raise HTTPException(
        400,
        f"search_model {model_id} is not one of the models being embedded "
        f"({', '.join(str(m.id) for m in chosen)}).",
    )


@router.get("/models", response_model=list[ModelOut])
def list_models_endpoint(user_id: AdminUser):
    """Every registered embedding model, for the upload form's pickers.

    `documents` says how many of the user's documents each has embedded.
    """
    with connection() as conn:
        try:
            upload = upload_model_for_user(conn, user_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

        models = list_models(conn, active_only=False)
        counts = document_counts(conn, user_id, [m.id for m in models])

        return [
            {
                "model_id": m.id,
                "label": m.label,
                "provider": m.provider,
                "model_name": m.model_name,
                "dimensions": m.dimensions,
                "is_active": m.is_active,
                "documents": counts.get(m.id, 0),
                "is_upload_default": m.id == upload.id,
            }
            for m in models
        ]
