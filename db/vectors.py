"""The embedding service behind each registered model.

The service is chosen from the `embedding_models` row, so one document can
hold vectors from several models side by side in `chunk_embeddings`.

Services are cached per model: each owns an HTTP client worth reusing.
"""

import threading

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from db.tables.registry import EmbeddingModel

# A whole document is embedded in one call, so this is longer than chat's.
EMBEDDING_TIMEOUT = 120

_services: dict[int, Embeddings] = {}
_lock = threading.Lock()


def get_embeddings(model: EmbeddingModel) -> Embeddings:
    """Resolve a registry row to the embedding service that produced it."""
    if model.provider == "openai":
        return OpenAIEmbeddings(
            model=model.model_name,
            dimensions=model.dimensions,
            timeout=EMBEDDING_TIMEOUT,
            max_retries=3,
        )

    raise ValueError(
        f"Unknown provider {model.provider!r}. Add it to get_embeddings()."
    )


def embeddings_for(model: EmbeddingModel) -> Embeddings:
    """The cached service for `model`, built on first use."""
    with _lock:
        service = _services.get(model.id)
        if service is None:
            service = _services[model.id] = get_embeddings(model)
        return service


def clear_embeddings() -> None:
    """Forget every cached service."""
    with _lock:
        _services.clear()
