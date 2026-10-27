"""A stand-in for the embedding services, so tests cost nothing.

It imitates the two properties the rest of the system actually relies on:
the vectors come back at the model's declared width, and they are unit-length,
which is what OpenAI returns and what makes cosine distance meaningful.

Deterministic, so the same chunk embeds to the same vector on every run: a
test can assert that re-embedding changed nothing without the comparison
drifting underneath it.
"""

import hashlib
import math
import random

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """A unit vector derived from the text, the same one every time."""
        digest = hashlib.sha256(text.encode()).digest()[:8]
        noise = random.Random(int.from_bytes(digest, "big"))
        raw = [noise.gauss(0.0, 1.0) for _ in range(self.dimensions)]
        length = math.sqrt(sum(x * x for x in raw))
        return [x / length for x in raw]
