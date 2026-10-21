"""Splitting extracted text into the chunks that get embedded.

This is the *second* of two splits. `extract.py` already split the PDF on its
heading structure, so what arrives here is one Document per section. That is
still too coarse to embed -- a long section would exceed the embedding model's
context and dilute its vector -- so each section is split again by token count.

The chunking is deliberately not per-model. Chunks are shared by every
embedding model, so the token encoder below is a stable way to measure length,
not a statement about which model will embed the result.
"""

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 70


def make_splitter() -> RecursiveCharacterTextSplitter:
    """The one splitter used for every model."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_documents(texts: list[Document]) -> Iterator[tuple[str, dict]]:
    """Token-split each section, yielding (content, metadata) in reading order.

    A section's metadata is copied onto every chunk cut from it -- the chunks
    are what get retrieved, so each one has to carry its own provenance.
    """
    splitter = make_splitter()
    for text in texts:
        for content in splitter.split_text(text.page_content):
            yield content, text.metadata or {}
