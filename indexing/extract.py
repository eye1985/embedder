"""PDF to Markdown `Document`s, split on heading structure.

This is the *first* of two splits -- it yields one Document per heading
section, which `chunk.py` then splits again by token count.
"""

import logging

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


def simple_extractor(path: str) -> list[Document]:
    try:
        loader = PyMuPDF4LLMLoader(file_path=path)
        docs = loader.load()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        result = []
        for doc in docs:
            for chunk in md_splitter.split_text(doc.page_content):
                chunk.metadata = {
                    **doc.metadata,
                    **{
                        k: chunk.metadata[k]
                        for k in (
                            "producer",
                            "creator",
                            "creationdate",
                            "source",
                            "file_path",
                            "total_pages",
                            "format",
                            "title",
                            "author",
                            "subject",
                            "page",
                        )
                        if k in chunk.metadata
                    },
                }
                result.append(chunk)

        return result
    except ValueError:
        # Returning empty rather than raising: the caller reports "extracted no
        # text", which is also what a scanned PDF with no text layer produces.
        logger.exception("could not extract %s", path)
        return []
