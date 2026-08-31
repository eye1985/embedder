from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


# Too heavy, use this on local machine
def docling_converter():
    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        do_picture_classification=True,
        images_scale=2.0,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# Too heavy, use this on local machine
def extract_pdf(file_path: str):
    loader = DoclingLoader(file_path=file_path, converter=docling_converter())
    return loader.load()


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
    except ValueError as e:
        print(f"Invalid path: {e}")
        return []
