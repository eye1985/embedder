import os
import sys

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_extractors import simple_extractor


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)


def main():
    texts = simple_extractor("./sample/flyer.pdf")

    for text in texts:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            chunk_size=8000,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(text.page_content)
        print(f"{text.metadata}\n")
        print(chunks)


if __name__ == "__main__":
    main()
