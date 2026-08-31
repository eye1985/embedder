from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def add_to_db(store, texts: list[Document]):
    docs = []
    for text in texts:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            chunk_size=500,
            chunk_overlap=70,
        )
        chunks = splitter.split_text(text.page_content)
        # print(f"meta: {text.metadata}\n")
        for chunk in chunks:
            # print(f"chunk: {chunk}")
            docs.append(Document(page_content=chunk, metadata=text.metadata))

    store.add_documents(docs)
