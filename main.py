import os
import sys

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pdf_extractors import simple_extractor

from pg_vector import create_pg_vector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)


def main():
    CONNECTION_STRING = (
        "postgresql+psycopg://postgres:admin@localhost:5432/postgres"  # Uses psycopg3!
    )
    VECTOR_SIZE = 1536
    TABLE_NAME = "doc_collection"

    store = create_pg_vector(
        connection_string=CONNECTION_STRING,
        vector_size=VECTOR_SIZE,
        table_name=TABLE_NAME,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # texts = simple_extractor("./sample/storybook.pdf")

    retriever = store.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the provided 
        context: {context}
        Question: {question}"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "Tell me about the story When Yama Called"
    res = chain.invoke(query)
    print(res)


if __name__ == "__main__":
    main()
