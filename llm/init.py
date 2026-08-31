import os
import sys
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from pg_vector import create_pg_vector

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

session_store: dict[str, BaseChatMessageHistory] = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def init_llm():
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the provided context:\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | prompt
        | llm
        | StrOutputParser()
    )

    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "erik-1"}}

    return {"runnable_with_history": conversational_chain, "config": config}
