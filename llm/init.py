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

from db import connection
from pg_vector import get_vector_store
from registry import EmbeddingModel, default_model_for_user

load_dotenv()


if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable.", file=sys.stderr)
    sys.exit(1)

session_store: dict[str, BaseChatMessageHistory] = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def init_llm(user_id: int, model: EmbeddingModel | None = None):
    """Build the chat chain over one user's embeddings.

    `model` picks which embedding type to retrieve from; it defaults to the
    user's `default_embedding_model_id`, which every user row has to name.
    """
    if model is None:
        with connection() as conn:
            model = default_model_for_user(conn, user_id)

    store = get_vector_store(model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # The filter is what keeps one user's vectors out of another's results. It
    # has to be applied by the search itself -- HNSW picks its top-k before any
    # caller could filter the rows afterwards.
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50, "filter": {"user_id": user_id}},
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

    config = {"configurable": {"session_id": f"user-{user_id}"}}

    return {"runnable_with_history": conversational_chain, "config": config}
