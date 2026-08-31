from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

CONNECTION_STRING = (
    "postgresql+psycopg://postgres:admin@localhost:5432/postgres"  # Uses psycopg3!
)
EMBEDDING_MODEL = "text-embedding-3-small"
TABLE_NAME = "doc_collection"
VECTOR_SIZE = 1536


def create_engine(connection_string: str = CONNECTION_STRING) -> PGEngine:
    return PGEngine.from_connection_string(url=connection_string)


def create_pg_vector(connection_string: str, table_name: str, vector_size: int):
    """Connect to an existing vector table.

    The table is not created here -- run `setup_db.py` once to create it.
    """
    engine = create_engine(connection_string)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=vector_size)

    try:
        return PGVectorStore.create_sync(
            engine=engine, table_name=table_name, embedding_service=embeddings
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Table {table_name!r} is missing or has the wrong columns. "
            f"Create it with: uv run python setup_db.py"
        ) from exc
