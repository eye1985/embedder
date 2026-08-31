# embeddings

Ingest PDFs into a local Postgres/pgvector store for semantic search, using
LangChain and OpenAI embeddings.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres + pgAdmin)
- An OpenAI API key

## Setup

**1. Install dependencies**

```bash
uv sync
```

**2. Start Postgres**

The database image is `pgvector/pgvector`, which ships the `vector` extension
preinstalled.

```bash
docker compose up -d
```

| service  | port | notes                                        |
| -------- | ---- | -------------------------------------------- |
| postgres | 5432 | user `postgres`, password `admin`, db `postgres` |
| pgadmin  | 5050 | login `pgadmin4@pgadmin.org` / `admin`       |

Override any of these with `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_PORT`, `PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`, or
`PGADMIN_PORT`.

**3. Add your API key**

Create a `.env` file (git-ignored):

```
OPENAI_API_KEY=sk-...
```

**4. Create the database table**

Run once, before the first ingest:

```bash
uv run python setup_db.py
```

It is safe to re-run -- if the table already exists it reports that and exits
without touching your data.

## Usage

```bash
uv run python main.py
```

Reads a PDF from `sample/`, extracts it to Markdown, splits it into chunks, and
prints them.

## Layout

| file               | role                                                        |
| ------------------ | ----------------------------------------------------------- |
| `main.py`          | Entry point: extract, chunk, and (soon) store.               |
| `pg_vector.py`     | Connects to the vector store. Holds the shared config constants. |
| `setup_db.py`      | One-off schema creation. The only place that runs DDL.       |
| `pdf_extractors.py`| PDF to Markdown `Document`s, split on heading structure.     |

### Schema is created once, on purpose

`setup_db.py` is the only code that creates tables. The application connects to
an existing table and fails with a clear message if it is missing:

```
Table 'doc_collection' is missing or has the wrong columns.
Create it with: uv run python setup_db.py
```

Creating tables at application startup instead would mean any restart could
silently alter the schema, and concurrent starts would race. Keeping DDL in one
deliberate step is the same principle as a migration, without the machinery --
if this grows into a deployed service, Alembic is the natural next step.

## Configuration

Shared settings live at the top of `pg_vector.py` so setup and runtime cannot
disagree about the table shape:

```python
CONNECTION_STRING = "postgresql+psycopg://postgres:admin@localhost:5432/postgres"
EMBEDDING_MODEL   = "text-embedding-3-small"
TABLE_NAME        = "doc_collection"
VECTOR_SIZE       = 1536
```

The `+psycopg` in the connection URL selects psycopg **3**, which is required:
`PGEngine` builds an async SQLAlchemy engine, and psycopg2 cannot back one.

### Changing the embedding model

`VECTOR_SIZE` must match the model's output width, and the column dimension is
fixed at creation time. Changing the model therefore means rebuilding the table
and re-embedding everything -- vectors from different models are not comparable:

```bash
uv run python setup_db.py --recreate   # drops the table, asks for confirmation first
```

Two things to know before switching to `text-embedding-3-large`:

- Its native width is 3072, but **pgvector cannot index past 2000 dimensions**
  with HNSW or IVFFlat. Pass `dimensions=1024` (or 1536) to `OpenAIEmbeddings` --
  the `-3-*` models are Matryoshka-trained, so truncating degrades gracefully,
  and `-3-large` at 1024 still scores above `-3-small` at its full 1536.
- Re-embedding a large corpus is a real API bill. Settle on the model before
  ingesting in bulk.

## Chunking

Chunking is the caller's job. `add_documents` embeds whatever you hand it --
one `Document` becomes one row and one vector, with no splitting.

Oversized input is not rejected either: `OpenAIEmbeddings` splits anything past
8191 tokens, embeds each piece, and returns their weighted average. That is one
blurred vector rather than an error, so retrieval quality degrades silently.

Current settings in `main.py` are 500-token chunks with 70 tokens of overlap.
Smaller, focused chunks embed to a sharper point and retrieve more accurately;
very large ones average several topics together and match none of them well.

| content type        | chunk_size | overlap |
| ------------------- | ---------- | ------- |
| prose, mixed PDFs   | 500        | 70      |
| technical reference | 800        | 100     |
| short factual Q&A   | 300        | 50      |
