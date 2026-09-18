# embeddings

Ingest PDFs into a local Postgres/pgvector store for semantic search, using
LangChain. Each user can hold several kinds of embedding side by side -- OpenAI,
Qwen, or anything else you register -- with one vector table per model.

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

**4. Create the schema**

```bash
uv run python setup_db.py
```

This does two things:

1. Applies every numbered migration in `database/` in filename order, recording
   each one in a `schema_migrations` table.
2. Creates one vector table per **active** row in `embedding_models`, by
   rendering `database/005_vector_table_template.sql` with that model's name and
   dimension.

Safe to re-run -- applied migrations are skipped and existing tables are left
alone. Expect:

```
Applied 001_extensions.sql
Applied 002_embedding_models.sql
Applied 003_users.sql
Applied 004_documents.sql
Created 'emb_openai_te3_small' with vector(1536).
Created 'emb_qwen3_0_6b' with vector(1024).
```

**5. The seed user**

Retrieval is scoped per user, so the app needs a row in `users` before it will
start. `006_seed_user.sql` creates one for you:

| id | email | default model |
| -- | ----- | ------------- |
| 1 | `testuser@mail.com` | `emb_openai_te3_small` |

On a fresh schema this is the first insert, so it lands on id 1 -- which is what
`CURRENT_USER_ID` defaults to. Nothing else to do.

It is a **development seed**: drop that migration before deploying anywhere
real. Add your own users with a plain insert:

```bash
psql -h localhost -U postgres -d postgres \
  -c "insert into users (email) values ('you@example.com');"
```

### setup_db.py options

| command | effect |
| ------- | ------ |
| `uv run python setup_db.py` | Apply migrations, create tables for all active models. |
| `uv run python setup_db.py --list` | Show every registered model and whether its table exists. |
| `uv run python setup_db.py --model emb_qwen3_0_6b` | Set up one model's table only, active or not. |
| `uv run python setup_db.py --recreate` | **Drops** the vector tables first. Asks for confirmation. Destroys embeddings, not documents. |

`--recreate` only touches vector tables. `documents` and `document_chunks`
survive, so you can re-embed without re-extracting the PDFs.

## Migrations

| file | contents |
| ---- | -------- |
| `001_extensions.sql` | `create extension vector` |
| `002_embedding_models.sql` | The model registry, plus four seed rows. |
| `003_users.sql` | `users`. |
| `004_documents.sql` | `documents` and `document_chunks`. |
| `005_vector_table_template.sql` | **Not a migration.** A template rendered once per registered model. |
| `006_seed_user.sql` | Development seed user. Drop before deploying. |

Migrations are plain SQL, applied in order, and tracked in `schema_migrations`.
To add one, drop a new `00N_name.sql` into `database/` and re-run `setup_db.py`.

Files with `template` in the name are skipped by the migration runner -- the template
carries `{table_name}`, `{dimensions}` and `{index_ddl}` placeholders and is not
valid SQL on its own.

### Starting over

```bash
psql -h localhost -U postgres -d postgres \
  -c "drop schema public cascade; create schema public;"
uv run python setup_db.py
```

Dropping the schema also drops the `vector` extension; migration `001` puts it
back.

## How embeddings are stored

Text is chunked **once**, into `document_chunks`, independent of any model. Each
model then embeds those same chunks into its own table:

```
users ──1:N──> documents ──1:N──> document_chunks
                                        │ 1:N
                     ┌──────────────────┼──────────────────┐
                     ▼                                     ▼
       emb_openai_te3_small (1536)              emb_qwen3_0_6b (1024)
```

A chunk is embedded by a model at most once (`chunk_id` is unique per table), and
a chunk may be embedded by any number of models, including none. The vector
tables have no relationship to one another -- dropping one leaves the rest
untouched.

Why one table per model: a pgvector column has a **fixed dimension**, so 1536-
and 1024-wide vectors cannot share a column. Postgres partitioning does not help
either, since all partitions share one column definition.

### Adding an embedding model

Insert a row into `embedding_models` and re-run `setup_db.py`:

```sql
insert into embedding_models (provider, model_name, dimensions, table_name)
values ('openai', 'text-embedding-3-large', 1024, 'emb_openai_te3_large_1024');
```

Then teach `get_embeddings()` in `pg_vector.py` about the provider if it is a new
one. `openai`, `qwen` and `huggingface` are already handled; `qwen` and
`huggingface` need `uv add langchain-huggingface sentence-transformers`.

**Keep `dimensions` at or below 2000.** pgvector cannot build an HNSW index above
that, and `setup_db.py` will create the table without a vector index and warn
you -- every search then scans the table in full. The two seeded models above the
limit (`text-embedding-3-large` at 3072, `Qwen3-Embedding-8B` at 4096) ship
`is_active = false` for this reason.

Both are Matryoshka-trained, so truncating degrades gracefully: register
`-3-large` at 1024 or 2000 rather than 3072. At 1024 it still scores above
`-3-small` at its full 1536.

## Usage

```bash
uv run uvicorn api.server:app --reload
```

Serves the chat UI at `http://localhost:8000`. Retrieval is scoped to
`CURRENT_USER_ID` (default `1`) until real auth exists.

### Ingesting the sample PDF

```bash
curl -X POST http://localhost:8000/ingest
```

Extracts `sample/storybook.pdf`, stores its chunks, and embeds them with the
user's default model. Takes roughly 40 seconds, most of it PDF extraction.

**Safe to call twice.** The document is stored once per source path, and each
model skips chunks it has already embedded -- so a repeat call re-reads the PDF
but does not pay for the same vectors again:

```json
{"document_id": "97d1...", "source_uri": "sample/storybook.pdf",
 "reused_existing_document": true, "chunks": 155,
 "embedded": {"emb_openai_te3_small": 0}}
```

To embed the same document with another model, name its table:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"models": ["emb_qwen3_0_6b"]}'
```

`path` defaults to `sample/storybook.pdf` and is constrained to `sample/`;
anything escaping it returns 400.

This route **spends against your OpenAI key** and has no auth. It is a
development convenience -- gate or remove it before deploying.

### From Python

Ingestion is also driven directly from `add_docs.py`:

```python
from db import connection
from registry import get_model
from add_docs import ingest
from pdf_extractors import simple_extractor

with connection() as conn:
    models = [get_model(conn, table_name="emb_openai_te3_small")]
    ingest(conn, user_id=1, texts=simple_extractor("./sample/storybook.pdf"),
           models=models, title="storybook", source_uri="./sample/storybook.pdf")
```

`ingest` stores the document and its chunks, then embeds them with each model
given. To add a second model later, call `embed_document` -- chunks already
embedded by that model are skipped, so it only does the missing work.

## Layout

| file | role |
| ---- | ---- |
| `db.py` | Connection settings and the psycopg pool. |
| `registry.py` | Reads `embedding_models`; resolves a model to its table. |
| `pg_vector.py` | Embedding-service factory, cached engines and vector stores. |
| `setup_db.py` | Migration runner and vector-table creation. The only place that runs DDL. |
| `add_docs.py` | Chunking, `documents`/`document_chunks` writes, per-model embedding. |
| `pdf_extractors.py` | PDF to Markdown `Document`s, split on heading structure. |
| `llm/init.py` | Builds the retrieval chain for one user. |
| `api/server.py` | FastAPI app; opens and closes the pools via lifespan. |
| `database/` | SQL migrations and the vector-table template. |

### Schema is created once, on purpose

`setup_db.py` is the only code that creates tables. The application connects to
existing tables and fails with a clear message if one is missing:

```
Table 'emb_openai_te3_small' is missing or has the wrong columns.
Create it with: uv run python setup_db.py
```

Creating tables at application startup instead would mean any restart could
silently alter the schema, and concurrent starts would race. Keeping DDL in one
deliberate step is the same principle as a migration, without the machinery --
if this grows into a deployed service, Alembic is the natural next step.

## Configuration

Everything is environment variables, read in `db.py` and `pg_vector.py`:

| variable | default | purpose |
| -------- | ------- | ------- |
| `DATABASE_URL` | `postgresql://postgres:admin@localhost:5432/postgres` | Postgres DSN. The SQLAlchemy form is derived from it. |
| `OPENAI_API_KEY` | -- | Required. |
| `CURRENT_USER_ID` | `1` | Which user the API retrieves as, pending auth. |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `10` | psycopg pool bounds. |
| `DB_POOL_TIMEOUT` | `30` | Seconds to wait for a free connection. |
| `DB_ENGINE_POOL_SIZE` / `DB_ENGINE_MAX_OVERFLOW` | `5` / `10` | SQLAlchemy pool, used by `PGVectorStore`. |

The connection URL is rewritten to `postgresql+psycopg://` for SQLAlchemy, which
selects psycopg **3** -- `PGEngine` builds an async engine, and psycopg2 cannot
back one.

### Two pools

There are two connection pools: the psycopg one in `db.py` for plain SQL, and
SQLAlchemy's inside `PGEngine` for everything `PGVectorStore` does. Both run on
psycopg 3, so this is one driver used two ways, not two drivers. They cannot be
merged while `PGVectorStore` is in use -- see the note at the top of `db.py`.

Worst case per process is `10 + 5 + 10 = 25` connections, against a Postgres
default `max_connections` of 100. Size them down before running more than two
workers.

## Chunking

Chunking happens once, in `add_docs.py`, and is deliberately **not** per-model --
all models embed the same chunks so their results stay comparable.

Defaults are 500-token chunks with 70 tokens of overlap. Smaller, focused chunks
embed to a sharper point and retrieve more accurately; very large ones average
several topics together and match none of them well.

| content type        | chunk_size | overlap |
| ------------------- | ---------- | ------- |
| prose, mixed PDFs   | 500        | 70      |
| technical reference | 800        | 100     |
| short factual Q&A   | 300        | 50      |

Oversized input is not rejected: `OpenAIEmbeddings` splits anything past 8191
tokens, embeds each piece, and returns their weighted average. That is one
blurred vector rather than an error, so retrieval quality degrades silently.
