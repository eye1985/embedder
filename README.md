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
Applied 006_seed_user.sql
Applied 007_upload_jobs.sql
Created 'emb_openai_te3_small' with vector(1536).
Created 'emb_qwen3_0_6b' with vector(1024).
```

`005` is the vector-table template, not a migration -- the runner skips it.

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
| `007_upload_jobs.sql` | `upload_jobs`, and the one-running-upload index. |

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

`upload_jobs` sits beside this rather than inside it: one row per `/upload`
request, recording what is running and what happened. It references a document
once one exists, with `on delete set null`, so job history neither keeps a
deleted document alive nor breaks when one goes.

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

Three pages, all scoped to `CURRENT_USER_ID` (default `1`) until real auth
exists:

| page | what it does |
| ---- | ------------ |
| `/` | Chat over the ingested documents. |
| `/static/upload.html` | Upload a PDF and choose which models embed it. |
| `/static/documents.html` | List documents; download or delete them. |

The routes behind them:

| method | route | purpose |
| ------ | ----- | ------- |
| `POST` | `/chat` | Streams an answer over the user's embeddings. |
| `GET` | `/models` | Registered models, for the upload picker. |
| `POST` | `/upload` | Store a PDF, chunk it, embed it. |
| `GET` | `/upload/status` | Whether an upload is running, and its outcome. |
| `GET` | `/documents` | List documents with chunk and vector counts. |
| `GET` | `/documents/{id}/embeddings` | Download one document as JSON. |
| `GET` | `/embeddings` | Download every document as a zip. |
| `DELETE` | `/documents/{id}` | Delete a document, its vectors, and its PDF. |
| `DELETE` | `/documents/{id}/embeddings` | Delete vectors, keep the chunks. |

### Chat

`POST /chat` takes `{"prompt": "..."}` and streams plain text back, token by
token, which is what the page renders as it arrives.

Retrieval is built once per process in `llm/init.py`: `gpt-4o-mini` at
temperature 0, over an MMR search of the user's default model -- `k=10` results
drawn from `fetch_k=50` candidates, so the context favours passages that differ
from one another rather than ten phrasings of the same one.

The `user_id` filter is applied **by the search**, not after it. HNSW picks its
top-k before any caller could scope the rows, so filtering afterwards would
quietly return fewer than `k` results -- or none -- once a second user exists.

Conversation history is per user, in memory, and unbounded: it resets on
restart and grows for as long as the process lives. `get_by_session_id` in
`llm/init.py` is the one place to change for durable or trimmed history.

### Uploading a PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F 'file=@sample/storybook.pdf' \
  -F 'models=emb_openai_te3_small'
```

The file is written to `uploads/` (git-ignored), extracted, chunked, and
embedded. `models` is a repeated field of vector table names; omit it and the
user's default model is used. Naming several embeds the same chunks once per
model -- the text is extracted and stored only once either way.

```json
{"document_id": "97d1...", "source_uri": "uploads/storybook.pdf",
 "reused_existing_document": false, "chunks": 155,
 "embedded": {"emb_openai_te3_small": 155}}
```

Roughly 40 seconds for the sample PDF, most of it extraction. The whole
pipeline runs inside the request -- the point at which this would need a job
queue instead.

| response | cause |
| -------- | ----- |
| 400 | Not a `.pdf`, no filename, or a model with no vector table. |
| 404 | No `embedding_models` row for a named table. |
| 409 | A document with that filename already exists. |
| 413 | Over the 50MB limit. |
| 415 | The bytes do not start with `%PDF`. |
| 422 | No file posted, or the extractor found no text. |
| 502 | The embedding call failed. Nothing was stored. |
| 503 | The provider is unusable here -- e.g. its packages are not installed. |

A failed upload stores nothing: the document, its chunks and the file are
removed before the error returns, so the same filename can be retried.

**One upload at a time.** A second one while the first is running returns 409.
That is also what makes the filename check safe: two concurrent uploads of one
name would both pass the `find_document` check before either committed, and
store the document twice.

**The request outlives the page that started it.** `/upload` is a sync `def`,
so it runs in a threadpool thread that no client disconnect can cancel --
closing or reloading the tab does not stop the work. `GET /upload/status` is
how a reloaded page finds its way back:

```json
{"active": true, "filename": "storybook.pdf", "models": [],
 "elapsed_seconds": 12.4, "last": null}
```

Once finished, `active` is false and `last` carries the result (or the error)
for two minutes, so a page that reloads just as the upload ends still sees what
happened.

State lives in `upload_jobs`, one row per request, so it survives a restart and
is visible to every process. The one-upload rule is a partial unique index --

```sql
create unique index upload_jobs_one_running_per_user
    on upload_jobs (user_id) where status = 'running';
```

-- so a second concurrent upload fails its insert rather than racing the first
past a check-then-act.

Durable state brings its own failure: a process killed mid-upload leaves a row
claiming to be running, and that index would lock the user out for good. The
lifespan startup sweeps them, marking each `interrupted` and deleting any
document left with chunks but no vectors -- a crash between the commit and the
embedding, which `/upload` could not otherwise retry because the filename would
be taken. A graceful restart does not need this: uvicorn waits for in-flight
requests, so the job finishes normally.

The sweep assumes **one process**. Run several workers and each would clear the
others' live jobs at startup; that is the point to swap it for a heartbeat.

**A filename already taken is refused, not overwritten.** Chunks are extracted
once per document, so overwriting the file would leave the stored chunks
describing content that no longer exists. Rename, or delete the document first.

`GET /models` lists what the picker offers -- every registered model, with
`available` false for one whose vector table `setup_db.py` has not created.

### Listing and downloading

```bash
curl http://localhost:8000/documents
curl -OJ http://localhost:8000/documents/<document_id>/embeddings
curl -OJ http://localhost:8000/embeddings
```

`/documents` returns each document with its chunk count and per-model vector
counts. The second downloads one document as JSON; the third downloads every
document as a zip of JSON files plus a `manifest.json`.

A chunk carries its text and metadata once, with `embeddings` keyed by vector
table -- so a document embedded by three models is one file, not three:

```json
{"chunk_id": "c41d...", "chunk_index": 0,
 "content": "Once upon a time...",
 "metadata": {"page": 1, "total_pages": 24},
 "embeddings": {"emb_openai_te3_small": [0.013, ...],
                "emb_qwen3_0_6b": [-0.008, ...]}}
```

A `null` there means that model holds no vector for that chunk.

### Deleting

```bash
curl -X DELETE http://localhost:8000/documents/<document_id>
curl -X DELETE 'http://localhost:8000/documents/<document_id>/embeddings?model=emb_qwen3_0_6b'
```

The first removes the document, its chunks, every vector for them, and the
uploaded PDF -- one `DELETE` on `documents`, with `ON DELETE CASCADE` carrying
it through `document_chunks` into each vector table. A document ingested from
`sample/` keeps its file; those are the repo's, not a user's.

The second removes vectors only, for one model or (with `model` omitted) all of
them. The chunks survive, so the text does not need re-extracting -- but no
route re-embeds them yet, so use `embed_document` from Python for that.

These routes **spend against your OpenAI key** and have no auth. They are a
development convenience -- gate them before deploying.

### From Python

Ingestion is also driven directly from `add_docs.py`:

```python
from db import connection
from registry import get_model
from add_docs import ingest
from pdf_extractors import simple_extractor

with connection() as conn:
    models = [get_model(conn, table_name="emb_openai_te3_small")]
    ingest(conn, user_id=1, texts=simple_extractor("sample/storybook.pdf"),
           models=models, title="storybook", source_uri="sample/storybook.pdf")
```

`ingest` stores the document and its chunks, then embeds them with each model
given. To add a second model later, call `embed_document` -- chunks already
embedded by that model are skipped, so it only does the missing work.

`source_uri` is the key `/upload` dedupes on, and it is matched as an exact
string: write it exactly as the app would (`sample/storybook.pdf`, no `./`), or
the same file ends up stored twice.

This is also the only way to embed a document that already exists -- no HTTP
route does it, since `/upload` refuses a filename that is already taken:

```python
from add_docs import embed_document

with connection() as conn:
    embed_document(conn, document_id, get_model(conn, table_name="emb_qwen3_0_6b"))
    conn.commit()
```

## Layout

| file | role |
| ---- | ---- |
| `db.py` | Connection settings and the psycopg pool. |
| `registry.py` | Reads `embedding_models`; resolves a model to its table. |
| `pg_vector.py` | Embedding-service factory, cached engines and vector stores. |
| `setup_db.py` | Migration runner and vector-table creation. The only place that runs DDL. |
| `add_docs.py` | Chunking, `documents`/`document_chunks` writes, per-model embedding. |
| `pdf_extractors.py` | PDF to Markdown `Document`s, split on heading structure. |
| `main.py` | Empty stub. The app is started through `api.server`, not this. |
| `llm/init.py` | Builds the retrieval chain for one user. |
| `api/server.py` | FastAPI app; opens and closes the pools via lifespan. |
| `api/ingest.py` | Upload validation, job tracking, the extract/store/embed pipeline. |
| `api/documents.py` | Document listing, the JSON/zip export, and deletion. |
| `static/` | The three demo pages: chat, upload, documents. |
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
