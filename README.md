# embeddings

Ingest PDFs into a local Postgres/pgvector store and chat over them, using
LangChain. Each user can hold several kinds of embedding side by side -- one
vector table per model -- and chunks are stored once, independent of any model.

## Requirements

- Python >= 3.14, [uv](https://docs.astral.sh/uv/)
- Node >= 20, npm
- Docker (Postgres + pgAdmin)
- An OpenAI API key

> `typescript` is pinned to `^5`. TypeScript 7's native port drops the compiler
> factory API `openapi-typescript` uses, so `npm run api-types` fails on 7.x.

## Setup

```bash
uv sync                      # Python dependencies
npm install                  # frontend dependencies
npm run build                # bundle frontend/ into static/ -- required once
docker compose up -d         # Postgres on 5432, pgAdmin on 5050
echo 'OPENAI_API_KEY=sk-...' > .env
uv run python -m db.setup    # migrations, then one vector table per active model
```

| service  | port | credentials                                        |
| -------- | ---- | -------------------------------------------------- |
| postgres | 5432 | `postgres` / `admin`, database `postgres`           |
| pgadmin  | 5050 | `pgadmin4@pgadmin.org` / `admin`                   |

Override with `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`,
`PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`, `PGADMIN_PORT`.

Migration `005_seed_user.sql` creates user id 1 (`testuser@mail.com`), which is
what `CURRENT_USER_ID` defaults to. It is a development seed -- drop it before
deploying. Add your own:

```bash
psql -h localhost -U postgres -d postgres \
  -c "insert into users (email) values ('you@example.com');"
```

## Running

```bash
uv run uvicorn api.server:app --reload    # http://localhost:8000
npm run dev:api                           # same, pointed at the Vite dev server
npm run dev                               # Vite with hot reload, port 5173
```

Python reloads itself; the frontend does not. `npm run dev:api` sets
`VITE_DEV_SERVER` so assets come from Vite instead of `static/`.

| command | effect |
| ------- | ------ |
| `npm run build` | Typecheck, bundle, minify into `static/`. What deploys. |
| `npm run watch` | Rebuild on save. Reload the page yourself. |
| `npm run typecheck` | `tsc --noEmit`, nothing written. |
| `npm run openapi` | Dump the OpenAPI schema to `openapi.json`. |
| `npm run api-types` | That, then regenerate `frontend/lib/api-types.ts`. |

Run `npm run api-types` after changing any response model, so a renamed field
fails `npm run build` instead of reaching the browser.

## Schema

```bash
uv run python -m db.setup                              # apply migrations, create missing tables
uv run python -m db.setup --list                       # every model, and whether its table exists
uv run python -m db.setup --model emb_qwen3_0_6b       # one model's table, active or not
uv run python -m db.setup --recreate                   # DROPS the vector tables first
```

Plain SQL migrations live in `sql/`, applied in filename order and tracked in
`schema_migrations`. Add one by dropping in `00N_name.sql` and re-running.
Vector-table DDL is in `db/setup.py` instead, since its name and width come
from the registry.

`--recreate` destroys embeddings but not documents or chunks, so you can
re-embed without re-extracting. Starting over completely:

```bash
psql -h localhost -U postgres -d postgres \
  -c "drop schema public cascade; create schema public;"
uv run python -m db.setup
```

### Adding an embedding model

```sql
insert into embedding_models (provider, model_name, dimensions, table_name)
values ('openai', 'text-embedding-3-large', 1024, 'emb_openai_te3_large_1024');
```

Re-run `db/setup.py`, then teach `get_embeddings()` in `db/vectors.py` about the
provider if it is new. `openai`, `qwen` and `huggingface` are handled; the last
two need `uv add langchain-huggingface sentence-transformers`.

The column type follows `dimensions` and is not yours to pick: `vector(n)` up to
2000, `halfvec(n)` to 4000 (both HNSW-indexed), `halfvec(n)` unindexed above
that.

## API

Three pages, all scoped to `CURRENT_USER_ID` until there is real auth: `/`
(chat), `/upload`, `/documents`. The JSON API is prefixed with `/api` so a route
can never collide with a page.

| method | route | purpose |
| ------ | ----- | ------- |
| `POST` | `/api/chat` | Stream an answer, then its sources. |
| `GET` | `/api/models` | Registered models, for the upload picker. |
| `POST` | `/api/upload` | Store a PDF, chunk it, embed it. |
| `GET` | `/api/upload/status` | Whether an upload is running, and its outcome. |
| `GET` | `/api/documents` | List documents with chunk and vector counts. |
| `PATCH` | `/api/documents/{id}` | Take a document in or out of search. |
| `DELETE` | `/api/documents/{id}` | Delete the document, its vectors and its PDF. |
| `GET` | `/api/documents/{id}/source` | Download the original PDF. |
| `GET` | `/api/documents/{id}/embeddings` | Download one document as JSON. |
| `POST` | `/api/documents/{id}/embeddings` | Embed an existing document with more models. |
| `DELETE` | `/api/documents/{id}/embeddings` | Delete vectors, keep the chunks. |
| `PUT` | `/api/documents/{id}/search-model/{table}` | Choose which model retrieves it. |
| `GET` | `/api/embeddings` | Download every document as a zip. |

```bash
# upload -- `models` is repeatable; omitted, the user's default model is used
curl -X POST http://localhost:8000/api/upload \
  -F 'file=@sample/storybook.pdf' \
  -F 'models=emb_openai_te3_small'

curl http://localhost:8000/api/upload/status
curl http://localhost:8000/api/documents
curl -OJ http://localhost:8000/api/documents/<id>/embeddings
curl -OJ http://localhost:8000/api/documents/<id>/source
curl -OJ http://localhost:8000/api/embeddings

curl -X POST http://localhost:8000/api/documents/<id>/embeddings \
  -H 'Content-Type: application/json' -d '{"models":["emb_qwen3_0_6b"]}'
curl -X PATCH http://localhost:8000/api/documents/<id> \
  -H 'Content-Type: application/json' -d '{"searchable":false}'
curl -X PUT http://localhost:8000/api/documents/<id>/search-model/emb_qwen3_0_6b

curl -X DELETE http://localhost:8000/api/documents/<id>
curl -X DELETE 'http://localhost:8000/api/documents/<id>/embeddings?model=emb_qwen3_0_6b'
```

One upload at a time per user -- a second returns 409. An upload outlives the
page that started it, so `GET /api/upload/status` is how a reloaded page finds
its way back. A filename already taken is refused, not overwritten.

These routes spend against your OpenAI key and have no auth. Gate them before
deploying.

### From Python

```python
from db.pool import connection
from db.tables.registry import get_model
from indexing.pdf import import_pdf
from paths import PROJECT_ROOT

with connection() as conn:
    models = [get_model(conn, table_name="emb_openai_te3_small")]
    import_pdf(conn, 1, PROJECT_ROOT / "sample/storybook.pdf", models)
```

Pass an **absolute** path: `source_uri` is derived relative to `PROJECT_ROOT`
and is the key uploads dedupe on, so a direct call and an upload of the same
file agree on one document. A document already stored for that path is reused
rather than re-extracted.

## Layout

```
api/            Speaks HTTP. No SQL, no export format, no pipeline.
  routes/       One router per resource, composed in routes/__init__.py.
  services/     The work behind the routes. uploads.py owns the upload_jobs SQL.
  config.py     CURRENT_USER_ID and the frontend directories.
  schemas.py    Response models. What the generated frontend types come from.
  assets.py     Resolves a frontend entry to built URLs via the Vite manifest.
  server.py     The app: lifespan, static mount, error handlers.
db/             Talks to Postgres. All SQL lives here.
  tables/       Row access, one module per table group.
  pool.py       Connection settings and the psycopg pool.
  vectors.py    Embedding-service factory, cached engines and vector stores.
  setup.py      Migration runner and vector-table DDL. The only place running DDL.
indexing/       Files into embeddings. extract -> chunk -> embed, pdf.py on top.
llm/chat.py     Builds the retrieval chain for one user.
export.py       One document as JSON, a library as a zip. No SQL, no HTTP.
paths.py        PROJECT_ROOT, which every stored source_uri is relative to.
sql/            Migrations. Vector-table DDL is in db/setup.py.
templates/      Jinja2 page shells. base.html holds the only copy of the nav.
frontend/       styles/, components/, lib/, pages/, main.css. Built into static/.
static/         Build output. Git-ignored; npm run build recreates it.
```

`frontend/lib/api-types.ts` is generated -- do not edit.

## Configuration

| variable | default | purpose |
| -------- | ------- | ------- |
| `DATABASE_URL` | `postgresql://postgres:admin@localhost:5432/postgres` | Postgres DSN. The SQLAlchemy form is derived from it. |
| `OPENAI_API_KEY` | -- | Required. |
| `CURRENT_USER_ID` | `1` | Which user the API acts as, pending auth. |
| `VITE_DEV_SERVER` | -- | Serve assets from Vite instead of `static/`. |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `10` | psycopg pool bounds. |
| `DB_POOL_TIMEOUT` | `30` | Seconds to wait for a free connection. |
| `DB_ENGINE_POOL_SIZE` / `DB_ENGINE_MAX_OVERFLOW` | `5` / `10` | SQLAlchemy pool, used by `PGVectorStore`. |

Two pools, both on psycopg 3: the psycopg one in `db/pool.py` for plain SQL, and
SQLAlchemy's inside `PGEngine` for everything `PGVectorStore` does. Worst case
is `10 + 5 + 10 = 25` connections per process against a default
`max_connections` of 100 -- size them down before running more than two workers.

## Chunking

500-token chunks with 70 tokens of overlap, set in `indexing/chunk.py`. Done
once, not per model, so every model's results stay comparable. Input past 8191
tokens is averaged into one blurred vector rather than rejected, so retrieval
degrades silently.
