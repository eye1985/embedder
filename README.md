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

## Tests

```bash
uv run pytest
```

Integration tests, grouped by feature. They need Docker's Postgres running and
nothing else -- no OpenAI key, and no network.

```
test/
  conftest.py       Test database, migrations, truncation, the fake embedder,
                    and stored_pdf -- a real file under uploads/.
  fakes.py          Deterministic unit-vector embedder. No test reaches OpenAI.
  factories.py      a_document(), a_job(), an_export(), and the readers.
  integration/
    jobs/               One job at a time, and what happens when one dies.
      concurrency/      The rule: an upload, a re-embed and an import exclude
                        one another.
        test_mutual_exclusion.py   Both directions, enforced by the index.
        test_slot_release.py       Handed back on success and on failure.
      recovery/         A job whose process died, and the way back.
        test_startup_sweep.py      Status corrected, data kept, slot freed.
        test_replay_plan.py        What re-running would mean. Four answers.
        test_rerun.py              The repair endpoint.
      page/             What /jobs reads and what its buttons do.
        test_listing.py            Order, and whether Re-run is offered.
        test_current_job.py        What is in flight, and the last outcome.
        test_deleting.py           Pruning history vs freeing a stuck slot.
    re_embedding/       A stored document gaining a model it lacks.
      test_adding_a_model.py     The feature, and retrieval left alone.
      test_idempotency.py        A repeat writes nothing and duplicates nothing.
      test_stored_chunks.py      Chunks are the input; the PDF is never reread.
      test_refusals.py           No vector table, and someone else's document.
    chat_sources/       What an answer says about where a passage came from.
      test_what_a_source_offers.py  PDF, restored from JSON, or file gone.
    chat_history/       A stored conversation, and what its answers still cite.
      test_a_stored_answer_keeps_its_sources.py  Through a delete and a reimport.
    deleting/           Delete means gone: rows, vectors and the file.
      test_deleting_a_document.py   Asserted against the disk, not the reply.
    restore_from_json/  A downloaded export uploaded back.
      test_round_trip.py         The promise: the same file comes out again.
      test_refusing_imports.py   Already here, and files that are not exports.
      test_overwriting.py        Replacing one, and what survives it.
      test_the_job_row.py        The slot an import holds, and handing it back.
```

Run one feature's tests with `uv run pytest test/integration/jobs`, or one
facet with `uv run pytest test/integration/jobs/recovery`.

They are integration rather than unit tests because what they check lives in
Postgres: one job at a time is a partial unique index, idempotent embedding is
a `NOT EXISTS` plus a unique constraint, and restart recovery is an UPDATE.
Mocking psycopg would assert that the code called `insert`, which would still
pass with the index dropped.

Two things keep that safe and free, both in `test/conftest.py`:

- Their own database, `embeddings_test`, created and migrated by a fixture.
  A run never touches development data. Override the name with
  `TEST_DATABASE_NAME`.
- A fake embedding service (`test/fakes.py`) that returns deterministic
  unit vectors at the model's width. It is installed session-wide and
  autouse, so no test can reach OpenAI by accident -- but the real
  PGVectorStore still does the writing, so the foreign keys, the unique
  constraint on `chunk_id` and the `halfvec` column are genuinely exercised.

Tables are truncated between tests rather than wrapped in a rolled-back
transaction, and that is forced by the design under test: `begin_job` claims
the slot on its own connection and commits, so a running job is visible to
other requests. A test holding everything in one transaction could not see
those rows.

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

Four pages, all scoped to `CURRENT_USER_ID` until there is real auth: `/`
(chat), `/upload`, `/documents`, `/jobs`. The JSON API is prefixed with `/api`
so a route can never collide with a page.

| method | route | purpose |
| ------ | ----- | ------- |
| `POST` | `/api/chat` | Stream an answer, then its sources, and store the turn. |
| `GET` | `/api/conversations` | Stored conversations, most recently used first. |
| `GET` | `/api/conversations/{id}` | One transcript, each answer's sources resolved. |
| `PATCH` | `/api/conversations/{id}` | Rename a conversation. |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation and its messages. |
| `GET` | `/api/models` | Registered models, for the upload picker. |
| `POST` | `/api/upload` | Store a PDF, chunk it, embed it. |
| `GET` | `/api/documents` | List documents with chunk and vector counts. |
| `PATCH` | `/api/documents/{id}` | Take a document in or out of search. |
| `DELETE` | `/api/documents/{id}` | Delete the document, its vectors and its PDF. |
| `GET` | `/api/documents/{id}/source` | Download the original PDF. |
| `GET` | `/api/documents/{id}/embeddings` | Download one document as JSON. |
| `POST` | `/api/documents/import` | Upload such a download back, exactly as it was. |
| `POST` | `/api/documents/{id}/embeddings` | Embed an existing document with more models. |
| `DELETE` | `/api/documents/{id}/embeddings` | Delete vectors, keep the chunks. |
| `PUT` | `/api/documents/{id}/search-model/{table}` | Choose which model retrieves it. |
| `GET` | `/api/embeddings` | Download every document as a zip. |
| `GET` | `/api/jobs/current` | Whether a job is running, and the last outcome. |
| `GET` | `/api/jobs` | Every job, newest first, with what it wrote. |
| `POST` | `/api/jobs/{id}/rerun` | Run a finished or interrupted job again. |
| `DELETE` | `/api/jobs/{id}` | Forget a job row, freeing a stuck slot. |

```bash
# upload -- `models` is repeatable; omitted, the user's default model is used
curl -X POST http://localhost:8000/api/upload \
  -F 'file=@sample/storybook.pdf' \
  -F 'models=emb_openai_te3_small'

curl http://localhost:8000/api/jobs/current
curl http://localhost:8000/api/jobs
curl http://localhost:8000/api/documents
curl -OJ http://localhost:8000/api/documents/<id>/embeddings
curl -OJ http://localhost:8000/api/documents/<id>/source
curl -OJ http://localhost:8000/api/embeddings

# the reverse of that first download: no PDF is read and nothing is embedded,
# so the vectors that land are the ones in the file. A document already stored
# comes back 409; `overwrite` replaces it, keeping the PDF it came from.
#
# An export is NOT a backup of the PDF. A document deleted and then restored
# has its chunks and vectors but no file, so it is marked origin='import' and
# a chat answer's sources name it without offering a download.
curl -X POST http://localhost:8000/api/documents/import \
  -F 'file=@<id>-<title>.json'
curl -X POST http://localhost:8000/api/documents/import \
  -F 'file=@<id>-<title>.json' -F 'overwrite=true'

curl -X POST http://localhost:8000/api/documents/<id>/embeddings \
  -H 'Content-Type: application/json' -d '{"models":["emb_qwen3_0_6b"]}'
curl -X PATCH http://localhost:8000/api/documents/<id> \
  -H 'Content-Type: application/json' -d '{"searchable":false}'
curl -X PUT http://localhost:8000/api/documents/<id>/search-model/emb_qwen3_0_6b

curl -X DELETE http://localhost:8000/api/documents/<id>
curl -X DELETE 'http://localhost:8000/api/documents/<id>/embeddings?model=emb_qwen3_0_6b'

curl -X POST http://localhost:8000/api/jobs/<job-id>/rerun
curl -X DELETE http://localhost:8000/api/jobs/<job-id>
```

One job at a time per user, across uploads and re-embeds both: a second of
either returns 409, enforced by a partial unique index rather than by
application code. A job outlives the page that started it, so
`GET /api/jobs/current` is how a reloaded page finds its way back. A filename
already taken is refused, not overwritten.

`POST /api/documents/{id}/embeddings` is how a document gains a model it was
not uploaded with. Chunks are stored once and are model-independent, so nothing
is re-extracted and chunks a model already holds are skipped -- which also
makes it the repair path for a job that died midway. Which model *searches* the
document does not move; that stays a separate choice on the documents page.

A crash leaves a job row marked running with nothing alive to correct it.
Startup marks those `interrupted` and keeps what they wrote; `/jobs` is where
they are re-run, or deleted if the slot is stuck. Deleting a row that is
genuinely still running frees the slot but does not stop the work -- nothing
running reads that row.

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
api/            Speaks HTTP. No SQL of its own, no pipeline.
  routes/       One router per resource, composed in routes/__init__.py.
  services/     The work behind the routes. One module per feature, or a
                directory once a feature outgrows one file.
    documents.py  The shapes the document endpoints return.
    jobs.py       Owns the one-at-a-time slot.
    export.py     One document as JSON, a library as a zip. EXPORT_VERSION.
    from_json/    An export read back in. Split where the database starts:
      parse.py      Checks and takes apart the uploaded bytes. No SQL.
      restore.py    Matches the registry and writes the rows.
    conversations.py  Stored chat history read back: the list, one transcript.
    sources.py    A source_uri resolved to a file, and stored citations to chips.
    uploads.py    Where an uploaded file lands.
  config.py     CURRENT_USER_ID and the frontend directories.
  schemas.py    Response models. What the generated frontend types come from.
  assets.py     Resolves a frontend entry to built URLs via the Vite manifest.
  server.py     The app: lifespan, static mount, error handlers.
db/             Talks to Postgres. Every SQL statement lives here.
  tables/       Row access, one module per table group.
  pool.py       Connection settings and the psycopg pool.
  vectors.py    Embedding-service factory, cached engines and vector stores.
  setup.py      Migration runner and vector-table DDL. The only place running DDL.
indexing/       Files into embeddings. extract -> chunk -> embed, pdf.py on top.
llm/chat.py     Builds the retrieval graph for one user. Holds no history:
                every turn is handed the conversation it answers against.
  context_management/  How that history is kept bounded. One strategy is chosen.
paths.py        PROJECT_ROOT, which every stored source_uri is relative to.
sql/            Migrations. Vector-table DDL is in db/setup.py.
test/           Fixtures, fakes and builders; integration/ is one dir per feature.
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

## Chat history

Conversations are stored in `conversations` and `chat_messages`, and that is
the source of truth: the graph in `llm/chat.py` compiles without a
checkpointer, so every turn loads the history it answers against and writes
the turn back when the answer is done. History therefore survives a restart,
and an answer does not depend on which process served the previous question.

`conversations` also carries the compaction state -- the running summary and
how many messages it covers -- because that state belongs to the conversation
rather than to a process. It is written in the same transaction as the
messages: `summarized` is a count of them, so committing it separately could
leave it pointing past the end.

A turn is stored whole or not at all. The question and the answer land
together, so history never holds a question with no reply -- the next turn
would then send two user messages in a row, which some providers reject. A run
that fails partway still stores what arrived, matching what the page keeps on
screen.

### What an answer remembers about its sources

`chat_messages.citations` holds one entry per cited document, best-ranked
first, each naming the retrieved chunks that document contributed:

```json
[{"document_id": "...", "chunk_ids": ["...", "..."]}]
```

Rank is the position. The chunk ids are every retrieved chunk of that
document, not only the ones a citation marker matched, because that is the set
the page numbers come from -- storing less would replay an answer with fewer
pages than it was sent with.

Neither id is a foreign key, which is deliberate. A citation has to outlive
its rows:

* **Document deleted.** The ids resolve to nothing, so the answer offers no
  sources -- the text keeps its claims, the chips are gone. `on delete cascade`
  would instead erase the citation, which looks the same until the next case.
* **Deleted, then restored from a JSON export.** `restore_document` puts the
  document back under the id it had and its chunks under theirs, so the stored
  ids are the ids that come back and the sources return. An `on delete set
  null` would have broken that link permanently.

Because the ids are re-resolved on every read, a reopened conversation
describes its documents as they stand now rather than as they stood then: a
restored document's chips say "from JSON", since restoring brings back the
text but never the PDF.

A GIN index on `citations` answers the reverse question, deleted document or
not:

```sql
select * from chat_messages where citations @> '[{"document_id": "..."}]';
```
