-- Registry of every embedding type in the system.
--
-- One row == one physical vector table, named by `table_name`. Postgres cannot
-- enforce that link (a CHECK cannot reference the catalog), so `setup_db.py`
-- is what keeps the registry and the physical tables in sync.

create table embedding_models (
    id          serial primary key,
    provider    text not null,                      -- 'openai' | 'qwen' | 'cohere'
    model_name  text not null,                      -- 'text-embedding-3-small'
    dimensions  int  not null check (dimensions > 0),
    table_name  text not null unique,               -- physical vector table
    distance    text not null default 'cosine'
                check (distance in ('cosine', 'inner_product', 'l2')),
    is_active   boolean not null default true,
    created_at  timestamptz not null default now(),

    unique (provider, model_name, dimensions)
);

-- Only rows marked active get a vector table from `setup_db.py`, and the
-- upload form only offers a model that has one.
--
-- `dimensions` decides the column type, which `setup_db.py` derives: up to
-- 2000 it is vector(n), wider it is halfvec(n), which pgvector indexes to
-- 4000. So nothing registered here has to fit under 2000.
--
-- text-embedding-3-large is registered at its native 3072, which lands in the
-- halfvec tier: indexable, and 6152 bytes a row -- the same as -3-small at
-- vector(1536). Truncating it would cost ranking accuracy for no storage
-- saving, since fp16 over all 3072 dimensions reproduces the full-precision
-- ranking while dropping half the dimensions does not.
--
-- Qwen3-Embedding-8B at 4096 is past even the halfvec index limit, so it ships
-- inactive.
--
-- Qwen3-0.6B fits the index fine but ships inactive for a different reason:
-- the provider needs langchain-huggingface and sentence-transformers, which
-- are not dependencies yet, so `get_embeddings` cannot build it. Activate it
-- once those are installed.
insert into embedding_models
    (provider, model_name, dimensions, table_name, is_active)
values
    ('openai', 'text-embedding-3-small',     1536, 'emb_openai_te3_small',      true),
    ('openai', 'text-embedding-3-large',     3072, 'emb_openai_te3_large',      true),
    ('qwen',   'Qwen/Qwen3-Embedding-0.6B',  1024, 'emb_qwen3_0_6b',            false),
    ('qwen',   'Qwen/Qwen3-Embedding-8B',    4096, 'emb_qwen3_8b',              false);
