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

-- The two models above 2000 dimensions ship inactive: pgvector's HNSW index
-- stops at 2000, so their tables would be searched by exact scan. Both support
-- Matryoshka truncation, so the practical fix is to register them at a smaller
-- `dimensions` rather than to activate them as-is.
insert into embedding_models
    (provider, model_name, dimensions, table_name, is_active)
values
    ('openai', 'text-embedding-3-small',     1536, 'emb_openai_te3_small', true),
    ('openai', 'text-embedding-3-large',     3072, 'emb_openai_te3_large', false),
    ('qwen',   'Qwen/Qwen3-Embedding-0.6B',  1024, 'emb_qwen3_0_6b',       true),
    ('qwen',   'Qwen/Qwen3-Embedding-8B',    4096, 'emb_qwen3_8b',         false);
