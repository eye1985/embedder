-- Registry of every embedding model in the system. Vectors from all of them
-- live in `chunk_embeddings` (009), each model behind its own partial index.

create table embedding_models (
    id          serial primary key,
    provider    text not null,                      -- 'openai'
    model_name  text not null,                      -- 'text-embedding-3-small'
    dimensions  int  not null check (dimensions > 0),
    distance    text not null default 'cosine'
                check (distance in ('cosine', 'inner_product', 'l2')),
    is_active   boolean not null default true,
    created_at  timestamptz not null default now(),

    unique (provider, model_name, dimensions)
);

-- Fixed ids: the partial indexes in 009 name them literally.
--
-- Up to 2000 dimensions a model is stored as vector(n), wider as halfvec(n),
-- which pgvector indexes to 4000. text-embedding-3-large at its native 3072
-- lands in the halfvec tier: indexable, and 6152 bytes a row -- the same as
-- -3-small at vector(1536).
insert into embedding_models (id, provider, model_name, dimensions)
values
    (1, 'openai', 'text-embedding-3-small', 1536),
    (2, 'openai', 'text-embedding-3-large', 3072);

select setval(pg_get_serial_sequence('embedding_models', 'id'), max(id))
from embedding_models;
