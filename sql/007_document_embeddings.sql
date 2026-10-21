-- Which models hold vectors for which documents, and whether that pairing is
-- searchable. A junction table between `documents` and `embedding_models`:
-- the vectors themselves stay in the per-model tables, one row per chunk;
-- this is the same relationship at document granularity, with somewhere to
-- hang state.
--
-- A row exists only where vectors exist, which makes three states distinct:
--
--   no row            never embedded by that model -- switching it on means
--                     paying the embedding API
--   enabled = false   embedded, vectors on disk, left out of search -- free
--                     and instant to switch back on
--   enabled = true    searched whenever that model is the user's search model
--
-- `enabled` deliberately lives here and not on the vector tables. A search
-- filters on columns of its own table, so a flag there would be reachable by
-- the filter -- but it would also be mutable state duplicated across every
-- model, and the copies would drift. The searchable set is resolved from this
-- table and passed to the search as a document_id filter instead.

create table document_embeddings (
    document_id uuid not null references documents(id) on delete cascade,
    model_id    int  not null references embedding_models(id) on delete cascade,
    enabled     boolean not null default true,
    created_at  timestamptz not null default now(),

    primary key (document_id, model_id)
);

-- Serves the search's "which documents may this model return?" lookup.
create index on document_embeddings (model_id) where enabled;
