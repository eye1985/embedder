-- Template, not a migration. One table is created from this per row in
-- `embedding_models`, with {table_name} and {dimensions} substituted by
-- setup_db.py.
--
-- These tables are written by hand rather than by PGEngine.init_vectorstore_table
-- so the foreign keys can be declared inline. PGVectorStore.create_sync does not
-- create a table -- it reads information_schema.columns and checks only that the
-- id, content, embedding and declared metadata columns exist, with content being
-- text/char and embedding being vector. Constraints and indexes are not inspected,
-- so the additions below are invisible to it.
--
-- Vector tables have no relationship to one another. Dropping one leaves every
-- other table untouched.
--
-- Only two foreign keys are carried:
--   chunk_id -- the real relationship. Deleting a document cascades to its
--               chunks, which cascades to the vectors here. UNIQUE because a
--               chunk is embedded at most once per model.
--   user_id  -- denormalized from documents.user_id. Required, not convenience:
--               a similarity search filters on columns of THIS table, and HNSW
--               returns its top-k before a join to `documents` could scope the
--               results to one user.
--
-- Note the FK on user_id guarantees only that the user exists, NOT that it
-- matches the chunk's owner. Keeping the two in agreement is the writer's job.
--
-- {index_ddl} is filled in by setup_db.py. pgvector's HNSW index tops out at
-- 2000 dimensions, so models above that get no index and fall back to an exact
-- scan. A halfvec index would reach 4000, but langchain orders by
-- `embedding <=> :query` with no cast, so an expression index on
-- embedding::halfvec would never be chosen by the planner.

create table {table_name} (
    langchain_id uuid primary key default gen_random_uuid(),
    content      text not null,
    embedding    vector({dimensions}) not null,
    chunk_id     uuid not null unique
                 references document_chunks(id) on delete cascade,
    user_id      int not null
                 references users(id) on delete cascade,
    langchain_metadata jsonb
);

create index on {table_name} (user_id);
{index_ddl}
