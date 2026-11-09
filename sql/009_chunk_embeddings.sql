-- Every model's vectors, one row per chunk per model.
--
-- Models up to 2000 dimensions fill `embedding`, wider ones `embedding_half`.
-- Neither column has a fixed width, since models share them; each model's
-- partial index casts to its own width instead, and a search must repeat that
-- exact cast and `model_id` literal for the index to be used.
--
-- `document_id` is copied from document_chunks so a search can filter on it:
-- HNSW returns its top-k before a join could scope the results.

create table chunk_embeddings (
    chunk_id       uuid not null references document_chunks(id) on delete cascade,
    model_id       int  not null references embedding_models(id) on delete cascade,
    document_id    uuid not null references documents(id) on delete cascade,
    embedding      vector,
    embedding_half halfvec,

    primary key (chunk_id, model_id),
    check ((embedding is null) <> (embedding_half is null))
);

create index on chunk_embeddings (document_id, model_id);

create index chunk_embeddings_m1_hnsw on chunk_embeddings
    using hnsw ((embedding::vector(1536)) vector_cosine_ops)
    where model_id = 1;

create index chunk_embeddings_m2_hnsw on chunk_embeddings
    using hnsw ((embedding_half::halfvec(3072)) halfvec_cosine_ops)
    where model_id = 2;
