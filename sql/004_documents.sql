-- Provider-agnostic source of truth. Text is chunked once here, then embedded
-- by any number of models -- each into its own vector table.

create table documents (
    id         uuid primary key default gen_random_uuid(),
    user_id    int  not null references users(id) on delete cascade,
    title      text,
    source_uri text,
    -- Master switch: false takes the document out of search for every model.
    -- Per-model state lives in document_embeddings and is left untouched, so
    -- switching a document back on restores exactly the models it had.
    searchable boolean not null default true,
    -- Where these rows came from. 'upload' means a PDF this app extracted and
    -- still has on disk to offer; 'import' means a JSON export restored back
    -- in, which carries chunks and vectors but no file. The two are not the
    -- same as each other and neither is the same as a PDF deleted by hand, so
    -- a chat answer's sources can say which rather than guess from the
    -- absence of a file.
    origin     text not null default 'upload'
               check (origin in ('upload', 'import')),
    created_at timestamptz not null default now()
);

create index on documents (user_id);

create table document_chunks (
    id          uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    chunk_index int  not null,
    content     text not null,
    -- Page numbers and the like, from the PDF extractor. Kept here rather than
    -- duplicated into every vector table; join back on chunk_id for citations.
    metadata    jsonb not null default '{}'::jsonb,

    unique (document_id, chunk_index)
);
