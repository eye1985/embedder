-- Provider-agnostic source of truth. Text is chunked once here, then embedded
-- by any number of models -- each into its own vector table.

create table documents (
    id         uuid primary key default gen_random_uuid(),
    user_id    int  not null references users(id) on delete cascade,
    title      text,
    source_uri text,
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
