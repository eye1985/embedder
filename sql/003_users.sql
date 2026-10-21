-- Supersedes user.sql. `user` is a reserved keyword in Postgres, so the table
-- is `users`.

create table users (
    id         serial primary key,
    email      varchar not null unique,
    is_premium boolean not null default false,
    -- Which model the upload form pre-selects. There is no user-level search
    -- model: each document is retrieved through the one it was given, in
    -- `document_embeddings`, so a question can draw on several at once.
    upload_model_id int not null references embedding_models(id),
    created_at timestamptz not null default now()
);
