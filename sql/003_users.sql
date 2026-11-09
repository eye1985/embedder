-- Supersedes user.sql. `user` is a reserved keyword in Postgres, so the table
-- is `users`.
--
-- One row per Auth0 account, created the first time it signs in.

create table users (
    id         serial primary key,
    -- Auth0's `sub` claim: the stable id of the account across sign-ins.
    auth0_sub  text not null unique,
    -- As Auth0 reported it. Not unique: two logins with the same email (a
    -- password and Google, say) are two accounts, never merged by email. Null
    -- for a connection that sends none.
    email      varchar,
    is_premium boolean not null default false,
    -- Which model the upload form pre-selects. There is no user-level search
    -- model: each document is retrieved through the one it was given, in
    -- `document_embeddings`, so a question can draw on several at once.
    upload_model_id int not null references embedding_models(id),
    created_at timestamptz not null default now()
);
