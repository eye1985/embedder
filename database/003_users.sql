-- Supersedes user.sql. `user` is a reserved keyword in Postgres, so the table
-- is `users`.

create table users (
    id         serial primary key,
    email      varchar not null unique,
    is_premium boolean not null default false,
    default_embedding_model_id int not null references embedding_models(id),
    created_at timestamptz not null default now()
);
