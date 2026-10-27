-- One row per job that writes embeddings: an upload, or a re-embed of a
-- document already stored. Durable, so "is a job running?" outlives the
-- process that answered it.

create table jobs (
    -- A counter, not a uuid: this id is a handle for the jobs page and
    -- nothing else. `documents` uses a uuid because its id is in URLs.
    id          serial primary key,
    user_id     int  not null references users(id) on delete cascade,
    kind        text not null check (kind in ('upload', 'embed')),
    -- The filename for an upload, the document title for a re-embed.
    subject     text not null,
    models      jsonb not null default '[]'::jsonb,
    status      text not null
                check (status in ('running', 'succeeded', 'failed', 'interrupted')),
    source_uri  text,
    -- Null until the document exists, and again if it is later deleted: job
    -- history should not keep a document alive or break when one goes.
    document_id uuid references documents(id) on delete set null,
    result      jsonb,
    error       text,
    started_at  timestamptz not null default now(),
    finished_at timestamptz,

    -- A finished job has an end time; a running one does not.
    check ((status = 'running') = (finished_at is null))
);

-- One job at a time per user, enforced by the database rather than by a
-- process-local lock: a second concurrent job fails this insert instead of
-- racing the first past a check-then-act. This is what makes an upload and a
-- re-embed exclude each other -- the index does not care which kind the row
-- describes. Partial, so finished jobs pile up freely.
create unique index jobs_one_running_per_user
    on jobs (user_id)
    where status = 'running';

-- Serves the jobs page and the current-job lookup.
create index on jobs (user_id, started_at desc);
