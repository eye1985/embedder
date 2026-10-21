-- One row per /upload request, so "is an upload running?" outlives the process
-- that answered it. The previous version of this lived in a dict guarded by a
-- threading.Lock, which reset on restart and was invisible to other workers.

create table upload_jobs (
    -- A counter, not a uuid: this id never leaves the server. It is a handle
    -- passed from begin_upload() to finish_upload() and nothing else -- no
    -- route takes it, no response returns it. `documents` uses a uuid because
    -- its id IS in URLs and a counter there would be enumerable.
    id          serial primary key,
    user_id     int  not null references users(id) on delete cascade,
    filename    text not null,
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

-- The one-upload-per-user rule, enforced by the database rather than by a
-- process-local lock: a second concurrent upload fails this insert instead of
-- racing the first past a check-then-act. Partial, so finished jobs pile up
-- freely and only the running one is constrained.
create unique index upload_jobs_one_running_per_user
    on upload_jobs (user_id)
    where status = 'running';

-- Serves the status endpoint: most recent job for a user.
create index on upload_jobs (user_id, started_at desc);
