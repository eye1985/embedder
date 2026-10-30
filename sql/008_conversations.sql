-- Stored chat history: one row per conversation, one per message.
--
-- This is the source of truth for a conversation, not a cache of one. The
-- graph in `llm/chat.py` runs without a checkpointer: every turn loads its
-- history from here, and writes the turn back when the answer is done.

create table conversations (
    id         uuid primary key default gen_random_uuid(),
    user_id    int  not null references users(id) on delete cascade,
    -- From the first question, truncated. Null until that turn is stored.
    title      text,
    -- The compaction state `llm/context_management/summarize.py` keeps:
    -- the running notes, and how many of the messages below they cover.
    -- `summarized` is a prefix count, so it stays aligned as long as the
    -- transcript is loaded in order. A strategy that keeps no summary (the
    -- sliding window) leaves both at their defaults.
    summary    text,
    summarized int  not null default 0,
    created_at timestamptz not null default now(),
    -- Bumped on every stored turn: what the conversation list sorts by.
    updated_at timestamptz not null default now()
);

create index on conversations (user_id, updated_at desc);

create table chat_messages (
    id              uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    -- Reading order, not a timestamp. `now()` is transaction time, so a
    -- question and its answer written together would tie on created_at;
    -- `document_chunks.chunk_index` numbers its rows for the same reason.
    seq             int  not null,
    role            text not null check (role in ('user', 'assistant')),
    content         text not null,
    -- What the answer cited, one entry per document, best-ranked first:
    --
    --   [{"document_id": "...", "chunk_ids": ["...", "..."]}]
    --
    -- Position carries the rank. `chunk_ids` is every retrieved chunk of that
    -- document, not only the marker-matched ones, because that is the set the
    -- live answer's page numbers come from -- storing less would replay an
    -- answer with fewer pages than it was sent with.
    --
    -- Deliberately not foreign keys. The ids have to outlive their rows: a
    -- document deleted and later restored from a JSON export comes back under
    -- the same document id and the same chunk ids (`restore_document`), so a
    -- plain uuid re-resolves where a reference could not. `on delete cascade`
    -- would erase the citation outright, and `on delete set null` would break
    -- the link permanently and not re-attach. While the document is gone the
    -- ids resolve to nothing and the answer simply shows no sources.
    citations       jsonb not null default '[]'::jsonb,
    created_at      timestamptz not null default now(),

    unique (conversation_id, seq)
);

-- Answers the reverse question -- which answers cited this document --
-- including for a document that has since been deleted:
--
--   where citations @> '[{"document_id": "..."}]'
--
-- `jsonb_path_ops` indexes containment only, which is the one operator this
-- is for, and is smaller than the default opclass.
create index on chat_messages using gin (citations jsonb_path_ops);
