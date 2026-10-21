-- Development seed. Keeps a usable account around after a wipe, so the app can
-- start without a manual insert -- `CURRENT_USER_ID` defaults to 1, and on a
-- fresh schema this is the first row, so it lands on id 1.
--
-- Drop this migration before deploying anywhere real.
--
-- `upload_model_id` is required and never guessed, so every insert into
-- `users` has to name it.

insert into users (email, upload_model_id)
select 'testuser@mail.com', id
from embedding_models
where table_name = 'emb_openai_te3_small'
on conflict (email) do nothing;
