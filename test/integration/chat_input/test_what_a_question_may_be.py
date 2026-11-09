"""A question is refused before anything is searched or paid for when it is
empty or longer than the cap."""

import pytest

from api.routes.chat import MAX_PROMPT_CHARS


@pytest.mark.parametrize("prompt", ["", "x" * (MAX_PROMPT_CHARS + 1)])
def test_an_empty_or_oversized_question_is_refused(client, prompt):
    assert client.post("/api/chat", json={"prompt": prompt}).status_code == 422
