"""Keeping the last few exchanges and forgetting the rest."""

from dataclasses import dataclass

from langchain_core.messages import AnyMessage

from llm.context_management.base import Conversation, last_exchanges


@dataclass(frozen=True)
class SlidingWindow:
    """The last `keep_turns` exchanges, nothing older.

    Free and predictable -- no extra model call, no state to keep. The price
    is that it forgets outright: a question about something said before the
    window cannot be answered from history at all.
    """

    keep_turns: int = 6

    async def compact(self, convo: Conversation) -> dict:
        # Nothing to store. The window is applied when the prompt is built,
        # so the stored transcript stays whole and the strategy can be
        # swapped without having destroyed anything.
        return {}

    def view(self, convo: Conversation) -> list[AnyMessage]:
        return convo.messages[last_exchanges(convo.messages, self.keep_turns) :]
