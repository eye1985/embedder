"""What the model is shown of a conversation, and how it stays bounded."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import tiktoken
from langchain_core.messages import AnyMessage, HumanMessage


@dataclass(frozen=True)
class Conversation:
    """One turn's history, as the strategies see it.

    `summarized` is how many of `messages` are already folded into `summary`,
    so a strategy that keeps one knows what it has read before. A strategy
    that does not keep a summary ignores both.
    """

    messages: list[AnyMessage]
    summary: str | None = None
    summarized: int = 0


class Strategy(Protocol):
    """How a conversation is kept from growing without bound."""

    async def compact(self, convo: Conversation) -> dict:
        """State to store so later turns carry less. Empty when nothing changed."""
        ...

    def view(self, convo: Conversation) -> list[AnyMessage]:
        """The messages the model is shown this turn."""
        ...


@lru_cache(maxsize=4)
def _encoding(model: str) -> tiktoken.Encoding:
    # A model tiktoken has not heard of is assumed to be a recent one. The
    # guess is only close for recent models: o200k and the cl100k it
    # replaced differ by about a third on Norwegian, so an older model named
    # here would have its history undercounted.
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(messages: list[AnyMessage], model: str) -> int:
    """Tokens the messages cost, counted rather than estimated.

    The four per message is the role and delimiter framing the chat format
    adds around each one.
    """
    encoding = _encoding(model)
    return sum(len(encoding.encode(message.text)) + 4 for message in messages)


def start_on_human(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Drop a leading reply so a window never opens mid-exchange.

    A history that begins with an answer reads as though the model spoke
    first, and some providers reject it outright.
    """
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            return messages[index:]
    return []


def last_exchanges(messages: list[AnyMessage], count: int) -> int:
    """Index where the last `count` exchanges begin.

    An exchange starts at a user message, so a cut here never separates a
    question from its answer. Cutting by message count instead splits the
    pair -- compaction runs with the new question already appended, which
    makes the count odd -- and the orphaned reply then belongs to neither
    the summary nor the window.
    """
    starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if len(starts) <= count:
        return 0
    return starts[-count]
