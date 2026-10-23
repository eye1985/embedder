"""Folding older exchanges into a running summary."""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from llm.context_management.base import (
    Conversation,
    count_tokens,
    last_exchanges,
    start_on_human,
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You maintain a running summary of a conversation between a "
                "user and an assistant answering from the user's documents.\n"
                "Rewrite the summary so it also covers the new exchanges, "
                "staying under {limit} words.\n"
                "Keep what the user asked for, what they were told, their "
                "stated preferences, and any name or figure a later question "
                "might refer back to. Drop pleasantries and repetition.\n"
                "Write it as plain notes, not as a reply to anyone."
            ),
        ),
        (
            "human",
            "Summary so far:\n{existing}\n\nNew exchanges to fold in:\n{dropped}",
        ),
    ]
)


def _transcript(messages: list[AnyMessage]) -> str:
    return "\n".join(f"{message.type}: {message.text}" for message in messages)


@dataclass(frozen=True)
class Summarize:
    """Recent exchanges verbatim, everything older as a summary.

    `budget` is an allowance for the history, deliberately not the model's
    context limit: the limit is where the request fails, and waiting for it
    means paying for a prompt tens of times larger on every turn in the
    meantime, with recall that falls off as it fills. Compacting at a budget
    holds a turn at roughly the same size forever.

    `llm` is passed in rather than built here so the summary shares the
    process's client, and so this module needs to know nothing about which
    model the chat is using.

    `budget_model` is the model the budget is counted for, which is the one
    that answers -- not whichever model `llm` happens to be. The budget
    bounds the answer's prompt, and two models can disagree by a third on
    the same text, so counting with the wrong one compacts at the wrong size.
    """

    llm: Callable[[], BaseChatModel]
    budget_model: str
    budget: int = 3_000
    keep_turns: int = 4
    summary_words: int = 200

    async def compact(self, convo: Conversation) -> dict:
        unsummarized = convo.messages[convo.summarized :]
        if count_tokens(unsummarized, self.budget_model) <= self.budget:
            return {}

        # Everything but the most recent exchanges, and only what the
        # summary has not already absorbed.
        cutoff = max(convo.summarized, last_exchanges(convo.messages, self.keep_turns))
        dropped = convo.messages[convo.summarized : cutoff]
        if not dropped:
            return {}

        summary = await self.llm().ainvoke(
            SUMMARY_PROMPT.invoke(
                {
                    "existing": convo.summary or "(nothing summarized yet)",
                    "dropped": _transcript(dropped),
                    "limit": self.summary_words,
                }
            )
        )
        return {
            "summary": summary.text,
            "summarized": convo.summarized + len(dropped),
        }

    def view(self, convo: Conversation) -> list[AnyMessage]:
        recent = start_on_human(convo.messages[convo.summarized :])
        if not convo.summary:
            return recent
        return [
            SystemMessage(f"Summary of earlier conversation:\n{convo.summary}"),
            *recent,
        ]
