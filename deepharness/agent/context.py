"""Keeping a long run inside the model's context window.

A think/act loop grows its own transcript: every turn appends the model's
request and whatever its tools returned, and a tool that reads a file or greps
a tree can return more text than the window holds. Two bounds handle that from
opposite ends - truncate what one tool may contribute, and prune what the
transcript as a whole may carry - and both live here so the loop only has to
consult one object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ConfigurationError
from .state import Message

_CHARS_PER_TOKEN = 4
"""Roughly what a token costs in English prose. A heuristic on purpose: a real
count means a per-vendor tokenizer, which is a dependency this library will not
take, and pruning only needs to know when a transcript is getting close."""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate how many tokens a transcript will cost to send.

    Deliberately an estimate, so treat a budget built on it as a soft bound -
    it is there to keep a long run from walking into a hard provider error, not
    to predict a bill.
    """
    return sum(_message_chars(message) for message in messages) // _CHARS_PER_TOKEN


def _message_chars(message: dict[str, Any]) -> int:
    chars = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or ():
        chars += len(str(call.get("name", ""))) + len(str(call.get("arguments", "")))
    return chars


def truncate(text: str, limit: int | None) -> str:
    """Shorten text to about `limit` characters, eliding the middle.

    The middle rather than the tail: a long tool result usually says what it is
    at the top and how it ended at the bottom, and a plain head-cut throws the
    ending away. The marker counts what went missing so the model can tell it is
    reading an excerpt and ask for the rest a different way.
    """
    if limit is None or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    elided = len(text) - head - tail
    kept_tail = text[-tail:] if tail else ""
    return f"{text[:head]}\n... [{elided} characters elided] ...\n{kept_tail}"


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """How much transcript a run may send, and how much one tool may add.

    `tool_result_chars` bounds each recorded tool result and is on by default,
    because an unbounded result is the usual way a run dies: one oversized
    result poisons every later turn, since it is sent again with each of them.

    `max_tokens` bounds the whole transcript and is off by default. Pruning
    drops history, and the estimate behind it is a heuristic - not something to
    apply to a caller who never asked. Set it to somewhere under the model's
    real window and the loop sends a pruned view of the transcript while
    `AgentState.messages` keeps every message, so nothing is lost to the caller
    that only had to be kept from the provider.

    Pruning never touches the leading system prompt or the last `keep_last`
    messages, so a transcript whose tail alone exceeds the budget is sent over
    it rather than mangled - a bound is not worth breaking a turn's tool-call
    pairing for.

    Frozen, so one policy can be shared between agents without a run
    re-tuning another's limits.
    """

    max_tokens: int | None = None
    tool_result_chars: int | None = 8000
    keep_last: int = 4

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigurationError(
                f"ContextPolicy.max_tokens must be at least 1 when set, "
                f"got {self.max_tokens}"
            )
        if self.tool_result_chars is not None and self.tool_result_chars < 1:
            raise ConfigurationError(
                f"ContextPolicy.tool_result_chars must be at least 1 when set, "
                f"got {self.tool_result_chars}"
            )
        if self.keep_last < 1:
            raise ConfigurationError(
                f"ContextPolicy.keep_last must be at least 1, got {self.keep_last}"
            )

    def prune(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The transcript as it should be sent, oldest turns dropped if needed.

        Returns the list unchanged when it already fits, which is the common
        case - a short run should pay nothing for a policy it never needs.
        Override this to prune differently; the loop asks for a view of the
        transcript and does not care how it was chosen.
        """
        if self.max_tokens is None or estimate_tokens(messages) <= self.max_tokens:
            return messages

        split = 0
        while split < len(messages) and messages[split].get("role") == "system":
            split += 1
        head, body = messages[:split], messages[split:]

        # A tool result cannot outlive the assistant turn that asked for it -
        # providers reject an orphan - so only a non-tool message starts a unit
        # that may be dropped whole.
        cuts = [
            index
            for index, message in enumerate(body)
            if index > 0
            and index <= len(body) - self.keep_last
            and message.get("role") != "tool"
        ]
        pruned = messages
        for cut in cuts:
            pruned = [*head, self._elision(cut), *body[cut:]]
            if estimate_tokens(pruned) <= self.max_tokens:
                break
        return pruned

    @staticmethod
    def _elision(count: int) -> dict[str, Any]:
        """Stand in for the dropped messages, so the gap is visible to the model.

        A user-role note rather than a second system message: vendors that lift
        system messages into a separate field would hoist this one to the front,
        away from the gap it describes.
        """
        return Message.human(
            f"[{count} earlier messages elided to fit the context budget.]"
        ).to_dict()
