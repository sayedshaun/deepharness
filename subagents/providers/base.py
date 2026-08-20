from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any


def token_usage(usage: Any) -> TokenUsage | None:
    """Normalize a vendor's parsed Usage into TokenUsage, if it sent one."""
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def without_none(payload: Any) -> dict[str, Any]:
    """A request payload as a dict, minus fields that were never set.

    Vendors do not treat an explicit null the same as an absent key - sending
    "tools": null where the API expects a list is an error at several of them -
    so unset optionals are dropped rather than serialized.
    """
    return {
        f.name: value
        for f in fields(payload)
        if (value := getattr(payload, f.name)) is not None
    }


class ReasoningLevel(StrEnum):
    """StrEnum so members serialize as plain strings (JSON payloads, dict
    keys) without extra conversion at the call sites."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def budget(self) -> int:
        """Anthropic and Gemini take a raw thinking-token budget, not an
        effort label; this is the token count each level maps to for them."""
        match self:
            case ReasoningLevel.LOW:
                return 1024
            case ReasoningLevel.MEDIUM:
                return 4096
            case ReasoningLevel.HIGH:
                return 16000


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model.

    id is the vendor's identifier for this specific call (OpenAI's
    tool_calls[].id, Anthropic's tool_use block id) - carried through so the
    result can be linked back to it on the next turn. None for vendors with
    no such concept (Gemini correlates by name/position instead).
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(slots=True)
class TokenUsage:
    """Token counts for one completion, normalized across vendors."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True)
class CompletionResponse:
    """Normalized result of a provider completion, independent of vendor format."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None


@dataclass(slots=True)
class TextDelta:
    """A chunk of the model's prose, as it arrives."""

    text: str


@dataclass(slots=True)
class Completed:
    """The whole turn, once the stream ends: text, tool calls and usage."""

    response: CompletionResponse


StreamEvent = TextDelta | Completed
"""What a streaming call emits.

Text alone is not enough to drive an agent: a turn may ask for tools instead of
answering, and the vendor sends those in the same stream, fragmented. So a
stream yields deltas as they arrive and finishes with the assembled response.
"""


class LLM(ABC):
    """The interface agent/ and graph/ depend on: send messages, get a reply.

    Deliberately narrow and transport-agnostic. A provider does not have to
    speak HTTP - a local model, a fake for tests, or a queue-backed worker
    implements these four methods and works everywhere. Vendors that do speak
    HTTP share their request sequence through RestCompletions (see rest.py)
    rather than through this class.

    __slots__ is empty here rather than absent: a base class without it hands
    every subclass a __dict__, which would make the providers' own __slots__
    declarations save nothing.
    """

    __slots__ = ()

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Send messages and optional tool schemas; return a normalized response."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Synchronous counterpart to agenerate(), for use outside an event loop."""

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a turn as TextDeltas, ending with a Completed.

        The default is the whole turn in one delta, because a backend that cannot
        stream still has to be usable here: callers - Agent included - then need
        one code path instead of two, and get the text either way rather than an
        error or an empty iterator. Providers that really stream override this.
        """
        response = await self.agenerate(messages, tools=tools)
        if response.content:
            yield TextDelta(response.content)
        yield Completed(response)

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Synchronous counterpart to astream_events()."""
        response = self.generate(messages, tools=tools)
        if response.content:
            yield TextDelta(response.content)
        yield Completed(response)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Just the text, for the common "print as it types" case."""
        async for event in self.astream_events(messages, tools=tools):
            if isinstance(event, TextDelta):
                yield event.text

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Synchronous counterpart to astream()."""
        for event in self.stream_events(messages, tools=tools):
            if isinstance(event, TextDelta):
                yield event.text
