from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field, fields
from enum import Enum
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


class ReasoningLevel(str, Enum):
    """Str subclass so members serialize as plain strings (JSON payloads,
    dict keys) without extra conversion at the call sites."""

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


class LLM(ABC):
    """The interface agent/ and graph/ depend on: send messages, get a reply.

    Deliberately narrow and transport-agnostic. A provider does not have to
    speak HTTP - a local model, a fake for tests, or a queue-backed worker
    implements these four methods and works everywhere. Vendors that do speak
    HTTP share their request sequence through RestCompletions (see rest.py)
    rather than through this class.
    """

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Send messages (and optional tool schemas) and return a normalized response."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Synchronous counterpart to agenerate(), for use outside an event loop."""

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the model's text response as it arrives, one content delta at a time.

        Optional, unlike agenerate/generate: not every backend can stream, and a
        provider that cannot should not be forced to write a stub. Callers that
        need streaming get a clear error instead of an empty iterator.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        yield  # pragma: no cover - marks this as an async generator for type checkers

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Synchronous counterpart to astream(), for use outside an event loop."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        yield  # pragma: no cover - marks this as a generator for type checkers
