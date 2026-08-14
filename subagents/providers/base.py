from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any


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
class CompletionResponse:
    """Normalized result of a provider completion, independent of vendor format."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(ABC):
    """Base interface for LLM providers.

    Keeps the core framework decoupled from any specific vendor. Concrete
    providers live under myagents.providers as optional integrations —
    import only the one you need, so unused vendor SDKs are never required.
    """

    @abstractmethod
    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Send messages (and optional tool schemas) to the model and return a normalized response."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        """Synchronous counterpart to agenerate(), for use outside an event loop."""

    @abstractmethod
    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the model's text response as it arrives, one content delta at a time."""
        raise NotImplementedError
        yield  # pragma: no cover - marks this as an async generator for type checkers

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Synchronous counterpart to astream(), for use outside an event loop."""
        raise NotImplementedError
        yield  # pragma: no cover - marks this as a generator for type checkers
