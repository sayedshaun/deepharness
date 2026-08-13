from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


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
    def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the model's text response as it arrives, one content delta at a time."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Synchronous counterpart to astream(), for use outside an event loop."""
