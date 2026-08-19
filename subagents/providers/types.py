"""Vendor response shapes, parsed from JSON into plain dataclasses.

Each type reads only the fields its provider actually uses, and `from_json`
raises ProviderError when a required one is absent. That is the point of these
types: a vendor changing or renaming a field should fail loudly on the first
response rather than quietly yielding an empty completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ProviderError


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    try:
        return data[key]
    except (KeyError, TypeError):
        raise ProviderError(
            f"{where} response is missing '{key}': {_clip(data)}"
        ) from None


def _clip(data: Any, limit: int = 200) -> str:
    """Response bodies can be long, and may carry keys we should not log."""
    text = str(data)
    return text if len(text) <= limit else f"{text[:limit]}..."


@dataclass(slots=True)
class OpenAIToolCall:
    id: str
    name: str
    arguments: str = "{}"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIToolCall:
        function = _require(data, "function", "OpenAI")
        return cls(
            id=_require(data, "id", "OpenAI"),
            name=_require(function, "name", "OpenAI"),
            arguments=function.get("arguments") or "{}",
        )


@dataclass(slots=True)
class OpenAIMessage:
    content: str | None = None
    tool_calls: list[OpenAIToolCall] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIMessage:
        return cls(
            content=data.get("content"),
            tool_calls=[
                OpenAIToolCall.from_json(call) for call in data.get("tool_calls") or []
            ],
        )


@dataclass(slots=True)
class Usage:
    """Token counts, named per vendor at the edges and normalized here."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class OpenAIChatCompletion:
    message: OpenAIMessage
    usage: Usage | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIChatCompletion:
        choices = _require(data, "choices", "OpenAI")
        if not choices:
            raise ProviderError(f"OpenAI response has no choices: {_clip(data)}")
        usage = data.get("usage")
        return cls(
            message=OpenAIMessage.from_json(_require(choices[0], "message", "OpenAI")),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            if usage
            else None,
        )


def openai_stream_delta(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    return choices[0].get("delta", {}).get("content") if choices else None


@dataclass(slots=True)
class AnthropicContentBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnthropicContentBlock:
        return cls(
            type=_require(data, "type", "Anthropic"),
            text=data.get("text"),
            id=data.get("id"),
            name=data.get("name"),
            input=data.get("input") or {},
        )


@dataclass(slots=True)
class AnthropicMessage:
    content: list[AnthropicContentBlock] = field(default_factory=list)
    usage: Usage | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnthropicMessage:
        usage = data.get("usage")
        blocks = _require(data, "content", "Anthropic")
        return cls(
            content=[AnthropicContentBlock.from_json(block) for block in blocks],
            usage=Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            )
            if usage
            else None,
        )


def anthropic_stream_delta(data: dict[str, Any]) -> str | None:
    if data.get("type") != "content_block_delta":
        return None
    delta = data.get("delta") or {}
    return delta.get("text") if delta.get("type") == "text_delta" else None


@dataclass(slots=True)
class GeminiPart:
    text: str | None = None
    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GeminiPart:
        call = data.get("functionCall") or {}
        return cls(
            text=data.get("text"),
            name=call.get("name"),
            args=call.get("args") or {},
        )


@dataclass(slots=True)
class GeminiResponse:
    parts: list[GeminiPart] = field(default_factory=list)
    usage: Usage | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GeminiResponse:
        candidates = data.get("candidates")
        if candidates is None:
            raise ProviderError(f"Gemini response has no candidates: {_clip(data)}")
        content = candidates[0].get("content") if candidates else None
        usage = data.get("usageMetadata")
        return cls(
            parts=[
                GeminiPart.from_json(part) for part in (content or {}).get("parts", [])
            ],
            usage=Usage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
            )
            if usage
            else None,
        )
