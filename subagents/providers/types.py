"""Vendor response shapes, parsed from JSON into plain dataclasses.

Each type reads only the fields its provider actually uses, and `from_json`
raises ProviderError when a required one is absent. That is the point of these
types: a vendor changing or renaming a field should fail loudly on the first
response rather than quietly yielding an empty completion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..errors import ProviderError
from .base import CompletionResponse, ToolCall, token_usage


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


def _usage(
    usage: Any, *, prompt: str, completion: str, total: str | None = None
) -> Usage | None:
    """One vendor's token counts under its own key names, or None if it sent none.

    total is optional because Anthropic reports only the two halves; summing
    them here keeps that quirk out of the response types.
    """
    if not usage:
        return None
    prompt_tokens = usage.get(prompt, 0)
    completion_tokens = usage.get(completion, 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.get(total, 0)
        if total
        else prompt_tokens + completion_tokens,
    )


@dataclass(slots=True)
class OpenAIChatCompletion:
    message: OpenAIMessage
    usage: Usage | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIChatCompletion:
        choices = _require(data, "choices", "OpenAI")
        if not choices:
            raise ProviderError(f"OpenAI response has no choices: {_clip(data)}")
        return cls(
            message=OpenAIMessage.from_json(_require(choices[0], "message", "OpenAI")),
            usage=_usage(
                data.get("usage"),
                prompt="prompt_tokens",
                completion="completion_tokens",
                total="total_tokens",
            ),
        )


class OpenAIStream:
    """Folds OpenAI's chat-completion chunks into text plus tool calls.

    Tool calls arrive spread over many chunks: the first carries an index, id and
    function name, and later ones append fragments of the argument JSON. They are
    keyed by index because that is the only field present on every fragment.
    """

    __slots__ = ("_calls", "_text", "_usage")

    def __init__(self) -> None:
        self._text: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}
        self._usage: Usage | None = None

    def feed(self, data: dict[str, Any]) -> str | None:
        if usage := data.get("usage"):
            self._usage = Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        choices = data.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        for fragment in delta.get("tool_calls") or []:
            call = self._calls.setdefault(
                fragment.get("index", 0), {"id": "", "name": "", "arguments": ""}
            )
            call["id"] = fragment.get("id") or call["id"]
            function = fragment.get("function") or {}
            call["name"] = function.get("name") or call["name"]
            call["arguments"] += function.get("arguments") or ""
        text = delta.get("content")
        if text:
            self._text.append(text)
        return text

    def response(self) -> CompletionResponse:
        return CompletionResponse(
            content="".join(self._text),
            tool_calls=[
                ToolCall(
                    id=call["id"] or None,
                    name=call["name"],
                    arguments=_load_arguments(call["arguments"]),
                )
                for call in self._calls.values()
                if call["name"]
            ],
            usage=token_usage(self._usage),
        )


def _load_arguments(raw: str) -> dict[str, Any]:
    """Argument JSON assembled from fragments, empty if the model sent none.

    A truncated stream can leave this unparseable; an empty dict lets the tool
    report the real problem (a missing argument) rather than the stream dying.
    """
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


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
        blocks = _require(data, "content", "Anthropic")
        return cls(
            content=[AnthropicContentBlock.from_json(block) for block in blocks],
            usage=_usage(
                data.get("usage"), prompt="input_tokens", completion="output_tokens"
            ),
        )


class AnthropicStream:
    """Folds Anthropic's block events into text plus tool calls.

    Anthropic streams content as numbered blocks: content_block_start announces a
    block's type (text or tool_use), the deltas that follow belong to whichever
    block is open, and tool arguments arrive as partial_json fragments.
    """

    __slots__ = ("_blocks", "_text", "_usage")

    def __init__(self) -> None:
        self._text: list[str] = []
        self._blocks: dict[int, dict[str, Any]] = {}
        self._usage: Usage | None = None

    def feed(self, data: dict[str, Any]) -> str | None:
        event = data.get("type")
        index = data.get("index", 0)

        if event == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._blocks[index] = {
                    "id": block.get("id"),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
            return None

        if event == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text")
                if text:
                    self._text.append(text)
                return text
            if delta.get("type") == "input_json_delta" and index in self._blocks:
                self._blocks[index]["arguments"] += delta.get("partial_json") or ""
            return None

        if event == "message_delta" and (usage := data.get("usage")):
            self._usage = Usage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            )
        elif event == "message_start":
            message = data.get("message") or {}
            if usage := message.get("usage"):
                self._usage = Usage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                )
        return None

    def response(self) -> CompletionResponse:
        return CompletionResponse(
            content="".join(self._text),
            tool_calls=[
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=_load_arguments(block["arguments"]),
                )
                for block in self._blocks.values()
                if block["name"]
            ],
            usage=token_usage(self._usage),
        )


class GeminiStream:
    """Folds Gemini's stream into text plus tool calls.

    Simpler than the others: every chunk is a complete GenerateContentResponse,
    so a functionCall arrives whole rather than in fragments - nothing to
    reassemble, only to collect.
    """

    __slots__ = ("_calls", "_text", "_usage")

    def __init__(self) -> None:
        self._text: list[str] = []
        self._calls: list[ToolCall] = []
        self._usage: Usage | None = None

    def feed(self, data: dict[str, Any]) -> str | None:
        chunk = GeminiResponse.from_json(data)
        if chunk.usage is not None:
            self._usage = chunk.usage
        self._calls.extend(
            ToolCall(name=part.name, arguments=dict(part.args))
            for part in chunk.parts
            if part.name
        )
        text = "".join(part.text for part in chunk.parts if part.text)
        if text:
            self._text.append(text)
        return text or None

    def response(self) -> CompletionResponse:
        return CompletionResponse(
            content="".join(self._text),
            tool_calls=list(self._calls),
            usage=token_usage(self._usage),
        )


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
        return cls(
            parts=[
                GeminiPart.from_json(part) for part in (content or {}).get("parts", [])
            ],
            usage=_usage(
                data.get("usageMetadata"),
                prompt="promptTokenCount",
                completion="candidatesTokenCount",
                total="totalTokenCount",
            ),
        )
