from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from deepharness.http import HTTPClient
from deepharness.providers.base import (
    CompletionResponse,
    FinishReason,
    ReasoningLevel,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    token_usage,
    without_none,
)
from deepharness.providers.content import (
    Document,
    Image,
    Text,
    Thinking,
    merge,
    parse,
    text_of,
)
from deepharness.providers.rest import RestCompletions, RestLLM
from deepharness.providers.wire import (
    Usage,
    finish_reason_from,
    load_arguments,
    require,
    usage_from,
)

_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096
_ENV_KEY = "ANTHROPIC_API_KEY"
_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "stop",
    "max_tokens": "length",
    "refusal": "filtered",
}
"""Anthropic's stop_reason values, normalized. "max_tokens" matters more here
than at other vendors: max_tokens is required on every request, so a long answer
runs into the default cap rather than only an unusually long one."""


@dataclass(slots=True)
class AnthropicPayload:
    """Request body for POST /messages, with unset optional fields
    (system, tools, thinking, stream) dropped rather than sent as null."""

    model: str
    max_tokens: int
    messages: list[dict[str, Any]]
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    thinking: dict[str, Any] | None = None
    stream: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return without_none(self)


class Anthropic(RestLLM):
    """Provider backed by Anthropic's Messages REST API.

    Anthropic's wire format differs from OpenAI/Gemini in one structural way
    this provider bridges but doesn't fully hide: the system prompt is a
    top-level `system` field (not a message), not a role in `messages`.
    Tool calls round-trip properly: an assistant turn with tool_calls becomes
    a `tool_use` content block (carrying ToolCall.id), and a tool-role
    message becomes a `tool_result` block referencing that same id via
    `tool_call_id` - required, since Anthropic enforces strict user/assistant
    alternation and will reject a request where that link is missing.
    """

    __slots__ = ("_http", "_max_tokens", "_model", "_reasoning_effort", "_rest")

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        *,
        reasoning_effort: ReasoningLevel | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get(_ENV_KEY)
        headers = {"x-api-key": api_key or "", "anthropic-version": _ANTHROPIC_VERSION}
        self._http = HTTPClient(
            _BASE_URL, headers=headers, client=client, sync_client=sync_client
        )
        self._rest = RestCompletions(self._http, self)
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    def payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
    ) -> AnthropicPayload:
        payload = _build_payload(
            self._model, self._max_tokens, messages, tools, self._reasoning_effort
        )
        payload.stream = stream or None
        return payload

    def endpoint(self, *, stream: bool = False) -> str:
        return "/messages"

    def parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_anthropic_response(AnthropicMessage.from_json(response.json()))

    def accumulator(self) -> AnthropicStream:
        return AnthropicStream()


def _build_payload(
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: ReasoningLevel | None = None,
) -> AnthropicPayload:
    system, converted = _to_anthropic_messages(messages)
    thinking: dict[str, Any] | None = None
    if reasoning_effort:
        budget = ReasoningLevel(reasoning_effort).budget
        thinking = {"type": "enabled", "budget_tokens": budget}
        # Anthropic requires max_tokens to exceed the thinking budget.
        max_tokens = max(max_tokens, budget + 1024)
    return AnthropicPayload(
        model=model,
        max_tokens=max_tokens,
        messages=converted,
        system=system,
        tools=[_to_anthropic_tool(tool) for tool in tools] if tools else None,
        thinking=thinking,
    )


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(text_of(message.get("content")))
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": text_of(message.get("content")),
                        }
                    ],
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            blocks = _to_anthropic_blocks(message.get("content"))
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                }
                for call in message["tool_calls"]
            )
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append(
                {"role": role, "content": _to_anthropic_content(message.get("content"))}
            )

    return ("\n".join(system_parts) if system_parts else None), converted


def _to_anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    """Blocks, or a plain string when text is all the content is.

    Anthropic takes either, and the string form keeps an ordinary conversation's
    payload identical to what it was before blocks existed.
    """
    blocks = parse(content)
    if all(isinstance(block, Text) for block in blocks):
        return text_of(blocks)
    return _to_anthropic_blocks(content)


def _to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    """One message's content as Anthropic content blocks.

    Thinking is replayed with its signature, unmodified: Anthropic requires the
    thinking block back on the request that follows a tool call, and a run that
    drops it loses the model's chain exactly where a long task depends on it. A
    thinking block with no signature is left out rather than sent unsigned,
    which the API rejects.
    """
    blocks: list[dict[str, Any]] = []
    for block in parse(content):
        match block:
            case Text():
                blocks.append({"type": "text", "text": block.text})
            case Thinking() if block.signature:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.text,
                        "signature": block.signature,
                    }
                )
            case Thinking():
                continue
            case Image() if block.url:
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": block.url}}
                )
            case Image():
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    }
                )
            case Document():
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    }
                )
    return blocks


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def _from_anthropic_response(message: AnthropicMessage) -> CompletionResponse:
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in message.content
        if block.type == "tool_use" and block.name
    ]
    return CompletionResponse(
        blocks=[
            block for raw in message.content if (block := _to_block(raw)) is not None
        ],
        tool_calls=tool_calls,
        usage=token_usage(message.usage),
        finish_reason=message.finish_reason or "stop",
    )


def _to_block(raw: AnthropicContentBlock) -> Text | Thinking | None:
    """One response block as normalized content, or None if it is a tool call.

    A tool call is already carried as a ToolCall, and repeating it as content
    would have the loop record the same request twice.
    """
    if raw.type == "text" and raw.text:
        return Text(raw.text)
    if raw.type == "thinking" and raw.thinking:
        return Thinking(raw.thinking, signature=raw.signature)
    return None


@dataclass(slots=True)
class AnthropicContentBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    thinking: str | None = None
    signature: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnthropicContentBlock:
        return cls(
            type=require(data, "type", "Anthropic"),
            text=data.get("text"),
            id=data.get("id"),
            name=data.get("name"),
            input=data.get("input") or {},
            thinking=data.get("thinking"),
            signature=data.get("signature"),
        )


@dataclass(slots=True)
class AnthropicMessage:
    content: list[AnthropicContentBlock] = field(default_factory=list)
    usage: Usage | None = None
    finish_reason: FinishReason | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnthropicMessage:
        blocks = require(data, "content", "Anthropic")
        return cls(
            content=[AnthropicContentBlock.from_json(block) for block in blocks],
            usage=usage_from(
                data.get("usage"), prompt="input_tokens", completion="output_tokens"
            ),
            finish_reason=finish_reason_from(data.get("stop_reason"), _FINISH_REASONS),
        )


class AnthropicStream:
    """Folds Anthropic's block events into content plus tool calls.

    Anthropic streams content as numbered blocks: content_block_start announces a
    block's type (text, thinking or tool_use), the deltas that follow belong to
    whichever block is open, and tool arguments arrive as partial_json
    fragments. Content is kept per index rather than in one buffer because a
    thinking block's signature arrives after its text, and it has to land on
    that block - unsigned thinking cannot be replayed.
    """

    __slots__ = ("_content", "_finish_reason", "_tools", "_usage")

    def __init__(self) -> None:
        self._content: dict[int, dict[str, Any]] = {}
        self._tools: dict[int, dict[str, Any]] = {}
        self._usage: Usage | None = None
        self._finish_reason: FinishReason = "stop"

    def feed(self, data: dict[str, Any]) -> TextDelta | ThinkingDelta | None:
        event = data.get("type")
        index = data.get("index", 0)

        if event == "content_block_start":
            block = data.get("content_block") or {}
            kind = block.get("type")
            if kind == "tool_use":
                self._tools[index] = {
                    "id": block.get("id"),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
            elif kind in ("text", "thinking"):
                self._content[index] = {"type": kind, "text": "", "signature": None}
            return None

        if event == "content_block_delta":
            return self._delta(index, data.get("delta") or {})

        if event == "message_start":
            self._record_usage((data.get("message") or {}).get("usage"))
        elif event == "message_delta":
            self._record_usage(data.get("usage"))
            raw_finish = (data.get("delta") or {}).get("stop_reason")
            if (finish := finish_reason_from(raw_finish, _FINISH_REASONS)) is not None:
                self._finish_reason = finish
        return None

    def _delta(
        self, index: int, delta: dict[str, Any]
    ) -> TextDelta | ThinkingDelta | None:
        """One delta into its open block, and the event it carried.

        The block is created on demand: a vendor that ever sends a delta before
        its content_block_start would otherwise drop that text on the floor.
        """
        match delta.get("type"):
            case "text_delta" if delta.get("text"):
                self._append(index, "text", delta["text"])
                return TextDelta(delta["text"])
            case "thinking_delta" if delta.get("thinking"):
                self._append(index, "thinking", delta["thinking"])
                return ThinkingDelta(delta["thinking"])
            case "signature_delta" if delta.get("signature"):
                block = self._content.setdefault(
                    index, {"type": "thinking", "text": "", "signature": None}
                )
                block["signature"] = (block["signature"] or "") + delta["signature"]
            case "input_json_delta" if index in self._tools:
                self._tools[index]["arguments"] += delta.get("partial_json") or ""
        return None

    def _append(self, index: int, kind: str, text: str) -> None:
        block = self._content.setdefault(
            index, {"type": kind, "text": "", "signature": None}
        )
        block["text"] += text

    def _record_usage(self, usage: dict[str, Any] | None) -> None:
        """Fold in one event's counts, keeping the ones it left out.

        Anthropic splits them across the stream: input_tokens comes with
        message_start and the final output_tokens with message_delta. Replacing
        rather than merging would drop the prompt half - the expensive one.
        """
        if not usage:
            return
        current = self._usage or Usage()
        prompt = usage.get("input_tokens", current.prompt_tokens)
        completion = usage.get("output_tokens", current.completion_tokens)
        self._usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    def response(self) -> CompletionResponse:
        return CompletionResponse(
            blocks=merge(
                Thinking(block["text"], signature=block["signature"])
                if block["type"] == "thinking"
                else Text(block["text"])
                for _, block in sorted(self._content.items())
                if block["text"]
            ),
            tool_calls=[
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=load_arguments(block["arguments"]),
                )
                for block in self._tools.values()
                if block["name"]
            ],
            usage=token_usage(self._usage),
            finish_reason=self._finish_reason,
        )
