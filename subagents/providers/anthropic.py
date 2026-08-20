from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from subagents.providers.base import (
    CompletionResponse,
    ReasoningLevel,
    ToolCall,
    token_usage,
    without_none,
)
from subagents.providers.client import HTTPClient
from subagents.providers.rest import RestCompletions, RestLLM
from subagents.providers.wire import Usage, load_arguments, require, usage_from

_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


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
            system_parts.append(message["content"])
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": message["content"],
                        }
                    ],
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
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
            converted.append({"role": role, "content": message["content"]})

    return ("\n".join(system_parts) if system_parts else None), converted


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def _from_anthropic_response(message: AnthropicMessage) -> CompletionResponse:
    text = "".join(
        block.text for block in message.content if block.type == "text" and block.text
    )
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in message.content
        if block.type == "tool_use" and block.name
    ]
    return CompletionResponse(
        content=text, tool_calls=tool_calls, usage=token_usage(message.usage)
    )


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
            type=require(data, "type", "Anthropic"),
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
        blocks = require(data, "content", "Anthropic")
        return cls(
            content=[AnthropicContentBlock.from_json(block) for block in blocks],
            usage=usage_from(
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
                    arguments=load_arguments(block["arguments"]),
                )
                for block in self._blocks.values()
                if block["name"]
            ],
            usage=token_usage(self._usage),
        )
