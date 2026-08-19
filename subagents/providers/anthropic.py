from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from subagents.providers.base import (
    LLM,
    CompletionResponse,
    ReasoningLevel,
    ToolCall,
    token_usage,
    without_none,
)
from subagents.providers.client import HTTPClient
from subagents.providers.types import AnthropicMessage, anthropic_stream_delta

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


class Anthropic(LLM):
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
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> AnthropicPayload:
        return _build_payload(
            self._model, self._max_tokens, messages, tools, self._reasoning_effort
        )

    def _parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_anthropic_response(AnthropicMessage.from_json(response.json()))

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = await self._http.post("/messages", json=payload.to_json())
        return self._parse_response(response)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = self._http.post_sync("/messages", json=payload.to_json())
        return self._parse_response(response)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, tools)
        payload.stream = True
        async with self._http.stream(
            "POST", "/messages", json=payload.to_json()
        ) as response:
            async for line in response.aiter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                delta = _extract_delta(data)
                if delta:
                    yield delta

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        payload = self._payload(messages, tools)
        payload.stream = True
        with self._http.stream_sync(
            "POST", "/messages", json=payload.to_json()
        ) as response:
            for line in response.iter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                delta = _extract_delta(data)
                if delta:
                    yield delta


def _parse_sse_line(line: str) -> str | None:
    if not line or not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def _extract_delta(data: str) -> str | None:
    return anthropic_stream_delta(json.loads(data))


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
