from __future__ import annotations

import json
import os
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
from subagents.providers.types import OpenAIChatCompletion, openai_stream_delta

_BASE_URL = "https://api.openai.com/v1"


@dataclass(slots=True)
class OpenAIPayload:
    """Request body for POST /chat/completions, with unset optional fields
    dropped rather than sent as null."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    reasoning_effort: ReasoningLevel | None = None
    stream: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return without_none(self)


class OpenAI(LLM):
    """Provider backed by OpenAI's Chat Completions REST API.

    Also the base for any OpenAI-compatible gateway (see providers/gateways.py):
    a subclass overriding default_base_url/env_key gets that endpoint and
    reads its API key from that environment variable with no other code.
    """

    provider: str = "openai"
    default_base_url: str = _BASE_URL
    env_key: str = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningLevel | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        if api_key is None and self.env_key:
            api_key = os.environ.get(self.env_key)

        headers = {"Authorization": f"Bearer {api_key or ''}"}
        resolved_base_url = base_url or self.default_base_url
        self._http = HTTPClient(
            resolved_base_url, headers=headers, client=client, sync_client=sync_client
        )
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> OpenAIPayload:
        return _build_payload(
            self._model, messages, tools, self._temperature, self._reasoning_effort
        )

    def _parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_openai_response(OpenAIChatCompletion.from_json(response.json()))

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = await self._http.post("/chat/completions", json=payload.to_json())
        return self._parse_response(response)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = self._http.post_sync("/chat/completions", json=payload.to_json())
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
            "POST", "/chat/completions", json=payload.to_json()
        ) as response:
            async for line in response.aiter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
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
            "POST", "/chat/completions", json=payload.to_json()
        ) as response:
            for line in response.iter_lines():
                data = _parse_sse_line(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                delta = _extract_delta(data)
                if delta:
                    yield delta


def _parse_sse_line(line: str) -> str | None:
    if not line or not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def _extract_delta(data: str) -> str | None:
    return openai_stream_delta(json.loads(data))


def _build_payload(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None = None,
    reasoning_effort: ReasoningLevel | None = None,
) -> OpenAIPayload:
    return OpenAIPayload(
        model=model,
        messages=_to_openai_messages(messages),
        tools=[_to_openai_tool(tool) for tool in tools] if tools else None,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            converted.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        elif message["role"] == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", ""),
                    "content": message["content"],
                }
            )
        else:
            converted.append(dict(message))

    return converted


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _from_openai_response(completion: OpenAIChatCompletion) -> CompletionResponse:
    tool_calls = [
        ToolCall(id=call.id, name=call.name, arguments=json.loads(call.arguments))
        for call in completion.message.tool_calls
    ]
    return CompletionResponse(
        content=completion.message.content or "",
        tool_calls=tool_calls,
        usage=token_usage(completion.usage),
    )
