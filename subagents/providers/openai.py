from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from subagents.providers.base import CompletionResponse, LLM, ToolCall
from subagents.providers.client import HTTPClient
from subagents.providers.types import OpenAIChatCompletion, OpenAIStreamChunk

_BASE_URL = "https://api.openai.com/v1"


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

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = _build_payload(self._model, messages, tools, self._temperature)
        response = await self._http.post("/chat/completions", json=payload)
        return _from_openai_response(
            OpenAIChatCompletion.model_validate(response.json())
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = _build_payload(self._model, messages, tools, self._temperature)
        response = self._http.post_sync("/chat/completions", json=payload)
        return _from_openai_response(
            OpenAIChatCompletion.model_validate(response.json())
        )

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = _build_payload(self._model, messages, tools, self._temperature)
        payload["stream"] = True
        async with self._http.stream(
            "POST", "/chat/completions", json=payload
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
        payload = _build_payload(self._model, messages, tools, self._temperature)
        payload["stream"] = True
        with self._http.stream_sync(
            "POST", "/chat/completions", json=payload
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
    chunk = OpenAIStreamChunk.model_validate(json.loads(data))
    return chunk.choices[0].delta.content if chunk.choices else None


def _build_payload(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _to_openai_messages(messages),
    }
    if tools:
        payload["tools"] = [_to_openai_tool(tool) for tool in tools]
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


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
    message = completion.choices[0].message
    tool_calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=json.loads(call.function.arguments),
        )
        for call in (message.tool_calls or [])
    ]
    return CompletionResponse(content=message.content or "", tool_calls=tool_calls)
