from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from subagents.providers.base import CompletionResponse, LLM, ToolCall
from subagents.providers.client import HTTPClient
from subagents.providers.types import OpenAIChatCompletion, OpenAIStreamChunk

_BASE_URL = "https://api.openai.com/v1"


class OpenAI(LLM):
    """Provider backed by OpenAI's Chat Completions REST API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"}
        self._http = HTTPClient(_BASE_URL, headers=headers, client=client, sync_client=sync_client)
        self._model = model

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = _build_payload(self._model, messages, tools)
        response = await self._http.post("/chat/completions", json=payload)
        return _from_openai_response(OpenAIChatCompletion.model_validate(response.json()))

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = _build_payload(self._model, messages, tools)
        response = self._http.post_sync("/chat/completions", json=payload)
        return _from_openai_response(OpenAIChatCompletion.model_validate(response.json()))

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = _build_payload(self._model, messages, tools)
        payload["stream"] = True
        async with self._http.stream("POST", "/chat/completions", json=payload) as response:
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
        payload = _build_payload(self._model, messages, tools)
        payload["stream"] = True
        with self._http.stream_sync("POST", "/chat/completions", json=payload) as response:
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
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = [_to_openai_tool(tool) for tool in tools]
    return payload


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
        ToolCall(name=call.function.name, arguments=json.loads(call.function.arguments))
        for call in (message.tool_calls or [])
    ]
    return CompletionResponse(content=message.content or "", tool_calls=tool_calls)
