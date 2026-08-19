from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .base import (
    LLM,
    CompletionResponse,
    ReasoningLevel,
    ToolCall,
    token_usage,
    without_none,
)
from .client import HTTPClient
from .types import GeminiResponse

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_ROLE_MAP = {"assistant": "model", "system": "user", "user": "user"}


@dataclass(slots=True)
class GeminiPayload:
    """Request body for :generateContent / :streamGenerateContent, with unset
    optional fields dropped rather than sent as null."""

    contents: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    generationConfig: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return without_none(self)


class Gemini(LLM):
    """Provider backed by Google's Gemini REST API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        reasoning_effort: ReasoningLevel | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        self._http = HTTPClient(_BASE_URL, client=client, sync_client=sync_client)
        self._api_key = api_key
        self._model = model
        self._reasoning_effort = reasoning_effort

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> GeminiPayload:
        return _build_payload(messages, tools, self._reasoning_effort)

    def _parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_gemini_response(GeminiResponse.from_json(response.json()))

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = await self._http.post(
            f"/models/{self._model}:generateContent",
            params={"key": self._api_key},
            json=payload.to_json(),
        )
        return self._parse_response(response)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        payload = self._payload(messages, tools)
        response = self._http.post_sync(
            f"/models/{self._model}:generateContent",
            params={"key": self._api_key},
            json=payload.to_json(),
        )
        return self._parse_response(response)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, tools)
        async with self._http.stream(
            "POST",
            f"/models/{self._model}:streamGenerateContent",
            params={"key": self._api_key, "alt": "sse"},
            json=payload.to_json(),
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
        with self._http.stream_sync(
            "POST",
            f"/models/{self._model}:streamGenerateContent",
            params={"key": self._api_key, "alt": "sse"},
            json=payload.to_json(),
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
    parts = GeminiResponse.from_json(json.loads(data)).parts
    return "".join(part.text for part in parts if part.text) or None


def _build_payload(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: ReasoningLevel | None = None,
) -> GeminiPayload:
    generation_config: dict[str, Any] | None = None
    if reasoning_effort:
        budget = ReasoningLevel(reasoning_effort).budget
        generation_config = {"thinkingConfig": {"thinkingBudget": budget}}
    return GeminiPayload(
        contents=[_to_gemini_content(m) for m in messages],
        tools=[_to_gemini_tool(t) for t in tools] if tools else None,
        generationConfig=generation_config,
    )


def _to_gemini_content(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]

    if role == "tool":
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": message.get("name", ""),
                        "response": {"result": message["content"]},
                    }
                }
            ],
        }

    if role == "assistant" and message.get("tool_calls"):
        parts: list[dict[str, Any]] = []
        if message.get("content"):
            parts.append({"text": message["content"]})
        parts.extend(
            {"functionCall": {"name": call["name"], "args": call["arguments"]}}
            for call in message["tool_calls"]
        )
        return {"role": "model", "parts": parts}

    return {
        "role": _ROLE_MAP.get(role, "user"),
        "parts": [{"text": message["content"]}],
    }


def _to_gemini_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "functionDeclarations": [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        ]
    }


def _from_gemini_response(response: GeminiResponse) -> CompletionResponse:
    text = "".join(part.text for part in response.parts if part.text)
    tool_calls = [
        ToolCall(name=part.name, arguments=dict(part.args))
        for part in response.parts
        if part.name
    ]
    return CompletionResponse(
        content=text, tool_calls=tool_calls, usage=token_usage(response.usage)
    )
