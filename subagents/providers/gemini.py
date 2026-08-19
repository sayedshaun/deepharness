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
from .rest import RestCompletions
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
        self._rest = RestCompletions(self._http, self)
        self._model = model
        self._reasoning_effort = reasoning_effort

    def payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
    ) -> GeminiPayload:
        return _build_payload(messages, tools, self._reasoning_effort)

    def endpoint(self, *, stream: bool = False) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        return f"/models/{self._model}:{action}"

    def request_args(self, *, stream: bool = False) -> dict[str, Any]:
        """Gemini authenticates by query parameter, and needs alt=sse to stream."""
        params = {"key": self._api_key}
        if stream:
            params["alt"] = "sse"
        return {"params": params}

    def parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_gemini_response(GeminiResponse.from_json(response.json()))

    def extract_delta(self, data: str) -> str | None:
        parts = GeminiResponse.from_json(json.loads(data)).parts
        return "".join(part.text for part in parts if part.text) or None

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        return await self._rest.agenerate(messages, tools)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        return self._rest.generate(messages, tools)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        async for delta in self._rest.astream(messages, tools):
            yield delta

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        yield from self._rest.stream(messages, tools)


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
