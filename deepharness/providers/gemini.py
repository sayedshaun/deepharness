from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import ProviderError
from .base import (
    CompletionResponse,
    ReasoningLevel,
    ToolCall,
    token_usage,
    without_none,
)
from .client import HTTPClient
from .rest import RestCompletions, RestLLM
from .wire import Usage, clip, usage_from

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_ROLE_MAP = {"assistant": "model", "system": "user", "user": "user"}
_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


@dataclass(slots=True)
class GeminiPayload:
    """Request body for :generateContent / :streamGenerateContent, with unset
    optional fields dropped rather than sent as null."""

    contents: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    generationConfig: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return without_none(self)


class Gemini(RestLLM):
    """Provider backed by Google's Gemini REST API."""

    __slots__ = ("_http", "_model", "_reasoning_effort", "_rest")

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        reasoning_effort: ReasoningLevel | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        if api_key is None:
            api_key = next(
                (key for name in _ENV_KEYS if (key := os.environ.get(name))), None
            )
        # In a header rather than the documented ?key= query parameter: httpx
        # puts the full URL in its error messages, so a query-string credential
        # ends up in every ProviderError a failed request raises.
        self._http = HTTPClient(
            _BASE_URL,
            headers={"x-goog-api-key": api_key or ""},
            client=client,
            sync_client=sync_client,
        )
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
        """Gemini needs alt=sse to send a stream as server-sent events."""
        return {"params": {"alt": "sse"}} if stream else {}

    def parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_gemini_response(GeminiResponse.from_json(response.json()))

    def accumulator(self) -> GeminiStream:
        return GeminiStream()


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
            raise ProviderError(f"Gemini response has no candidates: {clip(data)}")
        content = candidates[0].get("content") if candidates else None
        return cls(
            parts=[
                GeminiPart.from_json(part) for part in (content or {}).get("parts", [])
            ],
            usage=usage_from(
                data.get("usageMetadata"),
                prompt="promptTokenCount",
                completion="candidatesTokenCount",
                total="totalTokenCount",
            ),
        )
