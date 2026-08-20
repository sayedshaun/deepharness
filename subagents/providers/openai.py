from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from subagents.errors import ProviderError
from subagents.providers.base import (
    CompletionResponse,
    ReasoningLevel,
    ToolCall,
    token_usage,
    without_none,
)
from subagents.providers.client import HTTPClient
from subagents.providers.rest import RestCompletions, RestLLM
from subagents.providers.wire import Usage, clip, load_arguments, require, usage_from

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


class OpenAI(RestLLM):
    """Provider backed by OpenAI's Chat Completions REST API.

    Also the base for any OpenAI-compatible gateway (see providers/gateways.py):
    a subclass overriding default_base_url/env_key gets that endpoint and
    reads its API key from that environment variable with no other code.
    """

    provider: str = "openai"
    default_base_url: str = _BASE_URL
    env_key: str = "OPENAI_API_KEY"

    __slots__ = ("_http", "_model", "_reasoning_effort", "_rest", "_temperature")

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
        self._rest = RestCompletions(self._http, self)
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
    ) -> OpenAIPayload:
        payload = _build_payload(
            self._model, messages, tools, self._temperature, self._reasoning_effort
        )
        payload.stream = stream or None
        return payload

    def endpoint(self, *, stream: bool = False) -> str:
        return "/chat/completions"

    def parse_response(self, response: httpx.Response) -> CompletionResponse:
        return _from_openai_response(OpenAIChatCompletion.from_json(response.json()))

    def accumulator(self) -> OpenAIStream:
        return OpenAIStream()


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


@dataclass(slots=True)
class OpenAIToolCall:
    id: str
    name: str
    arguments: str = "{}"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIToolCall:
        function = require(data, "function", "OpenAI")
        return cls(
            id=require(data, "id", "OpenAI"),
            name=require(function, "name", "OpenAI"),
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
class OpenAIChatCompletion:
    message: OpenAIMessage
    usage: Usage | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OpenAIChatCompletion:
        choices = require(data, "choices", "OpenAI")
        if not choices:
            raise ProviderError(f"OpenAI response has no choices: {clip(data)}")
        return cls(
            message=OpenAIMessage.from_json(require(choices[0], "message", "OpenAI")),
            usage=usage_from(
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
                    arguments=load_arguments(call["arguments"]),
                )
                for call in self._calls.values()
                if call["name"]
            ],
            usage=token_usage(self._usage),
        )
