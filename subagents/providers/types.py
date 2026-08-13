from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class OpenAIToolCall(BaseModel):
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(BaseModel):
    message: OpenAIMessage


class OpenAIChatCompletion(BaseModel):
    choices: list[OpenAIChoice]


class OpenAIStreamDelta(BaseModel):
    content: str | None = None


class OpenAIStreamChoice(BaseModel):
    delta: OpenAIStreamDelta


class OpenAIStreamChunk(BaseModel):
    choices: list[OpenAIStreamChoice]


class GeminiFunctionCall(BaseModel):
    name: str
    args: dict[str, Any] = {}


class GeminiPart(BaseModel):
    text: str | None = None
    functionCall: GeminiFunctionCall | None = None


class GeminiContent(BaseModel):
    parts: list[GeminiPart] = []


class GeminiCandidate(BaseModel):
    content: GeminiContent | None = None


class GeminiGenerateContentResponse(BaseModel):
    candidates: list[GeminiCandidate] = []
