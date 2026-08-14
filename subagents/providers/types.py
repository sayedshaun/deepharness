from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class OpenAIToolCall(BaseModel):
    id: str
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(BaseModel):
    message: OpenAIMessage


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletion(BaseModel):
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None


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


class GeminiUsageMetadata(BaseModel):
    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0


class GeminiGenerateContentResponse(BaseModel):
    candidates: list[GeminiCandidate] = []
    usageMetadata: GeminiUsageMetadata | None = None


class AnthropicContentBlock(BaseModel):
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = {}


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessage(BaseModel):
    content: list[AnthropicContentBlock] = []
    usage: AnthropicUsage | None = None


class AnthropicStreamDelta(BaseModel):
    type: str
    text: str | None = None


class AnthropicStreamEvent(BaseModel):
    type: str
    delta: AnthropicStreamDelta | None = None
