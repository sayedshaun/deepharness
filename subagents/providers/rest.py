"""The request sequence every REST vendor shares, as a collaborator.

Each vendor answers a completion the same way - build a payload, POST it,
normalize the reply - and streams it the same way too. That sequence lives here
once and is composed into a provider rather than inherited from LLM, so LLM
stays a narrow interface that a non-HTTP provider can still implement.

A provider supplies the parts that actually differ by implementing Wire.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

import httpx

from .base import CompletionResponse
from .client import HTTPClient

_SSE_DONE = "[DONE]"
"""Sentinel some vendors send to close a stream; others just end the body."""


class Wire(Protocol):
    """What one vendor's REST API does differently from another's."""

    def payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool = False,
    ) -> Any:
        """The request body, as an object exposing to_json()."""

    def endpoint(self, *, stream: bool = False) -> str:
        """Path to POST to. Some vendors have a separate streaming endpoint."""

    def request_args(self, *, stream: bool = False) -> dict[str, Any]:
        """Extra request arguments - query parameters, mostly."""

    def parse_response(self, response: httpx.Response) -> CompletionResponse:
        """One completion response, normalized."""

    def extract_delta(self, data: str) -> str | None:
        """The text in one SSE data payload, or None if it carries none."""


class RestCompletions:
    """Runs a Wire's requests over an HTTPClient."""

    __slots__ = ("_http", "_wire")

    def __init__(self, http: HTTPClient, wire: Wire):
        self._http = http
        self._wire = wire

    async def agenerate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> CompletionResponse:
        payload = self._wire.payload(messages, tools)
        response = await self._http.post(
            self._wire.endpoint(), json=payload.to_json(), **self._wire.request_args()
        )
        return self._wire.parse_response(response)

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> CompletionResponse:
        payload = self._wire.payload(messages, tools)
        response = self._http.post_sync(
            self._wire.endpoint(), json=payload.to_json(), **self._wire.request_args()
        )
        return self._wire.parse_response(response)

    async def astream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[str]:
        payload = self._wire.payload(messages, tools, stream=True)
        async with self._http.stream(
            "POST",
            self._wire.endpoint(stream=True),
            json=payload.to_json(),
            **self._wire.request_args(stream=True),
        ) as response:
            async for line in response.aiter_lines():
                for delta in self._deltas(line):
                    yield delta

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Iterator[str]:
        payload = self._wire.payload(messages, tools, stream=True)
        with self._http.stream_sync(
            "POST",
            self._wire.endpoint(stream=True),
            json=payload.to_json(),
            **self._wire.request_args(stream=True),
        ) as response:
            for line in response.iter_lines():
                yield from self._deltas(line)

    def _deltas(self, line: str) -> Iterator[str]:
        """Zero or one delta for one SSE line, so both stream loops stay identical.

        A generator rather than a str | None so that "this line ends the stream"
        and "this line carries no text" are the same empty result to the caller.
        """
        if not line.startswith("data:"):
            return
        data = line[len("data:") :].strip()
        if not data or data == _SSE_DONE:
            return
        delta = self._wire.extract_delta(data)
        if delta:
            yield delta
