"""The request sequence every REST vendor shares, as a collaborator.

Each vendor answers a completion the same way - build a payload, POST it,
normalize the reply - and streams it the same way too. That sequence lives here
once and is composed into a provider rather than inherited from LLM, so LLM
stays a narrow interface that a non-HTTP provider can still implement.

A provider supplies the parts that actually differ by implementing Wire, and
inherits the four LLM methods from RestLLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

import httpx

from .base import LLM, CompletionResponse
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


class RestLLM(LLM):
    """An LLM served by one Wire over HTTP.

    The four LLM methods are the same forwarding call for every REST vendor, so
    they live here once. A provider subclasses this, builds a RestCompletions
    into self._rest, and implements only its Wire methods.
    """

    __slots__ = ()

    _rest: RestCompletions

    def request_args(self, *, stream: bool = False) -> dict[str, Any]:
        """No extra arguments; most vendors authenticate with a header set once."""
        return {}

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
