from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from typing import Any

import httpx

from .errors import ConfigurationError, ProviderError

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3
_MAX_ATTEMPTS = _MAX_RETRIES + 1
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
"""httpx defaults to five seconds for every phase, which a completion routinely
exceeds - a long generation is a working request, not a stalled one. Connecting
is the one phase that should still fail fast."""


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ProviderError(str(exc)) from exc


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """How long to wait before the next attempt, capped.

    Retry-After is honored when the server sends one, but a rate limiter asking
    for an hour is not something a library should silently sleep through.
    """
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), _MAX_DELAY_SECONDS)
            except ValueError:
                pass
    return min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)


def _transport_failure(url: str, exc: httpx.TransportError) -> ProviderError:
    """A connection-level failure as a ProviderError, so callers never catch httpx."""
    return ProviderError(f"request to {url} failed: {exc!r}")


class HTTPClient:
    """Pairs an async and sync httpx client for one base URL.

    Centralizes client construction and request/stream mechanics so the rest of
    the library never imports or calls httpx directly. It lives here rather than
    under providers/ because it is not provider-specific: tools/ reaches for the
    same retries and timeouts when it calls an API of its own. Requests are retried with
    exponential backoff on transient failures - 429 rate limits, 5xx server
    errors, and connection-level errors - honoring a Retry-After header when the
    server sends one. Whatever still fails after the last attempt surfaces as a
    ProviderError, so a caller has one exception type to handle rather than two.
    """

    __slots__ = ("_async_client", "_gates", "_max_concurrency", "_sync_client")

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
        timeout: httpx.Timeout | float | None = None,
        max_concurrency: int | None = None,
    ):
        self._async_client = client or httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout or DEFAULT_TIMEOUT
        )
        self._sync_client = sync_client or httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout or DEFAULT_TIMEOUT
        )
        if max_concurrency is not None and max_concurrency < 1:
            raise ConfigurationError(
                f"max_concurrency must be at least 1 when set, got {max_concurrency}"
            )
        self._max_concurrency = max_concurrency
        # One semaphore per event loop, made on first use rather than in
        # __init__: a contended asyncio primitive binds to the loop that
        # awaited it and raises in any other, and a provider built at import
        # time routinely outlives a script's first asyncio.run().
        self._gates: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    async def aclose(self) -> None:
        """Release both clients' connection pools."""
        await self._async_client.aclose()
        self._sync_client.close()

    def close(self) -> None:
        """Release the sync client's pool.

        Only the sync half: closing the async client needs a running event loop,
        so a caller outside one can still clean up what generate() opened.
        """
        self._sync_client.close()

    def _gate(self) -> Any:
        """The concurrency gate for the running loop, or a no-op when uncapped.

        In-flight requests are what a rate limiter counts, so the gate is held
        across retries and for a stream's whole body - releasing it early would
        let the cap be exceeded by exactly the requests already backing off.
        """
        if self._max_concurrency is None:
            return nullcontext()
        loop = asyncio.get_running_loop()
        gate = self._gates.get(loop)
        if gate is None:
            gate = asyncio.Semaphore(self._max_concurrency)
            self._gates[loop] = gate
        return gate

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        async with self._gate():
            return await self._post(url, **kwargs)

    async def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._async_client.post(url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES:
                    raise _transport_failure(url, exc) from exc
                await asyncio.sleep(_retry_delay(attempt))
                continue
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == _MAX_RETRIES
            ):
                _raise_for_status(response)
                return response
            await asyncio.sleep(_retry_delay(attempt, response))
        raise AssertionError("unreachable: last attempt always returns or raises")

    def post_sync(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._sync_client.post(url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES:
                    raise _transport_failure(url, exc) from exc
                time.sleep(_retry_delay(attempt))
                continue
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == _MAX_RETRIES
            ):
                _raise_for_status(response)
                return response
            time.sleep(_retry_delay(attempt, response))
        raise AssertionError("unreachable: last attempt always returns or raises")

    @asynccontextmanager
    async def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[httpx.Response]:
        async with self._gate(), self._stream(method, url, **kwargs) as response:
            yield response

    @asynccontextmanager
    async def _stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[httpx.Response]:
        started = False
        for attempt in range(_MAX_ATTEMPTS):
            delay = None
            try:
                async with self._async_client.stream(method, url, **kwargs) as response:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and attempt < _MAX_RETRIES
                    ):
                        delay = _retry_delay(attempt, response)
                    else:
                        _raise_for_status(response)
                        started = True
                        yield response
                        return
            except httpx.TransportError as exc:
                # A stream that breaks mid-body is not retried: its deltas are
                # already with the caller, and starting over would repeat them.
                if started or attempt == _MAX_RETRIES:
                    raise _transport_failure(url, exc) from exc
                delay = _retry_delay(attempt)

            await asyncio.sleep(delay)

    @contextmanager
    def stream_sync(
        self, method: str, url: str, **kwargs: Any
    ) -> Generator[httpx.Response]:
        started = False
        for attempt in range(_MAX_ATTEMPTS):
            delay = None
            try:
                with self._sync_client.stream(method, url, **kwargs) as response:
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and attempt < _MAX_RETRIES
                    ):
                        delay = _retry_delay(attempt, response)
                    else:
                        _raise_for_status(response)
                        started = True
                        yield response
                        return
            except httpx.TransportError as exc:
                if started or attempt == _MAX_RETRIES:
                    raise _transport_failure(url, exc) from exc
                delay = _retry_delay(attempt)

            time.sleep(delay)
