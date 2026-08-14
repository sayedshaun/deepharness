from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from ..errors import ProviderError

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ProviderError(str(exc)) from exc


def _retry_delay(attempt: int, response: httpx.Response) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return _BASE_DELAY_SECONDS * (2**attempt)


class HTTPClient:
    """Pairs an async and sync httpx client for one base URL.

    Centralizes client construction and request/stream mechanics so provider
    modules never import or call httpx directly. Requests are retried with
    exponential backoff on transient failures (429 rate limits, 5xx server
    errors), honoring a Retry-After header when the server sends one.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        self.async_client = client or httpx.AsyncClient(
            base_url=base_url, headers=headers
        )
        self.sync_client = sync_client or httpx.Client(
            base_url=base_url, headers=headers
        )

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            response = await self.async_client.post(url, **kwargs)
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == _MAX_RETRIES
            ):
                _raise_for_status(response)
                return response
            await asyncio.sleep(_retry_delay(attempt, response))

    def post_sync(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            response = self.sync_client.post(url, **kwargs)
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == _MAX_RETRIES
            ):
                _raise_for_status(response)
                return response
            time.sleep(_retry_delay(attempt, response))

    @asynccontextmanager
    async def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[httpx.Response]:
        for attempt in range(_MAX_RETRIES + 1):
            delay = None
            async with self.async_client.stream(method, url, **kwargs) as response:
                if (
                    response.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < _MAX_RETRIES
                ):
                    delay = _retry_delay(attempt, response)
                else:
                    _raise_for_status(response)
                    yield response
                    return

            await asyncio.sleep(delay)

    @contextmanager
    def stream_sync(
        self, method: str, url: str, **kwargs: Any
    ) -> Generator[httpx.Response]:
        for attempt in range(_MAX_RETRIES + 1):
            delay = None
            with self.sync_client.stream(method, url, **kwargs) as response:
                if (
                    response.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < _MAX_RETRIES
                ):
                    delay = _retry_delay(attempt, response)
                else:
                    _raise_for_status(response)
                    yield response
                    return

            time.sleep(delay)
