from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx


class HTTPClient:
    """Pairs an async and sync httpx client for one base URL.

    Centralizes client construction and request/stream mechanics so provider
    modules never import or call httpx directly.
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
        response = await self.async_client.post(url, **kwargs)
        response.raise_for_status()
        return response

    def post_sync(self, url: str, **kwargs: Any) -> httpx.Response:
        response = self.sync_client.post(url, **kwargs)
        response.raise_for_status()
        return response

    @asynccontextmanager
    async def stream(
        self, method: str, url: str, **kwargs: Any
    ) -> AsyncGenerator[httpx.Response]:
        async with self.async_client.stream(method, url, **kwargs) as response:
            response.raise_for_status()
            yield response

    @contextmanager
    def stream_sync(
        self, method: str, url: str, **kwargs: Any
    ) -> Generator[httpx.Response]:
        with self.sync_client.stream(method, url, **kwargs) as response:
            response.raise_for_status()
            yield response
