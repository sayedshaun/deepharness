from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from deepharness.errors import ProviderError
from deepharness.providers.client import DEFAULT_TIMEOUT, HTTPClient


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("deepharness.providers.client.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("deepharness.providers.client.time.sleep", MagicMock())


def make_response(status_code, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.raise_for_status = MagicMock(
        side_effect=(
            httpx.HTTPStatusError("error", request=MagicMock(), response=response)
            if status_code >= 400
            else None
        )
    )
    return response


async def test_post_retries_on_429_then_succeeds():
    ok_response = make_response(200)
    rate_limited = make_response(429)
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[rate_limited, ok_response])

    http = HTTPClient("https://example.com", client=client)
    response = await http.post("/thing")

    assert response is ok_response
    assert client.post.await_count == 2


async def test_post_raises_after_exhausting_retries():
    client = AsyncMock()
    client.post = AsyncMock(return_value=make_response(500))

    http = HTTPClient("https://example.com", client=client)

    with pytest.raises(ProviderError):
        await http.post("/thing")

    assert client.post.await_count == 4  # initial attempt + 3 retries


async def test_post_does_not_retry_on_non_retryable_error():
    client = AsyncMock()
    client.post = AsyncMock(return_value=make_response(400))

    http = HTTPClient("https://example.com", client=client)

    with pytest.raises(ProviderError):
        await http.post("/thing")

    assert client.post.await_count == 1


async def test_post_honors_retry_after_header(monkeypatch):
    ok_response = make_response(200)
    rate_limited = make_response(429, headers={"retry-after": "2.5"})
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[rate_limited, ok_response])
    sleep_mock = AsyncMock()
    monkeypatch.setattr("deepharness.providers.client.asyncio.sleep", sleep_mock)

    http = HTTPClient("https://example.com", client=client)
    await http.post("/thing")

    sleep_mock.assert_awaited_once_with(2.5)


def test_post_sync_retries_on_429_then_succeeds():
    ok_response = make_response(200)
    rate_limited = make_response(429)
    client = MagicMock()
    client.post = MagicMock(side_effect=[rate_limited, ok_response])

    http = HTTPClient("https://example.com", sync_client=client, client=AsyncMock())
    response = http.post_sync("/thing")

    assert response is ok_response
    assert client.post.call_count == 2


def test_clients_get_a_timeout_long_enough_for_a_completion():
    http = HTTPClient("https://example.com")

    assert http._sync_client.timeout == DEFAULT_TIMEOUT
    assert http._async_client.timeout == DEFAULT_TIMEOUT
    assert DEFAULT_TIMEOUT.read is not None and DEFAULT_TIMEOUT.read > 60


def test_explicit_timeout_is_passed_through():
    http = HTTPClient("https://example.com", timeout=12.0)

    assert http._sync_client.timeout == httpx.Timeout(12.0)


async def test_post_retries_transport_errors_then_succeeds():
    ok_response = make_response(200)
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[httpx.ConnectError("no route"), ok_response],
    )

    http = HTTPClient("https://example.com", client=client)
    response = await http.post("/thing")

    assert response is ok_response
    assert client.post.await_count == 2


async def test_post_reports_a_transport_error_as_a_provider_error():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.ConnectTimeout("too slow"))

    http = HTTPClient("https://example.com", client=client)

    with pytest.raises(ProviderError, match="request to /thing failed"):
        await http.post("/thing")

    assert client.post.await_count == 4


def test_post_sync_reports_a_transport_error_as_a_provider_error():
    client = MagicMock()
    client.post = MagicMock(side_effect=httpx.ReadTimeout("too slow"))

    http = HTTPClient("https://example.com", sync_client=client, client=AsyncMock())

    with pytest.raises(ProviderError, match="request to /thing failed"):
        http.post_sync("/thing")


async def test_retry_after_is_capped(monkeypatch):
    ok_response = make_response(200)
    rate_limited = make_response(429, headers={"retry-after": "3600"})
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[rate_limited, ok_response])
    sleep_mock = AsyncMock()
    monkeypatch.setattr("deepharness.providers.client.asyncio.sleep", sleep_mock)

    http = HTTPClient("https://example.com", client=client)
    await http.post("/thing")

    sleep_mock.assert_awaited_once_with(30.0)


async def test_aclose_releases_both_pools():
    client = AsyncMock()
    sync_client = MagicMock()

    http = HTTPClient("https://example.com", client=client, sync_client=sync_client)
    await http.aclose()

    client.aclose.assert_awaited_once()
    sync_client.close.assert_called_once()
