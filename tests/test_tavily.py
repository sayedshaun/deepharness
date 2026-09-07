import json

import httpx
import pytest

from deepharness.errors import ConfigurationError, ProviderError
from deepharness.tools.tavily import SearchResult, TavilySearch, format_results

PAYLOAD = {
    "results": [
        {
            "title": "First",
            "url": "https://example.com/1",
            "content": "About the first thing.",
            "score": 0.9,
        },
        {
            "title": "Second",
            "url": "https://example.com/2",
            "content": "About the second thing.",
        },
    ]
}


def make_search(handler, **kwargs):
    """A TavilySearch whose requests are answered by handler, never the network."""
    transport = httpx.MockTransport(handler)
    return TavilySearch(
        api_key="test-key",
        client=httpx.AsyncClient(
            transport=transport, base_url="https://api.tavily.com"
        ),
        sync_client=httpx.Client(
            transport=transport, base_url="https://api.tavily.com"
        ),
        **kwargs,
    )


def respond_with(payload, status_code=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status_code, json=payload)

    return handler


async def test_search_returns_typed_results():
    search = make_search(respond_with(PAYLOAD))

    results = await search.search("anything")

    assert results == [
        SearchResult("First", "https://example.com/1", "About the first thing.", 0.9),
        SearchResult(
            "Second", "https://example.com/2", "About the second thing.", None
        ),
    ]


def test_search_sync_returns_typed_results():
    search = make_search(respond_with(PAYLOAD))

    results = search.search_sync("anything")

    assert [result.url for result in results] == [
        "https://example.com/1",
        "https://example.com/2",
    ]


async def test_settings_are_sent_and_query_is_the_models_only_choice():
    seen = []
    search = make_search(
        respond_with(PAYLOAD, seen=seen), max_results=2, search_depth="advanced"
    )

    await search.search("bangladesh election")

    assert seen[0].url.path == "/search"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(seen[0].content) == {
        "query": "bangladesh election",
        "max_results": 2,
        "search_depth": "advanced",
    }


async def test_missing_results_key_is_not_an_error():
    search = make_search(respond_with({}))

    assert await search.search("nothing") == []


async def test_http_failure_surfaces_as_provider_error():
    search = make_search(respond_with({"detail": "nope"}, status_code=401))

    with pytest.raises(ProviderError):
        await search.search("anything")


def test_requires_a_credential(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ConfigurationError):
        TavilySearch()


def test_reads_credential_from_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")

    search = TavilySearch()

    assert search.max_results == 5
    search.close()


def test_rejects_a_max_results_below_one():
    with pytest.raises(ConfigurationError):
        TavilySearch(api_key="test-key", max_results=0)


def test_format_results_names_the_query_when_empty():
    assert format_results([], "ghosts") == "No results for 'ghosts'."


def test_format_results_puts_each_url_on_its_own_line():
    text = format_results([SearchResult.from_json(PAYLOAD["results"][0])], "anything")

    assert text == "First\nhttps://example.com/1\nAbout the first thing."


async def test_as_tool_exposes_only_query_and_returns_prose():
    search = make_search(respond_with(PAYLOAD))
    tool = search.as_tool()

    assert tool._tool_spec.name == "web_search"
    assert tool._tool_spec.parameters["required"] == ["query"]
    assert set(tool._tool_spec.parameters["properties"]) == {"query"}
    assert "https://example.com/2" in await tool("anything")


def test_as_sync_tool_is_callable_without_an_event_loop():
    search = make_search(respond_with(PAYLOAD))

    assert "https://example.com/1" in search.as_sync_tool()("anything")


def test_as_tool_accepts_a_name_and_description():
    search = make_search(respond_with(PAYLOAD))

    tool = search.as_tool(name="lookup", description="Look things up.")

    assert tool._tool_spec.name == "lookup"
    assert tool._tool_spec.description == "Look things up."
