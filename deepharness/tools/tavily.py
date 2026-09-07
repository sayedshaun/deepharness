"""Web search via Tavily, as a tool an Agent can be given.

A class rather than a bare @tool function because a search needs configuration -
a credential, a result count, a depth - and a tool's signature is the model's
prompt: every parameter added there is one more thing the model can get wrong.
Settings belong to the object, and only `query` is left for the model to choose.

Results come back as SearchResult objects rather than the raw payload, so a
caller that wants the URLs can read a field instead of parsing prose, and only
as_tool() flattens them into the text a model reads.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..errors import ConfigurationError
from ..providers.client import HTTPClient
from .toolbox import ToolSpec

_BASE_URL = "https://api.tavily.com"
_ENV_KEY = "TAVILY_API_KEY"

SearchDepth = Literal["basic", "advanced"]
"""Tavily's two modes: "advanced" reads further into each page and costs more
credits per call."""

_DESCRIPTION = (
    "Search the web for current information. "
    "Returns titles, URLs and excerpts from the top results."
)


@dataclass(slots=True)
class SearchResult:
    """One hit, with the fields every Tavily result carries.

    score is optional because Tavily omits it on some result kinds, and a
    missing relevance is not the same as a relevance of zero.
    """

    title: str
    url: str
    content: str
    score: float | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            title=data.get("title") or "",
            url=data.get("url") or "",
            content=data.get("content") or "",
            score=data.get("score"),
        )

    def to_prompt(self) -> str:
        """This result as the model reads it: title, URL, then the excerpt.

        The URL goes on its own line so a model asked to cite sources has an
        unambiguous string to copy rather than one buried in a sentence.
        """
        return f"{self.title}\n{self.url}\n{self.content}"


def format_results(results: list[SearchResult], query: str) -> str:
    """Results as one block of text for a model.

    An empty search says so in words: handing a model an empty string invites
    it to fill the silence from its own weights, which is the failure this
    tool exists to prevent.
    """
    if not results:
        return f"No results for {query!r}."
    return "\n\n".join(result.to_prompt() for result in results)


class TavilySearch:
    """Tavily's search API, usable directly or as an Agent tool.

    The credential is read from TAVILY_API_KEY when not passed, matching how
    providers resolve theirs. Requests go through the same HTTPClient the
    providers use, so a rate-limited search is retried with backoff instead of
    failing the tool call and costing the agent a turn.

        search = TavilySearch()
        agent = Agent(model, tools=[search.as_tool()])

    Sync and async both work, but a tool is one or the other: as_tool() is for
    an agent driven with arun(), as_sync_tool() for one driven with run().
    """

    __slots__ = ("_headers", "_http", "_max_results", "_search_depth")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_results: int = 5,
        search_depth: SearchDepth = "basic",
        client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get(_ENV_KEY)
        if not api_key:
            raise ConfigurationError(
                f"TavilySearch needs an API key: pass api_key= or set {_ENV_KEY}"
            )
        if max_results < 1:
            raise ConfigurationError(
                f"max_results must be at least 1, got {max_results}"
            )

        # Sent per request rather than baked into the client, so an injected
        # client (a test transport, a shared pool) still carries the credential.
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = HTTPClient(_BASE_URL, client=client, sync_client=sync_client)
        self._max_results = max_results
        self._search_depth = search_depth

    @property
    def max_results(self) -> int:
        return self._max_results

    async def aclose(self) -> None:
        """Release both connection pools."""
        await self._http.aclose()

    def close(self) -> None:
        """Release the sync connection pool; see HTTPClient.close()."""
        self._http.close()

    def _payload(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "max_results": self._max_results,
            "search_depth": self._search_depth,
        }

    async def search(self, query: str) -> list[SearchResult]:
        response = await self._http.post(
            "/search", json=self._payload(query), headers=self._headers
        )
        return self._parse(response)

    def search_sync(self, query: str) -> list[SearchResult]:
        response = self._http.post_sync(
            "/search", json=self._payload(query), headers=self._headers
        )
        return self._parse(response)

    @staticmethod
    def _parse(response: httpx.Response) -> list[SearchResult]:
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        return [SearchResult.from_json(item) for item in results or []]

    def as_tool(
        self, *, name: str = "web_search", description: str | None = None
    ) -> Callable[..., Any]:
        """This search as an async tool, for an agent driven with arun()."""

        async def call(query: str) -> str:
            return format_results(await self.search(query), query)

        return self._spec(call, name, description)

    def as_sync_tool(
        self, *, name: str = "web_search", description: str | None = None
    ) -> Callable[..., Any]:
        """This search as a blocking tool, for an agent driven with run()."""

        def call(query: str) -> str:
            return format_results(self.search_sync(query), query)

        return self._spec(call, name, description)

    def _spec(
        self, call: Callable[..., Any], name: str, description: str | None
    ) -> Callable[..., Any]:
        call.__name__ = name
        call._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name,
            description=description or _DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search the web for.",
                    }
                },
                "required": ["query"],
            },
            func=call,
        )
        return call
