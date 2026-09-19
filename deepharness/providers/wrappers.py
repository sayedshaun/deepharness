"""Providers that stand in front of another provider.

Retrying, caching, rate limiting and falling back are all the same shape: do
something around a model call, then delegate. LLM is already a narrow interface,
so each of these is simply another implementation of it - which means one
wrapper works on the synchronous path, the asynchronous one and the streaming
one, and they compose at the call site where the order is visible:

    llm = Caching(RateLimited(Fallback(OpenAI(...), Anthropic(...)), rps=2))

State here is guarded with a threading.Lock rather than an asyncio one. A
contended asyncio primitive binds to the loop that awaited it and raises in any
other, and a provider built once at import time routinely outlives a script's
first asyncio.run(); a lock held for a few microseconds with no await inside is
safe on either path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import Any

from ..errors import ConfigurationError, ProviderError
from .base import LLM, Completed, CompletionResponse, StreamEvent, as_deltas


class Wrapping(LLM):
    """An LLM that delegates to another one, overriding only what it changes.

    Inherited to declare the subtype - each wrapper *is* an LLM standing in
    front of an LLM - and to hold the forwarding once rather than four
    near-identical methods per wrapper.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: LLM):
        self._inner = inner

    @property
    def inner(self) -> LLM:
        """The provider underneath, so a caller can still reach it."""
        return self._inner

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        return await self._inner.agenerate(messages, tools=tools)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        return self._inner.generate(messages, tools=tools)

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._inner.astream_events(messages, tools=tools):
            yield event

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        yield from self._inner.stream_events(messages, tools=tools)

    async def aclose(self) -> None:
        """Close the provider underneath, if it holds anything to close.

        Asked for rather than declared on LLM: a fake or an in-process model has
        nothing to release, and a wrapper should not force one to pretend.
        """
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if closer is not None:
            closer()


class Fallback(LLM):
    """Try each provider in turn, moving on when one fails.

    `on` is ProviderError by default, not Exception: catching everything turns
    a bug in your own code into "the primary model is flaky" and silently sends
    your traffic elsewhere.

    A stream that fails before its first event falls back; one that fails
    partway through does not, because those deltas are already with the caller
    and starting over would repeat them.
    """

    __slots__ = ("_models", "_on")

    def __init__(
        self,
        primary: LLM,
        *others: LLM,
        on: tuple[type[BaseException], ...] = (ProviderError,),
    ):
        if not others:
            raise ConfigurationError("Fallback needs a second provider to fall back to")
        self._models = (primary, *others)
        self._on = on

    @property
    def models(self) -> tuple[LLM, ...]:
        return self._models

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        last: BaseException | None = None
        for model in self._models:
            try:
                return await model.agenerate(messages, tools=tools)
            except self._on as exc:
                last = exc
        raise self._exhausted(last)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        last: BaseException | None = None
        for model in self._models:
            try:
                return model.generate(messages, tools=tools)
            except self._on as exc:
                last = exc
        raise self._exhausted(last)

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        last: BaseException | None = None
        for model in self._models:
            started = False
            try:
                async for event in model.astream_events(messages, tools=tools):
                    started = True
                    yield event
                return
            except self._on as exc:
                if started:
                    raise
                last = exc
        raise self._exhausted(last)

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        last: BaseException | None = None
        for model in self._models:
            started = False
            try:
                for event in model.stream_events(messages, tools=tools):
                    started = True
                    yield event
                return
            except self._on as exc:
                if started:
                    raise
                last = exc
        raise self._exhausted(last)

    def _exhausted(self, last: BaseException | None) -> ProviderError:
        """Every provider failed, reported as the last failure's cause."""
        error = ProviderError(
            f"all {len(self._models)} providers failed; last error: {last!r}"
        )
        error.__cause__ = last
        return error

    async def aclose(self) -> None:
        for model in self._models:
            closer = getattr(model, "aclose", None)
            if closer is not None:
                await closer()

    def close(self) -> None:
        for model in self._models:
            closer = getattr(model, "close", None)
            if closer is not None:
                closer()


class Caching(Wrapping):
    """Serve a repeated request from memory instead of the vendor.

    Only worth putting in front of a deterministic setup - at temperature 0, or
    a classify/extract step. A cache in front of a sampling model hands every
    caller the first answer it happened to get, which is not what sampling is
    for, so this stays an explicit decision rather than a default.

    Two limits worth knowing. Identical calls made concurrently all miss and all
    reach the vendor: there is no single-flight, because coordinating that needs
    an asyncio primitive per loop for a case a retry already handles. And a
    cached response is copied on the way out, so a caller that edits what it got
    cannot corrupt the entry behind it.
    """

    __slots__ = ("_hits", "_lock", "_maxsize", "_misses", "_store", "_ttl")

    def __init__(self, inner: LLM, *, maxsize: int = 256, ttl: float | None = None):
        super().__init__(inner)
        if maxsize < 1:
            raise ConfigurationError(
                f"Caching.maxsize must be at least 1, got {maxsize}"
            )
        if ttl is not None and ttl <= 0:
            raise ConfigurationError(
                f"Caching.ttl must be positive when set, got {ttl}"
            )
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, CompletionResponse]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        key = _fingerprint(messages, tools)
        hit = self._take(key)
        if hit is not None:
            return hit
        response = await self._inner.agenerate(messages, tools=tools)
        self._put(key, response)
        return response

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        key = _fingerprint(messages, tools)
        hit = self._take(key)
        if hit is not None:
            return hit
        response = self._inner.generate(messages, tools=tools)
        self._put(key, response)
        return response

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        key = _fingerprint(messages, tools)
        hit = self._take(key)
        if hit is not None:
            # A hit has nothing left to stream, so it is replayed in block
            # order: a caller's event loop looks the same either way.
            for event in as_deltas(hit):
                yield event
            yield Completed(hit)
            return
        async for event in self._inner.astream_events(messages, tools=tools):
            if isinstance(event, Completed):
                self._put(key, event.response)
            yield event

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        key = _fingerprint(messages, tools)
        hit = self._take(key)
        if hit is not None:
            yield from as_deltas(hit)
            yield Completed(hit)
            return
        for event in self._inner.stream_events(messages, tools=tools):
            if isinstance(event, Completed):
                self._put(key, event.response)
            yield event

    def _take(self, key: str) -> CompletionResponse | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, response = entry
            if self._ttl is not None and time.monotonic() - stored_at > self._ttl:
                del self._store[key]  # dropped here, so a stale entry cannot linger
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return _copy(response)

    def _put(self, key: str, response: CompletionResponse) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), _copy(response))
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # least recently used


class RateLimited(Wrapping):
    """Spend one token per request from a bucket that refills at `rps`.

    A slot is reserved the moment a caller asks for one, and the caller waits
    out its own turn - so ten simultaneous requests leave in order at the
    configured rate instead of all retrying against each other.
    """

    __slots__ = ("_burst", "_lock", "_rps", "_tokens", "_updated")

    def __init__(self, inner: LLM, *, rps: float, burst: int | None = None):
        super().__init__(inner)
        if rps <= 0:
            raise ConfigurationError(f"RateLimited.rps must be positive, got {rps}")
        if burst is not None and burst < 1:
            raise ConfigurationError(
                f"RateLimited.burst must be at least 1 when set, got {burst}"
            )
        self._rps = rps
        self._burst = burst if burst is not None else max(1, int(rps))
        self._tokens = float(self._burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        """Take a slot, returning how long this caller must wait to use it."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                float(self._burst), self._tokens + (now - self._updated) * self._rps
            )
            self._updated = now
            wait = 0.0 if self._tokens >= 1.0 else (1.0 - self._tokens) / self._rps
            self._tokens -= 1.0
            return wait

    async def _await_turn(self) -> None:
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)

    def _wait_turn(self) -> None:
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        await self._await_turn()
        return await self._inner.agenerate(messages, tools=tools)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        self._wait_turn()
        return self._inner.generate(messages, tools=tools)

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        await self._await_turn()
        async for event in self._inner.astream_events(messages, tools=tools):
            yield event

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        self._wait_turn()
        yield from self._inner.stream_events(messages, tools=tools)


class Retrying(Wrapping):
    """Ask again when a turn comes back with nothing in it.

    Transport failures are already retried by the HTTP client; this is the other
    kind, where the request succeeded and the model said nothing at all - no
    text and no tool call. That turn is unusable to an agent loop, which would
    otherwise spend a step on it.

    A streamed turn is retried only while nothing has been emitted, which an
    empty turn satisfies by definition - so streaming stays as responsive as it
    was.
    """

    __slots__ = ("_attempts", "_backoff")

    def __init__(self, inner: LLM, *, attempts: int = 2, backoff: float = 0.5):
        super().__init__(inner)
        if attempts < 1:
            raise ConfigurationError(
                f"Retrying.attempts must be at least 1, got {attempts}"
            )
        if backoff < 0:
            raise ConfigurationError(
                f"Retrying.backoff cannot be negative, got {backoff}"
            )
        self._attempts = attempts
        self._backoff = backoff

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        response = await self._inner.agenerate(messages, tools=tools)
        for attempt in range(1, self._attempts):
            if not _is_empty(response):
                return response
            if self._backoff:
                await asyncio.sleep(self._backoff * attempt)
            response = await self._inner.agenerate(messages, tools=tools)
        return response

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResponse:
        response = self._inner.generate(messages, tools=tools)
        for attempt in range(1, self._attempts):
            if not _is_empty(response):
                return response
            if self._backoff:
                time.sleep(self._backoff * attempt)
            response = self._inner.generate(messages, tools=tools)
        return response

    async def astream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        for attempt in range(1, self._attempts + 1):
            emitted = False
            retry = False
            async for event in self._inner.astream_events(messages, tools=tools):
                if isinstance(event, Completed):
                    # Completed is the last event, so breaking here leaves
                    # nothing unread on the stream being abandoned.
                    if (
                        not emitted
                        and _is_empty(event.response)
                        and attempt < self._attempts
                    ):
                        retry = True
                        break
                    yield event
                    return
                emitted = True
                yield event
            if not retry:
                return
            if self._backoff:
                await asyncio.sleep(self._backoff * attempt)

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        for attempt in range(1, self._attempts + 1):
            emitted = False
            retry = False
            for event in self._inner.stream_events(messages, tools=tools):
                if isinstance(event, Completed):
                    if (
                        not emitted
                        and _is_empty(event.response)
                        and attempt < self._attempts
                    ):
                        retry = True
                        break
                    yield event
                    return
                emitted = True
                yield event
            if not retry:
                return
            if self._backoff:
                time.sleep(self._backoff * attempt)


def _is_empty(response: CompletionResponse) -> bool:
    """A turn with no text and no tool call: nothing a loop can act on."""
    return not response.content.strip() and not response.tool_calls


def _copy(response: CompletionResponse) -> CompletionResponse:
    """A shallow copy with its own lists, so sharing an entry cannot leak edits."""
    return replace(
        response,
        tool_calls=list(response.tool_calls),
        blocks=list(response.blocks),
    )


def _fingerprint(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> str:
    """A stable key for one request. Unserializable values fall back to repr."""
    payload = json.dumps([messages, tools], sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()
