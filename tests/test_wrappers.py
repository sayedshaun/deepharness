"""Providers that wrap a provider: fallback, caching, rate limiting, retrying."""

import asyncio
import time

import pytest

from deepharness.errors import ConfigurationError, ProviderError
from deepharness.providers.base import (
    LLM,
    Completed,
    CompletionResponse,
    TextDelta,
    ToolCall,
)
from deepharness.providers.wrappers import Caching, Fallback, RateLimited, Retrying


class Counting(LLM):
    """Answers with a fixed reply, counting how often it was asked."""

    def __init__(self, text="ok", *, tool_calls=(), fail=None):
        self.text = text
        self.tool_calls = list(tool_calls)
        self.fail = fail
        self.calls = 0

    def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return CompletionResponse(content=self.text, tool_calls=list(self.tool_calls))

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)


class Emptying(LLM):
    """Answers with nothing until the nth call, like a model that fumbled a turn."""

    def __init__(self, good_on=2):
        self.good_on = good_on
        self.calls = 0

    def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.calls < self.good_on:
            return CompletionResponse(content="")
        return CompletionResponse(content="finally")

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)


class MidStreamFailure(LLM):
    """Yields one delta, then breaks - a stream that cannot be restarted."""

    def generate(self, messages, *, tools=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def agenerate(self, messages, *, tools=None):  # pragma: no cover - unused
        raise NotImplementedError

    def stream_events(self, messages, *, tools=None):
        yield TextDelta("half a ")
        raise ProviderError("connection dropped")

    async def astream_events(self, messages, *, tools=None):
        yield TextDelta("half a ")
        raise ProviderError("connection dropped")


HELLO = [{"role": "user", "content": "hi"}]


# --- Fallback -----------------------------------------------------------------


def test_the_primary_is_used_when_it_works():
    primary, backup = Counting("first"), Counting("second")

    assert Fallback(primary, backup).generate(HELLO).content == "first"
    assert backup.calls == 0


def test_a_failing_primary_falls_through():
    primary = Counting(fail=ProviderError("down"))
    backup = Counting("second")

    assert Fallback(primary, backup).generate(HELLO).content == "second"


@pytest.mark.asyncio
async def test_fallback_works_on_the_async_path():
    primary = Counting(fail=ProviderError("down"))

    result = await Fallback(primary, Counting("second")).agenerate(HELLO)

    assert result.content == "second"


def test_an_error_not_listed_is_not_caught():
    """A bug in your own code must not read as a flaky provider."""
    primary = Counting(fail=TypeError("bug in my tool schema"))

    with pytest.raises(TypeError):
        Fallback(primary, Counting("second")).generate(HELLO)


def test_everything_failing_reports_the_last_error():
    models = [Counting(fail=ProviderError(f"down {n}")) for n in range(3)]

    with pytest.raises(ProviderError, match="all 3 providers failed"):
        Fallback(*models).generate(HELLO)


def test_fallback_needs_something_to_fall_back_to():
    with pytest.raises(ConfigurationError):
        Fallback(Counting())


def test_a_stream_that_fails_before_its_first_event_falls_back():
    primary = Counting(fail=ProviderError("down"))

    events = list(Fallback(primary, Counting("second")).stream_events(HELLO))

    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("second")]


def test_a_stream_that_fails_partway_is_not_restarted():
    """Those deltas already reached the caller; starting over would repeat them."""
    backup = Counting("second")
    stream = Fallback(MidStreamFailure(), backup).stream_events(HELLO)

    assert next(stream) == TextDelta("half a ")
    with pytest.raises(ProviderError, match="connection dropped"):
        next(stream)
    assert backup.calls == 0


# --- Caching ------------------------------------------------------------------


def test_a_repeated_request_is_served_from_memory():
    inner = Counting("cached")
    llm = Caching(inner)

    assert llm.generate(HELLO).content == "cached"
    assert llm.generate(HELLO).content == "cached"
    assert inner.calls == 1
    assert (llm.hits, llm.misses) == (1, 1)


def test_a_different_request_is_a_different_key():
    inner = Counting()
    llm = Caching(inner)

    llm.generate(HELLO)
    llm.generate([{"role": "user", "content": "different"}])

    assert inner.calls == 2


def test_the_tool_schemas_are_part_of_the_key():
    inner = Counting()
    llm = Caching(inner)

    llm.generate(HELLO, tools=[{"name": "a"}])
    llm.generate(HELLO, tools=[{"name": "b"}])

    assert inner.calls == 2


@pytest.mark.asyncio
async def test_caching_works_on_the_async_path():
    inner = Counting()
    llm = Caching(inner)

    await llm.agenerate(HELLO)
    await llm.agenerate(HELLO)

    assert inner.calls == 1


def test_a_cached_response_cannot_be_edited_from_outside():
    inner = Counting("cached", tool_calls=[ToolCall(name="t", arguments={}, id="1")])
    llm = Caching(inner)

    first = llm.generate(HELLO)
    first.content = "tampered"
    first.tool_calls.clear()

    second = llm.generate(HELLO)
    assert second.content == "cached"
    assert len(second.tool_calls) == 1


def test_an_expired_entry_is_dropped():
    inner = Counting()
    llm = Caching(inner, ttl=0.01)

    llm.generate(HELLO)
    time.sleep(0.02)
    llm.generate(HELLO)

    assert inner.calls == 2


def test_the_least_recently_used_entry_is_evicted():
    inner = Counting()
    llm = Caching(inner, maxsize=2)

    for text in ("a", "b", "c"):
        llm.generate([{"role": "user", "content": text}])
    llm.generate([{"role": "user", "content": "a"}])  # evicted, so a miss

    assert inner.calls == 4


def test_a_stream_is_cached_and_replayed():
    inner = Counting("streamed")
    llm = Caching(inner)

    first = list(llm.stream_events(HELLO))
    second = list(llm.stream_events(HELLO))

    assert inner.calls == 1
    assert [e for e in second if isinstance(e, TextDelta)] == [TextDelta("streamed")]
    assert isinstance(second[-1], Completed)
    assert second[-1].response.content == first[-1].response.content


def test_clearing_the_cache_makes_the_next_call_a_miss():
    inner = Counting()
    llm = Caching(inner)

    llm.generate(HELLO)
    llm.clear()
    llm.generate(HELLO)

    assert inner.calls == 2


@pytest.mark.parametrize("kwargs", [{"maxsize": 0}, {"ttl": 0}])
def test_caching_rejects_meaningless_limits(kwargs):
    with pytest.raises(ConfigurationError):
        Caching(Counting(), **kwargs)


# --- RateLimited --------------------------------------------------------------


def test_requests_within_the_burst_do_not_wait():
    llm = RateLimited(Counting(), rps=10, burst=3)

    started = time.monotonic()
    for _ in range(3):
        llm.generate(HELLO)

    assert time.monotonic() - started < 0.05


def test_going_over_the_burst_waits_for_the_bucket():
    llm = RateLimited(Counting(), rps=20, burst=1)

    started = time.monotonic()
    for _ in range(3):
        llm.generate(HELLO)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.09  # two refills at 20/s


@pytest.mark.asyncio
async def test_concurrent_callers_are_spaced_out_not_stacked():
    llm = RateLimited(Counting(), rps=20, burst=1)

    started = time.monotonic()
    await asyncio.gather(*(llm.agenerate(HELLO) for _ in range(3)))
    elapsed = time.monotonic() - started

    assert elapsed >= 0.09


def test_streaming_takes_a_slot_too():
    inner = Counting()
    llm = RateLimited(inner, rps=50, burst=1)

    list(llm.stream_events(HELLO))
    list(llm.stream_events(HELLO))

    assert inner.calls == 2


@pytest.mark.parametrize("kwargs", [{"rps": 0}, {"rps": 1, "burst": 0}])
def test_ratelimited_rejects_meaningless_limits(kwargs):
    with pytest.raises(ConfigurationError):
        RateLimited(Counting(), **kwargs)


# --- Retrying -----------------------------------------------------------------


def test_an_empty_turn_is_asked_again():
    inner = Emptying(good_on=2)

    result = Retrying(inner, attempts=3, backoff=0).generate(HELLO)

    assert result.content == "finally"
    assert inner.calls == 2


def test_a_usable_turn_is_not_retried():
    inner = Counting("good")

    Retrying(inner, attempts=3, backoff=0).generate(HELLO)

    assert inner.calls == 1


def test_a_turn_that_only_calls_a_tool_counts_as_usable():
    inner = Counting("", tool_calls=[ToolCall(name="t", arguments={}, id="1")])

    Retrying(inner, attempts=3, backoff=0).generate(HELLO)

    assert inner.calls == 1


@pytest.mark.asyncio
async def test_retrying_works_on_the_async_path():
    inner = Emptying(good_on=3)

    result = await Retrying(inner, attempts=3, backoff=0).agenerate(HELLO)

    assert result.content == "finally"
    assert inner.calls == 3


def test_attempts_are_bounded():
    inner = Emptying(good_on=99)

    result = Retrying(inner, attempts=2, backoff=0).generate(HELLO)

    assert result.content == ""
    assert inner.calls == 2


def test_an_empty_stream_is_retried():
    inner = Emptying(good_on=2)

    events = list(Retrying(inner, attempts=3, backoff=0).stream_events(HELLO))

    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("finally")]
    assert inner.calls == 2


@pytest.mark.parametrize("kwargs", [{"attempts": 0}, {"backoff": -1}])
def test_retrying_rejects_meaningless_limits(kwargs):
    with pytest.raises(ConfigurationError):
        Retrying(Counting(), **kwargs)


# --- Composition --------------------------------------------------------------


def test_wrappers_compose_and_expose_what_they_wrap():
    inner = Counting("composed")
    llm = Caching(RateLimited(Retrying(inner, backoff=0), rps=50))

    assert llm.generate(HELLO).content == "composed"
    assert llm.generate(HELLO).content == "composed"
    assert inner.calls == 1
    assert isinstance(llm.inner, RateLimited)
    assert llm.inner.inner.inner is inner


def test_closing_a_wrapper_closes_what_it_wraps():
    class Closable(Counting):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    inner = Closable()
    Caching(inner).close()

    assert inner.closed


@pytest.mark.asyncio
async def test_an_empty_stream_is_retried_on_the_async_path():
    inner = Emptying(good_on=2)
    llm = Retrying(inner, attempts=3, backoff=0)

    events = [event async for event in llm.astream_events(HELLO)]

    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("finally")]
    assert inner.calls == 2


def test_a_usable_stream_is_not_asked_again():
    """The bug this covers: falling through the retry loop after succeeding."""
    inner = Counting("once")

    events = list(Retrying(inner, attempts=3, backoff=0).stream_events(HELLO))

    assert inner.calls == 1
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("once")]
