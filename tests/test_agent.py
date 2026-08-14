import asyncio

import pytest

from subagents.agent import Agent, tool
from subagents.providers.base import CompletionResponse, LLM, ToolCall


class ScriptedProvider(LLM):
    """Returns a fixed sequence of responses, one per call to agenerate()."""

    def __init__(self, responses: list[CompletionResponse]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    async def agenerate(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        return self.responses[len(self.calls) - 1]

    def generate(self, messages, *, tools=None):
        self.calls.append([dict(message) for message in messages])
        return self.responses[len(self.calls) - 1]

    async def astream(self, messages, *, tools=None):
        raise NotImplementedError("ScriptedProvider only supports agenerate() in tests")
        yield  # pragma: no cover - makes this an async generator

    def stream(self, messages, *, tools=None):
        raise NotImplementedError("ScriptedProvider only supports agenerate() in tests")
        yield  # pragma: no cover - makes this a generator


def test_agent_stores_name():
    agent = Agent("researcher")
    assert agent.name == "researcher"


async def test_agent_run_returns_state():
    agent = Agent("researcher")
    state = {"input": "topic"}

    result = await agent.arun(state)

    assert result == state


async def test_returns_content_when_no_tool_call_needed():
    provider = ScriptedProvider([CompletionResponse(content="hello there")])
    agent = Agent("assistant", provider)

    state = await agent.arun({"messages": [{"role": "user", "content": "hi"}]})

    assert state["output"] == "hello there"
    assert state["messages"][-1] == {"role": "assistant", "content": "hello there"}


async def test_prepends_system_prompt_once():
    provider = ScriptedProvider([CompletionResponse(content="hello")])
    agent = Agent("assistant", provider, system_prompt="be helpful")

    state = await agent.arun({"messages": []})

    assert state["messages"][0] == {"role": "system", "content": "be helpful"}


async def test_dispatches_tool_calls_and_continues_loop():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            CompletionResponse(content="", tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})]),
            CompletionResponse(content="the answer is 3"),
        ]
    )
    agent = Agent("assistant", provider, tools=[add])

    state = await agent.arun({"messages": [{"role": "user", "content": "what is 1+2?"}]})

    assert state["output"] == "the answer is 3"
    tool_messages = [m for m in state["messages"] if m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "name": "add", "content": "3"}]
    assert len(provider.calls) == 2


async def test_multiple_tool_calls_in_one_turn_run_concurrently():
    order: list[str] = []

    @tool
    async def slow(ms: int) -> str:
        """Sleep for ms milliseconds then report done."""
        await asyncio.sleep(ms / 1000)
        order.append(f"slow-{ms}")
        return f"slept {ms}ms"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(name="slow", arguments={"ms": 50}),
                    ToolCall(name="slow", arguments={"ms": 10}),
                ],
            ),
            CompletionResponse(content="done"),
        ]
    )
    agent = Agent("assistant", provider, tools=[slow])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    # the shorter sleep finishes first if they ran concurrently
    assert order == ["slow-10", "slow-50"]
    # but result messages stay in call order, not completion order
    tool_messages = [m for m in state["messages"] if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "name": "slow", "content": "slept 50ms"},
        {"role": "tool", "name": "slow", "content": "slept 10ms"},
    ]


def test_sync_run_returns_content_when_no_tool_call_needed():
    provider = ScriptedProvider([CompletionResponse(content="hello there")])
    agent = Agent("assistant", provider)

    state = agent.run({"messages": [{"role": "user", "content": "hi"}]})

    assert state["output"] == "hello there"
    assert state["messages"][-1] == {"role": "assistant", "content": "hello there"}


def test_sync_run_dispatches_tool_calls():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            CompletionResponse(content="", tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})]),
            CompletionResponse(content="the answer is 3"),
        ]
    )
    agent = Agent("assistant", provider, tools=[add])

    state = agent.run({"messages": [{"role": "user", "content": "what is 1+2?"}]})

    assert state["output"] == "the answer is 3"
    tool_messages = [m for m in state["messages"] if m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "name": "add", "content": "3"}]


def test_sync_run_raises_for_async_tool():
    @tool
    async def slow(ms: int) -> str:
        """An async tool, incompatible with the sync run() path."""
        return "never gets here"

    provider = ScriptedProvider(
        [CompletionResponse(content="", tool_calls=[ToolCall(name="slow", arguments={"ms": 1})])]
    )
    agent = Agent("assistant", provider, tools=[slow])

    with pytest.raises(RuntimeError, match="async"):
        agent.run({"messages": [{"role": "user", "content": "go"}]})


async def test_stops_after_max_steps():
    @tool
    def noop() -> str:
        """Do nothing."""
        return "noop"

    provider = ScriptedProvider(
        [
            CompletionResponse(content="", tool_calls=[ToolCall(name="noop", arguments={})]),
            CompletionResponse(content="", tool_calls=[ToolCall(name="noop", arguments={})]),
        ]
    )
    agent = Agent("assistant", provider, tools=[noop], max_steps=2)

    state = await agent.arun({"messages": [{"role": "user", "content": "loop"}]})

    assert len(provider.calls) == 2
    assert state["output"] == "noop"
