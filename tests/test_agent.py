import asyncio
import threading
import time

import pytest

from subagents.agent import (
    Agent,
    Budget,
    Message,
    PendingHumanInput,
    TokenBudgetExceeded,
    Toolbox,
    tool,
)
from subagents.errors import ConfigurationError, HumanInputRequired
from subagents.providers.base import LLM, CompletionResponse, TokenUsage, ToolCall


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


@tool
def confirm(question: str) -> str:
    """Pause and ask a human to confirm."""
    raise HumanInputRequired(question)


def test_agent_stores_name():
    agent = Agent(name="researcher")
    assert agent.name == "researcher"


async def test_a_model_less_agent_passes_its_state_through():
    agent = Agent(name="researcher")

    result = await agent.arun("topic")

    assert result.messages == [{"role": "user", "content": "topic"}]
    assert result.stop_reason is None


async def test_state_keys_an_agent_does_not_own_are_rejected():
    agent = Agent(name="researcher")

    with pytest.raises(ConfigurationError, match="unknown state keys: input"):
        await agent.arun({"input": "topic"})


async def test_returns_content_when_no_tool_call_needed():
    provider = ScriptedProvider([CompletionResponse(content="hello there")])
    agent = Agent(provider, name="assistant")

    state = await agent.arun({"messages": [{"role": "user", "content": "hi"}]})

    assert state.output == "hello there"
    assert state.messages[-1] == {"role": "assistant", "content": "hello there"}


async def test_prepends_system_prompt_once():
    provider = ScriptedProvider([CompletionResponse(content="hello")])
    agent = Agent(provider, name="assistant", system="be helpful")

    state = await agent.arun({"messages": []})

    assert state.messages[0] == {"role": "system", "content": "be helpful"}


async def test_dispatches_tool_calls_and_continues_loop():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})],
            ),
            CompletionResponse(content="the answer is 3"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[add])

    state = await agent.arun(
        {"messages": [{"role": "user", "content": "what is 1+2?"}]}
    )

    assert state.output == "the answer is 3"
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
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
    agent = Agent(provider, name="assistant", tools=[slow])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    # the shorter sleep finishes first if they ran concurrently
    assert order == ["slow-10", "slow-50"]
    # but result messages stay in call order, not completion order
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "name": "slow", "content": "slept 50ms"},
        {"role": "tool", "name": "slow", "content": "slept 10ms"},
    ]


def test_sync_run_returns_content_when_no_tool_call_needed():
    provider = ScriptedProvider([CompletionResponse(content="hello there")])
    agent = Agent(provider, name="assistant")

    state = agent.run({"messages": [{"role": "user", "content": "hi"}]})

    assert state.output == "hello there"
    assert state.messages[-1] == {"role": "assistant", "content": "hello there"}


def test_sync_run_dispatches_tool_calls():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})],
            ),
            CompletionResponse(content="the answer is 3"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[add])

    state = agent.run({"messages": [{"role": "user", "content": "what is 1+2?"}]})

    assert state.output == "the answer is 3"
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "name": "add", "content": "3"}]


def test_sync_run_raises_for_async_tool():
    @tool
    async def slow(ms: int) -> str:
        """An async tool, incompatible with the sync run() path."""
        return "never gets here"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="slow", arguments={"ms": 1})]
            )
        ]
    )
    agent = Agent(provider, name="assistant", tools=[slow])

    with pytest.raises(ConfigurationError, match="async"):
        agent.run({"messages": [{"role": "user", "content": "go"}]})


async def test_accumulates_usage_across_turns():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(name="noop", arguments={})],
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            ),
            CompletionResponse(
                content="done",
                usage=TokenUsage(
                    prompt_tokens=20, completion_tokens=8, total_tokens=28
                ),
            ),
        ]
    )

    @tool
    def noop() -> str:
        """Do nothing."""
        return "noop"

    agent = Agent(provider, name="assistant", tools=[noop])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    assert state.usage == TokenUsage(
        prompt_tokens=30, completion_tokens=13, total_tokens=43
    )
    assert agent.total_usage == TokenUsage(
        prompt_tokens=30, completion_tokens=13, total_tokens=43
    )


async def test_raises_when_token_budget_exceeded():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="hi",
                usage=TokenUsage(
                    prompt_tokens=50, completion_tokens=60, total_tokens=110
                ),
            )
        ]
    )
    agent = Agent(provider, name="assistant", budget=Budget(tokens=100))

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    assert exc_info.value.agent_name == "assistant"
    assert exc_info.value.budget == 100
    assert exc_info.value.usage.total_tokens == 110


async def test_token_budget_error_carries_partial_state():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="partial answer",
                usage=TokenUsage(
                    prompt_tokens=50, completion_tokens=60, total_tokens=110
                ),
            )
        ]
    )
    agent = Agent(provider, name="assistant", budget=Budget(tokens=100))

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    # the tokens were paid for, so the work must survive the exception
    state = exc_info.value.state
    assert state.stop_reason == "token_budget"
    assert state.output == "partial answer"
    assert state.messages == [{"role": "user", "content": "go"}]
    assert state.usage.total_tokens == 110


def test_sync_run_also_accumulates_usage():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="hi",
                usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ]
    )
    agent = Agent(provider, name="assistant")

    state = agent.run({"messages": [{"role": "user", "content": "hi"}]})

    assert state.usage == TokenUsage(
        prompt_tokens=3, completion_tokens=2, total_tokens=5
    )


async def test_stops_when_step_budget_is_spent():
    @tool
    def noop() -> str:
        """Do nothing."""
        return "noop"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="noop", arguments={})]
            ),
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="noop", arguments={})]
            ),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[noop], budget=Budget(steps=2))

    state = await agent.arun({"messages": [{"role": "user", "content": "loop"}]})

    assert len(provider.calls) == 2
    assert state.stop_reason == "step_budget"
    # not the last tool's return value - the agent never actually answered
    assert state.output == ""


def test_as_tool_builds_schema_from_agent_name_and_prompt():
    agent = Agent(name="researcher", system="Finds facts.")

    researcher_tool = agent.as_tool()

    spec = researcher_tool._tool_spec
    assert spec.name == "researcher"
    assert spec.description == "Finds facts."
    assert spec.parameters == {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }


def test_as_tool_overrides_name_and_description():
    agent = Agent(name="researcher", system="Finds facts.")

    researcher_tool = agent.as_tool(name="lookup", description="Look things up.")

    spec = researcher_tool._tool_spec
    assert spec.name == "lookup"
    assert spec.description == "Look things up."


async def test_as_tool_delegates_to_sub_agent():
    sub_provider = ScriptedProvider([CompletionResponse(content="Paris")])
    sub_agent = Agent(sub_provider, name="geo")

    main_provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(name="geo", arguments={"input": "capital of France?"})
                ],
            ),
            CompletionResponse(content="It's Paris."),
        ]
    )
    main_agent = Agent(main_provider, name="assistant", tools=[sub_agent.as_tool()])

    state = await main_agent.arun(
        {"messages": [{"role": "user", "content": "what's the capital of France?"}]}
    )

    assert state.output == "It's Paris."
    assert sub_provider.calls[0][-1] == {
        "role": "user",
        "content": "capital of France?",
    }
    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "name": "geo", "content": "Paris"}]


async def test_pauses_when_a_tool_asks_for_a_human():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="confirm",
                        arguments={"question": "ok to proceed?"},
                    )
                ],
            )
        ]
    )
    agent = Agent(provider, name="assistant", tools=[confirm])

    state = await agent.arun(
        {"messages": [{"role": "user", "content": "delete the logs"}]}
    )

    assert state.stop_reason == "paused"
    assert state.paused == [
        PendingHumanInput(call_id="call_1", name="confirm", question="ok to proceed?")
    ]
    assert len(provider.calls) == 1


async def test_resumes_after_human_answer():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="confirm",
                        arguments={"question": "ok to proceed?"},
                    )
                ],
            ),
            CompletionResponse(content="done, proceeded"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[confirm])

    state = await agent.arun(
        {"messages": [{"role": "user", "content": "delete the logs"}]}
    )
    pending = state.paused[0]
    state.messages.append(
        Message.tool("yes", name=pending.name, call_id=pending.call_id)
    )
    state = await agent.arun(state)

    assert state.output == "done, proceeded"
    assert state.stop_reason == "answer"
    assert state.paused == []


async def test_other_tool_calls_still_run_when_one_asks_for_a_human():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}),
                    ToolCall(
                        id="call_2", name="confirm", arguments={"question": "ok?"}
                    ),
                ],
            )
        ]
    )
    agent = Agent(provider, name="assistant", tools=[add, confirm])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "name": "add", "content": "3", "tool_call_id": "call_1"}
    ]
    assert state.paused == [
        PendingHumanInput(call_id="call_2", name="confirm", question="ok?")
    ]


async def test_failing_tool_is_reported_to_model_and_loop_continues():
    @tool
    def flaky() -> str:
        """Always fails."""
        raise ValueError("upstream is down")

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="flaky", arguments={})]
            ),
            CompletionResponse(content="I couldn't reach it, sorry"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[flaky])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert "upstream is down" in tool_messages[0]["content"]
    # the run survived and the model got a turn to recover
    assert state.output == "I couldn't reach it, sorry"
    assert state.stop_reason == "answer"
    assert len(provider.calls) == 2


async def test_one_failing_tool_does_not_discard_its_siblings_results():
    @tool
    def ok() -> str:
        """Succeeds."""
        return "fine"

    @tool
    def boom() -> str:
        """Fails."""
        raise RuntimeError("nope")

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(name="ok", arguments={}),
                    ToolCall(name="boom", arguments={}),
                ],
            ),
            CompletionResponse(content="done"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[ok, boom])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "fine"
    assert "nope" in tool_messages[1]["content"]
    assert state.stop_reason == "answer"


async def test_unknown_tool_name_is_reported_to_model():
    @tool
    def real() -> str:
        """A registered tool."""
        return "fine"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="imaginary", arguments={})]
            ),
            CompletionResponse(content="my mistake"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[real])

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert "imaginary" in tool_messages[0]["content"]
    assert state.output == "my mistake"


def test_sync_run_reports_failing_tool_to_model():
    @tool
    def flaky() -> str:
        """Always fails."""
        raise ValueError("upstream is down")

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="flaky", arguments={})]
            ),
            CompletionResponse(content="recovered"),
        ]
    )
    agent = Agent(provider, name="assistant", tools=[flaky])

    state = agent.run({"messages": [{"role": "user", "content": "go"}]})

    tool_messages = [m for m in state.messages if m["role"] == "tool"]
    assert "upstream is down" in tool_messages[0]["content"]
    assert state.output == "recovered"


def test_sync_run_pauses_when_tool_asks_human():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="confirm", arguments={"question": "ok?"})
                ],
            )
        ]
    )
    agent = Agent(provider, name="assistant", tools=[confirm])

    state = agent.run({"messages": [{"role": "user", "content": "delete the logs"}]})

    assert state.paused == [
        PendingHumanInput(call_id="call_1", name="confirm", question="ok?")
    ]
    assert len(provider.calls) == 1


async def test_raises_when_model_asks_for_tools_but_none_are_registered():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="ghost", arguments={})]
            )
        ]
    )
    agent = Agent(provider, name="assistant")

    with pytest.raises(ConfigurationError, match="no tools registered"):
        await agent.arun({"messages": [{"role": "user", "content": "hi"}]})


def test_tools_is_always_a_toolbox():
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert len(Agent(tools=[add]).tools) == 1
    assert not Agent().tools


def test_an_existing_toolbox_is_used_as_is():
    toolbox = Toolbox()
    agent = Agent(tools=toolbox)

    assert agent.tools is toolbox


def test_defaults_to_a_generic_name():
    assert Agent().name == "agent"


async def test_sync_tools_in_one_turn_run_concurrently():
    """A blocking tool must not stall the others gathered alongside it.

    The barrier is the assertion: it only releases once both tools are inside it
    at the same time, so a serialized dispatch raises BrokenBarrierError.
    """
    gate = threading.Barrier(2, timeout=5)

    @tool
    def first() -> str:
        """Wait for the second tool."""
        gate.wait()
        return "first"

    @tool
    def second() -> str:
        """Wait for the first tool."""
        gate.wait()
        return "second"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="first", arguments={}),
                    ToolCall(id="b", name="second", arguments={}),
                ],
            ),
            CompletionResponse(content="both done"),
        ]
    )
    agent = Agent(provider, tools=[first, second])

    state = await agent.arun("go")

    assert state.output == "both done"
    assert [m["content"] for m in state.messages if m["role"] == "tool"] == [
        "first",
        "second",
    ]


async def test_a_sync_tool_does_not_block_the_event_loop():
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    @tool
    def slow_blocking() -> str:
        """Block the calling thread for a moment."""
        time.sleep(0.1)
        return "done"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="a", name="slow_blocking", arguments={})],
            ),
            CompletionResponse(content="finished"),
        ]
    )
    agent = Agent(provider, tools=[slow_blocking])

    heartbeat = asyncio.create_task(ticker())
    state = await agent.arun("go")
    heartbeat.cancel()

    assert state.output == "finished"
    assert ticks > 0, "the event loop never got a turn while the tool ran"
