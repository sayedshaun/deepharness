from subagents.agent import Agent, Toolbox, tool
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
        raise NotImplementedError("ScriptedProvider only supports agenerate() in tests")

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

    result = await agent.run(state)

    assert result == state


async def test_returns_content_when_no_tool_call_needed():
    provider = ScriptedProvider([CompletionResponse(content="hello there")])
    agent = Agent("assistant", provider)

    state = await agent.run({"messages": [{"role": "user", "content": "hi"}]})

    assert state["output"] == "hello there"
    assert state["messages"][-1] == {"role": "assistant", "content": "hello there"}


async def test_prepends_system_prompt_once():
    provider = ScriptedProvider([CompletionResponse(content="hello")])
    agent = Agent("assistant", provider, system_prompt="be helpful")

    state = await agent.run({"messages": []})

    assert state["messages"][0] == {"role": "system", "content": "be helpful"}


async def test_dispatches_tool_calls_and_continues_loop():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    toolbox = Toolbox()
    toolbox.register(add)

    provider = ScriptedProvider(
        [
            CompletionResponse(content="", tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})]),
            CompletionResponse(content="the answer is 3"),
        ]
    )
    agent = Agent("assistant", provider, toolbox=toolbox)

    state = await agent.run({"messages": [{"role": "user", "content": "what is 1+2?"}]})

    assert state["output"] == "the answer is 3"
    tool_messages = [m for m in state["messages"] if m["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "name": "add", "content": "3"}]
    assert len(provider.calls) == 2


async def test_stops_after_max_steps():
    @tool
    def noop() -> str:
        """Do nothing."""
        return "noop"

    toolbox = Toolbox()
    toolbox.register(noop)

    provider = ScriptedProvider(
        [
            CompletionResponse(content="", tool_calls=[ToolCall(name="noop", arguments={})]),
            CompletionResponse(content="", tool_calls=[ToolCall(name="noop", arguments={})]),
        ]
    )
    agent = Agent("assistant", provider, toolbox=toolbox, max_steps=2)

    state = await agent.run({"messages": [{"role": "user", "content": "loop"}]})

    assert len(provider.calls) == 2
    assert state["output"] == "noop"
