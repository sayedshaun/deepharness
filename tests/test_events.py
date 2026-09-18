"""What streaming a run reports besides prose: steps and tool lifecycle."""

import pytest

from deepharness.agent import (
    Agent,
    ContextPolicy,
    Finished,
    StepStarted,
    ToolFinished,
    ToolStarted,
)
from deepharness.errors import HumanInputRequired
from deepharness.providers.base import LLM, CompletionResponse, TextDelta, ToolCall
from deepharness.tools.toolbox import tool


class Scripted(LLM):
    """Replays scripted responses, one per model call."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)

    def generate(self, messages, *, tools=None):
        self.calls += 1
        return self.responses[self.calls - 1]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def explode() -> str:
    """Fail, so the model gets a turn to react."""
    raise ValueError("no")


@tool
def confirm(question: str) -> str:
    """Ask a human before going on."""
    raise HumanInputRequired(question)


def tool_turn(name, arguments, call_id="1"):
    return CompletionResponse(
        content="", tool_calls=[ToolCall(name=name, arguments=arguments, id=call_id)]
    )


def a_tool_run():
    return Scripted([tool_turn("add", {"a": 2, "b": 3}), CompletionResponse("5")])


def test_stream_events_reports_steps_and_the_tool_it_ran():
    events = list(Agent(a_tool_run(), tools=[add]).stream_events("2+3?"))

    assert [type(event) for event in events] == [
        StepStarted,
        ToolStarted,
        ToolFinished,
        StepStarted,
        TextDelta,
        Finished,
    ]
    assert [e.step for e in events if isinstance(e, StepStarted)] == [1, 2]
    started = events[1]
    assert (started.name, started.arguments, started.call_id) == (
        "add",
        {"a": 2, "b": 3},
        "1",
    )
    assert events[2] == ToolFinished("add", "5", False, "1")


@pytest.mark.asyncio
async def test_astream_events_reports_the_same_sequence():
    events = [
        event async for event in Agent(a_tool_run(), tools=[add]).astream_events("2+3?")
    ]

    assert [type(event) for event in events] == [
        StepStarted,
        ToolStarted,
        ToolFinished,
        StepStarted,
        TextDelta,
        Finished,
    ]


def test_a_failing_tool_is_reported_as_failed_without_ending_the_run():
    model = Scripted([tool_turn("explode", {}), CompletionResponse("it failed")])

    events = list(Agent(model, tools=[explode]).stream_events("go"))
    finished = next(event for event in events if isinstance(event, ToolFinished))

    assert finished.failed
    assert "no" in finished.result
    assert events[-1].state.answered


def test_a_tool_result_is_reported_as_the_model_will_read_it():
    @tool
    def dump() -> str:
        """Return more than the policy allows."""
        return "y" * 500

    model = Scripted([tool_turn("dump", {}), CompletionResponse("done")])
    agent = Agent(model, tools=[dump], context=ContextPolicy(tool_result_chars=50))

    events = list(agent.stream_events("go"))
    finished = next(event for event in events if isinstance(event, ToolFinished))
    transcript = events[-1].state.messages
    recorded = next(m for m in transcript if m["role"] == "tool")

    assert finished.result == recorded["content"]


def test_a_tool_asking_a_human_reports_no_outcome():
    model = Scripted([tool_turn("confirm", {"question": "ok?"})])

    events = list(Agent(model, tools=[confirm]).stream_events("go"))

    assert not any(isinstance(event, ToolFinished) for event in events)
    assert isinstance(events[1], ToolStarted)
    assert events[-1].state.stop_reason == "paused"


def test_a_modelless_agent_emits_only_its_final_state():
    events = list(Agent().stream_events("hello"))

    assert [type(event) for event in events] == [Finished]


def test_streaming_text_alone_skips_the_progress_events():
    assert list(Agent(a_tool_run(), tools=[add]).stream("2+3?")) == ["5"]
