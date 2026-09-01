import enum
from dataclasses import dataclass, field
from typing import Literal

import pytest

from deepharness.agent import Agent, Budget
from deepharness.agent.output import FINAL_TOOL, coerce, final_tool_schema
from deepharness.errors import ConfigurationError, OutputValidationError
from deepharness.providers.base import CompletionResponse, ToolCall

from .test_agent import ScriptedProvider


@dataclass
class Weather:
    city: str
    celsius: int


def final(**arguments):
    return CompletionResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name=FINAL_TOOL, arguments=arguments)],
    )


def test_output_must_be_a_dataclass():
    class NotADataclass:
        pass

    with pytest.raises(ConfigurationError, match="must be a dataclass"):
        Agent(output=NotADataclass)


def test_the_final_tool_describes_the_dataclass():
    schemas = Agent(output=Weather)._schemas()

    assert [schema["name"] for schema in schemas] == [FINAL_TOOL]
    assert schemas[0]["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}, "celsius": {"type": "integer"}},
        "required": ["city", "celsius"],
    }


def test_defaulted_fields_are_not_required():
    @dataclass
    class Report:
        title: str
        tags: list[str] = field(default_factory=list)
        score: int = 0

    assert final_tool_schema(Report)["parameters"]["required"] == ["title"]


def test_no_final_tool_without_output():
    assert Agent()._schemas() is None


async def test_returns_a_validated_instance():
    provider = ScriptedProvider([final(city="Oslo", celsius=22)])
    agent = Agent(provider, output=Weather)

    state = await agent.arun({"messages": [{"role": "user", "content": "Oslo?"}]})

    assert state.output == Weather(city="Oslo", celsius=22)
    assert state.stop_reason == "answer"


async def test_invalid_fields_are_handed_back_to_the_model_to_retry():
    provider = ScriptedProvider(
        [final(city="Oslo", celsius="warm"), final(city="Oslo", celsius=22)]
    )
    agent = Agent(provider, output=Weather)

    state = await agent.arun({"messages": [{"role": "user", "content": "Oslo?"}]})

    assert state.output == Weather(city="Oslo", celsius=22)
    assert len(provider.calls) == 2
    assert any(
        message.get("role") == "tool" and "celsius" in message["content"]
        for message in state.messages
    )


async def test_prose_is_not_an_answer_when_output_is_requested():
    provider = ScriptedProvider(
        [
            CompletionResponse(content="It is 22 degrees."),
            final(city="Oslo", celsius=22),
        ]
    )
    agent = Agent(provider, output=Weather)

    state = await agent.arun({"messages": [{"role": "user", "content": "Oslo?"}]})

    assert state.output == Weather(city="Oslo", celsius=22)
    assert any(FINAL_TOOL in str(message.get("content")) for message in state.messages)


async def test_a_model_that_never_calls_the_final_tool_runs_out_of_steps():
    provider = ScriptedProvider([CompletionResponse(content="prose")] * 2)
    agent = Agent(provider, output=Weather, budget=Budget(steps=2))

    state = await agent.arun({"messages": [{"role": "user", "content": "Oslo?"}]})

    assert state.stop_reason == "step_budget"


def test_missing_and_wrong_fields_are_reported_together():
    with pytest.raises(OutputValidationError) as excinfo:
        coerce(Weather, {"celsius": "warm"})

    message = str(excinfo.value)
    assert "city: missing" in message
    assert "celsius: expected int, got str" in message


def test_nested_dataclasses_are_built_recursively():
    @dataclass
    class Point:
        x: int
        y: int

    @dataclass
    class Line:
        start: Point
        end: Point

    line = coerce(Line, {"start": {"x": 1, "y": 2}, "end": {"x": 3, "y": 4}})

    assert line == Line(start=Point(1, 2), end=Point(3, 4))
    assert "$ref" not in str(final_tool_schema(Line))


def test_lists_coerce_their_items():
    @dataclass
    class Team:
        members: list[str]

    assert coerce(Team, {"members": ["a", "b"]}) == Team(members=["a", "b"])

    with pytest.raises(OutputValidationError, match="expected str, got int"):
        coerce(Team, {"members": ["a", 2]})


def test_literal_and_enum_values_are_checked():
    class Level(enum.Enum):
        LOW = "low"
        HIGH = "high"

    @dataclass
    class Rating:
        confidence: Literal["low", "high"]
        level: Level

    assert coerce(Rating, {"confidence": "low", "level": "high"}) == Rating(
        confidence="low", level=Level.HIGH
    )

    with pytest.raises(OutputValidationError, match="confidence"):
        coerce(Rating, {"confidence": "medium", "level": "low"})


def test_optional_fields_accept_null():
    @dataclass
    class Maybe:
        note: str | None

    assert coerce(Maybe, {"note": None}) == Maybe(note=None)
    assert coerce(Maybe, {"note": "hi"}) == Maybe(note="hi")


def test_an_int_is_accepted_where_a_float_is_declared():
    @dataclass
    class Score:
        value: float

    assert coerce(Score, {"value": 22}) == Score(value=22.0)


def test_a_bool_is_not_an_integer():
    @dataclass
    class Flagged:
        count: int

    with pytest.raises(OutputValidationError, match="count"):
        coerce(Flagged, {"count": True})


def test_unknown_keys_are_ignored():
    assert coerce(Weather, {"city": "Oslo", "celsius": 22, "chatter": "!"}) == Weather(
        city="Oslo", celsius=22
    )
