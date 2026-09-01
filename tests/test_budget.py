import pytest

from deepharness.agent import Agent, Budget, tool
from deepharness.errors import ConfigurationError
from deepharness.providers.base import CompletionResponse, ToolCall

from .test_agent import ScriptedProvider


def test_defaults_bound_steps_but_not_tokens():
    budget = Budget()

    assert budget.steps == 10
    assert budget.tokens is None


def test_agent_without_a_budget_gets_the_default():
    assert Agent(name="assistant").budget == Budget()


def test_budget_is_frozen_so_agents_can_share_one():
    budget = Budget(steps=3)

    with pytest.raises(AttributeError):
        budget.steps = 99


@pytest.mark.parametrize("steps", [0, -1])
def test_rejects_non_positive_steps(steps):
    with pytest.raises(ConfigurationError, match="steps must be at least 1"):
        Budget(steps=steps)


def test_rejects_non_positive_tokens_when_set():
    with pytest.raises(ConfigurationError, match="tokens must be at least 1"):
        Budget(tokens=0)


async def test_single_shot_budget_makes_exactly_one_model_call():
    provider = ScriptedProvider([CompletionResponse(content="42")])
    agent = Agent(provider, name="assistant", budget=Budget(steps=1))

    state = await agent.arun({"messages": [{"role": "user", "content": "2+2?"}]})

    assert len(provider.calls) == 1
    assert state.stop_reason == "answer"
    assert state.output == "42"


async def test_single_shot_agent_that_calls_a_tool_stops_without_answering():
    """The documented trade-off: steps=1 leaves no turn to react to a tool."""

    @tool
    def noop() -> str:
        """Do nothing."""
        return "noop"

    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="noop", arguments={})]
            )
        ]
    )
    agent = Agent(provider, name="assistant", tools=[noop], budget=Budget(steps=1))

    state = await agent.arun({"messages": [{"role": "user", "content": "go"}]})

    assert state.stop_reason == "step_budget"
    assert state.output == ""
