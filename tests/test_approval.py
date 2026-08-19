"""requires_approval: a gated call defers until a human allows it."""

import pytest

from subagents.agent import Agent, tool
from subagents.errors import ConfigurationError
from subagents.providers.base import CompletionResponse, ToolCall

from .test_agent import ScriptedProvider

SENT: list[int] = []


@tool(requires_approval=True)
def wire_transfer(amount_usd: int) -> str:
    """Send money."""
    SENT.append(amount_usd)
    return f"sent ${amount_usd}"


@tool
def balance() -> str:
    """Report the balance."""
    return "$1,000,000"


def transfer_turn(amount=50_000, call_id="t1"):
    return CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="wire_transfer", arguments={"amount_usd": amount})
        ],
    )


@pytest.fixture(autouse=True)
def _clear():
    SENT.clear()


def test_the_decorator_records_the_gate_on_the_tool():
    assert wire_transfer._tool_spec.requires_approval is True
    assert balance._tool_spec.requires_approval is False


async def test_a_gated_call_pauses_without_running():
    provider = ScriptedProvider([transfer_turn()])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")

    assert state.stop_reason == "paused"
    assert SENT == [], "the tool must not have run before approval"
    pending = state.paused[0]
    assert pending.name == "wire_transfer"
    assert pending.needs_approval
    assert pending.arguments == {"amount_usd": 50_000}


async def test_approving_actually_runs_the_call():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")
    state = await agent.arun(state.approve())

    assert SENT == [50_000], "approval must run the gated tool"
    assert state.output == "Sent."
    assert state.answered
    assert any(
        m["role"] == "tool" and m["content"] == "sent $50000" for m in state.messages
    )


async def test_rejecting_tells_the_model_instead_of_running():
    provider = ScriptedProvider(
        [transfer_turn(), CompletionResponse(content="Understood, cancelled.")]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")
    state = await agent.arun(state.reject())

    assert SENT == []
    assert state.output == "Understood, cancelled."
    assert any(
        m["role"] == "tool" and m["content"] == "Denied by the user."
        for m in state.messages
    )


async def test_the_arguments_the_model_sent_are_the_ones_that_run():
    provider = ScriptedProvider(
        [transfer_turn(amount=25), CompletionResponse(content="done")]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay $25")
    await agent.arun(state.approve())

    assert SENT == [25]


async def test_resuming_without_deciding_is_an_error():
    provider = ScriptedProvider([transfer_turn()])
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay Acme $50,000")

    with pytest.raises(ConfigurationError, match="state.approve"):
        await agent.arun(state)


def test_approving_a_call_that_is_not_paused_is_an_error():
    from subagents.agent import AgentState

    with pytest.raises(ConfigurationError, match="no paused call"):
        AgentState().approve()


async def test_one_call_can_be_approved_by_id_and_another_rejected():
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="wire_transfer", arguments={"amount_usd": 1}),
                    ToolCall(id="b", name="wire_transfer", arguments={"amount_usd": 2}),
                ],
            ),
            CompletionResponse(content="one sent, one cancelled"),
        ]
    )
    agent = Agent(provider, tools=[wire_transfer])

    state = await agent.arun("pay both")
    assert len(state.paused) == 2

    state.approve("a")
    state.reject("b")
    state = await agent.arun(state)

    assert SENT == [1]
    assert any(m["content"] == "Denied by the user." for m in state.messages)
    assert any(m["content"] == "sent $1" for m in state.messages)


async def test_an_ungated_tool_in_the_same_turn_waits_for_the_ruling():
    """A turn is not half-applied while a human is deciding."""
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="a", name="balance", arguments={}),
                    ToolCall(id="b", name="wire_transfer", arguments={"amount_usd": 5}),
                ],
            ),
            CompletionResponse(content="done"),
        ]
    )
    agent = Agent(provider, tools=[balance, wire_transfer])

    state = await agent.arun("check then pay")

    assert state.stop_reason == "paused"
    assert not any(m["role"] == "tool" for m in state.messages)


def test_approval_works_on_the_sync_path_too():
    provider = ScriptedProvider([transfer_turn(), CompletionResponse(content="Sent.")])
    agent = Agent(provider, tools=[wire_transfer])

    state = agent.run("pay Acme $50,000")
    state = agent.run(state.approve())

    assert SENT == [50_000]
    assert state.output == "Sent."
