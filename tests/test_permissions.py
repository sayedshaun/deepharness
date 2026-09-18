"""Deciding per call what runs, what waits for a human, and what is refused."""

import pytest

from deepharness.agent import Agent, tool
from deepharness.errors import ConfigurationError
from deepharness.providers.base import CompletionResponse, ToolCall
from deepharness.tools import Permissions, Rule

from .test_agent import ScriptedProvider

RAN: list[str] = []


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


@tool
def read_file(path: str) -> str:
    """Read a file."""
    RAN.append(f"read_file:{path}")
    return "contents"


@tool
def run_command(command: str) -> str:
    """Run a command."""
    RAN.append(f"run_command:{command}")
    return "output"


@tool(requires_approval=True)
def deploy() -> str:
    """Ship it."""
    RAN.append("deploy")
    return "shipped"


def call_turn(*calls):
    return CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id=str(index), name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(calls)
        ],
    )


def test_a_rule_without_arguments_matches_any_call():
    assert Rule("read_file").matches("read_file", {"path": "anything"})


def test_a_rule_narrows_on_an_argument_pattern():
    rule = Rule("run_command", {"command": "git log*"})

    assert rule.matches("run_command", {"command": "git log --oneline"})
    assert not rule.matches("run_command", {"command": "git push"})


def test_a_rule_does_not_match_an_argument_the_call_omitted():
    assert not Rule("run_command", {"command": "*"}).matches("run_command", {})


def test_a_tool_name_may_be_a_pattern():
    assert Rule("read_*").matches("read_file", {})


def test_an_empty_rule_is_refused():
    with pytest.raises(ConfigurationError):
        Rule("")


def test_deny_beats_allow_beats_ask():
    permissions = Permissions(
        deny=[Rule("run_command", {"command": "*rm -rf*"})],
        allow=[Rule("run_command", {"command": "git *"})],
        ask=["run_command"],
    )

    assert permissions.decide("run_command", {"command": "rm -rf /"}) == "deny"
    assert permissions.decide("run_command", {"command": "git status"}) == "allow"
    assert permissions.decide("run_command", {"command": "curl x"}) == "ask"


def test_no_rule_means_no_opinion():
    assert Permissions(allow=["read_file"]).decide("write_file", {}) is None


def test_an_allowed_call_runs_without_pausing():
    provider = ScriptedProvider(
        [call_turn(("read_file", {"path": "a.txt"})), CompletionResponse("done")]
    )
    agent = Agent(
        provider, tools=[read_file], permissions=Permissions(allow=["read_file"])
    )

    state = agent.run("read a.txt")

    assert state.answered
    assert RAN == ["read_file:a.txt"]


def test_a_denied_call_never_runs_and_the_model_is_told():
    provider = ScriptedProvider(
        [
            call_turn(("run_command", {"command": "rm -rf /"})),
            CompletionResponse("I cannot do that"),
        ]
    )
    agent = Agent(
        provider,
        tools=[run_command],
        permissions=Permissions(deny=[Rule("run_command", {"command": "*rm -rf*"})]),
    )

    state = agent.run("wipe the disk")

    assert RAN == []
    assert state.answered
    assert any("Denied by policy" in m["content"] for m in state.messages)


def test_a_turn_of_only_denials_goes_back_to_the_model():
    provider = ScriptedProvider(
        [
            call_turn(("run_command", {"command": "rm -rf /"})),
            call_turn(("read_file", {"path": "a.txt"})),
            CompletionResponse("done"),
        ]
    )
    agent = Agent(
        provider,
        tools=[run_command, read_file],
        permissions=Permissions(deny=["run_command"]),
    )

    state = agent.run("wipe it, then read a.txt")

    assert RAN == ["read_file:a.txt"]
    assert state.answered


def test_a_policy_allow_overrides_a_tool_that_asks_by_default():
    provider = ScriptedProvider([call_turn(("deploy", {})), CompletionResponse("done")])
    agent = Agent(provider, tools=[deploy], permissions=Permissions(allow=["deploy"]))

    state = agent.run("deploy")

    assert RAN == ["deploy"]
    assert state.answered


def test_a_policy_ask_gates_a_tool_that_would_otherwise_run():
    provider = ScriptedProvider(
        [call_turn(("read_file", {"path": "a.txt"})), CompletionResponse("done")]
    )
    agent = Agent(
        provider, tools=[read_file], permissions=Permissions(ask=["read_file"])
    )

    state = agent.run("read a.txt")

    assert state.stop_reason == "paused"
    assert RAN == []

    state = agent.run(state.approve())

    assert RAN == ["read_file:a.txt"]
    assert state.answered


def test_the_tool_flag_still_applies_where_no_rule_does():
    provider = ScriptedProvider([call_turn(("deploy", {}))])
    agent = Agent(
        provider, tools=[deploy], permissions=Permissions(allow=["read_file"])
    )

    assert agent.run("deploy").stop_reason == "paused"
