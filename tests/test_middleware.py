"""Middleware: stepping into a run without forking the loop."""

import pytest

from deepharness.agent import (
    Agent,
    Message,
    Middleware,
    ToolFinished,
    tool,
)
from deepharness.errors import HumanInputRequired
from deepharness.providers.base import CompletionResponse, ToolCall
from deepharness.tools import Permissions, Rule

from .test_agent import ScriptedProvider

RAN: list[dict] = []


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()


@tool
def read_file(path: str) -> str:
    """Read a file."""
    RAN.append({"path": path})
    return f"contents of {path}"


@tool
def confirm(question: str) -> str:
    """Ask a human."""
    raise HumanInputRequired(question)


def call_turn(name="read_file", arguments=None, call_id="1"):
    return CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments={"path": "a.txt"} if arguments is None else arguments,
            )
        ],
    )


def a_run(*responses):
    return ScriptedProvider(list(responses))


def test_default_middleware_changes_nothing():
    provider = a_run(call_turn(), CompletionResponse("done"))
    agent = Agent(provider, tools=[read_file], middleware=Middleware())

    state = agent.run("read it")

    assert state.answered
    assert RAN == [{"path": "a.txt"}]


def test_an_agent_has_middleware_even_when_none_was_given():
    assert isinstance(Agent().middleware, Middleware)


def test_before_model_changes_what_is_sent_without_recording_it():
    class Reminding(Middleware):
        def before_model(self, messages):
            return [*messages, Message.human("Remember to cite paths.").to_dict()]

    provider = a_run(CompletionResponse("done"))
    agent = Agent(provider, middleware=Reminding())

    state = agent.run("go")

    assert provider.calls[0][-1]["content"] == "Remember to cite paths."
    assert not any("Remember" in m["content"] for m in state.messages)


def test_before_model_sees_every_turn():
    seen = []

    class Counting(Middleware):
        def before_model(self, messages):
            seen.append(len(messages))
            return messages

    agent = Agent(
        a_run(call_turn(), CompletionResponse("done")),
        tools=[read_file],
        middleware=Counting(),
    )

    agent.run("read it")

    assert len(seen) == 2


def test_before_tool_rewrites_the_arguments_a_call_runs_with():
    class Confining(Middleware):
        def before_tool(self, call):
            call.arguments["path"] = call.arguments["path"].removeprefix("/")
            return call

    agent = Agent(
        a_run(call_turn(arguments={"path": "/etc/passwd"}), CompletionResponse("done")),
        tools=[read_file],
        middleware=Confining(),
    )

    agent.run("read it")

    assert RAN == [{"path": "etc/passwd"}]


def test_before_tool_can_refuse_a_call_and_the_model_is_told():
    class Refusing(Middleware):
        def before_tool(self, call):
            return None

    provider = a_run(call_turn(), CompletionResponse("understood"))
    agent = Agent(provider, tools=[read_file], middleware=Refusing())

    state = agent.run("read it")

    assert RAN == []
    assert state.answered
    assert any("Refused before running" in m["content"] for m in state.messages)


def test_a_rewritten_call_is_what_the_policy_rules_on():
    """Middleware must not be able to rewrite a call past a deny rule."""

    class Escalating(Middleware):
        def before_tool(self, call):
            call.arguments["path"] = "/etc/shadow"
            return call

    provider = a_run(call_turn(), CompletionResponse("understood"))
    agent = Agent(
        provider,
        tools=[read_file],
        permissions=Permissions(deny=[Rule("read_file", {"path": "/etc/*"})]),
        middleware=Escalating(),
    )

    state = agent.run("read it")

    assert RAN == []
    assert any("Denied by policy" in m["content"] for m in state.messages)


def test_after_tool_changes_what_the_model_and_the_caller_both_see():
    class Redacting(Middleware):
        def after_tool(self, call, result):
            return "[redacted]"

    agent = Agent(
        a_run(call_turn(), CompletionResponse("done")),
        tools=[read_file],
        middleware=Redacting(),
    )

    events = list(agent.stream_events("read it"))
    state = events[-1].state
    finished = next(e for e in events if isinstance(e, ToolFinished))

    assert finished.result == "[redacted]"
    recorded = next(m for m in state.messages if m["role"] == "tool")
    assert recorded["content"] == "[redacted]"


def test_after_tool_sees_a_failure_as_a_value():
    seen = []

    @tool
    def explode() -> str:
        """Fail."""
        raise ValueError("no")

    class Noting(Middleware):
        def after_tool(self, call, result):
            seen.append(type(result))
            return "handled"

    agent = Agent(
        a_run(call_turn(name="explode", arguments={}), CompletionResponse("done")),
        tools=[explode],
        middleware=Noting(),
    )

    state = agent.run("go")

    assert seen == [ValueError]
    assert any(m["content"] == "handled" for m in state.messages)


def test_after_tool_leaves_a_question_alone():
    class Rewriting(Middleware):
        def after_tool(self, call, result):
            return "answered"

    agent = Agent(
        a_run(call_turn(name="confirm", arguments={"question": "ok?"})),
        tools=[confirm],
        middleware=Rewriting(),
    )

    state = agent.run("go")

    assert state.stop_reason == "paused"
    assert state.paused[0].question == "ok?"


def test_after_step_can_stop_the_run():
    class Once(Middleware):
        def after_step(self, step, state):
            return step < 1

    provider = a_run(call_turn(), call_turn(), CompletionResponse("done"))
    agent = Agent(provider, tools=[read_file], middleware=Once())

    state = agent.run("read it twice")

    assert state.stop_reason == "stopped"
    assert not state.answered
    assert len(RAN) == 1


def test_after_step_gets_the_run_so_far():
    snapshots = []

    class Watching(Middleware):
        def after_step(self, step, state):
            snapshots.append((step, len(state.messages), state.stop_reason))
            return True

    agent = Agent(
        a_run(call_turn(), CompletionResponse("done")),
        tools=[read_file],
        middleware=Watching(),
    )

    agent.run("read it")

    assert snapshots == [(1, 3, None)]


def test_after_step_is_not_called_once_the_model_answers():
    class Loud(Middleware):
        def __init__(self):
            self.calls = 0

        def after_step(self, step, state):
            self.calls += 1
            return True

    middleware = Loud()
    Agent(a_run(CompletionResponse("done")), middleware=middleware).run("go")

    assert middleware.calls == 0


@pytest.mark.asyncio
async def test_middleware_works_the_same_on_the_async_path():
    class Redacting(Middleware):
        def before_tool(self, call):
            call.arguments["path"] = "safe.txt"
            return call

        def after_tool(self, call, result):
            return result.upper()

    agent = Agent(
        a_run(call_turn(), CompletionResponse("done")),
        tools=[read_file],
        middleware=Redacting(),
    )

    state = await agent.arun("read it")

    assert RAN == [{"path": "safe.txt"}]
    assert any(m["content"] == "CONTENTS OF SAFE.TXT" for m in state.messages)
