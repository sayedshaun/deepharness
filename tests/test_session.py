"""Saving a run to disk and picking it up again, pause and all."""

import pytest

from deepharness.agent import (
    Agent,
    AgentState,
    Message,
    PendingHumanInput,
    load_session,
    save_session,
    tool,
)
from deepharness.errors import ConfigurationError
from deepharness.providers.base import LLM, CompletionResponse, TokenUsage, ToolCall


class Scripted(LLM):
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)

    def generate(self, messages, *, tools=None):
        self.calls += 1
        return self.responses[self.calls - 1]


@tool(requires_approval=True)
def deploy(target: str) -> str:
    """Ship it, once a human says so."""
    return f"deployed to {target}"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "session.json"
    messages = [
        Message.system("be helpful").to_dict(),
        Message.human("hi").to_dict(),
        Message.ai("hello!").to_dict(),
    ]

    save_session(str(path), messages)
    loaded = load_session(str(path))

    assert loaded.messages == messages


def test_load_missing_session_returns_an_empty_state(tmp_path):
    path = tmp_path / "does-not-exist.json"

    assert load_session(str(path)) == AgentState()


def test_save_creates_human_readable_json(tmp_path):
    path = tmp_path / "session.json"

    save_session(str(path), [Message.human("hi")])

    assert '"role": "user"' in path.read_text()


def test_a_bare_message_array_still_loads(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('[{"role": "user", "content": "hi"}]')

    assert load_session(str(path)).messages == [{"role": "user", "content": "hi"}]


def test_saving_keeps_usage_and_stop_reason(tmp_path):
    path = tmp_path / "session.json"
    state = AgentState(
        messages=[Message.human("hi").to_dict()],
        output="hello",
        usage=TokenUsage(3, 4, 7),
        stop_reason="answer",
    )

    save_session(str(path), state)

    assert load_session(str(path)) == state


def test_structured_output_is_saved_as_plain_data(tmp_path):
    from dataclasses import dataclass

    @dataclass
    class Answer:
        city: str

    path = tmp_path / "session.json"
    save_session(str(path), AgentState(output=Answer(city="Oslo")))

    assert load_session(str(path)).output == {"city": "Oslo"}


def test_a_run_paused_on_an_approval_resumes_from_disk(tmp_path):
    path = tmp_path / "session.json"
    model = Scripted(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(name="deploy", arguments={"target": "prod"}, id="1")
                ],
            ),
            CompletionResponse(content="shipped"),
        ]
    )
    agent = Agent(model, tools=[deploy])

    paused = agent.run("deploy to prod")
    assert paused.stop_reason == "paused"
    save_session(str(path), paused)

    # A different process: the pause has to survive the file, not memory.
    resumed = load_session(str(path))
    assert resumed.paused == [
        PendingHumanInput(
            call_id="1",
            name="deploy",
            question="Run deploy with {'target': 'prod'}?",
            arguments={"target": "prod"},
        )
    ]

    state = agent.run(resumed.approve())

    assert state.answered
    assert "deployed to prod" in state.messages[-2]["content"]


def test_loading_a_file_with_an_unexpected_key_is_an_error(tmp_path):
    path = tmp_path / "session.json"
    path.write_text('{"messages": [], "temperature": 0.5}')

    with pytest.raises(ConfigurationError, match="unknown state keys"):
        load_session(str(path))
