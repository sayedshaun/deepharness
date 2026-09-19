"""Bounding what a long run sends: tool-result truncation and pruning."""

import pytest

from deepharness.agent import Agent, ContextPolicy, Message, estimate_tokens
from deepharness.agent.context import truncate
from deepharness.errors import ConfigurationError
from deepharness.providers.base import LLM, CompletionResponse, ToolCall
from deepharness.tools.toolbox import tool


class Recorder(LLM):
    """Replays scripted responses and keeps every transcript it was sent."""

    def __init__(self, responses):
        self.responses = responses
        self.sent = []

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)

    def generate(self, messages, *, tools=None):
        self.sent.append([dict(message) for message in messages])
        return self.responses[len(self.sent) - 1]


@tool
def dump() -> str:
    """Return far more text than a policy will allow through."""
    return "y" * 5000


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"
    assert truncate("short", None) == "short"


def test_truncate_keeps_both_ends_and_says_what_went_missing():
    result = truncate("a" * 50 + "b" * 50, 30)

    assert result.startswith("a" * 20)
    assert result.endswith("b" * 10)
    assert "[70 characters elided]" in result


def test_estimate_tokens_counts_content_and_tool_calls():
    messages = [
        Message.human("a" * 40).to_dict(),
        Message.ai(
            "", tool_calls=[{"name": "search", "arguments": "b" * 36}]
        ).to_dict(),
    ]

    assert estimate_tokens(messages) == 20


@pytest.mark.parametrize(
    "kwargs",
    [{"max_tokens": 0}, {"tool_result_chars": 0}, {"keep_last": 0}],
)
def test_policy_rejects_meaningless_limits(kwargs):
    with pytest.raises(ConfigurationError):
        ContextPolicy(**kwargs)


def _transcript(turns: int) -> list[dict]:
    messages = [Message.system("You are terse.").to_dict()]
    for index in range(turns):
        messages.append(Message.human(f"question {index} " + "q" * 100).to_dict())
        messages.append(Message.ai(f"answer {index} " + "a" * 100).to_dict())
    return messages


def test_prune_returns_the_transcript_untouched_when_it_fits():
    messages = _transcript(2)

    assert ContextPolicy(max_tokens=10_000).prune(messages) is messages
    assert ContextPolicy().prune(messages) is messages


def test_prune_drops_oldest_turns_and_keeps_the_system_prompt():
    messages = _transcript(10)

    pruned = ContextPolicy(max_tokens=150, keep_last=4).prune(messages)

    assert len(pruned) < len(messages)
    assert pruned[0] == messages[0]
    assert "elided to fit the context budget" in pruned[1]["content"]
    assert pruned[-4:] == messages[-4:]
    assert estimate_tokens(pruned) <= 150


def test_prune_never_orphans_a_tool_result():
    messages = [
        Message.human("start " + "s" * 400).to_dict(),
        Message.ai(
            "", tool_calls=[{"id": "1", "name": "dump", "arguments": {}}]
        ).to_dict(),
        Message.tool("d" * 400, name="dump", call_id="1").to_dict(),
        Message.human("again " + "g" * 400).to_dict(),
        Message.ai("done").to_dict(),
    ]

    pruned = ContextPolicy(max_tokens=50, keep_last=1).prune(messages)

    roles = [message["role"] for message in pruned]
    assert "tool" not in roles or roles.index("tool") > roles.index("assistant")


def test_prune_keeps_the_tail_even_when_it_cannot_fit():
    messages = _transcript(3)

    pruned = ContextPolicy(max_tokens=1, keep_last=2).prune(messages)

    assert pruned[-2:] == messages[-2:]
    assert estimate_tokens(pruned) > 1


def test_a_run_records_a_truncated_tool_result():
    model = Recorder(
        [
            CompletionResponse(
                content="", tool_calls=[ToolCall(name="dump", arguments={}, id="1")]
            ),
            CompletionResponse(content="done"),
        ]
    )
    agent = Agent(model, tools=[dump], context=ContextPolicy(tool_result_chars=100))

    state = agent.run("dump it")

    recorded = next(m for m in state.messages if m["role"] == "tool")
    assert len(recorded["content"]) < 5000
    assert "characters elided" in recorded["content"]


def test_a_run_sends_a_pruned_view_but_keeps_the_whole_transcript():
    model = Recorder([CompletionResponse(content="done")])
    agent = Agent(model, context=ContextPolicy(max_tokens=20, keep_last=1))

    state = agent.run(_transcript(5)[1:])

    assert len(model.sent[0]) < len(state.messages)
    assert len(state.messages) == len(_transcript(5)[1:]) + 1
