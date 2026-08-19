import pytest

from subagents.agent import AgentState, Message
from subagents.errors import ConfigurationError
from subagents.providers.base import TokenUsage


def test_a_fresh_state_is_empty_and_unanswered():
    state = AgentState()

    assert state.messages == []
    assert state.output is None
    assert state.paused == []
    assert state.stop_reason is None
    assert not state.answered
    assert state.usage == TokenUsage(0, 0, 0)


def test_each_state_gets_its_own_usage_counter():
    assert AgentState().usage is not AgentState().usage


@pytest.mark.parametrize(
    "stop_reason, answered",
    [("answer", True), ("step_budget", False), ("paused", False), (None, False)],
)
def test_only_an_answer_counts_as_answered(stop_reason, answered):
    assert AgentState(stop_reason=stop_reason).answered is answered


def test_a_string_becomes_one_user_message():
    assert AgentState.of("hi").messages == [{"role": "user", "content": "hi"}]


def test_a_list_of_messages_is_normalized_to_wire_form():
    state = AgentState.of([Message.human("hi"), {"role": "assistant", "content": "yo"}])

    assert state.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_none_becomes_an_empty_state():
    assert AgentState.of(None) == AgentState()


def test_an_existing_state_is_passed_through_unchanged():
    state = AgentState.of("hi")

    assert AgentState.of(state) is state


def test_a_dict_of_known_fields_still_works():
    state = AgentState.of({"messages": [{"role": "user", "content": "hi"}]})

    assert state.messages == [{"role": "user", "content": "hi"}]


def test_unknown_dict_keys_are_an_error_rather_than_dropped():
    with pytest.raises(ConfigurationError, match="unknown state keys: topic, user_id"):
        AgentState.of({"messages": [], "topic": "x", "user_id": 1})


def test_something_that_is_not_a_transcript_is_rejected():
    with pytest.raises(ConfigurationError, match="cannot build agent state from int"):
        AgentState.of(42)
