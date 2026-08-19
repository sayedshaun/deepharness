from subagents.agent import Message, as_dict


def test_system_message():
    assert Message.system("be helpful").to_dict() == {
        "role": "system",
        "content": "be helpful",
    }


def test_human_message():
    assert Message.human("hi").to_dict() == {"role": "user", "content": "hi"}


def test_ai_message():
    assert Message.ai("hello there").to_dict() == {
        "role": "assistant",
        "content": "hello there",
    }


def test_ai_message_with_tool_calls():
    calls = [{"id": "1", "name": "add", "arguments": {"a": 1}}]

    assert Message.ai("", tool_calls=calls).to_dict() == {
        "role": "assistant",
        "content": "",
        "tool_calls": calls,
    }


def test_tool_message():
    assert Message.tool("3", name="add").to_dict() == {
        "role": "tool",
        "name": "add",
        "content": "3",
    }


def test_tool_message_links_the_call_id():
    assert Message.tool("3", name="add", call_id="call_1").to_dict() == {
        "role": "tool",
        "name": "add",
        "content": "3",
        "tool_call_id": "call_1",
    }


def test_unused_fields_are_left_out_of_the_wire_form():
    assert "name" not in Message.human("hi").to_dict()
    assert "tool_calls" not in Message.ai("hi").to_dict()


def test_a_message_is_not_a_mutable_mapping():
    message = Message.human("hi")

    assert not isinstance(message, dict)
    assert message.role == "user"
    assert not hasattr(message, "update")


def test_as_dict_passes_plain_dicts_through():
    assert as_dict({"role": "user", "content": "hi"}) == {
        "role": "user",
        "content": "hi",
    }
    assert as_dict(Message.human("hi")) == {"role": "user", "content": "hi"}
