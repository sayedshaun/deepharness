from subagents.agent import Message


def test_system_message():
    assert Message.system("be helpful") == {"role": "system", "content": "be helpful"}


def test_human_message():
    assert Message.human("hi") == {"role": "user", "content": "hi"}


def test_ai_message():
    assert Message.ai("hello there") == {"role": "assistant", "content": "hello there"}


def test_tool_message():
    assert Message.tool("3", name="add") == {"role": "tool", "name": "add", "content": "3"}


def test_message_is_a_dict():
    message = Message.human("hi")

    assert isinstance(message, dict)
    assert message["role"] == "user"
