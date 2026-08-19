from subagents.agent import Message, load_session, save_session


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "session.json"
    messages = [
        Message.system("be helpful").to_dict(),
        Message.human("hi").to_dict(),
        Message.ai("hello!").to_dict(),
    ]

    save_session(str(path), messages)
    loaded = load_session(str(path))

    assert loaded == messages


def test_load_missing_session_returns_empty_list(tmp_path):
    path = tmp_path / "does-not-exist.json"

    assert load_session(str(path)) == []


def test_save_creates_human_readable_json(tmp_path):
    path = tmp_path / "session.json"

    save_session(str(path), [Message.human("hi")])

    assert '"role": "user"' in path.read_text()
