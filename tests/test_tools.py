import pytest

from subagents.agent import Toolbox
from subagents.tools.prebuilt.coding import (
    CodingToolbox,
    apply_patch,
    grep,
    read_file,
    run_shell,
    set_auto_approve,
    write_file,
)


@pytest.fixture(autouse=True)
def _auto_approve():
    set_auto_approve(True)
    yield
    set_auto_approve(False)


def test_read_write_roundtrip(tmp_path):
    path = tmp_path / "hello.txt"

    result = write_file(path=str(path), content="hello world")

    assert "hello.txt" in result
    assert read_file(path=str(path)) == "hello world"


def test_apply_patch_replaces_unique_match(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("foo bar\nfoo baz\n")

    apply_patch(path=str(path), old="foo bar", new="FOO BAR")

    assert path.read_text() == "FOO BAR\nfoo baz\n"


def test_apply_patch_raises_when_old_not_found(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("foo bar\n")

    with pytest.raises(ValueError, match="not found"):
        apply_patch(path=str(path), old="missing", new="x")


def test_apply_patch_raises_when_old_not_unique(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("dup\ndup\n")

    with pytest.raises(ValueError, match="not unique"):
        apply_patch(path=str(path), old="dup", new="x")


def test_grep_finds_matching_lines(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\n")
    (tmp_path / "b.txt").write_text("nothing here\n")

    result = grep(pattern="hello", path=str(tmp_path))

    assert "a.txt" in result
    assert "hello world" in result
    assert "b.txt" not in result


def test_grep_reports_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\n")

    assert grep(pattern="nothing-matches-this", path=str(tmp_path)) == "No matches."


def test_run_shell_returns_combined_output():
    result = run_shell(command="echo hi")

    assert "hi" in result


def test_run_shell_reports_nonzero_exit_code():
    result = run_shell(command="exit 1")

    assert "[exit code 1]" in result


def test_write_file_denied_when_not_approved(tmp_path, monkeypatch):
    set_auto_approve(False)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    path = tmp_path / "denied.txt"

    result = write_file(path=str(path), content="nope")

    assert result == "Denied by user."
    assert not path.exists()


def test_write_file_allowed_when_confirmed(tmp_path, monkeypatch):
    set_auto_approve(False)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    path = tmp_path / "allowed.txt"

    write_file(path=str(path), content="yep")

    assert path.read_text() == "yep"


def test_coding_toolbox_registers_all_tools():
    toolbox = CodingToolbox()

    assert isinstance(toolbox, Toolbox)
    names = {schema["name"] for schema in toolbox.schemas()}
    assert names == {"read_file", "write_file", "apply_patch", "run_shell", "grep"}


def test_coding_toolbox_tools_are_callable(tmp_path):
    toolbox = CodingToolbox()
    path = tmp_path / "hello.txt"

    result = toolbox.call_sync("write_file", path=str(path), content="hi")

    assert "hello.txt" in result
    assert path.read_text() == "hi"
