"""The shell tool: output shaping, timeouts, and the gate that is on by default."""

import pytest

from deepharness.tools import Workspace, shell_tool


@pytest.fixture
def run(tmp_path):
    return shell_tool(Workspace(tmp_path), requires_approval=False)


def test_it_returns_stdout(run):
    assert run(command="echo hello") == "hello"


def test_it_runs_in_the_workspace(tmp_path, run):
    (tmp_path / "marker.txt").write_text("x")

    assert "marker.txt" in run(command="ls")


def test_a_failing_command_reports_stderr_and_the_exit_code(run):
    output = run(command="ls /definitely-not-here")

    assert "[stderr]" in output
    assert "[exit " in output


def test_a_silent_command_says_so(run):
    assert run(command="true") == "(no output)"


def test_a_hanging_command_is_killed_and_reported(tmp_path):
    run = shell_tool(tmp_path, timeout=1, requires_approval=False)

    assert "timed out after 1s" in run(command="sleep 5")


def test_output_is_capped(tmp_path):
    run = shell_tool(tmp_path, max_chars=20, requires_approval=False)

    assert len(run(command="seq 1 1000")) <= 20


def test_it_is_gated_unless_told_otherwise(tmp_path):
    assert shell_tool(tmp_path)._tool_spec.requires_approval
    assert not shell_tool(
        tmp_path, requires_approval=False
    )._tool_spec.requires_approval
