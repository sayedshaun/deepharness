"""Running a command in the workspace, which is the sharpest tool here.

A factory like file_tools(): the command runs in an injected Workspace, and the
tool is gated by default. Gating is the default rather than a suggestion because
this is the one tool whose blast radius is not bounded by the workspace root -
a command can reach anything the process can.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from .toolbox import tool
from .workspace import Workspace

_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_MAX_CHARS = 30_000


class ShellTool(StrEnum):
    """The name shell_tool() registers. See FileTool for why this is a StrEnum."""

    RUN = "run_command"


def shell_tool(
    workspace: Workspace | str | Path = ".",
    *,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    requires_approval: bool = True,
) -> Callable[..., Any]:
    """A `run_command` tool that runs a shell command in the workspace.

    `requires_approval=True` (the default) pauses the run before each command
    so a human can rule on it; combine it with `Permissions` to allow the
    commands you trust by pattern and still be asked about the rest. Turning it
    off is a decision about your own environment - say so deliberately.

    The command runs through the shell, because a shell is what the model is
    being offered: pipes and redirection are the point. It is not a sandbox.
    """
    space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    @tool(requires_approval=requires_approval)
    def run_command(command: str) -> str:
        """Run a shell command in the workspace and return its output.

        Reports stdout, then stderr, then a non-zero exit code. A command that
        runs longer than the timeout is killed and reported as such.
        """
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=space.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # a failing command is a result, not an exception
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"

        return _report(completed, max_chars)

    return run_command


def _report(completed: subprocess.CompletedProcess[str], max_chars: int) -> str:
    """The parts of a finished command worth sending back, and nothing else.

    Silence is reported explicitly: a command that printed nothing and a command
    whose output was dropped look identical to a model otherwise.
    """
    parts: list[str] = []
    if completed.stdout.strip():
        parts.append(completed.stdout.rstrip()[:max_chars])
    if completed.stderr.strip():
        parts.append(f"[stderr]\n{completed.stderr.rstrip()[:max_chars]}")
    if completed.returncode != 0:
        parts.append(f"[exit {completed.returncode}]")
    return "\n".join(parts) if parts else "(no output)"
