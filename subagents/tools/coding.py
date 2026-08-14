from __future__ import annotations

import re
import subprocess
from pathlib import Path

from subagents.agent import Toolbox, tool

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
}

_auto_approve = False


def set_auto_approve(enabled: bool) -> None:
    """Skip confirmation prompts for risky tools (write_file, apply_patch, run_shell).

    Off by default, so a real coding agent asks before acting. Turn it on for
    tests, scripts, or any run where you've already decided to trust the agent.
    """
    global _auto_approve
    _auto_approve = enabled


def _confirm(action: str) -> bool:
    if _auto_approve:
        return True
    answer = input(f"{action}\nAllow? [y/n]: ").strip().lower()
    return answer == "y"


@tool
def read_file(path: str) -> str:
    """Read a file's contents."""
    return Path(path).read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Overwrite a file with new content, creating it if it doesn't exist."""
    if not _confirm(f"Write {len(content)} chars to {path!r}"):
        return "Denied by user."
    Path(path).write_text(content)
    return f"Wrote {len(content)} chars to {path}"


@tool
def apply_patch(path: str, old: str, new: str) -> str:
    """Replace one exact occurrence of `old` with `new` in a file."""
    text = Path(path).read_text()
    occurrences = text.count(old)

    if occurrences == 0:
        raise ValueError(f"'old' text not found in {path}")
    if occurrences > 1:
        raise ValueError(
            f"'old' text is not unique in {path} ({occurrences} matches) - add more context"
        )

    if not _confirm(f"Patch {path!r}: replace\n---\n{old}\n---\nwith\n---\n{new}\n---"):
        return "Denied by user."

    Path(path).write_text(text.replace(old, new, 1))
    return f"Patched {path}"


@tool
def run_shell(command: str, timeout: float = 30.0) -> str:
    """Run a shell command and return its combined stdout/stderr."""
    if not _confirm(f"Run shell command: {command!r}"):
        return "Denied by user."

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        output += f"\n[exit code {result.returncode}]"
    return output


@tool
def grep(pattern: str, path: str = ".", max_matches: int = 200) -> str:
    """Search files under a path for a regex pattern, returning matching lines."""
    regex = re.compile(pattern)
    root = Path(path)
    files = [root] if root.is_file() else _walk_files(root)

    matches: list[str] = []
    for file_path in files:
        if len(matches) >= max_matches:
            break
        try:
            lines = file_path.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{file_path}:{line_number}: {line.strip()}")
                if len(matches) >= max_matches:
                    break

    if not matches:
        return "No matches."
    return "\n".join(matches)


def _walk_files(root: Path):
    for entry in root.rglob("*"):
        if entry.is_file() and not any(part in _SKIP_DIRS for part in entry.parts):
            yield entry


class CodingToolbox(Toolbox):
    """A Toolbox pre-loaded with read_file, write_file, apply_patch, run_shell, and grep.

    Everything a coding agent needs to read, edit, and run a project - none
    of it is tied to a specific platform or programming language, since it
    operates on files and shell commands rather than any language's tooling.
    """

    def __init__(self) -> None:
        super().__init__()
        for fn in (read_file, write_file, apply_patch, run_shell, grep):
            self.register(fn)
