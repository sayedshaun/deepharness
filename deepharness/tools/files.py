"""Filesystem tools for an agent that works in a directory.

Built by a factory rather than exposed as module-level tools: each one closes
over the Workspace it may touch, so the confinement is injected like any other
collaborator instead of read from global state. The docstrings are what the
model is shown, so they are written for it.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from .toolbox import tool
from .workspace import Workspace

_DEFAULT_MAX_BYTES = 200_000
_DEFAULT_MAX_MATCHES = 200


class FileTool(StrEnum):
    """The names file_tools() registers, for writing permission rules against.

    A StrEnum because a member is a str: it can be compared, used as a Rule's
    tool, and matched by fnmatch with no conversion - while a typo is caught by
    an editor rather than by a rule that silently matches nothing.
    """

    READ = "read_file"
    LIST = "list_files"
    SEARCH = "search_files"
    WRITE = "write_file"
    EDIT = "edit_file"


def file_tools(
    workspace: Workspace | str | Path = ".",
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_matches: int = _DEFAULT_MAX_MATCHES,
    writable: bool = True,
) -> list[Callable[..., Any]]:
    """Read, write, edit, list and search files under one workspace root.

    `max_bytes` caps a single read; the rest of the file is a second call with
    an offset, which keeps one `cat` of a large file from filling the context
    window. `writable=False` returns the read-only three, for an agent that
    should look but not touch - a narrower toolbox is a stronger guarantee than
    a permission rule, because there is nothing to rule on.
    """
    space = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    @tool
    def read_file(path: str, offset: int = 0) -> str:
        """Read a UTF-8 text file. Paths are relative to the workspace root.

        Returns at most one window of the file; pass `offset` (a line number,
        1-based) to continue from where the last read stopped.
        """
        target = space.resolve(path)
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(offset - 1, 0)
        kept: list[str] = []
        size = 0
        for number, line in enumerate(lines[start:], start=start + 1):
            size += len(line) + 1
            if size > max_bytes and kept:
                remaining = len(lines) - number + 1
                kept.append(f"... [{remaining} more lines; read from offset {number}]")
                break
            kept.append(f"{number}\t{line}")
        return "\n".join(kept) if kept else "(empty file)"

    @tool
    def list_files(pattern: str = "**/*") -> str:
        """List files under the workspace root matching a glob pattern."""
        matches = sorted(
            space.relative(path)
            for path in space.root.glob(pattern)
            if path.is_file() and _inside(space, path)
        )
        if not matches:
            return f"No files match {pattern!r}."
        shown = matches[:max_matches]
        if len(matches) > len(shown):
            shown.append(f"... [{len(matches) - len(shown)} more]")
        return "\n".join(shown)

    @tool
    def search_files(pattern: str, glob: str = "**/*") -> str:
        """Find lines containing `pattern` in files matching `glob`.

        A plain substring search, case-sensitive, reported as
        `path:line: text`.
        """
        hits: list[str] = []
        for path in sorted(space.root.glob(glob)):
            if not path.is_file() or not _inside(space, path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # a binary or unreadable file is not a match
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    hits.append(f"{space.relative(path)}:{number}: {line.strip()}")
                    if len(hits) >= max_matches:
                        return "\n".join([*hits, "... [more matches not shown]"])
        return "\n".join(hits) if hits else f"No matches for {pattern!r}."

    if not writable:
        return [read_file, list_files, search_files]

    @tool(requires_approval=True)
    def write_file(path: str, content: str) -> str:
        """Create a file, or replace one entirely, with `content`.

        Replaces the whole file - prefer edit_file when changing part of an
        existing one, so the rest cannot be lost to a partial rewrite.
        """
        target = space.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        verb = "Replaced" if existed else "Created"
        return f"{verb} {space.relative(target)} ({len(content)} characters)."

    @tool(requires_approval=True)
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a file.

        `old` must appear exactly once, so an ambiguous edit is refused rather
        than applied to the wrong place: include surrounding lines to make it
        unique.
        """
        target = space.resolve(path)
        text = target.read_text(encoding="utf-8")
        found = text.count(old)
        if found == 0:
            raise ValueError(f"{space.relative(target)} does not contain that text")
        if found > 1:
            raise ValueError(
                f"that text appears {found} times in {space.relative(target)}; "
                f"include more surrounding lines to make it unique"
            )
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Edited {space.relative(target)}."

    return [read_file, list_files, search_files, write_file, edit_file]


def _inside(space: Workspace, path: Path) -> bool:
    """Whether a globbed path really lives in the workspace.

    glob() follows a symlinked directory, so a link inside the root can list
    files outside it - the same hole Workspace.resolve() closes for paths the
    model supplies.
    """
    resolved = path.resolve()
    return resolved == space.root or space.root in resolved.parents
