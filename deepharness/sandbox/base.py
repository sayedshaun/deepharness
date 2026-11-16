"""Where a shell command actually runs, separate from which directory it runs in.

A Workspace confines the paths a tool resolves, but a command can reach
anything the process can, so code that runs commands needs a second boundary:
the process it runs in. That boundary is a Sandbox, called instead of
`subprocess` - by a tool, or by a graph node directly.

A sandbox is a resource with a lifetime - a `with` block. Whatever a command
leaves behind outside the workspace (installed packages, caches) lasts until
the block ends and is thrown away with it. The workspace is bind-mounted at its
own absolute path, so code reading files on the host sees exactly what
commands wrote.
"""

from __future__ import annotations

import asyncio
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self

from ..tools.workspace import Workspace


class Sandbox(ABC):
    """Runs shell commands in one workspace, between `start()` and `stop()`.

    `run` follows the stdlib's contract: a CompletedProcess whatever the exit
    code, and `subprocess.TimeoutExpired` once the command has been killed.
    """

    def __init__(self, workspace: Workspace | str | Path = "."):
        self._workspace = (
            workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        )

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def start(self) -> None:
        """Acquire whatever the sandbox needs; a no-op where nothing is held."""

    def stop(self) -> None:
        """Release it, discarding everything outside the workspace."""

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    async def __aenter__(self) -> Self:
        await asyncio.to_thread(self.start)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.stop)

    @abstractmethod
    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]: ...


def run_process(
    argv: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # a failing command is a result, not an exception
    )
