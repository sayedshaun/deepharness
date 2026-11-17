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
import io
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self

from ..errors import ConfigurationError
from ..tools.workspace import Workspace

MAX_OUTPUT = 1_000_000
_CHUNK = 64 * 1024  # Linux's default pipe buffer: one read usually empties it


class Sandbox(ABC):
    """Runs shell commands in one workspace, between `start()` and `stop()`.

    `run` follows the stdlib's contract: a CompletedProcess whatever the exit
    code, and `subprocess.TimeoutExpired` once the command has been killed.
    Each of stdout and stderr keeps at most `max_output` bytes: a command can
    print far more than this process can hold before its timeout comes.
    """

    def __init__(
        self,
        workspace: Workspace | str | Path = ".",
        *,
        max_output: int = MAX_OUTPUT,
    ):
        self._workspace = (
            workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        )
        if self._workspace.root == Path("/"):
            raise ConfigurationError("a sandbox's workspace cannot be /")
        if max_output < 1:
            raise ConfigurationError("max_output must be at least 1")
        self._max_output = max_output

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
    argv: list[str], cwd: Path, timeout: float, max_output: int
) -> subprocess.CompletedProcess[str]:
    """`subprocess.run` with each stream capped at `max_output` bytes.

    Past the cap a stream is still read and thrown away, so the command never
    blocks on a full pipe, and the text ends with a note of how much was cut.
    """
    with subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        assert isinstance(process.stdout, io.BufferedReader)
        assert isinstance(process.stderr, io.BufferedReader)
        stdout = _Capture(process.stdout, max_output)
        stderr = _Capture(process.stderr, max_output)
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            stdout.join()
            stderr.join()
    return subprocess.CompletedProcess(
        argv, process.returncode, stdout.text(), stderr.text()
    )


class _Capture:
    """Reads a pipe on its own thread, keeping the first `limit` bytes."""

    def __init__(self, stream: io.BufferedReader, limit: int):
        self._kept = bytearray()
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._read, args=(stream, limit), daemon=True
        )
        self._thread.start()

    def _read(self, stream: io.BufferedReader, limit: int) -> None:
        while chunk := stream.read1(_CHUNK):
            room = max(limit - len(self._kept), 0)
            self._kept += chunk[:room]
            self._dropped += len(chunk) - min(room, len(chunk))

    def join(self) -> None:
        self._thread.join()

    def text(self) -> str:
        text = self._kept.decode("utf-8", errors="replace")
        if self._dropped:
            text += f"\n[output truncated: {self._dropped} more bytes]"
        return text
