"""Where a shell command actually runs, separate from which directory it runs in.

A Workspace confines the paths a tool resolves, but a command can reach
anything the process can, so a tool that runs commands needs a second
boundary: the process it runs in. That boundary is a Sandbox, which the tool
calls instead of `subprocess`.

A sandbox is a resource with a lifetime - a `with` block. Whatever a command
leaves behind outside the workspace (installed packages, caches) lasts until
the block ends and is thrown away with it. The workspace is bind-mounted at its
own absolute path, so tools that read files on the host see exactly what
commands wrote.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import uuid4

from ..errors import ConfigurationError
from .workspace import Workspace

_SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_TIMEOUT_EXIT = 124  # what coreutils and busybox `timeout` exit with


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


class Bubblewrap(Sandbox):
    """A Linux namespace sandbox via `bwrap`: no container image, no daemon.

    System directories are mounted read-only, the workspace read-write unless
    `writable=False`, and home directories not at all. The environment is cleared, so API keys held
    by this process are not visible to the command. `/tmp` and `$HOME` are a
    scratch directory that lives as long as the sandbox does.
    """

    def __init__(
        self,
        workspace: Workspace | str | Path = ".",
        *,
        writable: bool = True,
        network: bool = False,
        read_only: Sequence[str | Path] = (),
    ):
        super().__init__(workspace)
        if shutil.which("bwrap") is None:
            raise ConfigurationError("Bubblewrap needs the `bwrap` executable on PATH")
        self._writable = writable
        self._network = network
        self._read_only = tuple(str(Path(p).expanduser().resolve()) for p in read_only)
        self._scratch: Path | None = None

    def start(self) -> None:
        self._scratch = Path(tempfile.mkdtemp(prefix="deepharness-"))

    def stop(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if self._scratch is None:
            raise ConfigurationError("Bubblewrap.run outside its `with` block")
        root = str(self.workspace.root)
        argv = ["bwrap"]
        for path in (*_SYSTEM_DIRS, *self._read_only):
            argv += ["--ro-bind-try", path, path]
        argv += ["--dev", "/dev", "--proc", "/proc"]
        argv += ["--bind", str(self._scratch), "/tmp"]
        argv += ["--bind" if self._writable else "--ro-bind", root, root]
        argv += ["--chdir", root]
        argv += ["--unshare-all", *(["--share-net"] if self._network else [])]
        # The sandbox dies with bwrap, so the timeout kill reaches every child.
        argv += ["--die-with-parent", "--new-session", "--clearenv"]
        argv += ["--setenv", "PATH", _PATH, "--setenv", "HOME", "/tmp"]
        argv += ["sh", "-c", command]
        return _run(argv, self.workspace.root, timeout)


class Docker(Sandbox):
    """One container for the sandbox's lifetime, from an image you choose.

    The workspace is mounted read-write unless `writable=False`. Commands run
    as this process's uid and gid, so files written in the
    workspace are owned by you rather than by root. `options` is passed to
    `docker run` as-is, for limits such as `--memory` or `--cpus`. The image
    needs `sh`, `tail` and `timeout`, which coreutils and busybox both provide.
    """

    def __init__(
        self,
        image: str,
        workspace: Workspace | str | Path = ".",
        *,
        writable: bool = True,
        network: bool = False,
        options: Sequence[str] = (),
    ):
        super().__init__(workspace)
        if not image:
            raise ConfigurationError("Docker needs an image")
        if shutil.which("docker") is None:
            raise ConfigurationError("Docker needs the `docker` executable on PATH")
        self._image = image
        self._writable = writable
        self._network = network
        self._options = tuple(options)
        self._container: str | None = None

    def start(self) -> None:
        name = f"deepharness-{uuid4().hex[:12]}"
        root = str(self.workspace.root)
        argv = ["docker", "run", "--detach", "--init", "--name", name]
        argv += ["--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp"]
        argv += [] if self._network else ["--network", "none"]
        mount = f"{root}:{root}" if self._writable else f"{root}:{root}:ro"
        argv += ["--volume", mount, "--workdir", root, *self._options]
        argv += [self._image, "tail", "-f", "/dev/null"]
        subprocess.run(argv, capture_output=True, text=True, check=True)
        self._container = name

    def stop(self) -> None:
        if self._container is not None:
            subprocess.run(
                ["docker", "rm", "--force", self._container],
                capture_output=True,
                check=False,
            )
            self._container = None

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if self._container is None:
            raise ConfigurationError("Docker.run outside its `with` block")
        # Killing `docker exec` does not kill what it started, so the deadline
        # is enforced inside the container; the host timeout is only a backstop.
        argv = ["docker", "exec", "--workdir", str(self.workspace.root)]
        argv += [self._container, "timeout", "-k", "2", str(timeout)]
        argv += ["sh", "-c", command]
        completed = _run(argv, self.workspace.root, timeout + 10)
        if completed.returncode == _TIMEOUT_EXIT:
            raise subprocess.TimeoutExpired(command, timeout)
        return completed


def _run(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # a failing command is a result, not an exception
    )
