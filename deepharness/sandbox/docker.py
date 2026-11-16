"""A sandbox in a Docker container, for an image-defined toolchain."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from ..errors import ConfigurationError
from ..tools.workspace import Workspace
from .base import Sandbox, run_process

_TIMEOUT_EXIT = 124  # what coreutils and busybox `timeout` exit with


class Docker(Sandbox):
    """One container for the sandbox's lifetime, from an image you choose.

    The workspace is mounted read-write unless `writable=False`. Commands run
    as this process's uid and gid, so files written in the workspace are owned
    by you rather than by root. `options` is passed to
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
        completed = run_process(argv, self.workspace.root, timeout + 10)
        if completed.returncode == _TIMEOUT_EXIT:
            raise subprocess.TimeoutExpired(command, timeout)
        return completed
