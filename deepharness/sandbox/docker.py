"""A sandbox in a Docker container, for an image-defined toolchain."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from ..errors import ConfigurationError
from ..tools.workspace import Workspace
from .base import MAX_OUTPUT, Sandbox, run_process

_TIMEOUT_EXIT = 124  # what coreutils and busybox `timeout` exit with


class Docker(Sandbox):
    """One container for the sandbox's lifetime, from an image you choose.

    The workspace is mounted read-write unless `writable=False`. Commands run
    as this process's uid and gid, so files written in the workspace are owned
    by you rather than by root, with every capability dropped, so a setuid
    binary in the image cannot give them root's powers. `options` is passed
    to `docker run` as-is, for limits such as `--memory`, `--cpus` or
    `--pids-limit`; it is your configuration, never the model's. The image
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
        max_output: int = MAX_OUTPUT,
    ):
        super().__init__(workspace, max_output=max_output)
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
        if self._container is not None:
            raise ConfigurationError("this Docker sandbox is already started")
        name = f"deepharness-{uuid4().hex[:12]}"
        root = str(self.workspace.root)
        argv = ["docker", "run", "--detach", "--init", "--name", name]
        argv += ["--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp"]
        argv += ["--cap-drop=ALL"]
        argv += [] if self._network else ["--network", "none"]
        mount = f"{root}:{root}" if self._writable else f"{root}:{root}:ro"
        argv += ["--volume", mount, "--workdir", root, *self._options]
        argv += [self._image, "tail", "-f", "/dev/null"]
        started = subprocess.run(argv, capture_output=True, text=True, check=False)
        if started.returncode != 0:
            # A container that was created but failed to start still exists.
            self._remove(name)
            raise ConfigurationError(
                f"docker could not start {self._image}: {started.stderr.strip()}"
            )
        self._container = name
        # `docker run --detach` succeeds even when the container exits at once.
        probe = self.run("true", 30)
        if probe.returncode != 0:
            self.stop()
            raise ConfigurationError(
                f"the {self._image} container is not usable: {probe.stderr.strip()}"
            )

    def stop(self) -> None:
        if self._container is not None:
            self._remove(self._container)
            self._container = None

    @staticmethod
    def _remove(name: str) -> None:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, check=False
        )

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if self._container is None:
            raise ConfigurationError("Docker.run outside its `with` block")
        # Killing `docker exec` does not kill what it started, so the deadline
        # is enforced inside the container; the host timeout is only a backstop.
        argv = ["docker", "exec", "--workdir", str(self.workspace.root)]
        argv += [self._container, "timeout", "-k", "2", str(timeout)]
        argv += ["sh", "-c", command]
        started = time.monotonic()
        completed = run_process(
            argv, self.workspace.root, timeout + 10, self._max_output
        )
        # A command may exit 124 itself; only a deadline that passed is a timeout.
        timed_out = time.monotonic() - started >= timeout
        if completed.returncode == _TIMEOUT_EXIT and timed_out:
            raise subprocess.TimeoutExpired(command, timeout)
        return completed
