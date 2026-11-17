"""A sandbox from Linux namespaces, for machines without a container runtime."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ..errors import ConfigurationError
from ..tools.workspace import Workspace
from .base import MAX_OUTPUT, Sandbox, run_process

_SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class Bubblewrap(Sandbox):
    """A Linux namespace sandbox via `bwrap`: no container image, no daemon.

    System directories are mounted read-only, the workspace read-write unless
    `writable=False`, and home directories not at all. The environment is
    cleared, so API keys held by this process are not visible to the command.
    `/tmp` and `$HOME` are a scratch directory that lives as long as the
    sandbox does. Each command gets its own process namespace, so nothing it
    starts outlives it. There is no memory or process limit; use `Docker` with
    `options` where you need one.
    """

    def __init__(
        self,
        workspace: Workspace | str | Path = ".",
        *,
        writable: bool = True,
        network: bool = False,
        read_only: Sequence[str | Path] = (),
        max_output: int = MAX_OUTPUT,
    ):
        super().__init__(workspace, max_output=max_output)
        if shutil.which("bwrap") is None:
            raise ConfigurationError("Bubblewrap needs the `bwrap` executable on PATH")
        self._writable = writable
        self._network = network
        self._read_only = tuple(str(Path(p).expanduser().resolve()) for p in read_only)
        self._scratch: Path | None = None

    def start(self) -> None:
        if self._scratch is not None:
            raise ConfigurationError("this Bubblewrap is already started")
        self._scratch = Path(tempfile.mkdtemp(prefix="deepharness-"))
        # Hosts that restrict user namespaces fail inside bwrap, which would
        # otherwise look like every command failing.
        probe = self.run("true", 30)
        if probe.returncode != 0:
            self.stop()
            raise ConfigurationError(
                f"bwrap cannot create a sandbox on this host: {probe.stderr.strip()}"
            )

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
        return run_process(argv, self.workspace.root, timeout, self._max_output)
