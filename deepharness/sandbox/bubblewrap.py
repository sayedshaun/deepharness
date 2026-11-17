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
_MIB = 1024 * 1024
# Prints the memory.max and pids.max of the cgroup it runs in.
_READ_LIMITS = 'd=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup) && cat "$d/memory.max" "$d/pids.max"'


class Bubblewrap(Sandbox):
    """A Linux namespace sandbox via `bwrap`: no container image, no daemon.

    System directories are mounted read-only, the workspace read-write unless
    `writable=False`, and home directories not at all. The environment is
    cleared, so API keys held by this process are not visible to the command.
    `/tmp` and `$HOME` are a scratch directory that lives as long as the
    sandbox does. Each command gets its own process namespace, so nothing it
    starts outlives it.

    `max_memory_mb` and `max_processes` cap a command and everything it
    starts together, through a cgroup that systemd creates for each command
    (`systemd-run --user --scope`). Per-process rlimits cannot do this: the
    kernel counts RLIMIT_NPROC across every process the user owns. Without
    them, only time is limited.
    """

    def __init__(
        self,
        workspace: Workspace | str | Path = ".",
        *,
        writable: bool = True,
        network: bool = False,
        read_only: Sequence[str | Path] = (),
        max_output: int = MAX_OUTPUT,
        max_memory_mb: int | None = None,
        max_processes: int | None = None,
    ):
        super().__init__(workspace, max_output=max_output)
        if shutil.which("bwrap") is None:
            raise ConfigurationError("Bubblewrap needs the `bwrap` executable on PATH")
        for name, value in (
            ("max_memory_mb", max_memory_mb),
            ("max_processes", max_processes),
        ):
            if value is not None and value < 1:
                raise ConfigurationError(f"{name} must be at least 1")
        limited = max_memory_mb is not None or max_processes is not None
        if limited and shutil.which("systemd-run") is None:
            raise ConfigurationError(
                "Bubblewrap limits need `systemd-run` on PATH and a systemd user session"
            )
        self._memory = None if max_memory_mb is None else max_memory_mb * _MIB
        self._processes = max_processes
        self._writable = writable
        self._network = network
        self._read_only = tuple(str(Path(p).expanduser().resolve()) for p in read_only)
        self._scratch: Path | None = None

    def start(self) -> None:
        if self._scratch is not None:
            raise ConfigurationError("this Bubblewrap is already started")
        if self._memory is not None or self._processes is not None:
            self._check_limits()
        self._scratch = Path(tempfile.mkdtemp(prefix="deepharness-"))
        # Hosts that restrict user namespaces fail inside bwrap, which would
        # otherwise look like every command failing.
        probe = self.run("true", 30)
        if probe.returncode != 0:
            self.stop()
            raise ConfigurationError(
                f"bwrap cannot create a sandbox on this host: {probe.stderr.strip()}"
            )

    def _scope(self) -> list[str]:
        """The `systemd-run` prefix that puts a command in its own limited cgroup."""
        if self._memory is None and self._processes is None:
            return []
        argv = ["systemd-run", "--user", "--scope", "--quiet", "--collect"]
        if self._memory is not None:
            argv += [
                f"--property=MemoryMax={self._memory}",
                "--property=MemorySwapMax=0",
            ]
        if self._processes is not None:
            argv += [f"--property=TasksMax={self._processes}"]
        # Kill only the process that ran out of memory, as the kernel would
        # without a limit, rather than every process in the command.
        return [*argv, "--property=OOMPolicy=continue", "--"]

    def _check_limits(self) -> None:
        """Fail unless systemd really applies the limits, rather than run without them.

        `systemd-run` accepts a property it cannot enforce, for example when the
        memory controller is not delegated to the user, so read them back.
        """
        probe = subprocess.run(
            [*self._scope(), "sh", "-c", _READ_LIMITS],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        applied = probe.stdout.split()
        wanted = [self._memory, self._processes]
        if (
            probe.returncode != 0
            or len(applied) != 2
            or any(
                want is not None and got != str(want)
                for want, got in zip(wanted, applied, strict=True)
            )
        ):
            raise ConfigurationError(
                "systemd could not apply the memory and process limits: "
                f"{(probe.stderr or probe.stdout).strip()}"
            )

    def stop(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def run(self, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if self._scratch is None:
            raise ConfigurationError("Bubblewrap.run outside its `with` block")
        root = str(self.workspace.root)
        argv = [*self._scope(), "bwrap"]
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
