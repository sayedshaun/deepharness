"""Sandboxes: what a command inside one can reach, and what outlives it.

The tests run real bwrap and skip where it is missing; a sandbox is only worth
what it actually blocks, which a mock cannot show. DEEPHARNESS_REQUIRE_SANDBOXES=1
turns those skips into failures, so a CI job meant to test the sandbox cannot
pass without running it.
"""

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from deepharness.errors import ConfigurationError
from deepharness.sandbox import Bubblewrap

_REQUIRED = os.environ.get("DEEPHARNESS_REQUIRE_SANDBOXES") == "1"


def _bwrap_works() -> bool:
    if shutil.which("bwrap") is None:
        return False
    probe = subprocess.run(
        ["bwrap", "--ro-bind", "/", "/", "--unshare-all", "true"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(
    not _REQUIRED and not _bwrap_works(), reason="bwrap unavailable"
)


def _limits_work() -> bool:
    if not _bwrap_works():
        return False
    try:
        with (
            tempfile.TemporaryDirectory(dir=Path.cwd()) as root,
            Bubblewrap(root, max_memory_mb=64, max_processes=16),
        ):
            return True
    except ConfigurationError:
        return False


needs_limits = pytest.mark.skipif(
    not _REQUIRED and not _limits_work(),
    reason="systemd user scopes with memory and pids controllers unavailable",
)


def _fake(directory: Path, name: str, script: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(f"#!/bin/sh\n{script}\n")
    (directory / name).chmod(0o755)


def test_a_host_that_cannot_run_bwrap_fails_on_start(tmp_path, monkeypatch):
    _fake(
        tmp_path / "bin",
        "bwrap",
        "echo 'setting up uid map: Permission denied' >&2; exit 1",
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    box = Bubblewrap(tmp_path)

    with pytest.raises(ConfigurationError, match="uid map"):
        box.start()
    with pytest.raises(ConfigurationError):
        box.run("true", 10)


@pytest.fixture
def workspace():
    # Inside the repo, so a file just outside it is a real host file.
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
        yield Path(root)


@pytest.fixture
def box(workspace):
    with Bubblewrap(workspace) as sandbox:
        yield sandbox


def run(box, command):
    """Everything the command printed, and its exit code."""
    result = box.run(command, 30)
    return result.stdout + result.stderr, result.returncode


@needs_bwrap
def test_writes_in_the_workspace_reach_the_host(workspace, box):
    run(box, "echo hi > made.txt")

    assert (workspace / "made.txt").read_text() == "hi\n"


@needs_bwrap
def test_host_files_outside_the_workspace_are_not_visible(box):
    # The workspace sits inside the repo, so this file is just outside it.
    _, code = run(box, f"cat {Path.cwd() / 'pyproject.toml'}")

    assert code != 0


@needs_bwrap
def test_the_environment_does_not_leak(box, monkeypatch):
    monkeypatch.setenv("DEEPHARNESS_SECRET", "hunter2")

    output, _ = run(box, "env")

    assert "hunter2" not in output


@pytest.fixture
def host_port():
    """A port listening on the host's loopback, so reaching it needs no internet."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        socket.create_connection(("127.0.0.1", port), 2).close()  # reachable here
        yield port


def connect(port):
    return f"python3 -c \"import socket; socket.create_connection(('127.0.0.1', {port}), 2)\""


@needs_bwrap
def test_the_network_is_off_by_default(box, host_port):
    _, code = run(box, connect(host_port))

    assert code != 0


@needs_bwrap
def test_bwrap_with_network_shares_the_hosts(workspace, host_port):
    with Bubblewrap(workspace, network=True) as sandbox:
        assert sandbox.run(connect(host_port), 10).returncode == 0


@needs_bwrap
def test_state_outside_the_workspace_lasts_between_commands(box):
    run(box, "echo kept > ~/note")

    assert run(box, "cat ~/note") == ("kept\n", 0)


@needs_bwrap
def test_a_hanging_command_is_killed(box):
    with pytest.raises(subprocess.TimeoutExpired):
        box.run("sleep 30", 1)


@needs_bwrap
def test_exiting_124_is_not_a_timeout(box):
    assert run(box, "exit 124") == ("", 124)


@needs_bwrap
def test_nothing_a_command_starts_outlives_it(workspace):
    marker = "987.654"
    with Bubblewrap(workspace) as sandbox:
        with pytest.raises(subprocess.TimeoutExpired):
            sandbox.run(
                f"(setsid sleep {marker} &); nohup sleep {marker} & sleep 30", 1
            )
        sandbox.run(f"sleep {marker} &", 10)

    survivors = subprocess.run(
        ["pgrep", "-f", f"sleep {marker}"], capture_output=True, check=False
    )
    assert survivors.returncode == 1


@needs_bwrap
def test_output_is_capped(workspace):
    with Bubblewrap(workspace, max_output=1000) as sandbox:
        result = sandbox.run("head -c 3000000 /dev/zero | tr '\\0' a; echo err >&2", 30)

    assert result.stdout == "a" * 1000 + "\n[output truncated: 2999000 more bytes]"
    assert result.stderr == "err\n"


@needs_bwrap
def test_a_read_only_workspace_refuses_writes(workspace):
    (workspace / "kept.txt").write_text("original")

    with Bubblewrap(workspace, writable=False) as sandbox:
        assert sandbox.run("cat kept.txt", 10).stdout == "original"
        assert sandbox.run("echo changed > kept.txt", 10).returncode != 0

    assert (workspace / "kept.txt").read_text() == "original"


@needs_bwrap
async def test_async_with_starts_and_stops_the_sandbox(workspace):
    async with Bubblewrap(workspace) as sandbox:
        assert sandbox.run("echo hi", 10).stdout == "hi\n"

    with pytest.raises(ConfigurationError):
        sandbox.run("true", 10)


@needs_bwrap
def test_starting_twice_is_refused(workspace):
    with Bubblewrap(workspace) as sandbox:
        with pytest.raises(ConfigurationError):
            sandbox.start()
        assert sandbox.run("echo still", 10).stdout == "still\n"


@needs_bwrap
def test_the_root_cannot_be_the_workspace():
    with pytest.raises(ConfigurationError):
        Bubblewrap("/")


@needs_bwrap
def test_leaving_the_block_discards_the_sandbox(workspace):
    with Bubblewrap(workspace) as sandbox:
        sandbox.run("echo gone > ~/note", 10)

    with pytest.raises(ConfigurationError):
        sandbox.run("true", 10)
    with Bubblewrap(workspace) as fresh:
        assert fresh.run("cat ~/note", 10).returncode != 0


def test_limits_need_systemd_run(tmp_path, monkeypatch):
    _fake(tmp_path / "bin", "bwrap", 'exec "$@"')
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))

    Bubblewrap(tmp_path)  # no limits, no systemd needed
    with pytest.raises(ConfigurationError, match="systemd-run"):
        Bubblewrap(tmp_path, max_processes=16)


@pytest.mark.parametrize("limits", [{"max_memory_mb": 0}, {"max_processes": 0}])
def test_limits_must_be_positive(tmp_path, monkeypatch, limits):
    _fake(tmp_path / "bin", "bwrap", 'exec "$@"')
    _fake(tmp_path / "bin", "systemd-run", 'exec "$@"')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(ConfigurationError, match="at least 1"):
        Bubblewrap(tmp_path, **limits)


def test_limits_that_are_not_applied_fail_on_start(tmp_path, monkeypatch):
    # A systemd-run that runs the command without creating a limited cgroup.
    _fake(tmp_path / "bin", "bwrap", 'exec "$@"')
    _fake(
        tmp_path / "bin",
        "systemd-run",
        'while [ "$1" != "--" ]; do shift; done; shift; exec "$@"',
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    box = Bubblewrap(tmp_path, max_memory_mb=64, max_processes=16)

    with pytest.raises(ConfigurationError, match="could not apply"):
        box.start()
    with pytest.raises(ConfigurationError):
        box.run("true", 10)


@needs_limits
def test_memory_beyond_the_limit_is_refused(workspace):
    allocate = "python3 -c 'bytearray(300 * 1024 * 1024)'"
    with Bubblewrap(workspace, max_memory_mb=64) as sandbox:
        result = sandbox.run(f"{allocate}; echo exit=$?", 30)
        assert result.stdout == "exit=137\n"  # killed, and the shell carried on
        assert sandbox.run("echo fine", 10).stdout == "fine\n"


@needs_limits
def test_processes_beyond_the_limit_are_refused(workspace):
    with Bubblewrap(workspace, max_processes=16) as sandbox:
        result = sandbox.run("for i in $(seq 40); do sleep 2 & done; wait", 30)

    assert result.returncode != 0
    assert "fork" in result.stderr.lower()


@needs_limits
def test_a_limited_command_is_still_killed_on_timeout(workspace):
    with (
        Bubblewrap(workspace, max_memory_mb=64, max_processes=16) as sandbox,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        sandbox.run("sleep 30", 1)
