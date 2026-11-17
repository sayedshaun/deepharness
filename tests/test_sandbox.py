"""Sandboxes: what a command inside one can reach, and what outlives it.

The bwrap and docker tests run the real thing and skip where it is missing;
a sandbox is only worth what it actually blocks, which a mock cannot show.
DEEPHARNESS_REQUIRE_SANDBOXES=1 turns those skips into failures, so a CI job
meant to test the backends cannot pass without running them.
"""

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from deepharness.errors import ConfigurationError
from deepharness.sandbox import Bubblewrap, Docker

_IMAGE = os.environ.get("DEEPHARNESS_TEST_IMAGE", "python:3.12-slim")
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


def _docker_image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", _IMAGE], capture_output=True, check=False
    )
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(
    not _REQUIRED and not _bwrap_works(), reason="bwrap unavailable"
)
needs_docker = pytest.mark.skipif(
    not _REQUIRED and not _docker_image_present(),
    reason=f"docker or {_IMAGE} unavailable",
)
BACKENDS = [
    pytest.param(Bubblewrap, id="bwrap", marks=needs_bwrap),
    pytest.param(
        lambda root, **kw: Docker(_IMAGE, root, **kw), id="docker", marks=needs_docker
    ),
]


def test_docker_needs_an_image(tmp_path):
    with pytest.raises(ConfigurationError):
        Docker("", tmp_path)


def test_a_host_that_cannot_run_bwrap_fails_on_start(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/sh\necho 'setting up uid map: Permission denied' >&2\nexit 1\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    box = Bubblewrap(tmp_path)

    with pytest.raises(ConfigurationError, match="uid map"):
        box.start()
    with pytest.raises(ConfigurationError):
        box.run("true", 10)


@pytest.fixture
def workspace():
    # Not tmp_path: a snap-installed docker cannot see the host's /tmp, and
    # would mount an empty directory instead of the workspace.
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
        yield Path(root)


@pytest.fixture(params=BACKENDS)
def box(request, workspace):
    with request.param(workspace) as sandbox:
        yield sandbox


def run(box, command):
    """Everything the command printed, and its exit code."""
    result = box.run(command, 30)
    return result.stdout + result.stderr, result.returncode


def test_writes_in_the_workspace_reach_the_host(workspace, box):
    run(box, "echo hi > made.txt")

    assert (workspace / "made.txt").read_text() == "hi\n"


def test_host_files_outside_the_workspace_are_not_visible(box):
    # The workspace sits inside the repo, so this file is just outside it.
    _, code = run(box, f"cat {Path.cwd() / 'pyproject.toml'}")

    assert code != 0


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


def test_the_network_is_off_by_default(box, host_port):
    _, code = run(box, connect(host_port))

    assert code != 0


@needs_bwrap
def test_bwrap_with_network_shares_the_hosts(workspace, host_port):
    with Bubblewrap(workspace, network=True) as sandbox:
        assert sandbox.run(connect(host_port), 10).returncode == 0


def test_state_outside_the_workspace_lasts_between_commands(box):
    run(box, "echo kept > ~/note")

    assert run(box, "cat ~/note") == ("kept\n", 0)


def test_a_hanging_command_is_killed(box):
    with pytest.raises(subprocess.TimeoutExpired):
        box.run("sleep 30", 1)


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


@pytest.mark.parametrize("make", BACKENDS)
def test_output_is_capped(workspace, make):
    with make(workspace, max_output=1000) as sandbox:
        result = sandbox.run("head -c 3000000 /dev/zero | tr '\\0' a; echo err >&2", 30)

    assert result.stdout == "a" * 1000 + "\n[output truncated: 2999000 more bytes]"
    assert result.stderr == "err\n"


@pytest.mark.parametrize("make", BACKENDS)
def test_a_read_only_workspace_refuses_writes(workspace, make):
    (workspace / "kept.txt").write_text("original")

    with make(workspace, writable=False) as sandbox:
        assert sandbox.run("cat kept.txt", 10).stdout == "original"
        assert sandbox.run("echo changed > kept.txt", 10).returncode != 0

    assert (workspace / "kept.txt").read_text() == "original"


@pytest.mark.parametrize("make", BACKENDS)
async def test_async_with_starts_and_stops_the_sandbox(workspace, make):
    async with make(workspace) as sandbox:
        assert sandbox.run("echo hi", 10).stdout == "hi\n"

    with pytest.raises(ConfigurationError):
        sandbox.run("true", 10)


@pytest.mark.parametrize("make", BACKENDS)
def test_starting_twice_is_refused(workspace, make):
    with make(workspace) as sandbox:
        with pytest.raises(ConfigurationError):
            sandbox.start()
        assert sandbox.run("echo still", 10).stdout == "still\n"


@pytest.mark.parametrize("make", BACKENDS)
def test_the_root_cannot_be_the_workspace(make):
    with pytest.raises(ConfigurationError):
        make("/")


def _containers() -> set[str]:
    listed = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", "name=deepharness-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(listed.stdout.split())


@needs_docker
def test_a_container_that_fails_to_start_is_removed(workspace):
    before = _containers()
    box = Docker(_IMAGE, workspace, options=["--entrypoint", "/no/such/program"])

    try:
        with pytest.raises(ConfigurationError):
            box.start()
    finally:
        box.stop()
    assert _containers() == before


@pytest.mark.parametrize("make", BACKENDS)
def test_leaving_the_block_discards_the_sandbox(workspace, make):
    with make(workspace) as sandbox:
        sandbox.run("echo gone > ~/note", 10)

    with pytest.raises(ConfigurationError):
        sandbox.run("true", 10)
    with make(workspace) as fresh:
        assert fresh.run("cat ~/note", 10).returncode != 0
