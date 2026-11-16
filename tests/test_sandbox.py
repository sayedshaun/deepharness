"""Sandboxes: what a command inside one can reach, and what outlives it.

The bwrap and docker tests run the real thing and skip where it is missing;
a sandbox is only worth what it actually blocks, which a mock cannot show.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from deepharness.errors import ConfigurationError
from deepharness.tools import Bubblewrap, Docker, NoSandbox

_IMAGE = os.environ.get("DEEPHARNESS_TEST_IMAGE", "python:3.12-slim")


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


needs_bwrap = pytest.mark.skipif(not _bwrap_works(), reason="bwrap unavailable")
needs_docker = pytest.mark.skipif(
    not _docker_image_present(), reason=f"docker or {_IMAGE} unavailable"
)
BACKENDS = [
    pytest.param(Bubblewrap, id="bwrap", marks=needs_bwrap),
    pytest.param(
        lambda root, **kw: Docker(_IMAGE, root, **kw), id="docker", marks=needs_docker
    ),
]


def test_no_sandbox_runs_in_the_workspace(tmp_path):
    assert NoSandbox(tmp_path).run("pwd", 5).stdout.strip() == str(tmp_path.resolve())


def test_a_failing_command_is_a_result_not_an_exception(tmp_path):
    assert NoSandbox(tmp_path).run("exit 3", 5).returncode == 3


def test_no_sandbox_kills_a_hanging_command(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        NoSandbox(tmp_path).run("sleep 30", 1)


def test_no_sandbox_needs_no_with_block(tmp_path):
    assert NoSandbox(tmp_path).run("echo hi", 5).stdout == "hi\n"


def test_docker_needs_an_image(tmp_path):
    with pytest.raises(ConfigurationError):
        Docker("", tmp_path)


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


def test_the_network_is_off_by_default(box):
    _, code = run(
        box,
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"",
    )

    assert code != 0


def test_state_outside_the_workspace_lasts_between_commands(box):
    run(box, "echo kept > ~/note")

    assert run(box, "cat ~/note") == ("kept\n", 0)


def test_a_hanging_command_is_killed(box):
    with pytest.raises(subprocess.TimeoutExpired):
        box.run("sleep 30", 1)


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
def test_leaving_the_block_discards_the_sandbox(workspace, make):
    with make(workspace) as sandbox:
        sandbox.run("echo gone > ~/note", 10)

    with pytest.raises(ConfigurationError):
        sandbox.run("true", 10)
    with make(workspace) as fresh:
        assert fresh.run("cat ~/note", 10).returncode != 0
