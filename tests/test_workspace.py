"""Confinement: a path from a model is only ever resolved inside the root."""

import pytest

from deepharness.errors import ConfigurationError, OutsideWorkspace
from deepharness.tools import Workspace


def test_a_relative_path_resolves_inside_the_root(tmp_path):
    (tmp_path / "notes.md").write_text("hi")
    space = Workspace(tmp_path)

    assert space.resolve("notes.md") == (tmp_path / "notes.md").resolve()
    assert space.relative(space.resolve("notes.md")) == "notes.md"


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "sub/../../escape"])
def test_a_path_leaving_the_root_is_refused(tmp_path, path):
    space = Workspace(tmp_path)

    with pytest.raises(OutsideWorkspace):
        space.resolve(path)


def test_a_symlink_pointing_out_of_the_root_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s3cret")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside / "secret")

    with pytest.raises(OutsideWorkspace):
        Workspace(root).resolve("link")


def test_a_missing_root_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="not an existing directory"):
        Workspace(tmp_path / "nope")
