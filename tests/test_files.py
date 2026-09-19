"""The filesystem toolbox: what the model can read, write and find."""

import pytest

from deepharness.errors import OutsideWorkspace
from deepharness.tools import Workspace, file_tools


@pytest.fixture
def space(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\nprint('hi')\n")
    (tmp_path / "README.md").write_text("# Title\n")
    return Workspace(tmp_path)


def named(tools):
    return {tool.__name__: tool for tool in tools}


def test_the_read_only_set_omits_the_writing_tools(space):
    assert set(named(file_tools(space, writable=False))) == {
        "read_file",
        "list_files",
        "search_files",
    }


def test_read_file_numbers_lines(space):
    tools = named(file_tools(space))

    assert tools["read_file"](path="README.md") == "1\t# Title"


def test_read_file_reports_an_empty_file_rather_than_nothing(space):
    (space.root / "blank.txt").write_text("")
    tools = named(file_tools(space))

    assert tools["read_file"](path="blank.txt") == "(empty file)"


def test_read_file_stops_at_the_byte_cap_and_says_where_to_continue(space):
    (space.root / "big.txt").write_text("\n".join(f"line {n}" for n in range(1, 200)))
    tools = named(file_tools(space, max_bytes=50))

    output = tools["read_file"](path="big.txt")

    assert "more lines; read from offset" in output
    assert tools["read_file"](path="big.txt", offset=100).startswith("100\tline 100")


def test_read_file_will_not_leave_the_workspace(space):
    tools = named(file_tools(space))

    with pytest.raises(OutsideWorkspace):
        tools["read_file"](path="../secrets.txt")


def test_list_files_matches_a_glob(space):
    tools = named(file_tools(space))

    assert tools["list_files"](pattern="**/*.py") == "src/app.py"
    assert "No files match" in tools["list_files"](pattern="**/*.rs")


def test_search_files_reports_path_and_line(space):
    tools = named(file_tools(space))

    assert tools["search_files"](pattern="import") == "src/app.py:1: import os"
    assert "No matches" in tools["search_files"](pattern="nowhere")


def test_write_file_creates_then_replaces(space):
    tools = named(file_tools(space))

    assert "Created" in tools["write_file"](path="new.txt", content="one")
    assert "Replaced" in tools["write_file"](path="new.txt", content="two")
    assert (space.root / "new.txt").read_text() == "two"


def test_writing_tools_are_gated_by_default(space):
    tools = named(file_tools(space))

    assert tools["write_file"]._tool_spec.requires_approval
    assert tools["edit_file"]._tool_spec.requires_approval
    assert not tools["read_file"]._tool_spec.requires_approval


def test_edit_file_replaces_one_occurrence(space):
    tools = named(file_tools(space))

    tools["edit_file"](path="src/app.py", old="print('hi')", new="print('bye')")

    assert "print('bye')" in (space.root / "src" / "app.py").read_text()


def test_an_ambiguous_edit_is_refused_rather_than_guessed(space):
    (space.root / "dup.txt").write_text("same\nsame\n")
    tools = named(file_tools(space))

    with pytest.raises(ValueError, match="appears 2 times"):
        tools["edit_file"](path="dup.txt", old="same", new="other")

    assert (space.root / "dup.txt").read_text() == "same\nsame\n"


def test_an_edit_that_matches_nothing_is_an_error(space):
    tools = named(file_tools(space))

    with pytest.raises(ValueError, match="does not contain"):
        tools["edit_file"](path="README.md", old="absent", new="x")


def test_a_symlinked_directory_does_not_widen_a_listing(space, tmp_path):
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    (outside / "secret.py").write_text("token = 1")
    (space.root / "linked").symlink_to(outside)
    tools = named(file_tools(space))

    listing = tools["list_files"](pattern="**/*.py")

    assert "secret.py" not in listing
