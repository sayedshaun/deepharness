import unicodedata
import unicodedata
from dataclasses import dataclass, field

import pytest

from deepharness import Graph


@dataclass
class State:
    rounds: int = 0
    trace: list[str] = field(default_factory=list)


def _node(state: State) -> State:
    return state


def needs_rework(state: State) -> bool:
    return state.rounds < 3


def diamond() -> Graph:
    graph = Graph(State)
    for name in ("left", "right"):
        graph.add(start=True, name=name)(_node)
    graph.add(name="merge")(_node)
    graph.add(end=True, name="done")(_node)
    graph.connect("left", "merge")
    graph.connect("right", "merge")
    graph.connect("merge", "done")
    return graph


def skipping() -> Graph:
    """`a` reaches `c` both directly and through `b`, spanning two layers."""
    graph = Graph(State)
    graph.add(start=True, name="a")(_node)
    graph.add(name="b")(_node)
    graph.add(end=True, name="c")(_node)
    graph.connect("a", "b")
    graph.connect("b", "c")
    graph.connect("a", "c")
    return graph


def looping() -> Graph:
    graph = Graph(State)
    graph.add(start=True, name="agent")(_node)
    graph.add(name="tools")(_node)
    graph.connect("agent", "tools", condition=lambda s: s.rounds < 3)
    graph.connect("tools", "agent", loop=True)
    return graph


def test_every_node_is_drawn():
    out = diamond().build().diagram()
    for name in ("left", "right", "merge", "done"):
        assert name in out


def test_parallel_nodes_share_a_line():
    """Nodes the executor runs in one wave are laid out side by side."""
    line = next(ln for ln in diamond().build().diagram().splitlines() if "left" in ln)
    assert "right" in line


def test_later_wave_is_below_an_earlier_one():
    lines = diamond().build().diagram().splitlines()
    assert _row(lines, "left") < _row(lines, "merge") < _row(lines, "done")


def _row(lines, name):
    return next(i for i, ln in enumerate(lines) if name in ln)


def test_start_and_end_nodes_get_a_distinct_rule():
    out = diamond().build().diagram()
    assert "═" in out  # start/end
    assert "─" in out  # interior


def test_conditional_edge_uses_a_hollow_arrow_and_is_explained():
    out = looping().build().diagram()
    assert "▽" in out
    assert "conditional edge" in out


def test_unconditional_edge_uses_a_solid_arrow():
    out = diamond().build().diagram()
    assert "▼" in out
    assert "▽" not in out
    assert "conditional edge" not in out


def test_loop_edge_is_drawn_back_up_the_margin():
    out = looping().build().diagram()
    assert "↺ tools → agent" in out
    assert "◀" in out  # the back-edge arrow into the loop target


def test_named_loop_condition_is_spelled_out():
    graph = Graph(State)
    graph.add(start=True, name="draft")(_node)
    graph.add(end=True, name="review")(_node)
    graph.connect("draft", "review")
    graph.connect("review", "draft", loop=True, condition=needs_rework)
    assert "↺ review → draft when needs_rework" in graph.build().diagram()


def test_loop_is_not_drawn_through_a_sibling_box():
    """A line crossing a sibling would read as an edge that does not exist."""
    graph = Graph(State)
    graph.add(start=True, name="plan")(_node)
    graph.add(name="work")(_node)
    graph.add(name="sibling_on_the_right")(_node)
    graph.add(end=True, name="done")(_node)
    graph.connect("plan", "work")
    graph.connect("plan", "sibling_on_the_right")
    graph.connect("work", "done")
    graph.connect("sibling_on_the_right", "done")
    graph.connect("work", "plan", loop=True)

    out = graph.build().diagram()
    assert "(not drawn: no clear margin)" in out
    assert "◀" not in out
    work_line = next(ln for ln in out.splitlines() if "work" in ln and "│" in ln)
    assert not work_line.rstrip().endswith("╯")


def test_single_node_has_no_edges():
    graph = Graph(State)
    graph.add(start=True, name="only")(_node)
    out = graph.build().diagram()
    assert "only" in out
    assert "▼" not in out and "▽" not in out


def test_edge_spanning_layers_is_routed_past_the_layer_between():
    lines = skipping().build().diagram().splitlines()
    # The a->c edge must still be drawn on the rows occupied by b.
    b_row = _row(lines, "b")
    assert "│" in lines[b_row].replace("│ b │", "")


@pytest.mark.parametrize("build", [diamond, skipping, looping])
def test_no_unresolved_glyphs(build):
    """`·` is the fallback for a stroke combination with no box-drawing char."""
    assert "·" not in build().build().diagram()


def _columns(text: str) -> int:
    """Terminal columns the line occupies, not characters."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


@pytest.mark.parametrize("name", ["取得データ", "ship 🚀", "plain"])
def test_box_borders_line_up_with_double_width_names(name):
    graph = Graph(State)
    graph.add(start=True, name=name)(_node)
    top, middle, bottom = graph.build().diagram().splitlines()[:3]
    assert _columns(top) == _columns(middle) == _columns(bottom)


def test_a_wide_name_does_not_shift_its_neighbour():
    graph = Graph(State)
    graph.add(start=True, name="取得")(_node)
    graph.add(start=True, name="plain")(_node)
    graph.add(end=True, name="done")(_node)
    graph.connect("取得", "done")
    graph.connect("plain", "done")
    top, middle, _ = graph.build().diagram().splitlines()[:3]
    assert _columns(top) == _columns(middle)


def _columns(text: str) -> int:
    """Terminal columns the line occupies, not characters."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


@pytest.mark.parametrize("name", ["取得データ", "ship 🚀", "plain"])
def test_box_borders_line_up_with_double_width_names(name):
    graph = Graph(State)
    graph.add(start=True, name=name)(_node)
    top, middle, bottom = graph.build().diagram().splitlines()[:3]
    assert _columns(top) == _columns(middle) == _columns(bottom)


def test_a_wide_name_does_not_shift_its_neighbour():
    graph = Graph(State)
    graph.add(start=True, name="取得")(_node)
    graph.add(start=True, name="plain")(_node)
    graph.add(end=True, name="done")(_node)
    graph.connect("取得", "done")
    graph.connect("plain", "done")
    top, middle, _ = graph.build().diagram().splitlines()[:3]
    assert _columns(top) == _columns(middle)
