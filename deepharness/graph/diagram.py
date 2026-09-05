"""Box-drawing renderer for a built graph, for looking at in a terminal.

Nodes are placed in layers by longest path from a start node, so a layer is
exactly the set of nodes the executor can run in one wave. Edges are drawn onto
a character grid: each edge gets its own horizontal track, so crossings render
as crossings instead of silently merging into one misleading line.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from .builder import Condition, NodeSpec

_UP, _DOWN, _LEFT, _RIGHT = 1, 2, 4, 8

_GLYPHS = {
    _UP: "│",
    _DOWN: "│",
    _UP | _DOWN: "│",
    _LEFT: "─",
    _RIGHT: "─",
    _LEFT | _RIGHT: "─",
    _DOWN | _RIGHT: "╭",
    _DOWN | _LEFT: "╮",
    _UP | _RIGHT: "╰",
    _UP | _LEFT: "╯",
    _UP | _DOWN | _RIGHT: "├",
    _UP | _DOWN | _LEFT: "┤",
    _UP | _LEFT | _RIGHT: "┴",
    _DOWN | _LEFT | _RIGHT: "┬",
    _UP | _DOWN | _LEFT | _RIGHT: "┼",
}


class _Canvas:
    """A sparse character grid that merges box-drawing strokes as they meet."""

    def __init__(self) -> None:
        self._strokes: dict[tuple[int, int], int] = {}
        self._text: dict[tuple[int, int], str] = {}

    def text(self, row: int, col: int, value: str) -> None:
        for ch in value:
            self._text[(row, col)] = ch
            col += 1
            if _display_width(ch) == 2:
                # The glyph already covers this column; render it as nothing so
                # everything drawn to its right stays where the layout put it.
                self._text[(row, col)] = ""
                col += 1

    def stroke(self, row: int, col: int, bits: int) -> None:
        self._strokes[(row, col)] = self._strokes.get((row, col), 0) | bits

    def vertical(self, col: int, top: int, bottom: int) -> None:
        for row in range(top, bottom + 1):
            bits = (_DOWN if row < bottom else 0) | (_UP if row > top else 0)
            self.stroke(row, col, bits)

    def horizontal(self, row: int, start: int, end: int) -> None:
        lo, hi = sorted((start, end))
        for col in range(lo, hi + 1):
            bits = (_RIGHT if col < hi else 0) | (_LEFT if col > lo else 0)
            self.stroke(row, col, bits)

    def render(self) -> str:
        cells = {
            **{k: _GLYPHS.get(v, "·") for k, v in self._strokes.items()},
            **self._text,
        }
        if not cells:
            return ""
        by_row: dict[int, dict[int, str]] = {}
        for (row, col), ch in cells.items():
            by_row.setdefault(row, {})[col] = ch
        lines = []
        for row in range(max(by_row) + 1):
            cols = by_row.get(row, {})
            width = max(cols) + 1 if cols else 0
            lines.append("".join(cols.get(c, " ") for c in range(width)).rstrip())
        return "\n".join(lines)


_Edge = tuple[str, str, "Condition | None"]


def _forward(
    predecessors: dict[str, list[tuple[str, Condition | None]]],
) -> dict[str, list[tuple[str, Condition | None]]]:
    out: dict[str, list[tuple[str, Condition | None]]] = {n: [] for n in predecessors}
    for target, preds in predecessors.items():
        for source, condition in preds:
            out[source].append((target, condition))
    return out


def _layers(nodes: dict[str, NodeSpec], forward) -> dict[str, int]:
    """Longest path from a start node: the wave the executor would run it in."""
    level = dict.fromkeys(nodes, 0)
    for _ in range(len(nodes)):
        changed = False
        for source, targets in forward.items():
            for target, _ in targets:
                if level[target] < level[source] + 1:
                    level[target] = level[source] + 1
                    changed = True
        if not changed:
            return level
    return level


def _expand(nodes, forward, level):
    """Split edges that span several layers over pass-through columns.

    Without this a long edge would jump the layers in between, crossing whatever
    happens to be drawn there with no line to show for it.
    """
    layers: list[list[tuple[str, str]]] = [[] for _ in range(max(level.values()) + 1)]
    for name in nodes:
        layers[level[name]].append(("node", name))
    edges: list[_Edge] = []
    for source, targets in forward.items():
        for target, condition in targets:
            previous = source
            for step in range(1, level[target] - level[source]):
                dummy = f"\x00{source}>{target}#{step}"
                layers[level[source] + step].append(("dummy", dummy))
                level[dummy] = level[source] + step
                edges.append((previous, dummy, condition if step == 1 else None))
                previous = dummy
            edges.append((previous, target, condition if previous == source else None))
    return layers, edges


def _display_width(text: str) -> int:
    """Terminal columns rather than characters: CJK and emoji occupy two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _width(kind: str, name: str) -> int:
    return 1 if kind == "dummy" else _display_width(name) + 4


def _when(condition: Condition | None) -> str:
    """Name the condition when it has one; a lambda only warrants a marker."""
    if condition is None:
        return ""
    name = getattr(condition, "__name__", "")
    return f" when {name}" if name and name != "<lambda>" else " (conditional)"


def _positions(layers):
    """Centre each layer over the widest one and return every column's centre."""
    width = {name: _width(kind, name) for layer in layers for kind, name in layer}
    totals = [
        sum(width[n] for _, n in layer) + 3 * (len(layer) - 1) if layer else 0
        for layer in layers
    ]
    widest = max(totals)
    centre: dict[str, int] = {}
    for layer, total in zip(layers, totals):
        x = (widest - total) // 2
        for _, name in layer:
            centre[name] = x + width[name] // 2
            x += width[name] + 3
    return centre


def _extent(name: str, centre: dict[str, int]) -> tuple[int, int]:
    """Left and right columns of a node's box, as _draw_boxes lays it out."""
    size = _width("node", name)
    left = centre[name] - size // 2
    return left, left + size - 1


def _draw_boxes(canvas, layers, nodes, centre, tops):
    for layer in layers:
        for kind, name in layer:
            top = tops[name]
            if kind == "dummy":
                canvas.vertical(centre[name], top, top + 2)
                continue
            label = f"  {name}  "
            size = _display_width(label)
            left = centre[name] - size // 2
            rule = "═" if nodes[name].start or nodes[name].end else "─"
            canvas.text(top, left, "╭" + rule * (size - 2) + "╮")
            canvas.text(top + 1, left, "│" + label[1:-1] + "│")
            canvas.text(top + 2, left, "╰" + rule * (size - 2) + "╯")


def to_text(
    nodes: dict[str, NodeSpec],
    predecessors: dict[str, list[tuple[str, Condition | None]]],
    loops: Sequence[tuple[str, str, Condition | None, frozenset[str]]] = (),
) -> str:
    """Render the graph as box-drawing text, laid out top to bottom by wave."""
    if not nodes:
        return ""

    forward = _forward(predecessors)
    level = _layers(nodes, forward)
    layers, edges = _expand(nodes, forward, level)
    centre = _positions(layers)

    # Every edge in a gap gets its own track, so two edges crossing render as a
    # crossing rather than collapsing into one line that implies a link.
    tracks: dict[int, list[_Edge]] = {}
    for edge in edges:
        tracks.setdefault(level[edge[0]], []).append(edge)

    tops: dict[str, int] = {}
    row = 0
    for index, layer in enumerate(layers):
        for _, name in layer:
            tops[name] = row
        row += 3 + len(tracks.get(index, [])) + 1

    canvas = _Canvas()
    _draw_boxes(canvas, layers, nodes, centre, tops)

    for index, gap in tracks.items():
        bottom = tops[next(n for _, n in layers[index])] + 2
        for offset, (source, target, condition) in enumerate(gap):
            track = bottom + 1 + offset
            # Anchor to the box above: an edge on the first track has a
            # zero-length vertical and would otherwise float free of it.
            canvas.stroke(bottom + 1, centre[source], _UP)
            canvas.vertical(centre[source], bottom + 1, track)
            canvas.horizontal(track, centre[source], centre[target])
            canvas.vertical(centre[target], track, tops[target] - 1)
            if not target.startswith("\x00"):
                canvas.text(tops[target] - 1, centre[target], "▽" if condition else "▼")

    # A loop edge runs against the layer order, so it is routed up a margin
    # column on the right rather than back through the middle of the drawing.
    rightmost = max(
        _extent(n, centre)[1] for layer in layers for kind, n in layer if kind == "node"
    )
    layer_of = {n: i for i, layer in enumerate(layers) for _, n in layer}

    def blocked(name: str) -> bool:
        """True if a sibling sits to the right, so the margin run would cross it."""
        edge = _extent(name, centre)[1]
        return any(
            _extent(other, centre)[0] > edge
            for kind, other in layers[layer_of[name]]
            if kind == "node" and other != name
        )

    notes = []
    for offset, (source, target, condition, _body) in enumerate(loops):
        note = f"  ↺ {source} → {target}{_when(condition)}"
        if blocked(source) or blocked(target):
            # Drawing it would run the line through a sibling's box, which reads
            # as an edge that does not exist. Say it in words instead.
            notes.append(note + "  (not drawn: no clear margin)")
            continue
        margin = rightmost + 3 + offset * 2
        source_row, target_row = tops[source] + 1, tops[target] + 1
        canvas.horizontal(source_row, _extent(source, centre)[1] + 1, margin)
        canvas.vertical(margin, target_row, source_row)
        canvas.text(target_row, _extent(target, centre)[1] + 1, "◀")
        canvas.horizontal(target_row, _extent(target, centre)[1] + 2, margin)
        notes.append(note)

    if any(condition for _, _, condition in edges):
        notes.append("  ▽ conditional edge")
    body = canvas.render()
    return f"{body}\n\n" + "\n".join(notes) if notes else body
