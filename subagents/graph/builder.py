from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .executor import Executor

StateT = TypeVar("StateT")
Condition = Callable[[Any], bool]
NodeFunc = Callable[[Any], Any]
Edge = tuple[str, Condition | None, bool]
"""(target, condition, loop) - loop marks a back-edge, see Graph.connect."""


@dataclass(slots=True)
class NodeSpec:
    """A registered graph node: a plain function that transforms the state."""

    name: str
    func: NodeFunc
    start: bool = False
    end: bool = False


class Graph:
    """Plain functions wired into a graph over a shared, typed state object.

    Nodes are registered with @graph.add(...) and wired together with
    graph.connect(source, target). Ordinary edges must form a DAG; a cycle is
    declared explicitly with connect(..., loop=True), which re-enters the
    nodes between the two ends. graph.build() validates the graph
    (reachability, no undeclared cycles) and returns an Executor.
    """

    __slots__ = ("edges", "nodes", "state_type")

    def __init__(self, state_type: type[StateT]):
        self.state_type = state_type
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: dict[str, list[Edge]] = {}

    def add(
        self,
        *,
        name: str | None = None,
        start: bool = False,
        end: bool = False,
    ) -> Callable[[NodeFunc], NodeFunc]:
        def decorator(func: NodeFunc) -> NodeFunc:
            node_name = name or func.__name__
            if node_name in self.nodes:
                raise ValueError(f"Node already exists: {node_name}")
            self.nodes[node_name] = NodeSpec(node_name, func, start=start, end=end)
            self.edges[node_name] = []
            return func

        return decorator

    def connect(
        self,
        source: NodeFunc | str,
        target: NodeFunc | str,
        *,
        condition: Condition | None = None,
        loop: bool = False,
    ) -> None:
        """Wire source to target, optionally gated on a condition.

        loop=True marks this as a back-edge: when it is taken, every node
        between target and source re-runs, so the pair forms an iteration.
        Cycles must be declared this way - an undeclared one fails build().
        """
        self.edges[self._name_of(source)].append(
            (self._name_of(target), condition, loop)
        )

    def _name_of(self, ref: NodeFunc | str) -> str:
        if isinstance(ref, str):
            if ref not in self.nodes:
                raise ValueError(f"Unknown node: {ref}")
            return ref

        for name, spec in self.nodes.items():
            if spec.func is ref:
                return name
        raise ValueError(f"Function is not registered as a node: {ref!r}")

    def build(self) -> Executor:
        from .executor import Executor

        if not any(spec.start for spec in self.nodes.values()):
            raise ValueError("Graph has no start node")

        forward = {
            name: [(target, condition) for target, condition, loop in edges if not loop]
            for name, edges in self.edges.items()
        }

        # Loop edges are deliberately excluded: the scheduler only ever sees a
        # DAG, and re-entry is driven separately by the executor.
        predecessors: dict[str, list[tuple[str, Condition | None]]] = {
            name: [] for name in self.nodes
        }
        for source_name, edges in forward.items():
            for target_name, condition in edges:
                predecessors[target_name].append((source_name, condition))

        for name, spec in self.nodes.items():
            if not spec.start and not self._has_incoming(name):
                raise ValueError(
                    f"Node '{name}' is unreachable: no start flag and no incoming edges"
                )

        _check_acyclic(self.nodes, forward)

        loops = [
            (source_name, target_name, condition, _loop_body(forward, target_name))
            for source_name, edges in self.edges.items()
            for target_name, condition, loop in edges
            if loop
        ]

        return Executor(self.nodes, predecessors, loops)

    def _has_incoming(self, name: str) -> bool:
        return any(
            target == name for edges in self.edges.values() for target, _, _ in edges
        )


def _loop_body(
    forward: dict[str, list[tuple[str, Condition | None]]], head: str
) -> frozenset[str]:
    """Nodes that re-run when a loop edge back to head is taken: head and
    everything forward-reachable from it.

    Deliberately the whole downstream region rather than just the nodes on a
    path back to the loop's source. That keeps side-effect-only body nodes and
    branches skipped by a false condition on an earlier pass in the iteration.
    Work upstream of the head is outside the region, so it runs once. A node
    guarding the loop's exit is inside it, but only ever runs when the exit
    condition holds - at which point the back-edge no longer fires.
    """
    return frozenset(_reachable(forward, head))


def _reachable(
    edges: dict[str, list[tuple[str, Condition | None]]], start: str
) -> set[str]:
    seen = {start}
    queue = [start]
    while queue:
        name = queue.pop()
        for target, _ in edges.get(name, ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def _check_acyclic(
    nodes: dict[str, NodeSpec],
    forward: dict[str, list[tuple[str, Condition | None]]],
) -> None:
    in_degree = dict.fromkeys(nodes, 0)
    for source_edges in forward.values():
        for target_name, _ in source_edges:
            in_degree[target_name] += 1

    queue = [name for name, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        name = queue.pop()
        visited += 1
        for target_name, _ in forward[name]:
            in_degree[target_name] -= 1
            if in_degree[target_name] == 0:
                queue.append(target_name)

    if visited != len(nodes):
        raise ValueError(
            "Graph contains a cycle; declare the back-edge with "
            "connect(..., loop=True) if the iteration is intentional"
        )
