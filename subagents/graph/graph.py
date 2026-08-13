from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .executor import Executor

StateT = TypeVar("StateT")
Condition = Callable[[Any], bool]
NodeFunc = Callable[[Any], Any]


@dataclass(slots=True)
class NodeSpec:
    """A registered graph node: a plain function that transforms the state."""

    name: str
    func: NodeFunc
    start: bool = False
    end: bool = False


class Graph:
    """A DAG of plain functions that transform a shared, typed state object.

    Nodes are registered with @graph.add(...) and wired together with
    graph.connect(source, target). graph.build() validates the graph
    (reachability, cycles) and returns an Executor that runs it.
    """

    def __init__(self, state_type: type[StateT]):
        self.state_type = state_type
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: dict[str, list[tuple[str, Condition | None]]] = {}

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
    ) -> None:
        self.edges[self._name_of(source)].append((self._name_of(target), condition))

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

        predecessors: dict[str, list[tuple[str, Condition | None]]] = {
            name: [] for name in self.nodes
        }
        for source_name, edges in self.edges.items():
            for target_name, condition in edges:
                predecessors[target_name].append((source_name, condition))

        for name, spec in self.nodes.items():
            if not spec.start and not predecessors[name]:
                raise ValueError(
                    f"Node '{name}' is unreachable: no start flag and no incoming edges"
                )

        _check_acyclic(self.nodes, self.edges)

        return Executor(self.nodes, predecessors)


def _check_acyclic(
    nodes: dict[str, NodeSpec],
    edges: dict[str, list[tuple[str, Condition | None]]],
) -> None:
    in_degree = dict.fromkeys(nodes, 0)
    for source_edges in edges.values():
        for target_name, _ in source_edges:
            in_degree[target_name] += 1

    queue = [name for name, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        name = queue.pop()
        visited += 1
        for target_name, _ in edges[name]:
            in_degree[target_name] -= 1
            if in_degree[target_name] == 0:
                queue.append(target_name)

    if visited != len(nodes):
        raise ValueError("Graph contains a cycle")
