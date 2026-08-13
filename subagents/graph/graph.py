from __future__ import annotations

from collections.abc import Callable
from typing import Any

from subagents.agent import Agent

Condition = Callable[[dict[str, Any]], bool]


class Node:

    def __init__(
        self,
        agent: Agent,
        *,
        retry: int = 0,
        timeout: float | None = None,
    ):
        self.agent = agent
        self.edges: list[tuple[Node, Condition | None]] = []
        self.retry = retry
        self.timeout = timeout

    def connect(self, node: Node, *, condition: Condition | None = None) -> None:
        self.edges.append((node, condition))


class Graph:
    def __init__(self):
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        name = node.agent.name

        if name in self.nodes:
            raise ValueError(f"Node already exists: {name}")

        self.nodes[name] = node

    def connect(
        self,
        source: Node,
        target: Node,
        *,
        condition: Condition | None = None,
    ) -> None:
        if source not in self.nodes.values():
            raise ValueError("Source node is not part of the graph")

        if target not in self.nodes.values():
            raise ValueError("Target node is not part of the graph")

        source.connect(target, condition=condition)
