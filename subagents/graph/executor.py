from __future__ import annotations

import asyncio
from typing import Any

from subagents.graph import Graph, Node


class ExecutionError(Exception):
    """Raised when a node's agent fails after exhausting its retries."""

    def __init__(self, node: Node, attempts: int, original: Exception):
        self.node = node
        self.agent_name = node.agent.name
        self.attempts = attempts
        self.original = original
        super().__init__(
            f"Agent '{self.agent_name}' failed after {attempts} attempt(s): {original!r}"
        )


class Executor:
    """Runs a Graph starting from a given Node, threading state between agents."""

    async def run(self, graph: Graph, start: Node, state: dict[str, Any]) -> dict[str, Any]:
        if start not in graph.nodes.values():
            raise ValueError("Start node is not part of the graph")

        current: Node | None = start
        while current is not None:
            state = await self._execute_node(current, state)
            current = self._next_node(current, state)

        return state

    async def _execute_node(self, node: Node, state: dict[str, Any]) -> dict[str, Any]:
        attempts = 0
        while True:
            attempts += 1
            try:
                if node.timeout is not None:
                    return await asyncio.wait_for(node.agent.run(state), timeout=node.timeout)
                return await node.agent.run(state)
            except Exception as exc:
                if attempts > node.retry:
                    raise ExecutionError(node, attempts, exc) from exc

    def _next_node(self, node: Node, state: dict[str, Any]) -> Node | None:
        for target, condition in node.edges:
            if condition is None or condition(state):
                return target
        return None
