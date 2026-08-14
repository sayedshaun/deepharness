from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
from typing import Any

from ..errors import ExecutionError
from .graph import Condition, NodeSpec


class Executor:
    """Runs a built Graph to completion, merging concurrent branches' state.

    Nodes with no unmet dependencies run concurrently in each wave. After a
    wave completes, each branch's resulting state is merged back by field:
    a field is applied only if it changed relative to the state the wave
    started from, and later branches (in registration order) win ties.
    """

    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        predecessors: dict[str, list[tuple[str, Condition | None]]],
    ):
        self._nodes = nodes
        self._predecessors = predecessors

    async def run(self, state: Any) -> Any:
        current = copy.deepcopy(state)
        finished: set[str] = set()
        ready = [name for name, spec in self._nodes.items() if spec.start]

        while ready:
            snapshot = copy.deepcopy(current)
            results = await asyncio.gather(
                *(
                    _run_node(name, self._nodes[name].func, copy.deepcopy(snapshot))
                    for name in ready
                )
            )

            for result in results:
                _merge(current, snapshot, result)
            finished.update(ready)

            ready = self._next_ready(finished, current)

        return current

    def _next_ready(self, finished: set[str], state: Any) -> list[str]:
        ready: list[str] = []
        for name in self._nodes:
            if name in finished:
                continue

            preds = self._predecessors[name]
            if not preds or not all(source in finished for source, _ in preds):
                continue

            if any(condition is None or condition(state) for _, condition in preds):
                ready.append(name)
            else:
                finished.add(name)  # all predecessors done, but no active edge: skip

        return ready


async def _run_node(name: str, func: Any, state: Any) -> Any:
    try:
        if inspect.iscoroutinefunction(func):
            return await func(state)
        return await asyncio.to_thread(func, state)
    except Exception as exc:
        raise ExecutionError(name, exc) from exc


def _merge(current: Any, before: Any, branch_result: Any) -> None:
    for field in dataclasses.fields(current):
        new_value = getattr(branch_result, field.name)
        if new_value != getattr(before, field.name):
            setattr(current, field.name, new_value)
