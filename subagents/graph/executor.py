from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
from typing import Any

from ..errors import ConcurrentUpdateError, ExecutionError, StepLimitExceeded
from .builder import Condition, NodeSpec

Loop = tuple[str, str, Condition | None, frozenset[str]]
"""(source, target, condition, body) - see Graph.connect(loop=True)."""


class Executor:
    """Runs a built Graph to completion, merging concurrent branches' state.

    Nodes with no unmet dependencies run concurrently in each wave. After a
    wave completes, each branch's resulting state is merged back by field: a
    field is applied only if it changed relative to the state the wave started
    from. Two branches changing the same field need a reducer on it, otherwise
    the merge raises rather than silently drop one branch's work.

    A loop edge whose condition holds re-runs the nodes between its two ends,
    so the graph can iterate. max_steps bounds that.
    """

    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        predecessors: dict[str, list[tuple[str, Condition | None]]],
        loops: list[Loop] | None = None,
    ):
        self._nodes = nodes
        self._predecessors = predecessors
        self._loops = loops or []

    async def run(self, state: Any, *, max_steps: int = 50) -> Any:
        current = copy.deepcopy(state)
        finished: set[str] = set()
        ready = [name for name, spec in self._nodes.items() if spec.start]

        for _ in range(max_steps):
            if not ready:
                return current

            snapshot = copy.deepcopy(current)
            results = await asyncio.gather(
                *(
                    _run_node(name, self._nodes[name].func, copy.deepcopy(snapshot))
                    for name in ready
                )
            )

            _merge(current, snapshot, results)
            finished.update(ready)

            reentered = self._reenter_loops(set(ready), finished, current)
            ready = self._next_ready(finished, current)
            # A re-entered loop head has no unfinished predecessors to wait on,
            # so schedule it directly rather than through dependency readiness.
            ready += [name for name in reentered if name not in ready]

        raise StepLimitExceeded(max_steps, state=current)

    def _reenter_loops(
        self, just_ran: set[str], finished: set[str], state: Any
    ) -> list[str]:
        """Clear the bodies of any loop edges taken this wave, returning heads."""
        heads: list[str] = []
        for source, target, condition, body in self._loops:
            if source not in just_ran:
                continue
            if condition is not None and not condition(state):
                continue
            finished.difference_update(body)
            if target not in heads:
                heads.append(target)
        return heads

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


def _merge(current: Any, before: Any, branch_results: list[Any]) -> None:
    for field in dataclasses.fields(current):
        original = getattr(before, field.name)
        written = [
            getattr(result, field.name)
            for result in branch_results
            if getattr(result, field.name) != original
        ]
        if not written:
            continue
        if len(written) == 1:
            setattr(current, field.name, written[0])
            continue

        reducer = field.metadata.get("reducer")
        if reducer is None:
            raise ConcurrentUpdateError(field.name, len(written))
        setattr(current, field.name, reducer(original, written))
