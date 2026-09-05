from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
from typing import Any

from ..errors import (
    ConcurrentUpdateError,
    ConfigurationError,
    ExecutionError,
    StepLimitExceeded,
)
from .builder import Condition, NodeSpec
from .diagram import to_text

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

    __slots__ = ("_loops", "_nodes", "_predecessors", "_state_type")

    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        predecessors: dict[str, list[tuple[str, Condition | None]]],
        loops: list[Loop] | None = None,
        state_type: type | None = None,
    ):
        self._nodes = nodes
        self._predecessors = predecessors
        self._loops = loops or []
        self._state_type = state_type

    def diagram(self) -> str:
        """A box-drawing picture of the graph, laid out top to bottom by wave.

        Returns the text rather than printing it: where the drawing goes is the
        caller's business, not this object's.
        """
        return to_text(self._nodes, self._predecessors, self._loops)

    async def run(self, state: Any = None, *, max_steps: int = 50) -> Any:
        """Run the graph over one state object, returning it once no node is ready.

        Omitting the state builds one from the type the Graph was declared with,
        so a graph whose fields all have defaults needs no argument at all. The
        state is checked against that type before anything runs: the wrong shape
        otherwise surfaces as an AttributeError from whichever node happened to
        touch a missing field first.
        """
        if state is None:
            if self._state_type is None:
                raise ConfigurationError(
                    "this graph has no declared state type, so run() needs a state"
                )
            state = self._state_type()
        if self._state_type is not None and not isinstance(state, self._state_type):
            raise ConfigurationError(
                f"this graph runs on {self._state_type.__name__}, got "
                f"{type(state).__name__}"
            )
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
