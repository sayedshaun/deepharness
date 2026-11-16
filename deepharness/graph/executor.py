from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ..errors import (
    ConcurrentUpdateError,
    ConfigurationError,
    ExecutionError,
    StepLimitExceeded,
)
from .builder import Condition, NodeSpec
from .diagram import to_text
from .events import GraphEvent, GraphFinished, NodeFinished, NodeStarted

Loop = tuple[str, str, Condition | None, frozenset[str]]
"""(source, target, condition, body) - see Graph.connect(loop=True)."""

GraphEventHook = Callable[[GraphEvent], Awaitable[None] | None]
"""Called with every event a run emits, so run() can be observed without
iterating astream_events(). May be async."""


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

    __slots__ = ("_loops", "_nodes", "_on_event", "_predecessors", "_state_type")

    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        predecessors: dict[str, list[tuple[str, Condition | None]]],
        loops: list[Loop] | None = None,
        state_type: type | None = None,
        on_event: GraphEventHook | None = None,
    ):
        self._nodes = nodes
        self._predecessors = predecessors
        self._loops = loops or []
        self._state_type = state_type
        self._on_event = on_event

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
        async for event in self.astream_events(state, max_steps=max_steps):
            if isinstance(event, GraphFinished):
                return event.state
        raise AssertionError("unreachable: astream_events always ends finished")

    async def astream_events(
        self, state: Any = None, *, max_steps: int = 50
    ) -> AsyncIterator[GraphEvent]:
        """Run the graph like run(), reporting each node as it starts and ends.

        The only driver: run() consumes this and keeps the final state, so the
        scheduling exists once. Ends with one GraphFinished carrying the state,
        or raises what run() would. Each event goes to on_event, when one was
        given, before the caller sees it.
        """
        async for event in self._events(state, max_steps):
            if self._on_event is not None:
                notified = self._on_event(event)
                if inspect.isawaitable(notified):
                    await notified
            yield event

    async def _events(self, state: Any, max_steps: int) -> AsyncIterator[GraphEvent]:
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
        skipped: set[str] = set()
        ready = [name for name, spec in self._nodes.items() if spec.start]

        for _ in range(max_steps):
            if not ready:
                yield GraphFinished(current)
                return

            snapshot = copy.deepcopy(current)
            for name in ready:
                yield NodeStarted(name)
            # Merged in wave order, so a reducer sees branches in a stable order,
            # but reported in completion order, as each node actually ends.
            results: list[Any] = [None] * len(ready)
            async for index, result, took in self._run_wave(ready, snapshot):
                results[index] = result
                yield NodeFinished(ready[index], took)

            _merge(current, snapshot, results)
            finished.update(ready)

            reentered = self._reenter_loops(set(ready), finished, current)
            skipped &= finished  # a re-entered loop body may run this time
            ready = self._next_ready(finished, skipped, current)
            # A re-entered loop head has no unfinished predecessors to wait on,
            # so schedule it directly rather than through dependency readiness.
            ready += [name for name in reentered if name not in ready]

        raise StepLimitExceeded(max_steps, state=current)

    async def _run_wave(
        self, names: list[str], snapshot: Any
    ) -> AsyncIterator[tuple[int, Any, float]]:
        """Run one wave's nodes concurrently, yielding (position, state, seconds)."""
        tasks = [
            asyncio.create_task(
                _timed_node(name, self._nodes[name].func, copy.deepcopy(snapshot))
            )
            for name in names
        ]
        position = {task: index for index, task in enumerate(tasks)}
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in sorted(done, key=position.__getitem__):
                    result, took = task.result()
                    yield position[task], result, took
        finally:
            # A failed node ends the run, so its siblings' work has no reader.
            for task in tasks:
                task.cancel()

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

    def _next_ready(
        self, finished: set[str], skipped: set[str], state: Any
    ) -> list[str]:
        """The nodes to run next, marking the ones no active edge reaches skipped.

        An edge is active only when its source actually ran, so a skip carries
        on downstream instead of a skipped branch's successors running anyway.
        Repeated until nothing new is skipped, since a skip can settle a node
        this pass already looked at.
        """
        ready: list[str] = []
        changed = True
        while changed:
            changed = False
            for name in self._nodes:
                if name in finished or name in ready:
                    continue

                preds = self._predecessors[name]
                if not preds or not all(source in finished for source, _ in preds):
                    continue

                if any(
                    source not in skipped and (condition is None or condition(state))
                    for source, condition in preds
                ):
                    ready.append(name)
                else:
                    finished.add(name)
                    skipped.add(name)
                    changed = True

        return ready


async def _timed_node(name: str, func: Any, state: Any) -> tuple[Any, float]:
    started = time.monotonic()
    result = await _run_node(name, func, state)
    return result, time.monotonic() - started


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
