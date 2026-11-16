"""What running a graph emits, so a caller can watch it work.

A graph's run is otherwise silent until it returns, which hides the thing worth
knowing in a long one: which node is running now, and which took the time.
Plain data, like an agent's events, so a caller can match on them, log them or
turn them into trace spans without importing the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeStarted:
    """A node is about to run. Every node in a wave starts before any finishes."""

    name: str


@dataclass(slots=True)
class NodeFinished:
    """A node returned. Reported as each one ends, not in wave order.

    `duration` is its wall-clock time in seconds, left out of equality: it is a
    measurement of this run, not part of what happened.
    """

    name: str
    duration: float = field(default=0.0, compare=False)


@dataclass(slots=True)
class GraphFinished:
    """The run is over; `state` is what run() would have returned.

    Named apart from the agent's Finished because it carries a different thing:
    the graph's own state object rather than an AgentState.
    """

    state: Any


GraphEvent = NodeStarted | NodeFinished | GraphFinished
"""What Executor.astream_events() yields, ending with one GraphFinished."""
