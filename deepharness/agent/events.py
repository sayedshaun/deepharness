"""What a run emits besides prose, so a caller can watch it work.

Text alone describes a chat, not an agent: most of a long run's wall clock goes
on tool calls the model never narrates. These say a step began, a tool started
and how it ended - enough to render progress, log a run, or time a tool - and
they are plain data so a caller can match on them without importing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..providers.base import TokenUsage


@dataclass(slots=True)
class StepStarted:
    """A think/act turn is about to ask the model. 1-based, capped by Budget.steps."""

    step: int


@dataclass(slots=True)
class StepFinished:
    """The model's turn for a step is complete: what it cost and how long it took.

    Emitted once the turn's stream ends, before any tool it asked for runs, so
    `duration` is the model call alone. `usage` is None when the provider did
    not report one. `duration` is in seconds and left out of equality: it is a
    measurement of this run, not part of what happened.
    """

    step: int
    usage: TokenUsage | None = None
    duration: float = field(default=0.0, compare=False)


@dataclass(slots=True)
class ToolStarted:
    """A tool is about to run, with the arguments the model sent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(slots=True)
class ToolFinished:
    """A tool's outcome, as the text the model will read.

    `result` is what went into the transcript - already truncated by the
    ContextPolicy - so what a caller displays is what the model saw, rather
    than a fuller version that makes the model's next turn look unfounded.

    A tool that raised sets `failed`; the run continues either way, since the
    error is handed back to the model as that call's result. `duration` is the
    call's wall-clock time in seconds, left out of equality like StepFinished's.
    """

    name: str
    result: str
    failed: bool = False
    call_id: str | None = None
    duration: float = field(default=0.0, compare=False)
