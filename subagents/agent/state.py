from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from ..errors import ConfigurationError
from ..providers.base import TokenUsage
from .message import Message, as_dict

StopReason = Literal["answer", "step_budget", "paused", "token_budget"]
"""Why the think/act loop stopped. Only "answer" means the model actually
replied - the rest are early exits, so a caller that ignores this can't tell
a real answer from a truncated run."""


@dataclass(slots=True)
class PendingHumanInput:
    """One tool call that's waiting on a human answer."""

    call_id: str | None
    name: str
    question: str


@dataclass(slots=True)
class AgentState:
    """What a run consumed and produced, as fields rather than string keys.

    A typed object because stop_reason is the field a caller must check to know
    whether output is a real answer, and a mistyped string key would silently
    read as None - the exact mistake that makes a truncated run look finished.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    stop_reason: StopReason | None = None
    paused: list[PendingHumanInput] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        """True only when the model actually replied.

        Every other stop reason leaves output empty or partial, so this is the
        one check worth making before trusting it.
        """
        return self.stop_reason == "answer"

    @classmethod
    def of(cls, value: Any) -> AgentState:
        """Build a state from whatever the caller found convenient.

        A prompt string or a list of messages covers the common cases; a dict is
        accepted so existing callers keep working, but an unrecognized key is an
        error rather than a field silently dropped on the way in.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(messages=[Message.human(value).to_dict()])
        if isinstance(value, (list, tuple)):
            return cls(messages=[as_dict(message) for message in value])
        if isinstance(value, dict):
            unknown = set(value) - {f.name for f in fields(cls)}
            if unknown:
                raise ConfigurationError(
                    f"unknown state keys: {', '.join(sorted(unknown))}. An agent "
                    f"owns its own state - keep a graph's fields on the graph's state"
                )
            return cls(**value)
        raise ConfigurationError(
            f"cannot build agent state from {type(value).__name__}; pass a prompt, "
            f"a list of messages, or an AgentState"
        )


@dataclass(slots=True)
class Finished:
    """The run's final state, emitted once the last delta has gone out.

    Streaming a run yields prose as it arrives, but a caller still needs the
    stop reason and structured output at the end - and an async generator cannot
    return a value, so the state comes through as the final event.
    """

    state: AgentState
