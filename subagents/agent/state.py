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
    """One tool call waiting on a human, and what happens when one answers.

    Two kinds of pause end up here, and they resolve differently:

    * A tool that raised HumanInputRequired is *asking* something. The human's
      answer becomes that call's result - the tool has already run.
    * A tool marked requires_approval has not run at all. Approving it runs it
      now and records the real result; rejecting it records the refusal. That is
      the difference between "yes" meaning something and "yes" being recorded as
      the answer to a question nobody asked.

    arguments is kept for the second case, because re-dispatching an approved
    call needs the arguments the model sent.
    """

    call_id: str | None
    name: str
    question: str
    arguments: dict[str, Any] | None = None
    """Present only for an approval pause: the call to run once approved."""

    approved: bool | None = None
    """Set by AgentState.approve()/reject(); None means still waiting."""

    @property
    def needs_approval(self) -> bool:
        return self.arguments is not None


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

    def approve(self, call_id: str | None = None) -> AgentState:
        """Allow a paused call to run on the next run()/arun().

        Returns self so a resume reads as one line:
        `await agent.arun(state.approve())`. With no call_id every pending call
        is approved, which is the common case of a single gated tool.
        """
        return self._resolve(call_id, True)

    def reject(self, call_id: str | None = None) -> AgentState:
        """Refuse a paused call; the model is told it was denied."""
        return self._resolve(call_id, False)

    def _resolve(self, call_id: str | None, approved: bool) -> AgentState:
        matched = False
        for pending in self.paused:
            if call_id is None or pending.call_id == call_id:
                pending.approved = approved
                matched = True
        if not matched:
            raise ConfigurationError(
                "no paused call to resolve"
                + (f" with call_id {call_id!r}" if call_id else "")
            )
        return self

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
