"""The data a run carries: its messages, its limits, and its result.

One module because these only exist in relation to each other - a Message goes
into an AgentState's transcript, a Budget bounds how many more of them a run may
produce, and a session is that transcript on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from ..errors import ConfigurationError
from ..providers.base import TokenUsage


@dataclass(slots=True)
class Message:
    """A chat message, built via role-named constructors instead of a raw dict.

    A dataclass rather than a dict subclass: the wire format is a mapping, but a
    message is not a general-purpose mapping - nothing should be able to pop a
    role off one or update() it with arbitrary keys. to_dict() produces the wire
    form at the boundary, and Agent normalizes whatever it is handed, so a plain
    dict from a caller still works.
    """

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def human(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def ai(
        cls, content: str, *, tool_calls: list[dict[str, Any]] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or None)

    @classmethod
    def tool(cls, content: str, *, name: str, call_id: str | None = None) -> Message:
        return cls(role="tool", content=content, name=name, tool_call_id=call_id)

    def to_dict(self) -> dict[str, Any]:
        """The wire form, without the fields this role does not use."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


def as_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    """One transcript entry as a plain dict, whoever built it."""
    return message.to_dict() if isinstance(message, Message) else dict(message)


@dataclass(frozen=True, slots=True)
class Budget:
    """What an agent is allowed to spend on one run: turns and tokens.

    Both bounds exist because they fail differently. Running out of steps
    means the model was still calling tools and never answered - the run is
    truncated but paid for, so it returns normally with a "step_budget" stop
    reason. Running out of tokens is a cost ceiling the caller set, so it
    raises TokenBudgetExceeded with the partial state attached.

    steps=1 makes the agent single-shot: one model call, one round of tools,
    and the model gets no turn to react to the results. Useful for a
    classify-or-extract step where reflection buys nothing, but note that a
    single-shot agent given tools will stop with "step_budget" rather than an
    answer whenever it calls one.

    Frozen so a Budget can be shared between agents without one run's limits
    being mutated out from under another.
    """

    steps: int = 10
    tokens: int | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ConfigurationError(
                f"Budget.steps must be at least 1, got {self.steps}"
            )
        if self.tokens is not None and self.tokens < 1:
            raise ConfigurationError(
                f"Budget.tokens must be at least 1 when set, got {self.tokens}"
            )


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


def save_session(path: str, messages: list[Message | dict[str, Any]]) -> None:
    """Write a message history to a JSON file, so a session can be resumed later."""
    Path(path).write_text(json.dumps([as_dict(m) for m in messages], indent=2))


def load_session(path: str) -> list[dict[str, Any]]:
    """Read a message history written by save_session().

    Returns [] when the file does not exist yet, so a first run needs no
    special case at the call site.
    """
    file = Path(path)
    if not file.exists():
        return []
    return json.loads(file.read_text())
