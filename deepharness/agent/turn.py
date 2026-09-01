"""What happens to a single turn: recording it, and gating it on a human.

Both halves are about a turn's outcome rather than the loop's mechanics, and
neither needs an Agent - only the transcript, the tools, and the calls the model
asked for. Keeping them here leaves loop.py to the loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ConfigurationError, HumanInputRequired
from ..tools.toolbox import Toolbox
from .state import AgentState, Message, PendingHumanInput, as_dict


def prepare(state: AgentState, system: str | None) -> list[dict[str, Any]]:
    """The transcript to send, as wire-form dicts, system prompt in front.

    Entries are normalized because a caller resuming a paused run appends a
    Message of their own, and a provider must never be handed one.
    """
    messages = [as_dict(message) for message in state.messages]
    if system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, Message.system(system).to_dict())
    return messages


def record_request(messages: list[dict[str, Any]], response: Any) -> None:
    """Record the assistant turn that asked for tools."""
    messages.append(
        Message.ai(
            response.content,
            tool_calls=[
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ],
        ).to_dict()
    )


def record_results(
    messages: list[dict[str, Any]],
    calls: list[Any],
    results: list[Any],
) -> list[PendingHumanInput]:
    """Record one turn's tool outcomes, returning any that need a human.

    Tools are arbitrary user code, so the dispatcher catches broadly and hands
    the exception here as a value: a failing tool becomes an error message the
    model gets a turn to correct, instead of killing the run and taking the whole
    message history with it.
    """
    pending: list[PendingHumanInput] = []
    for call, result in zip(calls, results, strict=True):
        if isinstance(result, HumanInputRequired):
            pending.append(PendingHumanInput(call.id, call.name, result.question))
            continue
        content = f"Error: {result!r}" if isinstance(result, Exception) else str(result)
        messages.append(
            Message.tool(content, name=call.name, call_id=call.id).to_dict()
        )
    return pending


@dataclass(slots=True)
class ApprovedCall:
    """A gated call the human allowed, replayed with the model's arguments.

    Shaped like a provider's ToolCall so the dispatcher cannot tell the
    difference between a fresh call and a replayed one.
    """

    id: str | None
    name: str
    arguments: dict[str, Any]


def gated(tools: Toolbox, calls: list[Any]) -> list[PendingHumanInput]:
    """Calls a human must allow first, carrying their arguments for later.

    Checked before dispatch rather than inside the tool, so a tool marked
    requires_approval cannot run by accident - and the model cannot route around
    the gate by declining to ask.
    """
    pending: list[PendingHumanInput] = []
    for call in calls:
        if call.name in tools and tools.get(call.name).requires_approval:
            arguments = dict(call.arguments)
            pending.append(
                PendingHumanInput(
                    call_id=call.id,
                    name=call.name,
                    question=f"Run {call.name} with {arguments}?",
                    arguments=arguments,
                )
            )
    return pending


def settle(
    state: AgentState, messages: list[dict[str, Any]], agent_name: str
) -> list[ApprovedCall]:
    """Apply the caller's rulings, returning the calls still to run.

    Rejections are recorded here as that call's result, so the model learns it
    was refused and can say so instead of retrying forever.
    """
    outstanding = [pending for pending in state.paused if pending.needs_approval]
    if not outstanding:
        return []
    if any(pending.approved is None for pending in outstanding):
        raise ConfigurationError(
            f"{agent_name} is paused on {outstanding[0].name}; call "
            f"state.approve() or state.reject() before running it again"
        )
    for pending in outstanding:
        if not pending.approved:
            messages.append(
                Message.tool(
                    "Denied by the user.",
                    name=pending.name,
                    call_id=pending.call_id,
                ).to_dict()
            )
    return [
        ApprovedCall(pending.call_id, pending.name, pending.arguments or {})
        for pending in outstanding
        if pending.approved
    ]
