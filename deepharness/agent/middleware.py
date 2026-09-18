"""Where a caller can step into a run without forking the loop.

One object with four optional methods rather than a chain: a chain means
documenting precedence, and "which one ran first" is a class of bug a single
object does not have. Future steps into the loop arrive here as another method
with a no-op default, so an Agent gains capabilities without gaining a
constructor parameter each time.

A plain class rather than an ABC: every method is optional, and an ABC with
@abstractmethod would force a caller who wants one of them to write all four.

The methods are synchronous on purpose. Agent.run() is a genuinely synchronous
path, not arun() wrapped in an event loop, so an async method here would either
need a second form or silently work under arun() alone. Work that has to await
belongs in a tool, which the loop already dispatches both ways.
"""

from __future__ import annotations

from typing import Any

from ..providers.base import ToolCall
from .state import AgentState


class Middleware:
    """Subclass and override only what you need; the rest stay no-ops.

    What these methods may change, and what they must not: none of them decide
    whether a gated call runs. Permissions and requires_approval own that, so
    there stays one answer to "why did this call run?" - before_tool can refuse
    a call, which is recorded the same way a policy denial is, but it cannot
    approve one the policy asked a human about.
    """

    __slots__ = ()

    def before_model(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """The transcript about to be sent, or None to send it unchanged.

        Called before every model call, on the whole transcript and before the
        ContextPolicy shapes it to fit - so a reminder added here still leaves
        the request inside its token budget. The return value is what goes on
        the wire and is not recorded, which is what makes this the place for a
        per-turn reminder: it does not accumulate in state.messages.
        """
        return None

    def before_tool(self, call: ToolCall) -> ToolCall | None:
        """The call as it should run, or None to refuse it.

        Called for each tool call the model asked for, before the permission
        policy rules on it - so a rewritten argument is what the policy sees,
        rather than middleware being able to slip a call past a deny rule by
        rewriting it afterwards. A refused call never runs and the model is
        told so, exactly as a denial is.
        """
        return call

    def after_tool(self, call: ToolCall, result: Any) -> Any:
        """One tool's result on its way back to the model.

        Return it changed to redact, summarize or replace it. `result` is an
        Exception when the tool raised - the loop passes failures around as
        values - so this is where a failure can become an instruction. What is
        returned is what the transcript and the ToolFinished event both carry,
        so the model and the caller cannot be shown different things.
        """
        return result

    def after_step(self, step: int, state: AgentState) -> bool:
        """Whether the run should continue after a step that ran tools.

        Return False to stop it, which ends the run with stop_reason
        "stopped" - an early exit, so `answered` stays False and a caller
        cannot mistake it for a reply. `state` is a snapshot: the transcript so
        far and usage to date, with no stop reason, because the run has not
        finished.
        """
        return True
