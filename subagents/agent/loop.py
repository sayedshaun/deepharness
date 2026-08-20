from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from subagents.providers.base import (
    LLM,
    TextDelta,
    TokenUsage,
)

from ..errors import (
    ConfigurationError,
    OutputValidationError,
    TokenBudgetExceeded,
)
from ..tools.toolbox import Ctx, Toolbox, ToolSpec
from . import turn
from .output import FINAL_TOOL, coerce, final_tool_schema, find_final
from .state import (
    AgentState,
    Budget,
    Finished,
    Message,
    PendingHumanInput,
    StopReason,
)

AgentEvent = TextDelta | Finished
"""What streaming a run emits: prose as it arrives, then the final state."""


@dataclass(slots=True)
class _Ask:
    """A model call the loop needs its driver to perform."""

    messages: list[dict[str, Any]]


@dataclass(slots=True)
class _Dispatch:
    """One turn's tool calls, for the driver to run however it runs them."""

    calls: list[Any]


class Agent:
    """A model plus tools, run as a think/act loop over an AgentState.

    Ask the model, dispatch whatever tools it requests, repeat until it answers
    or the budget runs out. Without a model the run is a passthrough, which is
    what makes an Agent usable as a placeholder node in a Graph.

    The parts worth knowing before reading the loop:

    * Only stop_reason == "answer" means the model actually replied. Every other
      reason leaves output empty or partial, so state.answered is the check to
      make before trusting it.
    * run() is a real synchronous path, not arun() wrapped in an event loop, so
      it raises for a tool that turns out to be `async def`. arun() dispatches a
      turn's tools concurrently, sync ones included - those go to threads.
    * A failing tool does not end the run: the error becomes that call's result
      so the model gets a turn to correct itself.
    * total_usage accumulates across every model call this instance makes, not
      per run, and Budget(tokens=...) turns crossing it into TokenBudgetExceeded
      with the partial state attached.
    * Every public entry point - arun, run, astream, stream - is the same loop;
      only the I/O differs. See astream_events for the one async driver.

    Pausing has two flavours, and they resolve differently: a tool marked
    requires_approval has not run yet (approve it and it runs), while a tool
    raising HumanInputRequired is asking a question (your answer becomes its
    result). See subagents/agent/turn.py.
    """

    __slots__ = (
        "_budget",
        "_final_schema",
        "_model",
        "_name",
        "_output",
        "_system",
        "_tools",
        "_total_usage",
    )

    def __init__(
        self,
        model: LLM | None = None,
        *,
        tools: Iterable[Callable[..., Any]] | Toolbox = (),
        system: str | None = None,
        name: str = "agent",
        budget: Budget | None = None,
        output: type | None = None,
    ):
        self._model = model
        self._tools = tools if isinstance(tools, Toolbox) else Toolbox(tools)
        self._system = system
        self._name = name
        self._budget = budget or Budget()
        self._output = output
        self._final_schema = final_tool_schema(output) if output is not None else None
        self._total_usage = TokenUsage(0, 0, 0)

    # Read-only views: an agent's configuration is settled at construction, and
    # total_usage is live state no caller should be able to reset or inflate.

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> LLM | None:
        return self._model

    @property
    def tools(self) -> Toolbox:
        return self._tools

    @property
    def system(self) -> str | None:
        return self._system

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def output(self) -> type | None:
        return self._output

    @property
    def total_usage(self) -> TokenUsage:
        """Cumulative usage across every model call this agent has made."""
        return self._total_usage

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> Callable[..., Any]:
        """Wrap this Agent as a tool callable from another Agent's toolbox.

        The wrapped tool takes a single `input` string, runs it through
        arun() as a user message, and returns the resulting output text.
        It's async, so register it with an Agent that calls arun() -
        call_sync() raises ConfigurationError for async tools, same as any
        other async tool.

        The sub-agent inherits the caller's deps, so a delegated run keeps the
        database handle or tenant the parent was given.
        """

        async def call(input: str, ctx: Ctx) -> str:
            result = await self.arun(input, deps=ctx.deps)
            return result.output

        call.__name__ = name or self._name
        call._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name or self._name,
            description=description
            or self._system
            or f"Delegate a task to the '{self._name}' agent.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            func=call,
            ctx_params=("ctx",),
        )
        return call

    def _account_for_usage(
        self,
        response: Any,
        state: AgentState,
        messages: list[dict[str, Any]],
    ) -> None:
        if response.usage is None:
            return

        self._total_usage = self._total_usage + response.usage
        if (
            self._budget.tokens is not None
            and self._total_usage.total_tokens > self._budget.tokens
        ):
            raise TokenBudgetExceeded(
                self._name,
                self._total_usage,
                self._budget.tokens,
                state=self._result(state, messages, response.content, "token_budget"),
            )

    def _result(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        output: Any,
        stop_reason: StopReason,
        paused: list[PendingHumanInput] | None = None,
    ) -> AgentState:
        return AgentState(
            messages=messages,
            output=output,
            usage=self._total_usage,
            stop_reason=stop_reason,
            paused=paused or [],
        )

    def _turns(
        self, state: AgentState, messages: list[dict[str, Any]]
    ) -> Generator[_Ask | _Dispatch, Any, AgentState]:
        """The think/act loop itself, with the I/O lifted out of it.

        The loop yields the work it needs done - a model call, or a round of
        tool calls - and its driver performs that work and sends the result
        back. run() can then stay a genuinely synchronous path and arun() a
        concurrent one, while everything they agree on (the step budget, stop
        reasons, pausing for a human, feeding a failed tool back to the model)
        lives here once instead of being maintained in two copies.

        A run that starts from a paused state settles the approvals first: the
        allowed calls run now, with the arguments the model originally sent, and
        only then does the loop go back to the model with their results.
        """
        approved = turn.settle(state, messages, self._name)
        if approved:
            results = yield _Dispatch(approved)
            turn.record_results(messages, approved, results)

        for _ in range(self._budget.steps):
            response = yield _Ask(messages)
            self._account_for_usage(response, state, messages)

            final = find_final(response) if self._final_schema else None
            if final is not None:
                turn.record_request(messages, response)
                try:
                    answer = coerce(self._output, final.arguments)
                except OutputValidationError as exc:
                    # Same courtesy a failing tool gets: hand the model the
                    # error so it can call FINAL_TOOL again with valid fields.
                    messages.append(
                        Message.tool(
                            f"Error: {exc}", name=FINAL_TOOL, call_id=final.id
                        ).to_dict()
                    )
                    continue
                return self._result(state, messages, answer, "answer")

            if not response.tool_calls:
                messages.append(Message.ai(response.content).to_dict())
                if self._final_schema is not None:
                    # output= was asked for, so plain prose is not an answer yet.
                    messages.append(
                        Message.human(f"Answer by calling {FINAL_TOOL}.").to_dict()
                    )
                    continue
                return self._result(state, messages, response.content, "answer")

            if not self._tools:
                raise ConfigurationError(
                    f"{self._name} received tool calls but has no tools registered"
                )

            turn.record_request(messages, response)
            wanted = [call for call in response.tool_calls if call.name != FINAL_TOOL]
            gated = turn.gated(self._tools, wanted)
            if gated:
                # Nothing in this turn runs until the human rules on the gated
                # call: letting the rest run first would half-apply a turn the
                # human may be about to refuse.
                return self._result(state, messages, "", "paused", paused=gated)

            results = yield _Dispatch(wanted)
            dispatched = wanted
            pending = turn.record_results(messages, dispatched, results)
            if pending:
                return self._result(state, messages, "", "paused", paused=pending)

        return self._result(state, messages, "", "step_budget")

    def _passthrough(self, state: AgentState) -> AgentState:
        """Without a model an Agent is inert - a placeholder node in a Graph.

        Returns the state untouched rather than announcing itself: printing from
        inside the agent would make this path untestable without capturing
        stdout, and a library has no business writing to a caller's console.
        """
        return state

    def _schemas(self) -> list[dict[str, Any]] | None:
        schemas = self._tools.schemas()
        if self._final_schema is not None:
            schemas.append(self._final_schema)
        return schemas or None

    async def arun(self, state: Any = None, *, deps: Any = None) -> AgentState:
        """Run to completion. Tool calls in the same turn dispatch concurrently."""
        async for event in self.astream_events(state, deps=deps):
            if isinstance(event, Finished):
                return event.state
        raise AssertionError("a run always ends with Finished")  # pragma: no cover

    def run(self, state: Any = None, *, deps: Any = None) -> AgentState:
        """Synchronous counterpart to arun(). Raises if a tool is `async def`."""
        for event in self.stream_events(state, deps=deps):
            if isinstance(event, Finished):
                return event.state
        raise AssertionError("a run always ends with Finished")  # pragma: no cover

    async def astream_events(
        self, state: Any = None, *, deps: Any = None
    ) -> AsyncIterator[AgentEvent]:
        """Drive one run, emitting prose as it arrives then the final state.

        The only async driver: arun() consumes this and keeps the last event, so
        the loop's mechanics - budget, approvals, tool dispatch - exist once
        rather than once per public method. Providers that cannot really stream
        still work here; their turn simply arrives as a single delta.
        """
        state = AgentState.of(state)
        if self._model is None:
            yield Finished(self._passthrough(state))
            return

        ctx = Ctx(state=state, deps=deps)
        turns = self._turns(state, turn.prepare(state, self._system))
        schemas = self._schemas()
        outcome: Any = None
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    async for event in self._model.astream_events(
                        request.messages, tools=schemas
                    ):
                        if isinstance(event, TextDelta):
                            yield event
                        else:
                            outcome = event.response
                else:
                    outcome = await asyncio.gather(
                        *(
                            self._call_tool(call.name, call.arguments, ctx)
                            for call in request.calls
                        )
                    )
        except StopIteration as done:
            yield Finished(done.value)

    def stream_events(
        self, state: Any = None, *, deps: Any = None
    ) -> Iterator[AgentEvent]:
        """Synchronous counterpart to astream_events()."""
        state = AgentState.of(state)
        if self._model is None:
            yield Finished(self._passthrough(state))
            return

        ctx = Ctx(state=state, deps=deps)
        turns = self._turns(state, turn.prepare(state, self._system))
        schemas = self._schemas()
        outcome: Any = None
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    for event in self._model.stream_events(
                        request.messages, tools=schemas
                    ):
                        if isinstance(event, TextDelta):
                            yield event
                        else:
                            outcome = event.response
                else:
                    outcome = [
                        self._call_tool_sync(call.name, call.arguments, ctx)
                        for call in request.calls
                    ]
        except StopIteration as done:
            yield Finished(done.value)

    async def astream(
        self, state: Any = None, *, deps: Any = None
    ) -> AsyncIterator[str]:
        """Just the text, for the common "print as it types" case."""
        async for event in self.astream_events(state, deps=deps):
            if isinstance(event, TextDelta):
                yield event.text

    def stream(self, state: Any = None, *, deps: Any = None) -> Iterator[str]:
        """Synchronous counterpart to astream()."""
        for event in self.stream_events(state, deps=deps):
            if isinstance(event, TextDelta):
                yield event.text

    async def _call_tool(self, name: str, arguments: dict[str, Any], ctx: Ctx) -> Any:
        try:
            return await self._tools.call(name, ctx=ctx, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see transcript.record_results
            return exc

    def _call_tool_sync(self, name: str, arguments: dict[str, Any], ctx: Ctx) -> Any:
        try:
            return self._tools.call_sync(name, ctx=ctx, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see transcript.record_results
            return exc
