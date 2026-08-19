from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from subagents.providers.base import LLM, TokenUsage

from ..errors import (
    ConfigurationError,
    HumanInputRequired,
    OutputValidationError,
    TokenBudgetExceeded,
)
from ..tools.toolbox import Toolbox, ToolSpec
from .budget import Budget
from .message import Message
from .output import FINAL_TOOL, coerce, final_tool_schema

StopReason = Literal["answer", "step_budget", "paused", "token_budget"]
"""Why the think/act loop stopped. Only "answer" means the model actually
replied - the rest are early exits, so a caller that ignores this can't tell
a real answer from a truncated run."""


@dataclass(slots=True)
class _Ask:
    """A model call the loop needs its driver to perform."""

    messages: list[dict[str, Any]]


@dataclass(slots=True)
class _Dispatch:
    """One turn's tool calls, for the driver to run however it runs them."""

    calls: list[Any]


@dataclass(slots=True)
class PendingHumanInput:
    """One tool call that's waiting on a human answer."""

    call_id: str | None
    name: str
    question: str


class Agent:
    """Performs a unit of work, optionally reasoning and acting via an LLM model.

    Without a model, run()/arun() are a no-op passthrough. With a model, they
    run a think/act loop: ask the model for a response, dispatch any tool
    calls it requests, and repeat until the model replies with no tool calls
    or the Budget's step limit is reached.

    arun() is fully async: tool calls requested in the same turn run
    concurrently via asyncio.gather. run() is a real synchronous path (the
    model's generate(), tools called directly) - it raises if a registered
    tool turns out to be async, since there's no event loop here to await it.

    tools accepts either an iterable of functions (a Toolbox is built for
    you) or an existing Toolbox instance used as-is.

    The returned state carries a "stop_reason" (see StopReason) saying why
    the loop ended. Only "answer" means the model replied; "step_budget"
    means it was still calling tools when it ran out of turns, and "output"
    is "" rather than some arbitrary tool's return value. Check it before
    trusting "output" - a truncated run is otherwise indistinguishable from a
    finished one.

    A failing tool does not end the run: the error is fed back to the model
    as that call's result so it can correct itself.

    budget bounds both turns and tokens (see Budget); the default allows 10
    steps and unlimited tokens.

    total_usage accumulates token counts across every model call this agent
    makes (reset per Agent instance, not per run()/arun() call). Set
    Budget(tokens=...) to raise TokenBudgetExceeded once cumulative usage
    crosses it, checked right after each model response - so a run already
    past budget won't dispatch further tool calls or make another model call.
    The partial result rides along on the exception's .state.

    A tool can raise HumanInputRequired to pause the loop instead of
    returning a result. run()/arun() then return early with "paused" set to
    a list of PendingHumanInput - any other tool calls in the same turn
    still run and keep their results. To resume, append a Message.tool(...)
    answer for each pending call (matching name and call_id) and call
    run()/arun() again with the updated state.
    """

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
        self.model = model
        self.tools = tools if isinstance(tools, Toolbox) else Toolbox(tools)
        self.system = system
        self.name = name
        self.budget = budget or Budget()
        self.output = output
        self._final_schema = final_tool_schema(output) if output is not None else None
        self.total_usage = TokenUsage(0, 0, 0)

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> Callable[..., Any]:
        """Wrap this Agent as a tool callable from another Agent's toolbox.

        The wrapped tool takes a single `input` string, runs it through
        arun() as a user message, and returns the resulting output text.
        It's async, so register it with an Agent that calls arun() -
        call_sync() raises ConfigurationError for async tools, same as any
        other async tool.
        """

        async def call(input: str) -> str:
            result = await self.arun({"messages": [Message.human(input)]})
            return result["output"]

        call.__name__ = name or self.name
        call._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=name or self.name,
            description=description
            or self.system
            or f"Delegate a task to the '{self.name}' agent.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            func=call,
        )
        return call

    def _prepare_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        messages = list(state.get("messages", []))
        if self.system and not any(m["role"] == "system" for m in messages):
            messages.insert(0, Message.system(self.system))
        return messages

    @staticmethod
    def _record_tool_call_request(
        messages: list[dict[str, Any]], response: Any
    ) -> None:
        messages.append(
            Message.ai(
                response.content,
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
            )
        )

    def _account_for_usage(
        self,
        response: Any,
        state: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> None:
        if response.usage is None:
            return

        self.total_usage = self.total_usage + response.usage
        if (
            self.budget.tokens is not None
            and self.total_usage.total_tokens > self.budget.tokens
        ):
            raise TokenBudgetExceeded(
                self.name,
                self.total_usage,
                self.budget.tokens,
                state=self._result(state, messages, response.content, "token_budget"),
            )

    def _result(
        self,
        state: dict[str, Any],
        messages: list[dict[str, Any]],
        output: Any,
        stop_reason: StopReason,
        paused: list[PendingHumanInput] | None = None,
    ) -> dict[str, Any]:
        return {
            **state,
            "messages": messages,
            "output": output,
            "usage": self.total_usage,
            "stop_reason": stop_reason,
            "paused": paused,
        }

    @staticmethod
    def _record_tool_results(
        messages: list[dict[str, Any]],
        calls: list[Any],
        results: list[Any],
    ) -> list[PendingHumanInput]:
        """Turn one turn's tool outcomes into messages, returning any that
        need a human.

        Tools are arbitrary user code, so _call_tool catches broadly and
        hands the exception here as a value: a failing tool becomes an error
        message the model gets a turn to correct, instead of killing the run
        and taking the whole message history with it. ConfigurationError is
        the exception - that's a wiring mistake no amount of retrying fixes.
        """
        pending: list[PendingHumanInput] = []
        for call, result in zip(calls, results):
            if isinstance(result, HumanInputRequired):
                pending.append(PendingHumanInput(call.id, call.name, result.question))
                continue
            content = (
                f"Error: {result!r}" if isinstance(result, Exception) else str(result)
            )
            messages.append(Message.tool(content, name=call.name, call_id=call.id))
        return pending

    def _turns(
        self, state: dict[str, Any], messages: list[dict[str, Any]]
    ) -> Generator[_Ask | _Dispatch, Any, dict[str, Any]]:
        """The think/act loop itself, with the I/O lifted out of it.

        The loop yields the work it needs done - a model call, or a round of
        tool calls - and its driver performs that work and sends the result
        back. run() can then stay a genuinely synchronous path and arun() a
        concurrent one, while everything they agree on (the step budget, stop
        reasons, pausing for a human, feeding a failed tool back to the model)
        lives here once instead of being maintained in two copies.
        """
        for _ in range(self.budget.steps):
            response = yield _Ask(messages)
            self._account_for_usage(response, state, messages)

            final = self._find_final(response)
            if final is not None:
                self._record_tool_call_request(messages, response)
                try:
                    answer = coerce(self.output, final.arguments)
                except OutputValidationError as exc:
                    # Same courtesy a failing tool gets: hand the model the
                    # error so it can call FINAL_TOOL again with valid fields.
                    messages.append(
                        Message.tool(f"Error: {exc}", name=FINAL_TOOL, call_id=final.id)
                    )
                    continue
                return self._result(state, messages, answer, "answer")

            if not response.tool_calls:
                messages.append(Message.ai(response.content))
                if self._final_schema is not None:
                    # output= was asked for, so plain prose is not an answer yet.
                    messages.append(Message.human(f"Answer by calling {FINAL_TOOL}."))
                    continue
                return self._result(state, messages, response.content, "answer")

            if not self.tools:
                raise ConfigurationError(
                    f"{self.name} received tool calls but has no tools registered"
                )

            self._record_tool_call_request(messages, response)
            results = yield _Dispatch(
                [call for call in response.tool_calls if call.name != FINAL_TOOL]
            )
            pending = self._record_tool_results(messages, response.tool_calls, results)
            if pending:
                return self._result(state, messages, "", "paused", paused=pending)

        return self._result(state, messages, "", "step_budget")

    def _find_final(self, response: Any) -> Any | None:
        if self._final_schema is None:
            return None
        return next(
            (call for call in response.tool_calls if call.name == FINAL_TOOL), None
        )

    def _passthrough(self, state: dict[str, Any]) -> dict[str, Any]:
        """Without a model an Agent is inert - a placeholder node in a Graph."""
        print(f"{self.name} is running...")
        return state

    def _schemas(self) -> list[dict[str, Any]] | None:
        schemas = self.tools.schemas()
        if self._final_schema is not None:
            schemas.append(self._final_schema)
        return schemas or None

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return self._passthrough(state)

        turns = self._turns(state, self._prepare_messages(state))
        schemas = self._schemas()
        outcome: Any = None
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    outcome = await self.model.agenerate(
                        request.messages, tools=schemas
                    )
                else:
                    outcome = await asyncio.gather(
                        *(
                            self._call_tool(call.name, call.arguments)
                            for call in request.calls
                        )
                    )
        except StopIteration as done:
            return done.value

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return self._passthrough(state)

        turns = self._turns(state, self._prepare_messages(state))
        schemas = self._schemas()
        outcome: Any = None
        try:
            while True:
                request = turns.send(outcome)
                if isinstance(request, _Ask):
                    outcome = self.model.generate(request.messages, tools=schemas)
                else:
                    outcome = [
                        self._call_tool_sync(call.name, call.arguments)
                        for call in request.calls
                    ]
        except StopIteration as done:
            return done.value

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await self.tools.call(name, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see _record_tool_results
            return exc

    def _call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return self.tools.call_sync(name, **arguments)
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - see _record_tool_results
            return exc
