from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from subagents.providers.base import LLM, TokenUsage

from ..errors import ConfigurationError, TokenBudgetExceeded
from ..tools.toolbox import Toolbox
from .message import Message


class Agent:
    """Performs a unit of work, optionally reasoning and acting via an LLM model.

    Without a model, run()/arun() are a no-op passthrough. With a model, they
    run a think/act loop: ask the model for a response, dispatch any tool
    calls it requests, and repeat until the model replies with no tool calls
    or max_steps is reached.

    arun() is fully async: tool calls requested in the same turn run
    concurrently via asyncio.gather. run() is a real synchronous path (the
    model's generate(), tools called directly) - it raises if a registered
    tool turns out to be async, since there's no event loop here to await it.

    tools accepts either a plain list of functions (a Toolbox is built for
    you) or an existing Toolbox instance (e.g. CodingToolbox), used as-is.

    total_usage accumulates token counts across every model call this agent
    makes (reset per Agent instance, not per run()/arun() call). Pass
    token_budget to raise TokenBudgetExceeded once cumulative usage crosses
    it, checked right after each model response - so a run already past
    budget won't dispatch further tool calls or make another model call.
    """

    def __init__(
        self,
        name: str,
        model: LLM | None = None,
        *,
        system_prompt: str | None = None,
        tools: list[Callable[..., Any]] | Toolbox | None = None,
        max_steps: int = 10,
        token_budget: int | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.toolbox: Toolbox | None
        if isinstance(tools, Toolbox):
            self.toolbox = tools
        elif tools:
            self.toolbox = Toolbox()
            for fn in tools:
                self.toolbox.register(fn)
        else:
            self.toolbox = None
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.total_usage = TokenUsage(0, 0, 0)

    def _prepare_messages(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        messages = list(state.get("messages", []))
        if self.system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, Message.system(self.system_prompt))
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

    def _account_for_usage(self, response: Any) -> None:
        if response.usage is None:
            return

        self.total_usage = self.total_usage + response.usage
        if (
            self.token_budget is not None
            and self.total_usage.total_tokens > self.token_budget
        ):
            raise TokenBudgetExceeded(self.name, self.total_usage, self.token_budget)

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            print(f"{self.name} is running...")
            return state

        messages = self._prepare_messages(state)
        tools = self.toolbox.schemas() if self.toolbox else None
        output = ""

        for _ in range(self.max_steps):
            response = await self.model.agenerate(messages, tools=tools)
            self._account_for_usage(response)

            if not response.tool_calls:
                output = response.content
                messages.append(Message.ai(output))
                break

            if self.toolbox is None:
                raise ConfigurationError(
                    f"{self.name} received tool calls but has no toolbox configured"
                )

            self._record_tool_call_request(messages, response)
            results = await asyncio.gather(
                *(
                    self.toolbox.call(call.name, **call.arguments)
                    for call in response.tool_calls
                )
            )
            for call, result in zip(response.tool_calls, results):
                output = str(result)
                messages.append(Message.tool(output, name=call.name, call_id=call.id))

        return {
            **state,
            "messages": messages,
            "output": output,
            "usage": self.total_usage,
        }

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            print(f"{self.name} is running...")
            return state

        messages = self._prepare_messages(state)
        tools = self.toolbox.schemas() if self.toolbox else None
        output = ""

        for _ in range(self.max_steps):
            response = self.model.generate(messages, tools=tools)
            self._account_for_usage(response)

            if not response.tool_calls:
                output = response.content
                messages.append(Message.ai(output))
                break

            if self.toolbox is None:
                raise ConfigurationError(
                    f"{self.name} received tool calls but has no toolbox configured"
                )

            self._record_tool_call_request(messages, response)
            for call in response.tool_calls:
                result = self.toolbox.call_sync(call.name, **call.arguments)
                output = str(result)
                messages.append(Message.tool(output, name=call.name, call_id=call.id))

        return {
            **state,
            "messages": messages,
            "output": output,
            "usage": self.total_usage,
        }
