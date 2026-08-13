from __future__ import annotations

from typing import Any

from subagents.providers.base import LLM

from .toolbox import Toolbox


class Agent:
    """Performs a unit of work, optionally reasoning and acting via an LLM model.

    Without a model, run() is a no-op passthrough. With a model, it runs a
    think/act loop: ask the model for a response, dispatch any tool calls it
    requests, and repeat until the model replies with no tool calls or
    max_steps is reached.
    """

    def __init__(
        self,
        name: str,
        model: LLM | None = None,
        *,
        system_prompt: str | None = None,
        toolbox: Toolbox | None = None,
        max_steps: int = 10,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.toolbox = toolbox
        self.max_steps = max_steps

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            print(f"{self.name} is running...")
            return state

        messages = list(state.get("messages", []))
        if self.system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        tools = self.toolbox.schemas() if self.toolbox else None
        output = ""

        for _ in range(self.max_steps):
            response = await self.model.agenerate(messages, tools=tools)

            if not response.tool_calls:
                output = response.content
                messages.append({"role": "assistant", "content": output})
                break

            if self.toolbox is None:
                raise RuntimeError(f"{self.name} received tool calls but has no toolbox configured")

            for call in response.tool_calls:
                result = await self.toolbox.call(call.name, **call.arguments)
                output = str(result)
                messages.append({"role": "tool", "name": call.name, "content": output})

        return {**state, "messages": messages, "output": output}
