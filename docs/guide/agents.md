# Agents

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls or `max_steps` is hit.

```python
from subagents import Agent, Message, tool
from subagents import OpenAI


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    "assistant",
    model=OpenAI(model="gpt-4o-mini", api_key="sk-..."),
    system_prompt="You are a concise assistant.",
    tools=[get_weather],
)

result = await agent.arun({"messages": [Message.human("Weather in Oslo?")]})
print(result["output"])
```

`tools=` accepts any number of functions — decorated with [`@tool`](tools.md) or plain — an
agent isn't limited to one. See the [tools guide](tools.md) for how a function becomes a tool.

## `run()` vs `arun()`

Same split as providers: `arun()` is the async path — tool calls requested in the same turn
run **concurrently**. `run()` is a real synchronous path rather than a wrapper, so it raises
if a registered tool turns out to be `async def` — there's no event loop here to await it:

```python
result = agent.run(
    {"messages": [Message.human("Weather in Oslo?")]}
)  # sync, sync tools only
result = await agent.arun(
    {"messages": [Message.human("Weather in Oslo?")]}
)  # async, concurrent tool calls
```

## Token usage and budgets

Every provider normalizes the vendor's token counts into a `TokenUsage(prompt_tokens,
completion_tokens, total_tokens)`. `Agent` accumulates it across every model call on
`agent.total_usage`, and the final `state["usage"]` reflects that running total:

```python
agent = Agent("assistant", model=llm, token_budget=50_000)

state = await agent.arun({"messages": [Message.human("...")]})
print(
    state["usage"]
)  # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
print(agent.total_usage)  # same object — persists across multiple arun()/run() calls
```

Pass `token_budget` to raise `TokenBudgetExceeded` the moment cumulative usage crosses it —
checked right after each model response, so a run already over budget won't dispatch further
tool calls or make another model call.

## Session persistence

`state["messages"]` is just a list of dicts, so `save_session`/`load_session` round-trip it
through JSON — resume a conversation across process runs:

```python
from subagents import Agent, Message, load_session, save_session

messages = load_session("session.json")  # [] if the file doesn't exist yet
messages.append(Message.human("Continue where we left off."))

state = await agent.arun({"messages": messages})
save_session("session.json", state["messages"])
```

## Messages

`Message` is a `dict` subclass with role-named constructors, so it's a drop-in replacement for
`{"role": ..., "content": ...}` everywhere a message is expected:

```python
from subagents import Message

Message.system("You are a concise assistant.")  # {"role": "system", "content": "..."}
Message.human("What's the weather in Oslo?")  # {"role": "user", "content": "..."}
Message.ai("It's 22°C and sunny.")  # {"role": "assistant", "content": "..."}
Message.tool(
    "22°C, sunny", name="get_weather"
)  # {"role": "tool", "name": "...", "content": "..."}
```

You only need the `tool_calls=`/`call_id=` forms yourself if you're building messages by hand
instead of going through `Agent` — it manages that round-trip for you.

## Agents as nodes

See [Graphs → Agents as nodes](graph.md#agents-as-nodes) for wiring multiple agents into a
multi-agent workflow.
