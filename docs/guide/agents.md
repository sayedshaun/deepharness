# Agents

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls or the step budget is spent.

```python
from deepharness import Agent, Budget, Message, tool
from deepharness import OpenAI


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    OpenAI(model="gpt-4o-mini", api_key="sk-..."),
    system="You are a concise assistant.",
    tools=[get_weather],
)

result = await agent.arun("Weather in Oslo?")
print(result.output)
```

`tools=` accepts any number of functions — decorated with [`@tool`](tools.md) or plain — an
agent isn't limited to one. See the [tools guide](tools.md) for how a function becomes a tool.

## `run()` vs `arun()`

Same split as providers: `arun()` is the async path — tool calls requested in the same turn
run **concurrently**, sync tools included: those go to a thread so one blocking call cannot
stall the rest. `run()` is a real synchronous path rather than a wrapper, so it raises
if a registered tool turns out to be `async def` — there's no event loop here to await it:

```python
result = agent.run("Weather in Oslo?")  # sync, sync tools only
result = await agent.arun("Weather in Oslo?")  # async, concurrent tool calls
```

## State

`arun`/`run` return an `AgentState` — a dataclass, not a dict:

| Field | What it holds |
| --- | --- |
| `messages` | The transcript, as wire-form dicts. |
| `output` | The answer: text, or an `output=` instance when one is set. |
| `usage` | `TokenUsage` for this run. |
| `stop_reason` | Why the loop ended: `"answer"`, `"step_budget"`, `"paused"`, `"token_budget"`. |
| `paused` | Any `PendingHumanInput` waiting on a human; empty otherwise. |
| `answered` | `True` only when `stop_reason == "answer"` — check this before trusting `output`. |

For input, pass whatever is convenient — a prompt string, a list of messages, or an
`AgentState` when you are resuming one:

```python
await agent.arun("Weather in Oslo?")
await agent.arun([Message.human("Weather in Oslo?")])
await agent.arun(previous_state)
```

A dict of known fields still works, but a key the agent does not own raises
`ConfigurationError` rather than being dropped silently — an agent owns its own state, so keep
a graph's fields on the graph's state.

## Streaming

`astream()` yields the model's prose as it arrives, and tools still run:

```python
async for chunk in agent.astream("What is 17 * 23?"):
    print(chunk, end="", flush=True)
```

Every turn is streamed, not just the answering one — an agent cannot know in advance whether a
turn will answer or call a tool, so the provider hands back the assembled turn either way and a
tool turn simply yields no text.

When you need the result too, use `astream_events()`. It yields `TextDelta` as text arrives and
one final `Finished` carrying the `AgentState`, because an async generator cannot return a
value:

```python
from deepharness import Finished, TextDelta

async for event in agent.astream_events("What is 17 * 23?"):
    match event:
        case TextDelta(text):
            print(text, end="", flush=True)
        case Finished(state):
            print(f"\nstopped because: {state.stop_reason}")
```

`stream()`/`stream_events()` are the synchronous counterparts. A provider that cannot stream
raises `NotImplementedError` naming itself, rather than yielding nothing.

## Token usage and budgets

Every provider normalizes the vendor's token counts into a `TokenUsage(prompt_tokens,
completion_tokens, total_tokens)`. `Agent` accumulates it across every model call on
`agent.total_usage`, and the final `state.usage` reflects that running total:

```python
agent = Agent(llm, name="assistant", budget=Budget(tokens=50_000))

state = await agent.arun("...")
print(
    state.usage
)  # TokenUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
print(agent.total_usage)  # same object — persists across multiple arun()/run() calls
```

`Budget` bounds a run two ways, and they fail differently. `Budget(tokens=...)` raises
`TokenBudgetExceeded` the moment cumulative usage crosses it — checked right after each model
response, so a run already over budget won't dispatch further tool calls or make another model
call. `Budget(steps=...)` (default 10) caps think/act turns instead, and spending them returns
normally with `stop_reason == "step_budget"` and an empty `output`.

`Budget(steps=1)` is the single-shot case: one model call, no turn to react to a tool result.
Handy for a classify-or-extract step, but an agent with tools will stop at `"step_budget"`
rather than answering whenever it calls one.

## Structured output

Pass `output=` a dataclass and `state.output` becomes a validated instance of it instead of
prose:

```python
from dataclasses import dataclass


@dataclass
class Weather:
    city: str
    celsius: int


agent = Agent(llm, output=Weather, tools=[get_weather])

state = await agent.arun("Weather in Oslo?")
print(state.output.celsius)  # 22
```

It works by offering the model one extra tool, `final_answer`, whose parameters are the
model's schema — so it behaves the same on every provider, with no vendor-specific JSON mode
involved. Two consequences worth knowing:

- If the model replies with prose instead of calling `final_answer`, that is not treated as an
  answer: the agent asks it to call the tool and keeps going, bounded by the step budget. A
  model that never complies ends at `stop_reason == "step_budget"`.
- If the arguments don't fit, an `OutputValidationError` goes back to the model as that call's
  result — the same courtesy a failing tool gets — so it can try again with valid fields. Every
  bad field is reported at once, so one round-trip fixes them all.
- Fields are checked, not coerced from anything: `list[str]`, `Literal`, `Enum`, `X | None` and
  nested dataclasses all validate, an `int` is accepted where a `float` is declared, and `true`
  is rejected for an `int` field even though Python calls a bool an int.

## Human in the loop

Two different things a human can be needed for, and they resolve differently.

**Approval — the call has not run yet.** Mark the tool and the agent pauses *before* running it:

```python
@tool(requires_approval=True)
def wire_transfer(amount_usd: int, to: str) -> str:
    """Send money."""
    return f"sent ${amount_usd:,} to {to}"


state = await agent.arun("Pay the Acme invoice")
print(state.stop_reason)  # "paused"
print(state.paused[0].question)  # Run wire_transfer with {'amount_usd': 50000, ...}?

state = await agent.arun(state.approve())  # runs it now, with the model's arguments
# or
state = await agent.arun(state.reject())  # records "Denied by the user." instead
```

`approve()`/`reject()` return the state, so a resume is one line. With no `call_id` they resolve
every pending call; pass one to rule on a single call. Resuming a paused run without deciding
raises `ConfigurationError` rather than silently continuing.

The gate lives on the tool, not in the prompt, so a model cannot route around it by declining to
ask. And if a turn requests a gated call alongside ordinary ones, **nothing** in that turn runs
until the ruling — a half-applied turn the human is about to refuse would be worse than waiting.

**A question — the tool wants to ask you something.** Raise `HumanInputRequired` and the human's
answer becomes that call's result:

```python
@tool
def confirm(question: str) -> str:
    """Ask the operator something."""
    raise HumanInputRequired(question)


state = await agent.arun("...")
pending = state.paused[0]
state.messages.append(
    Message.tool("yes, proceed", name=pending.name, call_id=pending.call_id)
)
state = await agent.arun(state)
```

The difference matters: an approval defers execution, a question substitutes a result. Use
`pending.needs_approval` to tell them apart.

## Session persistence

`state.messages` is a list of wire-form dicts, so `save_session`/`load_session` round-trip it
through JSON — resume a conversation across process runs:

```python
from deepharness import Agent, Message, load_session, save_session

messages = load_session("session.json")  # [] if the file doesn't exist yet
messages.append(Message.human("Continue where we left off."))

state = await agent.arun({"messages": messages})
save_session("session.json", state.messages)
```

## Messages

`Message` is a `dict` subclass with role-named constructors, so it's a drop-in replacement for
`{"role": ..., "content": ...}` everywhere a message is expected:

```python
from deepharness import Message

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
