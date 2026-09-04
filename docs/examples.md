# Examples

Short, self-contained pieces, in the order they get harder. Each one is the smallest code that
shows the thing it is about — copy it, swap the model for a real provider, and it runs.

## An agent with one tool

The think/act loop at its smallest: one model, one tool, one answer.

```python
import asyncio

from deepharness import Agent, OpenAI, tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(OpenAI("gpt-4o-mini"), tools=[get_weather])

state = asyncio.run(agent.arun("Weather in Oslo?"))
print(state.answered, state.output)
```

`answered` is the check worth making: it is `True` only when the model actually replied. Every
other `stop_reason` leaves `output` empty or partial.

## A graph with no model at all

The clearest way to see what nodes, state, and edges are — no LLM involved.

```python
import asyncio
from dataclasses import dataclass

from deepharness import Graph


@dataclass
class State:
    total: int = 0


graph = Graph(State)


@graph.add(start=True)
def double(state: State) -> State:
    state.total = 21
    return state


@graph.add()
def add_one(state: State) -> State:
    state.total += 1
    return state


graph.connect(double, add_one)

print(asyncio.run(graph.build().run()))  # State(total=22)
```

## Tools that fail, and tools that run together

Two tool calls in one turn run concurrently, and a tool that raises is reported back to the
model rather than ending the run:

```python
@tool
def check_flights(city: str) -> str:
    """Look up flights. Fails when the airline API is down."""
    raise RuntimeError("airline API timed out")


agent = Agent(model, tools=[check_flights])
state = await agent.arun("Any flights to Paris?")
```

The failing call becomes `Error: RuntimeError('airline API timed out')` in the transcript, so
the model gets a turn to apologize or try something else. Sync tools run in a thread, so a
blocking one does not stall the others in the same turn.

## Specialists in parallel

Three agents run as concurrent `start=True` nodes, then a fourth node joins their output:

```python
graph.connect(sales, synthesize)
graph.connect(churn, synthesize)
graph.connect(support, synthesize)
```

Wall-clock time is close to the slowest branch, not the sum of the three — the point of marking
independent work `start=True`. Each branch writes its own field, so no reducer is needed.

## Many branches into one field

When branches write the **same** field, declare a reducer so they combine instead of
overwriting (see [state merging](guide/graph.md#parallel-execution-and-state-merging)):

```python
from dataclasses import dataclass, field

from deepharness import concat


@dataclass
class State:
    findings: list[str] = field(default_factory=list, metadata={"reducer": concat})
```

Without one, the merge raises `ConcurrentUpdateError` — a loud mistake rather than a silent lost
write. For a fan-out whose width is only known at runtime, the graph cannot help: nodes are
registered up front, so use `asyncio.gather` inside a single node instead.

## A refinement loop

A back-edge marked `loop=True` turns a chain into a loop. `draft` and `critique` re-run until
the score clears the bar, while `load_rubric` sits upstream of the head and runs once:

```python
graph.connect(load_rubric, critique)
graph.connect(draft, critique)
graph.connect(critique, draft, loop=True, condition=lambda s: s.score < 0.8)
```

See [loops](guide/graph.md#loops) for the re-entry rules and `max_steps`.

## Gating a risky tool on a human

Mark the tool and the agent pauses *before* running it, every time:

```python
from deepharness import tool


@tool(requires_approval=True)
def wire_transfer(amount_usd: int, to: str) -> str:
    """Send money."""
    return f"sent ${amount_usd:,} to {to}"


state = await agent.arun("Pay the Acme invoice")
print(state.stop_reason)  # "paused"
print(state.paused[0].question)  # Run wire_transfer with {'amount_usd': 50000, ...}?

state = await agent.arun(state.approve())  # now it runs
```

`reject()` instead of `approve()` records "Denied by the user." as the result, so the model can
say so rather than retrying. The gate is on the tool, not in the prompt, so a model cannot skip
it by not asking. The pause is an ordinary returned state, not an exception, so it can be
persisted and resumed in another process.

## Streaming, with tools still working

```python
async for chunk in agent.astream("What is 17 * 23?"):
    print(chunk, end="", flush=True)
```

Every turn is streamed, tool turns included — those simply yield no text. Use
`astream_events()` when you also need the final state; see
[streaming](guide/agents.md#streaming).

## Dependencies a tool can reach

A parameter annotated `Ctx` is filled by the runtime and hidden from the model, so a tool reads
per-request dependencies without a global:

```python
from deepharness import Ctx, tool


@tool
def lookup_plan(customer: str, ctx: Ctx) -> str:
    """Look up a customer's plan."""
    return ctx.deps.db.plan_for(customer, tenant=ctx.deps.tenant)


state = await agent.arun("What plan is Acme on?", deps=Deps(db=db, tenant="acme"))
```

## The agent loop, as a graph

`Agent` runs think/act as an internal Python loop. Rebuilding it out of graph nodes gives you a
place to insert an approval gate, a budget check, or a re-planning step mid-cycle:

```python
graph.connect(think, act, condition=lambda s: bool(s.pending_calls))
graph.connect(act, think, loop=True)
```
