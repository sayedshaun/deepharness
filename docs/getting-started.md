# Getting started

## Install

```bash
pip install subagents
```

Or from a clone, for local development:

```bash
git clone https://github.com/your-org/subagents.git
cd subagents
pip install -e .
```

Requires Python 3.11+. Runtime dependencies are just `httpx` and `pydantic`.

## Quickstart

Two lookups run at the same time, then a third node joins their results:

```python
import asyncio
from dataclasses import dataclass

from subagents import Graph


@dataclass
class State:
    sales: str = ""
    churn: str = ""
    summary: str = ""


graph = Graph(State)


@graph.add(start=True)
async def fetch_sales(state: State) -> State:
    await asyncio.sleep(0.5)  # a slow API call, an agent, whatever
    state.sales = "Sales up 12% QoQ."
    return state


@graph.add(start=True)
async def fetch_churn(state: State) -> State:
    await asyncio.sleep(0.3)
    state.churn = "Churn down to 4%."
    return state


@graph.add(end=True)
def summarize(state: State) -> State:
    state.summary = f"{state.sales} {state.churn}"
    return state


graph.connect(fetch_sales, summarize)
graph.connect(fetch_churn, summarize)

executor = graph.build()
result = asyncio.run(executor.run(State()))

print(result.summary)  # Sales up 12% QoQ. Churn down to 4%.
```

Both `fetch_*` nodes are marked `start=True`, so they fire concurrently. `summarize` has two
incoming edges, so it waits for **both** before running — a join, expressed by wiring alone.

## Adding an agent

Drop an [`Agent`](guide/agents.md) inside any node and the graph handles sequencing around it:

```python
from subagents import Agent, Message
from subagents import OpenAI

writer = Agent("writer", model=OpenAI("gpt-4o-mini"), system_prompt="You are concise.")


@graph.add(start=True, end=True)
async def write(state: State) -> State:
    out = await writer.arun({"messages": [Message.human(f"Summarize: {state.sales}")]})
    state.summary = out["output"]
    return state
```

## Next steps

- [Graphs](guide/graph.md) — state, nodes, edges, conditional routing, parallel merging
- [Agents](guide/agents.md) — the think/act loop, tools, token budgets, sessions
- [Tools](guide/tools.md) — turning functions into callable tools
- [Providers](guide/providers.md) — Anthropic, OpenAI, Gemini, and OpenAI-compatible gateways
