<div align="center">

<img src="docs/assets/logo.svg" width="160" alt="DeepHarness logo">

# DeepHarness

**Compose LLM agents into typed, concurrent workflows.**

A lightweight framework for wiring plain Python functions — and the agents inside them — into a
graph that runs branches in parallel, routes on conditions, and threads one typed state
object through the whole thing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Dependencies: httpx only](https://img.shields.io/badge/dependencies-httpx%20only-6E63F5)](pyproject.toml) [![Async native](https://img.shields.io/badge/async-native-0EA5E9)](#quickstart) [![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff) [![Read the docs](https://img.shields.io/badge/docs-read%20the%20docs-3776AB?logo=materialformkdocs&logoColor=white)](https://sayedshaun.github.io/subagents/)

</div>

---

## Why DeepHarness

- **One dependency. That's it.** `httpx`. Providers talk to vendor REST APIs
  directly — no vendor SDKs, no transitive dependency sprawl.
- **Your state is a dataclass.** Not an untyped dict. Your editor autocompletes it, your type
  checker checks it.
- **Nodes are just functions.** Sync or async, decorated in place. No base class to inherit,
  no runner object to construct.
- **Parallel by default.** Independent branches run concurrently via `asyncio`; sync functions
  are offloaded to threads so a blocking call never stalls the event loop.
- **One provider interface, a dozen vendors.** `OpenAI`, `Anthropic`, `Gemini`, and OpenAI-compatible
  gateways (`Groq`, `Together`, `Fireworks`, `DeepSeek`, `Mistral`, `Cerebras`, `OpenRouter`, `XAI`,
  `Ollama`, `LMStudio`, `VLLM`) all share the same `LLM` interface.

## Install

```bash
pip install deepharness
```

Or install from source:

```bash
pip install git+https://github.com/sayedshaun/subagents.git
```

## Quickstart

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls.

```python
from deepharness import Agent, Message, OpenAI, tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    OpenAI(model="gpt-4o-mini", api_key="sk-..."),
    tools=[get_weather],
)

result = await agent.arun("Weather in Oslo?")
print(result.output)
```

An `Agent` can itself be handed to another agent as a tool via `as_tool()`, so one agent can
delegate a task to another:

```python
researcher = Agent(OpenAI(model="gpt-4o-mini", api_key="sk-..."), name="researcher")

editor = Agent(
    OpenAI(model="gpt-4o-mini", api_key="sk-..."),
    name="editor",
    tools=[researcher.as_tool(description="Look up facts on a topic.")],
)
```

## Graphs

For fixed, known-shape workflows — run these steps, some in parallel, then merge — wire plain
functions (or agents) into a `Graph` instead:

```python
import asyncio
from dataclasses import dataclass

from deepharness import Graph


@dataclass
class State:
    sales: str = ""
    churn: str = ""
    summary: str = ""


graph = Graph(State)


@graph.add(start=True)
async def fetch_sales(state: State) -> State:
    await asyncio.sleep(0.5)
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

`fetch_sales` and `fetch_churn` both run in the first wave, concurrently. `summarize` waits
for both before running.

**Agent or Graph?** Reach for `as_tool()` when an LLM should decide at runtime whether and
which sub-agent to call. Reach for `Graph` when the workflow's shape — which steps, which run
in parallel, in what order — is known ahead of time.

See the **[full documentation](https://sayedshaun.github.io/subagents/)** for graphs, agents,
tools, providers, and the API reference.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff format . && ruff check .
```

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

<div align="center">
<sub>Simple, modular, typed, dependency-light.</sub>
</div>
