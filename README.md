<div align="center">
<h1><img src="https://raw.githubusercontent.com/sayedshaun/deepharness/main/docs/assets/logo.png" width="220" alt="DeepHarness logo"><br>DeepHarness</h1>

**Compose LLM agents into typed, concurrent workflows.**

A lightweight framework for wiring plain Python functions — and the agents inside them — into a
graph that runs branches in parallel, routes on conditions, and threads one typed state
object through the whole thing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyPI](https://img.shields.io/pypi/v/deepharness?logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/deepharness/) [![Dependencies: httpx only](https://img.shields.io/badge/dependencies-httpx%20only-6E63F5)](https://github.com/sayedshaun/deepharness/blob/main/pyproject.toml) [![Async native](https://img.shields.io/badge/async-native-0EA5E9)](#quickstart) [![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff) [![License: MIT](https://img.shields.io/badge/license-MIT-22C55E)](LICENSE) [![Read the docs](https://img.shields.io/badge/docs-read%20the%20docs-3776AB?logo=materialformkdocs&logoColor=white)](https://sayedshaun.github.io/deepharness/)

[Install](#install) · [Quickstart](#quickstart) · [Graphs](#graphs) · [Providers](#providers) · [Docs](https://sayedshaun.github.io/deepharness/) · [Contributing](CONTRIBUTING.md)

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

Or from source:

```bash
pip install git+https://github.com/sayedshaun/deepharness.git
```

Requires Python 3.11 or newer.

## Quickstart

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls
it requests, repeat until the model answers with no tool calls.

```python
from deepharness import Agent, OpenAI, tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."


agent = Agent(
    OpenAI(model="gpt-4o-mini"),  # reads OPENAI_API_KEY from the environment
    tools=[get_weather],
)

result = agent.run("Weather in Oslo?")
print(result.output)
```

Every entry point comes in both flavours — `run()` / `arun()` and `stream()` / `astream()` —
so the same agent drops into a script or an event loop unchanged:

```python
result = await agent.arun("Weather in Oslo?")
```

### Agents as tools

An `Agent` can itself be handed to another agent as a tool via `as_tool()`, so one agent can
delegate a task to another:

```python
researcher = Agent(OpenAI(model="gpt-4o-mini"), name="researcher")

editor = Agent(
    OpenAI(model="gpt-4o-mini"),
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
result = asyncio.run(executor.run())

print(result.summary)  # Sales up 12% QoQ. Churn down to 4%.
```

`fetch_sales` and `fetch_churn` both run in the first wave, concurrently. `summarize` waits
for both before running.

### Agent or Graph?

Reach for `as_tool()` when an LLM should decide at runtime whether and which sub-agent to
call. Reach for `Graph` when the workflow's shape — which steps, which run in parallel, in
what order — is known ahead of time.

## Providers

Pass `api_key=` explicitly, or leave it out and let the provider read its own environment
variable:

| Provider | Environment variable |
| --- | --- |
| `OpenAI` | `OPENAI_API_KEY` |
| `Anthropic` | `ANTHROPIC_API_KEY` |
| `Gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

OpenAI-compatible gateways each read their own key — `GROQ_API_KEY`, `TOGETHER_API_KEY`,
`DEEPSEEK_API_KEY`, and so on — and local runtimes (`Ollama`, `LMStudio`, `VLLM`) need no key
at all. Swapping vendors means swapping the constructor; nothing else changes:

```python
from deepharness import Anthropic, Gemini, Groq, Ollama

Anthropic(model="claude-sonnet-4-5")
Gemini(model="gemini-2.5-flash")
Groq(model="llama-3.3-70b-versatile")
Ollama(model="llama3.2")  # no key needed
```

## Contributing

Issues and pull requests are welcome. To get a working checkout:

```bash
git clone https://github.com/sayedshaun/deepharness.git
cd deepharness
pip install -e ".[dev]"
make test
```

Two conventions are worth knowing before you open a pull request:

- **The dependency budget is one.** `httpx`, and nothing else at runtime. A feature that needs
  a third-party library almost always has a standard-library shape — reach for that instead.
- **Tests must not touch the network.** Inject an `httpx` client or transport so the suite stays
  deterministic and offline.

Run `make fmt` and `make test` before pushing; CI runs the same checks on Python 3.11, 3.12 and
3.13. Keep each commit focused on a single change, with a one-line message prefixed by a
Conventional Commits type (`feat:`, `fix:`, `docs:`, and so on).

Full setup, commands, and design conventions live in **[CONTRIBUTING.md](CONTRIBUTING.md)**; the
rules the code itself is held to are in **[AGENTS.md](AGENTS.md)**.

## License

Released under the **[MIT License](LICENSE)** — Copyright (c) 2026 Sayed Shaun.

You may use, modify, distribute, and sell this software, in open-source or commercial work,
provided the copyright notice and licence text travel with it. It comes with no warranty.

---

<div align="center">
<sub>Simple, modular, typed, dependency-light.</sub>
</div>
