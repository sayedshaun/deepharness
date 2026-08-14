<div align="center">

<img src="docs/assets/logo.svg" width="160" alt="">

# Subagents

**Compose LLM agents into typed, concurrent workflows.**

A small framework for wiring plain Python functions — and the agents inside them — into a graph that runs branches in parallel, routes on conditions, and threads one typed state object through the whole thing.

<p>
<img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
<img src="https://img.shields.io/badge/dependencies-httpx%20%2B%20pydantic-6E63F5" alt="Dependencies: httpx + pydantic">
<img src="https://img.shields.io/badge/async-native-0EA5E9" alt="Async native">
</p>

**[Read the docs →](https://sayedshaun.github.io/subagents/)**

</div>

---

## Install

```bash
pip install subagents
```

Requires Python 3.11+.

## Quickstart

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
