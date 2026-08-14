<div align="center">

<img src="docs/assets/logo.svg" width="120" alt="">

# Subagents

**Compose LLM agents into typed, concurrent workflows.**

A small framework for wiring plain Python functions — and the agents inside them — into a graph that runs branches in parallel, routes on conditions, and threads one typed state object through the whole thing.

<p>
<img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
<img src="https://img.shields.io/badge/dependencies-httpx%20%2B%20pydantic-6E63F5" alt="Dependencies: httpx + pydantic">
<img src="https://img.shields.io/badge/async-native-0EA5E9" alt="Async native">
</p>

</div>

---

## Why Subagents

- **Two dependencies. That's it.** `httpx` and `pydantic`. Providers talk to vendor REST APIs directly — no OpenAI SDK, no `google-genai`, no transitive dependency sprawl.
- **Your state is a dataclass.** Not an untyped dict. Your editor autocompletes it, your type checker checks it.
- **Nodes are just functions.** Sync or async, decorated in place. No base class to inherit, no runner object to construct.
- **Parallel by default.** Independent branches run concurrently via `asyncio`; sync functions are offloaded to threads so a blocking call never stalls the loop.
- **Agents are optional.** The graph engine has no idea what an LLM is. Use it for plain orchestration, or drop an `Agent` inside a node.

---

## Install

```bash
git clone https://github.com/your-org/subagents.git
cd subagents
pip install -e .
```

Requires Python 3.11+.

---

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
    await asyncio.sleep(0.5)          # a slow API call, an agent, whatever
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

print(result.summary)   # Sales up 12% QoQ. Churn down to 4%.
```

Both `fetch_*` nodes are marked `start=True`, so they fire concurrently. `summarize` has two incoming edges, so it waits for **both** before running — a join, expressed by wiring alone.

---

## Core concepts

### State

One dataclass, threaded through every node. A node receives it, mutates it, returns it.

`executor.run()` deep-copies the state you hand it, so your original object is never mutated.

### Nodes

`@graph.add()` registers a function as a node. The node name defaults to the function's name.

```python
@graph.add(start=True, name="research")   # explicit name
def research_node(state: State) -> State:
    return state
```

| Flag | Meaning |
| --- | --- |
| `start=True` | Runs in the first wave, with no upstream dependency. At least one node needs it. |
| `end=True` | Marks a terminal node. Documentation only today — execution stops when nothing is left to run. |
| `name=...` | Overrides the node name (defaults to the function name). |

Both `def` and `async def` work. Sync functions are run through `asyncio.to_thread`, so a blocking HTTP call or a `input()` prompt won't block the event loop.

### Edges and conditional routing

`graph.connect(source, target)` declares "target runs after source." Pass either the function object or its name.

Add a `condition` to branch — the edge is only followed if it returns truthy against the current state:

```python
graph.connect(review, publish, condition=lambda s: s.approved)
graph.connect(review, revise,  condition=lambda s: not s.approved)
```

After `review` finishes, exactly one of `publish` / `revise` runs. If **every** incoming edge of a node evaluates falsy once its predecessors are done, that node is skipped rather than left hanging.

### Parallel execution and state merging

Each wave runs its ready nodes concurrently, then merges results back field by field:

- A field is written back only if the branch **changed** it relative to the state the wave started from.
- If two concurrent branches change the *same* field, the later one (in registration order) wins.

> [!IMPORTANT]
> Merging is whole-field, last-writer-wins — it is not additive. Two parallel branches that both `append` to the same list will clobber each other rather than combine. **Give each parallel branch its own field** (`sales` / `churn` above), and join them in a downstream node.

### Validation

`graph.build()` fails fast on structural mistakes, before anything runs:

- no node marked `start=True`
- a node that is unreachable (no `start` flag, no incoming edges)
- a cycle

The graph is a DAG: every node runs at most once. Iteration belongs *inside* a node — an `Agent` already loops over its own tool calls via `max_steps`.

### Human in the loop

A pause for human input is just a node. Because sync nodes run in a worker thread, a blocking prompt is fine:

```python
@graph.add(name="review")
def review(state: State) -> State:
    print(f"\nDraft:\n  {state.draft}\n")
    state.approved = input("Approve? [y/n]: ").strip().lower() == "y"
    return state


graph.connect(write, review)
graph.connect(review, publish, condition=lambda s: s.approved)
graph.connect(review, revise,  condition=lambda s: not s.approved)
```

This covers "block until a human responds while the process runs." Pausing a run and resuming it days later would need state persistence, which the executor does not do today.

### Errors

A node that raises is wrapped in `ExecutionError`, carrying the node name and the original exception:

```python
from subagents import ExecutionError

try:
    await executor.run(State())
except ExecutionError as exc:
    print(exc.node_name, exc.original)
```

---

## Agents

An `Agent` runs a think/act loop against a model: ask for a response, dispatch any tool calls it requests, repeat until the model answers with no tool calls or `max_steps` is hit.

```python
from subagents import Agent, Message, tool
from subagents.providers.openai import OpenAI


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

`@tool` derives the JSON schema from the signature and docstring — parameter types from annotations, the description from the docstring, `required` from which parameters lack defaults. Tool functions may be sync or async. `tools` accepts any number of functions — decorated with `@tool` or plain — an agent isn't limited to one.

### run() vs arun()

Same split as providers: `arun()` is the async path — tool calls requested in the same turn run **concurrently** via `asyncio.gather`. `run()` is a real synchronous path (calls the model's `generate()`, calls tools directly) rather than a wrapper — so it raises if a registered tool turns out to be `async def`, since there's no event loop here to await it:

```python
result = agent.run({"messages": [Message.human("Weather in Oslo?")]})       # sync, sync tools only
result = await agent.arun({"messages": [Message.human("Weather in Oslo?")]})  # async, concurrent tool calls
```

### Coding tools

`subagents.tools` ships a ready-made tool set for building a coding agent: `read_file`, `write_file`, `apply_patch`, `run_shell`, `grep` — plain functions, not tied to any language or platform. The risky ones (`write_file`, `apply_patch`, `run_shell`) ask for confirmation via `input()` before acting; call `set_auto_approve(True)` to skip that (tests, scripts, anywhere you've already decided to trust the agent).

`CodingToolbox` bundles all five as a ready-to-use `Toolbox` — pass it straight to `tools=`:

```python
from subagents import Agent
from subagents.providers.openai import OpenAI
from subagents.tools import CodingToolbox, set_auto_approve

set_auto_approve(True)  # skip confirmation prompts, e.g. for a CI script

coding_agent = Agent(
    "coder",
    OpenAI("gpt-4.1-mini"),
    system_prompt="You are a coding assistant. Use the tools to read, write, and run code.",
    tools=CodingToolbox(),
    max_steps=20,
)
```

`tools=` accepts either form — a plain list of functions (a `Toolbox` is built for you) or an existing `Toolbox`/subclass, used as-is.

### Session persistence

`state["messages"]` is just a list of dicts, so `save_session`/`load_session` round-trip it through JSON — resume a conversation across process runs:

```python
from subagents import Agent, Message, load_session, save_session

messages = load_session("session.json")  # [] if the file doesn't exist yet
messages.append(Message.human("Continue where we left off."))

state = await agent.arun({"messages": messages})
save_session("session.json", state["messages"])
```

### Messages

`Message` is a `dict` subclass with role-named constructors, so it's a drop-in replacement for `{"role": ..., "content": ...}` everywhere a message is expected:

```python
from subagents import Message

Message.system("You are a concise assistant.")   # {"role": "system", "content": "..."}
Message.human("What's the weather in Oslo?")     # {"role": "user", "content": "..."}
Message.ai("It's 22°C and sunny.")                # {"role": "assistant", "content": "..."}
Message.tool("22°C, sunny", name="get_weather")   # {"role": "tool", "name": "...", "content": "..."}
```

`Agent` uses two more constructors internally to keep tool-call round-trips correct: `Message.ai(content, tool_calls=[{"id": ..., "name": ..., "arguments": ...}])` records the model's tool-call request, and `Message.tool(content, name=..., call_id=...)` links the result back to it via `tool_call_id`. Every provider converts this into its own required shape (OpenAI's `tool_calls[].id`, Anthropic's `tool_use`/`tool_result` blocks, Gemini's `functionCall`/`functionResponse`) — you only need these yourself if you're building messages by hand instead of going through `Agent`.

### Agents as nodes

The graph engine is agent-agnostic: `Agent.arun()` speaks `dict` state, the graph speaks your dataclass, so a node is the adapter between them. One agent per node is the natural way to build a multi-agent system:

```python
researcher = Agent("researcher", model=llm, system_prompt="You research topics.")
writer     = Agent("writer",     model=llm, system_prompt="You write short reports.")

@graph.add(start=True)
async def research(state: State) -> State:
    out = await researcher.arun({"messages": [Message.human(state.topic)]})
    state.research = out["output"]
    return state


@graph.add(end=True)
async def write(state: State) -> State:
    out = await writer.arun({"messages": [Message.human(state.research)]})
    state.draft = out["output"]
    return state


graph.connect(research, write)
```

Each agent keeps its own prompt, tools, and model; the graph handles sequencing, branching, and fan-out between them.

---

## Providers

Every provider implements the same four methods, so swapping vendors is a one-line change:

| Method | Returns |
| --- | --- |
| `await agenerate(messages, tools=None)` | `CompletionResponse` |
| `generate(messages, tools=None)` | `CompletionResponse` |
| `async for chunk in astream(messages, tools=None)` | text deltas |
| `for chunk in stream(messages, tools=None)` | text deltas |

```python
from subagents import Anthropic, Gemini, OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key="sk-...")
llm = Gemini(model="gemini-2.0-flash", api_key="...")           # same interface
llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")  # same interface

response = await llm.agenerate([{"role": "user", "content": "Hi"}])
print(response.content, response.tool_calls)

async for chunk in llm.astream([{"role": "user", "content": "Write a haiku"}]):
    print(chunk, end="", flush=True)
```

Responses normalize to a vendor-neutral `CompletionResponse(content: str, tool_calls: list[ToolCall])`.

> [!NOTE]
> Streaming yields **text deltas only**. Tool calls are resolved through `generate` / `agenerate`.

> [!NOTE]
> `Anthropic`'s wire format differs more than Gemini/OpenAI do from each other: the system prompt is a separate top-level field (a `{"role": "system", ...}` message gets moved there automatically), and there's no `role: "tool"` — tool calls/results become `tool_use`/`tool_result` content blocks instead. `ToolCall`/`Message` carry the vendor's call id (`Message.ai(..., tool_calls=[...])`, `Message.tool(..., call_id=...)`) so this round-trips correctly across turns for all three providers, including Anthropic's strict user/assistant alternation requirement.

All three providers accept an injectable `client` / `sync_client` (`httpx.AsyncClient` / `httpx.Client`), which is what makes the test suite run without touching the network.

### Adding a provider

Subclass `LLM`, implement the four methods, and reuse `HTTPClient` for transport:

```python
from subagents.providers.base import CompletionResponse, LLM
from subagents.providers.client import HTTPClient


class MyProvider(LLM):
    def __init__(self, model: str, api_key: str | None = None):
        self._http = HTTPClient("https://api.example.com/v1", headers={"X-Key": api_key})
        self._model = model

    async def agenerate(self, messages, *, tools=None) -> CompletionResponse:
        response = await self._http.post("/chat", json={"model": self._model, "messages": messages})
        ...
```

`HTTPClient` owns client construction, `raise_for_status`, and the streaming context managers, so provider modules never touch `httpx` directly.

### OpenAI-compatible gateways

Groq, Together, DeepSeek, Mistral, xAI, OpenRouter, Fireworks, Cerebras, and local servers (Ollama, vLLM, LM Studio) all speak the same wire format as OpenAI's Chat Completions API — just a different base URL and API key. `subagents.providers.gateways` gives each one a four-line class:

```python
from subagents import Groq

model = Groq("llama-3.3-70b-versatile", temperature=0)   # reads GROQ_API_KEY automatically
```

`OpenAI` itself carries the `base_url`/`env_key` machinery: a subclass overrides `default_base_url` and `env_key` as class attributes, and `api_key` is read from that environment variable whenever it isn't passed explicitly (local servers set `env_key = ""` to skip that lookup and just send an empty bearer token). For a gateway that isn't listed, construct `OpenAI` directly — there's nothing to register:

```python
OpenAI("llama-3.3-70b", base_url="https://llm.internal/v1", api_key=key)
```

---

## Project layout

```
subagents/
├── agent/          Agent think/act loop, Toolbox, @tool
├── graph/          Graph (nodes + edges), Executor (waves + merging)
└── providers/      LLM interface, HTTPClient, wire types, OpenAI, Gemini
```

---

## Development

```bash
pip install -e . pytest pytest-asyncio
pytest -q
ruff format . && ruff check .
```

---

<div align="center">
<sub>Simple, modular, typed, dependency-light.</sub>
</div>
