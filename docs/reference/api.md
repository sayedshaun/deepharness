# API reference

The public surface, importable from `subagents` unless noted otherwise. This page covers
signatures and behavior only — see the [guide](../guide/agents.md) for narrative
explanations and examples.

## Agents

### `Agent`

```python
Agent(
    name: str,
    model: LLM | None = None,
    *,
    system_prompt: str | None = None,
    tools: list[Callable] | Toolbox | None = None,
    max_steps: int = 10,
    token_budget: int | None = None,
)
```

Runs a think/act loop against `model`: request a completion, dispatch any requested tool
calls, repeat until the model stops calling tools or `max_steps` is reached.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(state: dict) -> dict` | Async run. Tool calls in the same turn dispatch concurrently. |
| `run` | `def run(state: dict) -> dict` | Sync run. Raises if a registered tool is `async def`. |
| `total_usage` | `TokenUsage` | Cumulative token usage across every call made by this agent instance. |

`state` is a dict with a `messages` key (a list of [`Message`](#message)/dict). The returned
dict adds `output` (the model's final text) and `usage` (a `TokenUsage`).

Without a `model`, `run`/`arun` are a no-op passthrough — useful as a placeholder while
wiring a graph.

### `TokenBudgetExceeded`

```python
TokenBudgetExceeded(agent_name: str, usage: TokenUsage, budget: int)
```

Raised by `Agent` the moment cumulative `total_usage` crosses `token_budget`, checked right
after a model response — before any further tool dispatch or model call.

## Messages & sessions

### `Message`

A `dict` subclass — every constructor below returns a plain `{"role": ..., "content": ...}`
style dict, so it's interchangeable with hand-built message dicts anywhere one is expected.

| Constructor | Produces |
| --- | --- |
| `Message.system(content: str)` | `{"role": "system", "content": ...}` |
| `Message.human(content: str)` | `{"role": "user", "content": ...}` |
| `Message.ai(content: str, *, tool_calls: list[dict] \| None = None)` | `{"role": "assistant", "content": ...}` |
| `Message.tool(content: str, *, name: str, call_id: str \| None = None)` | `{"role": "tool", "name": ..., "content": ...}` |

### `save_session` / `load_session`

```python
save_session(path: str, messages: list[dict]) -> None
load_session(path: str) -> list[dict]  # [] if the file doesn't exist
```

Round-trips `state["messages"]` through JSON so a conversation can resume across process runs.

## Tools

### `tool`

```python
@tool
@tool(name: str | None = None, description: str | None = None)
```

Decorates a function so it can be registered as a callable tool. Builds a JSON schema from
the function's signature (parameter types, required-ness) and docstring (description).
Works on both sync and async functions.

### `Toolbox`

```python
Toolbox()
```

| Method | Signature | Description |
| --- | --- | --- |
| `register` | `def register(func: Callable) -> Callable` | Registers a function as a tool. |
| `get` | `def get(name: str) -> ToolSpec` | Looks up a registered tool by name. |
| `schemas` | `def schemas() -> list[dict]` | Returns tool schemas, ready to pass to a provider. |
| `call` | `async def call(name: str, **kwargs) -> Any` | Invokes a tool by name, awaiting it if async. |
| `call_sync` | `def call_sync(name: str, **kwargs) -> Any` | Invokes a tool synchronously; raises if it's async. |

### `CodingToolbox` and `set_auto_approve`

`from subagents.tools import CodingToolbox, set_auto_approve`

A ready-to-use `Toolbox` pre-registered with five file/shell tools: `read_file`, `write_file`,
`apply_patch`, `run_shell`, `grep`. `CodingToolbox.as_list()` returns them as a plain list of
functions, for `tools=[...]`-style construction.

`set_auto_approve(enabled: bool)` toggles whether the risky tools (`write_file`,
`apply_patch`, `run_shell`) skip their confirmation prompt — off by default.

## Graphs & execution

### `Graph`

```python
Graph(state_type: type)
```

| Method | Signature | Description |
| --- | --- | --- |
| `add` | `def add(*, name: str \| None = None, start: bool = False, end: bool = False)` | Decorator that registers a function as a node. |
| `connect` | `def connect(source: Callable \| str, target: Callable \| str, *, condition: Callable[[Any], bool] \| None = None)` | Declares an edge, optionally conditional. |
| `build` | `def build() -> Executor` | Validates the graph (has a start node, fully reachable, acyclic) and returns an `Executor`. |

### `Executor`

```python
async def run(state: Any) -> Any
```

Runs the graph wave by wave: each wave's ready nodes execute concurrently, results merge back
into the state field-by-field, and the next wave is whichever nodes now have all predecessors
satisfied. Returned by `Graph.build()` — not constructed directly.

### `ExecutionError`

```python
ExecutionError(node_name: str, original: Exception)
```

Raised by `Executor.run` when a node function raises; wraps the original exception with the
name of the node that failed.

## Providers

Every provider below implements the same interface:

| Method | Returns |
| --- | --- |
| `await agenerate(messages: list[dict], *, tools: list[dict] \| None = None)` | `CompletionResponse` |
| `generate(messages: list[dict], *, tools: list[dict] \| None = None)` | `CompletionResponse` |
| `async for chunk in astream(messages, *, tools=None)` | text deltas |
| `for chunk in stream(messages, *, tools=None)` | text deltas |

### Response types

```python
ToolCall(name: str, arguments: dict, id: str | None = None)
TokenUsage(prompt_tokens: int, completion_tokens: int, total_tokens: int)
CompletionResponse(content: str, tool_calls: list[ToolCall] = [], usage: TokenUsage | None = None)
```

### Direct providers

| Class | Signature |
| --- | --- |
| `Anthropic` | `Anthropic(model: str, api_key: str \| None = None, max_tokens: int = 4096)` |
| `OpenAI` | `OpenAI(model: str, api_key: str \| None = None, *, base_url: str \| None = None, temperature: float \| None = None)` |
| `Gemini` | `Gemini(model: str, api_key: str \| None = None)` |

`api_key` falls back to the vendor's standard environment variable (e.g. `OPENAI_API_KEY`)
when omitted.

### OpenAI-compatible gateways

All of these share `OpenAI`'s constructor shape and read their own API key from the
environment automatically:

`Groq`, `Together`, `DeepSeek`, `Mistral`, `XAI`, `OpenRouter`, `Fireworks`, `Cerebras`.

Local servers — no API key required: `Ollama`, `VLLM`, `LMStudio`.

For any other OpenAI-compatible endpoint, construct `OpenAI` directly with an explicit
`base_url` and `api_key`.
