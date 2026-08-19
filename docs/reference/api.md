# API reference

The public surface, importable from `subagents` unless noted otherwise. This page covers
signatures and behavior only — see the [guide](../guide/agents.md) for narrative
explanations and examples.

## Agents

### `Agent`

```python
Agent(
    model: LLM | None = None,
    *,
    tools: Iterable[Callable] | Toolbox = (),
    system: str | None = None,
    name: str = "agent",
    budget: Budget | None = None,
    output: type | None = None,
)
```

Runs a think/act loop against `model`: request a completion, dispatch any requested tool
calls, repeat until the model stops calling tools or the budget's step limit is reached.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(state: Any = None, *, deps: Any = None) -> AgentState` | Async run. Tool calls in the same turn dispatch concurrently. |
| `run` | `def run(state: Any = None, *, deps: Any = None) -> AgentState` | Sync run. Raises if a registered tool is `async def`. |
| `total_usage` | `TokenUsage` | Cumulative token usage across every call made by this agent instance. Read-only. |
| `budget` | `Budget` | The run's limits; defaults to `Budget()` when none is passed. Read-only. |
| `tools` | `Toolbox` | Always a `Toolbox` — an iterable passed as `tools=` is wrapped in one. Read-only. |
| `output` | `type \| None` | A dataclass; when set, `state.output` is a validated instance of it. |

### `AgentState`

```python
AgentState(
    messages: list[dict] = [],
    output: Any = None,
    usage: TokenUsage = TokenUsage(0, 0, 0),
    stop_reason: StopReason | None = None,
    paused: list[PendingHumanInput] = [],
)
```

What a run consumed and produced. `answered` is `True` only when `stop_reason == "answer"`.
`AgentState.of(value)` builds one from a prompt string, a list of messages, a dict of known
fields, or an existing state; an unknown dict key raises `ConfigurationError`.

### `Budget`

```python
Budget(steps: int = 10, tokens: int | None = None)
```

Frozen dataclass bounding one run. `steps` caps think/act turns — spending them stops the run
with a `"step_budget"` stop reason rather than raising, since the run is truncated but already
paid for. `tokens` caps cumulative usage and raises `TokenBudgetExceeded` when crossed.
`Budget(steps=1)` makes an agent single-shot: one model call, and no turn to react to a tool
result. Non-positive values raise `ConfigurationError`.

`state` is a dict with a `messages` key (a list of [`Message`](#message)/dict). The returned
dict adds `output` (the model's final text) and `usage` (a `TokenUsage`).

Without a `model`, `run`/`arun` are a no-op passthrough — useful as a placeholder while
wiring a graph.

### `TokenBudgetExceeded`

```python
TokenBudgetExceeded(agent_name: str, usage: TokenUsage, budget: int)
```

Raised by `Agent` the moment cumulative `total_usage` crosses `Budget.tokens`, checked right
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

Round-trips `state.messages` through JSON so a conversation can resume across process runs.

## Tools

### `tool`

```python
@tool
@tool(name: str | None = None, description: str | None = None)
```

Decorates a function so it can be registered as a callable tool. Builds a JSON schema from
the function's signature (parameter types, required-ness) and docstring (summary plus
`Args:`/`:param:` descriptions). Handles containers, `Literal`, `Enum` and unions — see [Tools](../guide/tools.md). Works on
both sync and async functions.

### `Ctx`

```python
Ctx(state: Any = None, deps: Any = None)
```

Injected into any tool parameter annotated `Ctx`, and hidden from that tool's schema. `deps` is
whatever the run was given; `state` is the run's `AgentState`.

### `Toolbox`

```python
Toolbox(tools: Iterable[Callable] = ())
```

| Method | Signature | Description |
| --- | --- | --- |
| `register` | `def register(func: Callable) -> Callable` | Registers a function as a tool. |
| `get` | `def get(name: str) -> ToolSpec` | Looks up a registered tool by name. |
| `schemas` | `def schemas() -> list[dict]` | Returns tool schemas, ready to pass to a provider. |
| `call` | `async def call(name: str, **kwargs) -> Any` | Invokes a tool by name, awaiting it if async. |
| `call_sync` | `def call_sync(name: str, **kwargs) -> Any` | Invokes a tool synchronously; raises if it's async. |

## Graphs & execution

### `Graph`

```python
Graph(state_type: type)
```

| Method | Signature | Description |
| --- | --- | --- |
| `add` | `def add(*, name: str \| None = None, start: bool = False, end: bool = False)` | Decorator that registers a function as a node. |
| `connect` | `def connect(source: Callable \| str, target: Callable \| str, *, condition: Callable[[Any], bool] \| None = None, loop: bool = False)` | Declares an edge, optionally conditional. `loop=True` marks a back-edge, re-running the loop head and everything downstream of it. |
| `build` | `def build() -> Executor` | Validates the graph (has a start node, fully reachable, no cycle whose back-edge is unmarked) and returns an `Executor`. |

### `Executor`

```python
async def run(state: Any, *, max_steps: int = 50) -> Any
```

Runs the graph wave by wave: each wave's ready nodes execute concurrently, results merge back
into the state field-by-field, and the next wave is whichever nodes now have all predecessors
satisfied. A field written by two or more concurrent branches is combined by its
[reducer](../guide/graph.md#reducers), or raises `ConcurrentUpdateError` if it declares none.
Taking a `loop=True` edge re-runs the loop head and everything downstream of it; exceeding
`max_steps` raises `StepLimitExceeded`. Returned by `Graph.build()` — not constructed
directly.

### `concat` / `merge_dicts`

```python
concat(base: list, values: list[list]) -> list
merge_dicts(base: dict, values: list[dict]) -> dict
```

Built-in reducers for merging concurrent writes to one field. Declare one with
`field(metadata={"reducer": concat})`.

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

`agenerate`/`generate` are the required pair — implement those two and a custom provider works
everywhere, transport regardless. The streaming pair is optional: the base class raises
`NotImplementedError` naming the provider, so a backend that cannot stream needs no stub.
Vendors that speak REST share their request sequence through `RestCompletions` rather than by
inheriting it (see `providers/rest.py`).

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
