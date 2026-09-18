# API reference

The public surface, importable from `deepharness` unless noted otherwise. This page covers
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
    context: ContextPolicy | None = None,
    output: type | None = None,
)
```

Runs a think/act loop against `model`: request a completion, dispatch any requested tool
calls, repeat until the model stops calling tools or the budget's step limit is reached.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(state: Any = None, *, deps: Any = None) -> AgentState` | Async run. Tool calls in the same turn dispatch concurrently. |
| `run` | `def run(state: Any = None, *, deps: Any = None) -> AgentState` | Sync run. Raises if a registered tool is `async def`. |
| `astream` | `async def astream(state=None, *, deps=None) -> AsyncIterator[str]` | Text deltas as they arrive; tools still dispatch. |
| `astream_events` | `async def astream_events(...) -> AsyncIterator[AgentEvent]` | Text deltas, progress events and a final `Finished(state)`. |
| `stream` / `stream_events` | sync counterparts | Same, outside an event loop. |
| `total_usage` | `TokenUsage` | Cumulative token usage across every call made by this agent instance. Read-only. |
| `budget` | `Budget` | The run's limits; defaults to `Budget()` when none is passed. Read-only. |
| `context` | `ContextPolicy` | What the transcript may cost; defaults to `ContextPolicy()`. Read-only. |
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
`to_dict()`/`from_dict(data)` round-trip the whole state as JSON-able data — what
`save_session`/`load_session` use.

`approve(call_id=None)` / `reject(call_id=None)` rule on calls waiting in `paused`, returning the
state so a resume reads as `await agent.arun(state.approve())`. Both raise `ConfigurationError`
when nothing matches.

### `PendingHumanInput`

```python
PendingHumanInput(
    call_id: str | None,
    name: str,
    question: str,
    arguments: dict | None = None,
    approved: bool | None = None,
)
```

One paused call. `needs_approval` is `True` when it came from a `requires_approval` tool — those
have not run yet and carry the `arguments` to run with. Otherwise the pause came from a tool
raising `HumanInputRequired`, and the human's answer becomes that call's result.

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

### `ContextPolicy`

```python
ContextPolicy(
    max_tokens: int | None = None,
    tool_result_chars: int | None = 8000,
    keep_last: int = 4,
)
```

Frozen dataclass bounding what a run sends. `tool_result_chars` truncates each tool result as
it is recorded, eliding the middle. `max_tokens` prunes the transcript the model is sent —
oldest turns first, with a note in their place — while `state.messages` keeps every message;
it is `None` (no pruning) by default. Pruning never drops the leading system prompt, never
orphans a tool result, and always keeps the last `keep_last` messages, so a transcript whose
tail alone exceeds the budget is sent over it. Non-positive values raise `ConfigurationError`.

| Member | Signature | Description |
| --- | --- | --- |
| `prune` | `def prune(messages: list[dict]) -> list[dict]` | The transcript as it should be sent; returns the list unchanged when it fits. Override to prune differently. |

### `estimate_tokens`

```python
estimate_tokens(messages: list[dict]) -> int
```

Approximate token cost of a transcript, at roughly four characters per token. A heuristic, not
a tokenizer — a real count would mean a per-vendor dependency.

### `AgentEvent`

```python
AgentEvent = TextDelta | StepStarted | ToolStarted | ToolFinished | Finished
```

What `astream_events()`/`stream_events()` emit.

| Event | Fields | Emitted |
| --- | --- | --- |
| `StepStarted` | `step: int` | Before each model call; 1-based, capped by `Budget.steps`. |
| `ToolStarted` | `name: str`, `arguments: dict`, `call_id: str \| None` | Before a tool runs, with the arguments the model sent. |
| `ToolFinished` | `name: str`, `result: str`, `failed: bool`, `call_id: str \| None` | After a tool returns or raises. `result` is the (truncated) text the model will read. Not emitted for a tool that asked a human — it has no result yet. |
| `TextDelta` | `text: str` | As the model's prose arrives. |
| `Finished` | `state: AgentState` | Once, last, carrying the run's result. |

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
save_session(path: str, session: AgentState | list[dict]) -> None
load_session(path: str) -> AgentState  # empty AgentState if the file doesn't exist
```

Round-trips a whole `AgentState` through JSON so a run can resume across process runs —
including `usage`, `stop_reason` and any call paused on an approval, so a run waiting on a
human can be resumed with `load_session(path).approve()`. A bare message list is accepted on
the way in, and a file holding a bare JSON array is read as a transcript. Structured `output`
is stored as plain data and returns as a dict.

## Tools

### `tool`

```python
@tool
@tool(
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
)
```

Decorates a function so it can be registered as a callable tool. Builds a JSON schema from
the function's signature (parameter types, required-ness) and docstring (summary plus
`Args:`/`:param:` descriptions). Handles containers, `Literal`, `Enum` and unions — see [Tools](../guide/tools.md). Works on
both sync and async functions. `requires_approval=True` makes the agent pause before every call
to it and run it only once approved.

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
async def run(state: Any = None, *, max_steps: int = 50) -> Any
def diagram() -> str
```

Omitting `state` builds one from the type the `Graph` was declared with, so a state whose
fields all have defaults needs no argument. Runs the graph wave by wave: each wave's ready
nodes execute concurrently, results merge back
into the state field-by-field, and the next wave is whichever nodes now have all predecessors
satisfied. A field written by two or more concurrent branches is combined by its
[reducer](../guide/graph.md#reducers), or raises `ConcurrentUpdateError` if it declares none.
Taking a `loop=True` edge re-runs the loop head and everything downstream of it; exceeding
`max_steps` raises `StepLimitExceeded`. Returned by `Graph.build()` — not constructed
directly.

`diagram()` returns a box-drawing picture of the graph as text — layers top to bottom by
wave, `═` for start and end nodes, `▽` for a conditional edge, and back-edges routed up the
right margin. It returns the string rather than printing it. See
[Seeing the shape](../guide/graph.md#seeing-the-shape).

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
| `async for event in astream_events(messages, *, tools=None)` | `TextDelta`, then `Completed(response)` |
| `for event in stream_events(messages, *, tools=None)` | `TextDelta`, then `Completed(response)` |
| `async for chunk in astream(messages, *, tools=None)` | text deltas (filters the above) |
| `for chunk in stream(messages, *, tools=None)` | text deltas |

A streaming turn ends with `Completed`, carrying the assembled `CompletionResponse` — tool
calls included, reassembled from however the vendor fragmented them. `agenerate`/`generate` are
the required pair — implement those two and a custom provider works
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
| `OpenAI` | `OpenAI(model: str, api_key: str \| None = None, *, base_url: str \| None = None, temperature: float \| None = None, stream_usage: bool = True)` |
| `Gemini` | `Gemini(model: str, api_key: str \| None = None)` |

`api_key` falls back to the vendor's standard environment variable when omitted:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

Each provider holds an HTTP connection pool. Call `await model.aclose()` — or
`model.close()` from synchronous code — when you are done with one.

### OpenAI-compatible gateways

All of these share `OpenAI`'s constructor shape and read their own API key from the
environment automatically:

`Groq`, `Together`, `DeepSeek`, `Mistral`, `XAI`, `OpenRouter`, `Fireworks`, `Cerebras`.

Local servers — no API key required: `Ollama`, `VLLM`, `LMStudio`, `LlamaCpp`.

For any other OpenAI-compatible endpoint, construct `OpenAI` directly with an explicit
`base_url` and `api_key`.

## Prebuilt workflows

### `DeepResearch`

```python
DeepResearch(
    model: LLM,
    *,
    tools: Iterable[Callable] | Toolbox = (),
    max_sub_questions: int = 5,
    n_parallel: int | None = None,
    sequential: bool = False,
    planner_system: str = PLANNER_SYSTEM,
    researcher_system: str = RESEARCHER_SYSTEM,
    synthesizer_system: str = SYNTHESIZER_SYSTEM,
    budget: Budget | None = None,
)
```

Plans sub-questions, researches each with its own `Agent` (`tools` reach every researcher, not
the planner or synthesizer), then synthesizes one report. See
[Deep research](../guide/research.md) for the full picture, including `n_parallel` vs
`sequential`.

| Member | Signature | Description |
| --- | --- | --- |
| `arun` | `async def arun(query: str) -> ResearchResult` | Drains `astream_events` and returns its `ResearchFinished` result. |
| `astream_events` | `async def astream_events(query: str) -> AsyncIterator[ResearchEvent]` | The actual driver. Yields `Planning`, `Planned`, `Researching`, `Researched`, `Synthesizing`, `TextDelta` (the report as it is written), then a final `ResearchFinished(result)`. |
| `tools` | `Toolbox` | Always a `Toolbox`. Read-only. |
| `max_sub_questions` | `int` | The planner's cap. Read-only. |
| `sequential` | `bool` | Whether researchers run one at a time, seeing earlier answers. Read-only. |
| `n_parallel` | `int \| None` | The concurrency cap; `None` leaves it to the planner. Read-only. |

Raises `ConfigurationError` at construction for `max_sub_questions < 1`, `n_parallel < 1`, or
passing both `sequential=True` and `n_parallel=`.

### `ResearchResult`

```python
ResearchResult(
    query: str,
    sub_questions: list[str],
    findings: list[Finding],
    report: str,
    usage: TokenUsage,
)
```

`findings` is in the order the answers arrived: completion order when parallel, planned order
when `sequential=True`. `usage` covers every agent the run made — planner, every researcher, and
the synthesizer.

### `Finding`

```python
Finding(question: str, answer: str)
```

One sub-question and the answer a researcher reached for it.

### Events

Importable from `deepharness.prebuilt.research` (not re-exported from `deepharness` itself):

```python
Planning(query: str)
Planned(sub_questions: list[str])
Researching(count: int)
Researched(index: int, question: str)
Synthesizing(findings: int)
ResearchFinished(result: ResearchResult)
```

`ResearchFinished` is named apart from `Agent`'s `Finished` on purpose, even though it fills the
same role: the two wrap different things (`result` vs `state`), and the two streams are commonly
read side by side — `from deepharness import Finished` would silently match nothing against a
`DeepResearch` stream, since it names the wrong class.

`ResearchEvent` is the union of these plus `TextDelta`. Typed rather than formatted prose, so a
caller can count, route, or record events instead of matching substrings against a sentence
meant for a human.
