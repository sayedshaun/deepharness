# Observability

deepharness doesn't ship an exporter, a tracing backend or a dashboard. Which one you use is
your application's choice, and each would be a dependency. What it ships instead is enough raw
data to feed any of them:

| Source | What you get |
| --- | --- |
| `Agent.astream_events()` | `StepStarted`, then `StepFinished(step, usage, duration)` once each model call ends; `ToolStarted`, then `ToolFinished(..., duration)` as each tool ends |
| `Executor.astream_events()` | `NodeStarted(name)`, `NodeFinished(name, duration)` as each node ends, and a final `GraphFinished(state)` |
| `state.usage`, `agent.total_usage` | Token totals for a run, and for everything an agent has run |
| The `deepharness` logger | A warning for each HTTP retry: rate limits, server errors, dropped connections |

Durations are wall-clock seconds. They are left out of event equality, so tests can compare
events without caring how long something took.

## Two ways to receive events

Iterate the stream when you are driving the run yourself, or pass `on_event` and keep calling
`arun()`/`run()` as before. Both see the same events in the same order:

```python
async for event in agent.astream_events(prompt):  # you drive the loop
    ...

agent = Agent(llm, tools=[...], on_event=observer)  # the run calls you
state = await agent.arun(prompt)

executor = graph.build(on_event=observer)  # the same for a graph
result = await executor.run()
```

`on_event` is called with each event before the caller sees it. It may be `async` under
`arun()` and a graph's `run()`; a sync `run()` refuses an async one with `ConfigurationError`.
It runs inside the loop, so keep it quick, and an exception it raises ends the run rather than
being swallowed. The recipes below are written as callables to pass as `on_event`.

## Where the time and tokens went

A summary of a run, collected as it happens:

```python
from collections import defaultdict

from deepharness.agent import StepFinished, ToolFinished


class RunSummary:
    """Totals of model time, tokens and per-tool time, from an agent's events."""

    def __init__(self):
        self.model_seconds = 0.0
        self.tokens = 0
        self.tool_seconds = defaultdict(float)

    def __call__(self, event):
        match event:
            case StepFinished(_, usage, duration):
                self.model_seconds += duration
                self.tokens += usage.total_tokens if usage else 0
            case ToolFinished(name, duration=duration):
                self.tool_seconds[name] += duration

    def report(self) -> str:
        lines = [f"model: {self.model_seconds:.1f}s, {self.tokens} tokens"]
        for name, seconds in sorted(self.tool_seconds.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name}: {seconds:.1f}s")
        return "\n".join(lines)


summary = RunSummary()
agent = Agent(llm, tools=[...], on_event=summary)
state = await agent.arun(prompt)
print(summary.report())
```

A graph reports its nodes the same way:

```python
from deepharness.graph import GraphFinished, NodeFinished, NodeStarted

async for event in graph.build().astream_events(State(task="...")):
    match event:
        case NodeStarted(name):
            print(f"▶ {name}")
        case NodeFinished(name, duration):
            print(f"✓ {name} ({duration:.1f}s)")
        case GraphFinished(state):
            result = state
```

Nodes in one wave all start before any finishes, and each `NodeFinished` arrives when that node
ends. To watch an agent running inside a node, iterate that agent's own `astream_events()` in
the node.

## Retries

The library logs through the standard `logging` module, under the `deepharness` logger, and is
silent until your application configures logging. Retries are logged as warnings, because a
run slowed by backoff otherwise looks exactly like a slow model:

```python
import logging

logging.basicConfig(level=logging.WARNING)
# WARNING:deepharness.http:retrying /chat/completions in 1.0s (attempt 2 of 4): HTTP 429
```

To route only deepharness's messages somewhere, attach a handler to
`logging.getLogger("deepharness")`. Query strings are dropped from logged URLs, so a credential
passed in one never reaches a log.

## Tracing with OpenTelemetry

Each started/finished pair maps onto a span. This needs `opentelemetry-api` in your application,
not in deepharness:

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from deepharness import Finished
from deepharness.agent import StepFinished, StepStarted, ToolFinished, ToolStarted


class SpanRecorder:
    """An agent's events as OpenTelemetry spans: one per model call and per tool."""

    def __init__(self, tracer):
        self._tracer = tracer
        self._spans = {}

    def __call__(self, event):
        match event:
            case StepStarted(step):
                self._spans["model"] = self._tracer.start_span(f"model step {step}")
            case StepFinished(_, usage, _):
                span = self._spans.pop("model")
                if usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", usage.completion_tokens
                    )
                span.end()
            case ToolStarted(name, _, call_id):
                self._spans[call_id or name] = self._tracer.start_span(f"tool {name}")
            case ToolFinished(name, _, failed, call_id):
                span = self._spans.pop(call_id or name)
                if failed:
                    span.set_status(Status(StatusCode.ERROR))
                span.end()
            case Finished():
                for span in self._spans.values():  # a tool waiting on a human
                    span.end()
                self._spans.clear()


tracer = trace.get_tracer("my-agent")
agent = Agent(llm, tools=[...], on_event=SpanRecorder(tracer))

with tracer.start_as_current_span("fix the tests"):
    state = await agent.arun("Make the test suite pass")
```

`on_event` runs inside the call to `arun()`, so the spans it starts become children of whatever
span is current there, and each run is one trace with a span per model call and per tool. A
recorder tracks one run at a time; give concurrent runs a recorder each.
`NodeStarted`/`NodeFinished` map onto spans the same way for a graph. Langfuse, Logfire and
similar tools accept OpenTelemetry, or can be fed from an `on_event` callable directly.
