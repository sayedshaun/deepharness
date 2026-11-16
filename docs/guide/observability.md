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

## Where the time and tokens went

A summary of one run, from its events:

```python
from collections import defaultdict

from deepharness import Finished
from deepharness.agent import StepFinished, ToolFinished


async def run_with_summary(agent, prompt):
    model_seconds = 0.0
    tool_seconds = defaultdict(float)
    tokens = 0
    async for event in agent.astream_events(prompt):
        match event:
            case StepFinished(_, usage, duration):
                model_seconds += duration
                tokens += usage.total_tokens if usage else 0
            case ToolFinished(name, duration=duration):
                tool_seconds[name] += duration
            case Finished(state):
                final = state
    print(f"model: {model_seconds:.1f}s, {tokens} tokens")
    for name, seconds in sorted(tool_seconds.items(), key=lambda item: -item[1]):
        print(f"  {name}: {seconds:.1f}s")
    return final
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

tracer = trace.get_tracer("my-agent")


async def traced_run(agent, prompt):
    with tracer.start_as_current_span(f"agent {agent.name}"):
        spans = {}
        async for event in agent.astream_events(prompt):
            match event:
                case StepStarted(step):
                    spans["model"] = tracer.start_span(f"model step {step}")
                case StepFinished(_, usage, _):
                    span = spans.pop("model")
                    if usage:
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", usage.prompt_tokens
                        )
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", usage.completion_tokens
                        )
                    span.end()
                case ToolStarted(name, _, call_id):
                    spans[call_id or name] = tracer.start_span(f"tool {name}")
                case ToolFinished(name, _, failed, call_id):
                    span = spans.pop(call_id or name)
                    if failed:
                        span.set_status(Status(StatusCode.ERROR))
                    span.end()
                case Finished(state):
                    final = state
        for span in spans.values():  # a tool waiting on a human has no ToolFinished
            span.end()
    return final
```

Spans started inside `start_as_current_span` become its children, so each run is one trace with
a span per model call and per tool. `NodeStarted`/`NodeFinished` map onto spans the same way.
Langfuse, Logfire and similar tools accept OpenTelemetry, or can be fed from the same `match`
statement directly.
