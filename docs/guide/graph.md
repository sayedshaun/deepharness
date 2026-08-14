# Graphs

`Graph` wires plain functions into a DAG that runs independent branches concurrently and
threads one typed state object through the whole run. It has no idea what an LLM is — use it
for plain orchestration, or put an [`Agent`](agents.md) inside a node.

## State

One dataclass, threaded through every node. A node receives it, mutates it, returns it.

```python
from dataclasses import dataclass
from subagents import Graph


@dataclass
class State:
    topic: str = ""
    output: str = ""


graph = Graph(State)
```

`executor.run()` deep-copies the state you hand it, so your original object is never mutated.

## Nodes

`@graph.add()` registers a function as a node. The node name defaults to the function's name.

```python
@graph.add(start=True, name="research")  # explicit name
def research_node(state: State) -> State:
    return state
```

| Flag        | Meaning                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------- |
| `start=True` | Runs in the first wave, with no upstream dependency. At least one node needs it.             |
| `end=True`   | Marks a terminal node. Documentation only today — execution stops when nothing is left to run. |
| `name=...`   | Overrides the node name (defaults to the function name).                                     |

Both `def` and `async def` work. Sync functions run in a worker thread, so a blocking HTTP
call or an `input()` prompt won't block the event loop.

## Edges and conditional routing

`graph.connect(source, target)` declares "target runs after source." Pass either the function
object or its name.

Add a `condition` to branch — the edge is only followed if it returns truthy against the
current state:

```python
graph.connect(review, publish, condition=lambda s: s.approved)
graph.connect(review, revise, condition=lambda s: not s.approved)
```

After `review` finishes, exactly one of `publish` / `revise` runs. If **every** incoming edge
of a node evaluates falsy once its predecessors are done, that node is skipped rather than
left hanging.

## Parallel execution and state merging

Each wave runs its ready nodes concurrently, then merges results back field by field:

- A field is written back only if the branch **changed** it relative to the state the wave
  started from.
- If two concurrent branches change the *same* field, the later one (in registration order)
  wins.

!!! warning "Merging is whole-field, last-writer-wins"
    It is not additive. Two parallel branches that both `append` to the same list will
    clobber each other rather than combine. **Give each parallel branch its own field**
    (`sales` / `churn` in the quickstart), and join them in a downstream node.

## Validation

`graph.build()` fails fast on structural mistakes, before anything runs:

- no node marked `start=True`
- a node that is unreachable (no `start` flag, no incoming edges)
- a cycle

The graph is a DAG: every node runs at most once. Iteration belongs *inside* a node — an
`Agent` already loops over its own tool calls via `max_steps`.

## Human in the loop

A pause for human input is just a node. Because sync nodes run in a worker thread, a blocking
prompt is fine:

```python
@graph.add(name="review")
def review(state: State) -> State:
    print(f"\nDraft:\n  {state.draft}\n")
    state.approved = input("Approve? [y/n]: ").strip().lower() == "y"
    return state


graph.connect(write, review)
graph.connect(review, publish, condition=lambda s: s.approved)
graph.connect(review, revise, condition=lambda s: not s.approved)
```

This covers "block until a human responds while the process runs." Pausing a run and resuming
it days later would need state persistence, which the executor does not do today.

## Errors

A node that raises is wrapped in `ExecutionError`, carrying the node name and the original
exception:

```python
from subagents import ExecutionError

try:
    await executor.run(State())
except ExecutionError as exc:
    print(exc.node_name, exc.original)
```

## Agents as nodes

The graph engine is agent-agnostic: `Agent.arun()` speaks `dict` state, the graph speaks your
dataclass, so a node is the adapter between them. One agent per node is the natural way to
build a multi-agent system:

```python
from subagents import Agent, Message

researcher = Agent("researcher", model=llm, system_prompt="You research topics.")
writer = Agent("writer", model=llm, system_prompt="You write short reports.")


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

Each agent keeps its own prompt, tools, and model; the graph handles sequencing, branching,
and fan-out between them.
