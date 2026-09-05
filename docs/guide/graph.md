# Graphs

`Graph` wires plain functions into a graph that runs independent branches concurrently and
threads one typed state object through the whole run. It has no idea what an LLM is — use it
for plain orchestration, or put an [`Agent`](agents.md) inside a node.

## State

One dataclass, threaded through every node. A node receives it, mutates it, returns it.

```python
from dataclasses import dataclass
from deepharness import Graph


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
- If **one** branch changed a field, its value is written.
- If **two or more** branches changed the same field, the field's *reducer* decides how to
  combine them. With no reducer declared, the merge raises `ConcurrentUpdateError` rather
  than silently dropping a branch's work.

### Reducers

Declare a reducer in the field's `metadata`, so the merge policy lives with the data:

```python
from dataclasses import dataclass, field

from deepharness import concat


@dataclass
class State:
    findings: list[str] = field(default_factory=list, metadata={"reducer": concat})
```

Now two parallel branches that both append to `findings` combine instead of clobbering.
Built-ins: `concat` for sequences and `merge_dicts` for dicts. A reducer is any
`Callable[[Any, list[Any]], Any]` — it receives the field's value at the start of the wave
plus each branch's value, and returns the merged result.

!!! note "Reducers are only consulted for genuine conflicts"
    A field written by a single branch is assigned directly, so declaring a reducer costs
    nothing on the common path. Fields with no reducer are still perfectly fine — as long as
    only one parallel branch writes each of them.

## Loops

Ordinary edges must form a DAG. To iterate, mark the back-edge with `loop=True`:

```python
graph.connect(think, act, condition=lambda s: bool(s.pending_calls))
graph.connect(act, think, loop=True)
```

When a loop edge is taken, the loop's **head** and everything downstream of it re-run.
Anything *upstream* of the head keeps its result and runs exactly once, so setup work stays
outside the iteration:

```python
graph.connect(load_rubric, critique)  # runs once
graph.connect(draft, critique)
graph.connect(critique, draft, loop=True, condition=lambda s: s.score < 0.8)
```

Requiring the flag is deliberate: it keeps loops visible in the wiring, and it means the
scheduler still only ever sees a DAG. An *undeclared* cycle is a build error.

`run()` builds the state from the type the graph was declared with when you omit it, so a
state whose fields all have defaults needs no argument; pass an instance when a run starts
with real input. It also takes `max_steps` (default 50) as the termination guard. Exceeding
it raises `StepLimitExceeded`, with the partial state on `.state` so a runaway loop is still
inspectable:

```python
from deepharness import StepLimitExceeded

try:
    result = await executor.run(max_steps=20)
except StepLimitExceeded as exc:
    print(exc.state)
```

## Seeing the shape

`executor.diagram()` returns a picture of the wiring, laid out top to bottom by wave, so
nodes drawn side by side are the ones that actually run concurrently:

```python
graph.connect(load_rubric, critique)
graph.connect(draft, critique)
graph.connect(critique, publish, condition=lambda s: s.score >= 0.8)
graph.connect(critique, draft, loop=True, condition=needs_work)

print(graph.build().diagram())
```

```text
╭═════════════╮   ╭═══════╮
│ load_rubric │   │ draft │◀─╮
╰═════════════╯   ╰═══════╯  │
       ╰─────╮        │      │
             ├────────╯      │
             ▼               │
       ╭──────────╮          │
       │ critique │──────────╯
       ╰──────────╯
             │
             ▽
        ╭═════════╮
        │ publish │
        ╰═════════╯

  ↺ critique → draft when needs_work
  ▽ conditional edge
```

A double rule (`═`) marks a `start` or `end` node, `▼` an unconditional edge and `▽` a
conditional one. A back-edge is routed up the right margin and named after its condition,
which is why giving the condition a `def` rather than a `lambda` reads better here.

It returns the text rather than printing it, so it is yours to route — a terminal, a log
line, or a test asserting on the shape of a graph you built dynamically.

Two limits worth knowing. A back-edge whose source or target has a sibling to its right
would have to cross that node's box, which would read as an edge that does not exist; rather
than draw a lie it falls back to `(not drawn: no clear margin)` in the notes. And a wide wave
produces a wide drawing — there is no wrapping or truncation to fit a terminal.

## Validation

`graph.build()` fails fast on structural mistakes, before anything runs:

- no node marked `start=True`
- a node that is unreachable (no `start` flag, no incoming edges)
- a cycle whose back-edge is not marked `loop=True`

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
from deepharness import ExecutionError

try:
    await executor.run()
except ExecutionError as exc:
    print(exc.node_name, exc.original)
```

## Agents as nodes

The graph engine is agent-agnostic: `Agent.arun()` takes a prompt and returns an `AgentState`,
the graph carries your dataclass, so a node is the adapter between them. One agent per node is
the natural way to build a multi-agent system:

```python
from deepharness import Agent, Message

researcher = Agent(llm, name="researcher", system="You research topics.")
writer = Agent(llm, name="writer", system="You write short reports.")


@graph.add(start=True)
async def research(state: State) -> State:
    out = await researcher.arun(state.topic)
    state.research = out.output
    return state


@graph.add(end=True)
async def write(state: State) -> State:
    out = await writer.arun(state.research)
    state.draft = out.output
    return state


graph.connect(research, write)
```

Each agent keeps its own prompt, tools, and model; the graph handles sequencing, branching,
and fan-out between them.
