# Examples

Runnable scripts live in [`examples/`](https://github.com/sayedshaun/subagents/tree/main/examples),
grouped by how much you need to know already. They share a scripted stand-in for
a provider, so every one runs offline with no API keys:

```bash
python examples/beginner/01_hello_agent.py   # one of them
make examples                                # all of them
```

## Beginner

`beginner/01_hello_agent.py` — the think/act loop at its smallest: one model,
one tool, one answer. `beginner/02_sequential_graph.py` — the graph with no LLM
involved at all, which is the clearest way to see what nodes, state, and edges
actually are.

## Medium

`medium/01_tools_and_errors.py` covers the parts that matter in production:
tool calls in one turn run concurrently, a tool that raises is reported back to
the model instead of ending the run, and `stop_reason` is what distinguishes a
real answer from a truncated one.

`medium/02_parallel_agents.py` — three specialist agents (`sales`, `churn`,
`support`) run as concurrent `start=True` nodes, then a `synthesize` node joins
their output:

```python
graph.connect(sales, synthesize)
graph.connect(churn, synthesize)
graph.connect(support, synthesize)
```

Wall-clock time is close to the slowest branch, not the sum of all three — the
whole point of marking independent work `start=True`. Each branch writes its
own field, so no reducer is needed.

`medium/03_conditional_routing.py` — branching with `condition=`, including the
detail that a branch skipped by a falsy condition does not wedge a downstream
join.

## Advanced

`advanced/01_human_in_the_loop.py` — gating a risky tool by raising
`HumanInputRequired`, then resuming from the returned state.

`advanced/02_many_parallel_agents.py` — many homogeneous agents collecting into
**one** shared field. Declare a reducer so the branches combine instead of
overwriting each other (see
[state merging](guide/graph.md#parallel-execution-and-state-merging)):

```python
from dataclasses import dataclass, field

from subagents import concat


@dataclass
class State:
    findings: list[str] = field(default_factory=list, metadata={"reducer": concat})
```

Without a reducer the merge raises `ConcurrentUpdateError`, so this is a loud
mistake rather than a silent one. The same script also shows the case the graph
can't help with — a fan-out whose width is only known at runtime, where nodes
are registered up front — and uses `asyncio.gather` inside a single node
instead.

`advanced/03_refinement_loop.py` — a back-edge marked `loop=True` turns a chain
into a refinement loop. `draft` and `critique` re-run until the score clears the
bar, while `load_rubric` sits upstream of the head and runs once:

```python
graph.connect(load_rubric, critique)
graph.connect(draft, critique)
graph.connect(critique, draft, loop=True, condition=lambda s: s.score < 0.8)
```

See [loops](guide/graph.md#loops) for the re-entry rules and `max_steps`.

`advanced/04_agent_loop_as_graph.py` — the think/act loop rebuilt out of graph
nodes rather than `Agent`'s internal Python loop, which is how you get a place
to insert an approval gate, a budget check, or a re-planning step mid-cycle.
