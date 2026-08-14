# Examples

Runnable scripts live in [`examples/`](https://github.com/sayedshaun/subagents/tree/main/examples)
in the repository.

## Parallel agents

`examples/parallel_agents.py` — three specialist agents (`sales`, `churn`, `support`) run as
concurrent `start=True` nodes, then a `synthesize` node joins their output. A stubbed model
requires no API keys, so it's a good template for testing a multi-agent graph without live
providers.

```python
graph.connect(sales, synthesize)
graph.connect(churn, synthesize)
graph.connect(support, synthesize)
```

Wall-clock time is close to the slowest branch, not the sum of all three — the whole point of
marking independent work `start=True`.

## Fan-out over many agents

`examples/many_parallel_agents.py` demonstrates a common pitfall: putting many homogeneous
agents into a `Graph` as separate nodes that all write into one shared field collapses to a
single surviving result, because [merging is last-writer-wins per field](guide/graph.md#parallel-execution-and-state-merging),
not additive.

For "run N agents, collect a list of results," skip the graph and use `asyncio.gather`
directly:

```python
import asyncio

results = await asyncio.gather(
    *(agent.arun(state) for agent, state in zip(agents, states))
)
```

Reach for `Graph` when branches write to *different* fields and need to be sequenced or
joined — not for simple homogeneous fan-out.
