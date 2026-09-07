# Tools

A tool is just a Python function. `@tool` derives a JSON schema from its signature and
docstring, so a model can call it.

```python
from deepharness import tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."
```

- Parameter types come from the annotations, including the shapes inside them: `list[int]`
  becomes an array of integers, `Literal["c", "f"]` and an `Enum` become an enum, `int | None`
  keeps its type but stops being required, and `int | str` becomes an `anyOf`.
- An annotation that can't be described — an arbitrary class, `Any`, or none at all — is left
  unconstrained rather than mislabelled as a string. The model falls back on the parameter name
  and your docstring.
- The description is the function's docstring, used as written.
- `required` is whichever parameters lack defaults.
- Tool functions may be sync or async. A sync tool runs in a thread during `arun()`, so a
  blocking one does not stall the other tools gathered alongside it.

`@tool(name=..., description=...)` overrides the inferred name or description if you need
something different from the function's own.

## Reaching the run: `Ctx`

Annotate a parameter `ctx: Ctx` and the runtime fills it in. The parameter is hidden from the
model — it never appears in the schema and the model cannot pass it — so a tool can reach the
run without a module-level global:

```python
from deepharness import Agent, Ctx, tool


@tool
def lookup_plan(customer: str, ctx: Ctx) -> str:
    """Look up a customer's plan."""
    return ctx.deps.db.plan_for(customer, tenant=ctx.deps.tenant)


agent = Agent(llm, tools=[lookup_plan])
state = await agent.arun("What plan is Acme on?", deps=Deps(db=db, tenant="acme"))
```

`Ctx` carries two things:

| Field | What it holds |
| --- | --- |
| `deps` | Whatever you passed as `deps=` to `run`/`arun` — a database handle, a tenant, a request scope. |
| `state` | The `AgentState` for this run, so a tool can read the transcript so far. |

`deps` belongs to the run, not the agent: one agent instance serves many requests, each with
its own dependencies. A sub-agent registered with `as_tool()` inherits the caller's `deps`, so
a delegated run keeps the same scope. Calling a tool outside a run gets an empty `Ctx`, so
`ctx.deps` is always readable without a guard.

## Gating a tool on a human

`@tool(requires_approval=True)` pauses the run before the tool executes, every time. Approving
runs it with the arguments the model sent; rejecting tells the model it was denied. See
[human in the loop](agents.md#human-in-the-loop).

```python
@tool(requires_approval=True)
def delete_records(table: str) -> str:
    """Delete every row in a table."""
    return f"cleared {table}"
```

## Passing tools to an agent

```python
agent = Agent(llm, name="assistant", tools=[get_weather])
```

`tools=` accepts either form:

- a plain list of functions — a `Toolbox` is built for you automatically
- an existing `Toolbox` (or subclass) — used as-is

## Built-in: web search

`TavilySearch` wraps [Tavily's](https://tavily.com) search API as a tool, so an agent can
answer from the live web instead of its training data.

```python
from deepharness import Agent, TavilySearch

search = TavilySearch()  # reads TAVILY_API_KEY, or pass api_key=

agent = Agent(
    llm,
    tools=[search.as_tool()],
    system="Answer from search results only, and cite the URLs you used.",
)
state = await agent.arun("Who won the Chuadanga-1 seat in 2026?")
```

The settings live on the object, not in the schema — the model chooses only `query`, which is
one less thing for it to get wrong:

```python
search = TavilySearch(max_results=10, search_depth="advanced")
```

- `as_tool()` is async, for an agent driven with `arun()`; `as_sync_tool()` blocks, for one
  driven with `run()`. Both take `name=` and `description=` overrides.
- `await search.search(query)` (or `search.search_sync`) returns `list[SearchResult]` —
  `title`, `url`, `content`, `score` — for when you want the URLs as data rather than prose.
  Only the tool flattens them into text.
- Requests go through the same retrying `HTTPClient` the providers use, so a rate-limited
  search backs off and retries instead of failing the call and costing the agent a turn.
  A failure that survives the retries raises `ProviderError`.
- Close it when you are done: `await search.aclose()`, or `search.close()` for the sync pool.
