# Tools

A tool is just a Python function. `@tool` derives a JSON schema from its signature and
docstring, so a model can call it.

```python
from subagents import tool


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
from subagents import Agent, Ctx, tool


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
