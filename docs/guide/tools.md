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

## Passing tools to an agent

```python
agent = Agent(llm, name="assistant", tools=[get_weather])
```

`tools=` accepts either form:

- a plain list of functions — a `Toolbox` is built for you automatically
- an existing `Toolbox` (or subclass) — used as-is
