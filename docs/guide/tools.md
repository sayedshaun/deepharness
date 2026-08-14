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

- Parameter types come from the annotations (`str`, `int`, `float`, `bool`, `list`, `dict`).
- The description comes from the docstring.
- `required` is whichever parameters lack defaults.
- Tool functions may be sync or async.

`@tool(name=..., description=...)` overrides the inferred name or description if you need
something different from the function's own.

## Passing tools to an agent

```python
agent = Agent("assistant", model=llm, tools=[get_weather])
```

`tools=` accepts either form:

- a plain list of functions — a `Toolbox` is built for you automatically
- an existing `Toolbox` (or subclass) — used as-is

## Coding tools

`subagents` ships a ready-made tool set for building a coding agent: `read_file`,
`write_file`, `apply_patch`, `run_shell`, `grep` — plain functions, not tied to any language or
platform. The risky ones (`write_file`, `apply_patch`, `run_shell`) ask for confirmation
before acting; call `set_auto_approve(True)` to skip that (tests, scripts, anywhere you've
already decided to trust the agent).

`CodingToolbox` bundles all five as a ready-to-use toolbox — pass it straight to `tools=`:

```python
from subagents import Agent, OpenAI
from subagents.tools import CodingToolbox, set_auto_approve

set_auto_approve(True)  # skip confirmation prompts, e.g. for a CI script

coding_agent = Agent(
    "coder",
    OpenAI("gpt-4.1-mini"),
    system_prompt="You are a coding assistant. Use the tools to read, write, and run code.",
    tools=CodingToolbox(),
    max_steps=20,
)
```
