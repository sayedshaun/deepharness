# Example tools

deepharness does not ship ready-made tools. A tool's name, its docstring and the text it returns
are what the model reads, so they are decisions about your agent, not the library's. What the
library ships instead are the parts that are easy to get subtly wrong:

- [`Workspace`](tools.md#workspace-confining-paths) confines the paths a tool resolves, symlinks
  included.
- [`Sandbox`](tools.md#sandboxes-bounding-what-a-command-can-reach) bounds what a command can
  reach.
- [`Permissions`](tools.md#permissions-deciding-per-call) and `requires_approval` decide which
  calls run unasked.

The tools below are built only from those. Copy them, then change the names, docstrings, limits
and output to suit your agent.

## Files

```python
from deepharness import tool
from deepharness.tools import Workspace

workspace = Workspace("./project")
MAX_CHARS = 50_000


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file. Paths are relative to the project root."""
    text = workspace.resolve(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n... [{len(text) - MAX_CHARS} more characters]"
    return text


@tool
def list_files(pattern: str = "**/*") -> str:
    """List files under the project root that match a glob pattern."""
    files = sorted(
        workspace.relative(path)
        for path in workspace.root.glob(pattern)
        # a symlink inside the root can point out of it
        if path.is_file() and path.resolve().is_relative_to(workspace.root)
    )
    return "\n".join(files[:200]) or f"No files match {pattern!r}."


@tool(requires_approval=True)
def write_file(path: str, content: str) -> str:
    """Create a file, or replace one entirely. Prefer edit_file for changes."""
    target = workspace.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}."


@tool(requires_approval=True)
def edit_file(path: str, old: str, new: str) -> str:
    """Replace the one place `old` occurs in a file with `new`."""
    target = workspace.resolve(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        return f"`old` occurs {count} times in {path}; it must occur exactly once."
    target.write_text(text.replace(old, new), encoding="utf-8")
    return f"Edited {path}."
```

Every path goes through `workspace.resolve()`, which raises `OutsideWorkspace` for
`../../.ssh/id_rsa` or a symlink pointing out of the root. A tool that raises reaches the model
as an error result, so it can correct itself, and the read never happens.

A few of the choices here are worth keeping however you change the rest:

- **Cap what a read returns.** Every tool result is re-sent with every later turn, so one
  unbounded read of a large file is paid for again and again.
- **Make an ambiguous edit fail.** Replacing the first of several matches edits the wrong place
  without anyone noticing.
- **Gate the writes.** `requires_approval=True` pauses the run before the call; see
  [the coding agent guide](coding-agent.md#ruling-on-it) for answering it.
- **Leave out tools the agent shouldn't have.** An agent that only needs to read gets
  `read_file` and `list_files` and nothing else, which guarantees more than any rule can.

## A shell

```python
import subprocess

from deepharness import tool
from deepharness.tools import Sandbox


def run_command_tool(box: Sandbox, timeout: int = 60, max_chars: int = 20_000):
    @tool(requires_approval=True)
    def run_command(command: str) -> str:
        """Run a shell command in the project and return its output."""
        try:
            result = box.run(command, timeout)
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout}s: {command}"
        streams = (result.stdout.rstrip(), result.stderr.rstrip())
        output = "\n".join(stream for stream in streams if stream)[-max_chars:]
        if result.returncode:
            output += f"\n[exit {result.returncode}]"
        return output or "(no output)"

    return run_command
```

The tool is a factory because it holds a sandbox, which you pass in like any other dependency:

```python
from deepharness import Agent
from deepharness.tools import Bubblewrap

async with Bubblewrap("./project") as box:
    agent = Agent(llm, tools=[read_file, list_files, run_command_tool(box)])
    state = await agent.arun("Why does the test suite fail?")
```

`NoSandbox("./project")` runs commands as your own process, with your environment and full
reach, and needs no `with` block. Keep the approval gate on with it. Inside `Bubblewrap` or
`Docker` a command can reach only the workspace, so removing the gate is a reasonable choice
there.

Keep the tail of long output, not the head, because errors are printed last. Report the exit
code and the case of no output explicitly, or the model can't tell success from silence.

## Web search

```python
import os

import httpx

from deepharness import tool


@tool
async def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs and excerpts from the top results."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": 5},
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return f"No results for {query!r}."
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in results)
```

The same shape works for any search API: call it, then return text the model can cite, with
URLs included. The tool is async, so drive the agent with `arun()`. An agent that runs many
searches should share one `httpx.AsyncClient` rather than open one per call.
