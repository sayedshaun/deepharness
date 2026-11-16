# Tools

A tool is just a Python function. `@tool` derives a JSON schema from its signature and
docstring, so a model can call it.

Snippets here use an `llm` such as `OpenAI("gpt-4o-mini")`. The ones that `await` belong inside
an `async def` run with `asyncio.run(...)`, or paste them into `python -m asyncio`.

```python
from deepharness import tool


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is 22°C and sunny in {city}."
```

- Parameter types come from the annotations, including the shapes inside them: `list[int]`
  becomes an array of integers, `Literal["c", "f"]` and an `Enum` become an enum, `int | None`
  keeps its type (give it a default such as `= None` to make it optional), and `int | str` becomes an `anyOf`.
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

## Workspace: confining paths

A tool that takes a path takes it from a model, and a model will happily ask for
`../../.ssh/id_rsa`. `Workspace` is the one place that check lives:

```python
from deepharness.tools import Workspace

workspace = Workspace("./project")


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file. Paths are relative to the project root."""
    return workspace.resolve(path).read_text(encoding="utf-8")
```

`resolve()` returns an absolute path inside the root or raises `OutsideWorkspace`. That
exception reaches the model as the call's result, so it can correct itself, and the read never
happens. Resolution follows symlinks, so a link inside the root that points out of it is refused
too. `relative(path)` turns a resolved path back into what the model should see.

deepharness ships no file or shell tools of its own: their names, docstrings and output are
decisions about your agent. [Example tools](example-tools.md) has a complete set to copy.

## Sandboxes: bounding what a command can reach

`Workspace` bounds where a command starts, not what it can reach: `cd / && rm -rf ~` reaches
anything your process can. A `Sandbox` bounds the process itself. It is a resource with a
lifetime: open it with `with` (or `async with`), and everything a command leaves outside the
workspace, such as installed packages and caches, is thrown away when the block ends.

```python
from deepharness.sandbox import Bubblewrap

async with Bubblewrap("./project") as box:
    result = box.run("pip install -r requirements.txt && pytest -q", timeout=300)
    print(result.returncode, result.stdout)
```

`run(command, timeout)` returns a `subprocess.CompletedProcess` whatever the exit code, and
raises `subprocess.TimeoutExpired` once a command is killed. A tool that runs commands calls it
instead of `subprocess`; [the shell example](example-tools.md#a-shell) wraps it in a
`run_command` tool. Nodes in a [graph](graph.md#running-nodes-in-a-sandbox) can call it directly.

The workspace is mounted at its own path inside the sandbox, so tools on the host using
`box.workspace` see exactly what commands wrote.

| | `Bubblewrap(workspace)` | `Docker(image, workspace)` |
| --- | --- | --- |
| Needs | `bwrap` on Linux | a Docker daemon and the image |
| Host files | system directories read-only, the workspace, no home directories | the workspace only |
| Environment | cleared, so this process's API keys are not visible | not inherited |
| Lasts for the block | a scratch `/tmp` and `$HOME` | the container's whole filesystem |

Both take `writable=False` to mount the workspace read-only and `network=True` to allow network
access; both default to a writable workspace and no network. `Bubblewrap(read_only=[...])`
mounts extra host paths read-only; `Docker(options=[...])` passes flags such as `--memory` to
`docker run`. A Docker image needs `sh`, `tail` and `timeout`, which coreutils and busybox both
provide. Docker runs commands as your uid and gid, so files it writes are owned by you.

A sandbox is also what makes it reasonable to drop `requires_approval` from a command tool: the
gate exists because an unsandboxed command can reach anything, and inside a sandbox it cannot.

Calling `run` outside the block raises `ConfigurationError` rather than quietly running
unsandboxed. A new backend, such as a cloud VM, subclasses `Sandbox` and implements
`run(command, timeout)`, plus `start()` and `stop()` if it holds anything.

!!! note
    A snap-installed Docker cannot see the host's `/tmp`; a workspace there is mounted as an
    empty directory.

## Permissions: deciding per call

`requires_approval` answers "may this run?" for a tool as a whole, which stops being enough
once the tool is `run_command`: `git log` and `rm -rf /` are the same tool. `Permissions`
decides per call, from the arguments the model actually sent.

```python
from deepharness.tools import Permissions, Rule

# read_file, list_files, write_file, edit_file and run_command: the example tools
permissions = Permissions(
    allow=[read_file, list_files, Rule("run_command", {"command": "git log*"})],
    ask=[write_file, edit_file, run_command],
    deny=[Rule("run_command", {"command": "*rm -rf*"})],
)

agent = Agent(
    llm,
    tools=[read_file, list_files, write_file, edit_file, run_command],
    permissions=permissions,
)
```

### Naming a tool in a rule

Two forms, and they mean the same thing:

```python
Permissions(ask=[write_file])  # the tool itself, when it is in scope
Permissions(ask=["write_file"])  # a name, or a pattern like "write_*"
```

Prefer the tool itself where you can: your editor renames it with the code, and a typo is a
`NameError` instead of a rule that silently matches nothing. Passing the tool reads the name it
was registered under, so a tool renamed with `@tool(name=...)` still matches what the model
sees.

A string stays necessary for patterns (`"read_*"`) and for tools defined elsewhere, so it is
never wrong — just unchecked.

**A deny or ask rule naming a tool the agent does not have is refused at construction:**

```python
Agent(llm, tools=[run_command], permissions=Permissions(deny=[Rule("run_comand")]))
# ConfigurationError: permission rules deny or ask about unregistered tools: run_comand.
# Registered tools: run_command. ...
```

That typo would otherwise match nothing, and a deny rule matching nothing silently allows
exactly what it was written to stop. An **allow** rule is not checked, because one that matches
nothing is inert — the call falls back to the tool's own `requires_approval` — so a policy
shared between agents may allow tools only some of them have. Pattern rules are never checked;
they are meant not to name one tool.

A rule narrowed to arguments matches both sides as `fnmatch` globs, and a rule that mentions an
argument the call did not send does not match — an absent argument cannot be vouched for.

`deny` wins over `allow`, which wins over `ask`. That ordering is what makes a policy safe to
widen: adding an `allow` can never quietly override a `deny` already written down.

A call **no** rule matches gets no opinion from the policy, and falls back to the tool's own
`requires_approval`. So a policy is a narrowing of what tools already declare, not a
replacement for it — and `Permissions()` with no rules changes nothing.

The three decisions play out like this:

- **allow** — the call runs, even if the tool is marked `requires_approval`.
- **ask** — the run stops with `stop_reason == "paused"` and the call in `state.paused`;
  resume with `state.approve()` or `state.reject()`.
- **deny** — the call never runs, and the model is told so as that call's result, so it can
  find another way instead of retrying. Nothing pauses; a turn where everything was refused
  simply goes back to the model.

## Tools from an MCP server

`MCPServer` connects to a [Model Context Protocol](https://modelcontextprotocol.io) server and
hands you its tools as ordinary callables — no protocol SDK, just `httpx` and the standard
library.

```python
from deepharness import Agent
from deepharness.tools import MCPServer

async with MCPServer.stdio(
    ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
) as mcp:
    agent = Agent(llm, tools=await mcp.tools())
    state = await agent.arun("What is in the repo?")
```

`MCPServer.stdio(command)` runs a server as a subprocess; `MCPServer.http(url, headers=...)`
reaches a remote one over streamable HTTP, handling both the JSON and SSE reply shapes and
carrying the session id a server may issue. The handshake happens on first use, so `connect()`
is optional; as a context manager the transport is closed for you.

The tools are async, so drive the agent with `arun()`. Each one keeps the schema the server
published rather than a re-derived one, and a call that the server reports as failed raises
`MCPError` — which reaches the model as that call's result, so the run continues.

A tool the server did **not** mark read-only is gated by default: it is code in another process
that this side cannot inspect, so asking is the right default. Widen it with
[`Permissions`](#permissions-deciding-per-call) so what you allowed stays written down, or pass
`requires_approval=False` to turn the gate off wholesale.

```python
mcp = MCPServer.http(
    "https://tools.internal/mcp", headers={"Authorization": f"Bearer {key}"}
)
tools = await mcp.tools()  # list[Callable], ready for Agent(tools=...)
# list[MCPTool]: name, description, input_schema, read_only
listed = await mcp.list_tools()
```

### What is covered, and how to cover the rest

Two transports, which is what current servers speak:

| Transport | Support |
| --- | --- |
| stdio (subprocess) | Yes — `MCPServer.stdio(...)`. What local servers use. |
| Streamable HTTP (spec `2025-06-18`) | Yes — `MCPServer.http(...)`, JSON or SSE replies, session id carried. |
| HTTP+SSE (spec `2024-11-05`, deprecated) | No. A separate `GET /sse` stream plus a POST endpoint. |

Tools are the whole of what is implemented: MCP's resources, prompts and sampling are not, and
authentication goes through headers you supply rather than an OAuth flow.

`Transport` is the extension point, so none of that needs a fork. It is three methods —
`request`, `notify`, `aclose` — and `MCPServer` takes any implementation:

```python
from deepharness.tools import MCPServer, Transport


class MyTransport(Transport):
    async def request(self, method: str, params: dict) -> dict: ...
    async def notify(self, method: str, params: dict) -> None: ...
    async def aclose(self) -> None: ...


mcp = MCPServer(MyTransport())
```

That is also the answer for the deprecated transport: it is the most intricate piece of the
protocol — a long-lived stream, a background reader, replies correlated by id — and it is being
retired, so it is left to the caller who actually needs it rather than carried here for
everyone.
