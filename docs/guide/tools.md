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

## Built-in: working in a directory

`file_tools()` and `shell_tool()` are what turns an agent into a harness: an agent that can
look at a codebase and change it. Both are factories, because each tool closes over the
`Workspace` it may touch — the confinement is injected, not read from global state.

```python
from deepharness import Agent, Permissions, Rule, file_tools, shell_tool

agent = Agent(
    llm,
    tools=[*file_tools("."), shell_tool(".")],
    system="Work in the repository you are given. Read before you edit.",
)
```

`file_tools(root)` gives the model five tools — `read_file`, `list_files`, `search_files`,
`write_file`, `edit_file`. Pass `writable=False` for the read-only three: a narrower toolbox is
a stronger guarantee than a rule, because there is nothing left to rule on.

Every path the model sends is resolved through `Workspace`, and a path that lands outside the
root raises `OutsideWorkspace` — which reaches the model as that call's result, so it can
correct itself, while the read never happens. Resolution is symlink-aware, so a link inside the
root pointing out of it is refused too, and a symlinked directory does not widen a listing.

`read_file` numbers lines and stops at `max_bytes`, telling the model which offset to continue
from; without that cap, one `read_file` of a large file fills the context window and every later
turn pays for it again. `edit_file` refuses an `old` string that appears more than once rather
than editing the wrong place. `write_file` and `edit_file` are gated (`requires_approval=True`)
out of the box.

`shell_tool(root)` adds `run_command`, which runs through a shell — pipes and redirection are
the point. It is gated by default and it is **not a sandbox**: the workspace bounds where the
command starts, not what it can reach. Leave the gate on, or narrow it with `Permissions`
below.

## Permissions: deciding per call

`requires_approval` answers "may this run?" for a tool as a whole, which stops being enough
once the tool is `run_command`: `git log` and `rm -rf /` are the same tool. `Permissions`
decides per call, from the arguments the model actually sent.

```python
from deepharness import FileTool, Permissions, Rule, ShellTool

permissions = Permissions(
    allow=[
        FileTool.READ,
        FileTool.LIST,
        FileTool.SEARCH,
        Rule(ShellTool.RUN, {"command": "git log*"}),
    ],
    ask=[FileTool.WRITE, FileTool.EDIT, ShellTool.RUN],
    deny=[Rule(ShellTool.RUN, {"command": "*rm -rf*"})],
)

agent = Agent(llm, tools=[*file_tools("."), shell_tool(".")], permissions=permissions)
```

### Naming a tool in a rule

Three forms, and they mean the same thing:

```python
Permissions(ask=[write_file])  # the tool itself, when it is in scope
Permissions(ask=[FileTool.WRITE])  # the built-in names, as a StrEnum
Permissions(ask=["write_file"])  # a name, or a pattern like "write_*"
```

Prefer a tool or an enum member where you can: your editor renames them with the code, and a
typo is a `NameError` instead of a rule that silently matches nothing. `FileTool` and
`ShellTool` are `StrEnum`s, so a member *is* a string — usable as a `Rule`'s tool and matched
by `fnmatch` with no conversion. Passing the tool itself reads the name it was registered
under, so a tool renamed with `@tool(name=...)` still matches what the model sees.

A string stays necessary for patterns (`"read_*"`) and for tools defined elsewhere, so it is
never wrong — just unchecked.

**A deny or ask rule naming a tool the agent does not have is refused at construction:**

```python
Agent(llm, tools=[shell_tool(".")], permissions=Permissions(deny=[Rule("run_comand")]))
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
from deepharness import Agent, MCPServer

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
listed = await mcp.list_tools()  # list[MCPTool] - name, description, schema, read_only
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
from deepharness import MCPServer, Transport


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
