# Building a coding agent

An agent that reads a codebase and changes it, assembled one decision at a time. Each stage
runs; each one fixes a problem the stage before it has.

deepharness ships no file or shell tools, because what they're called and what they return is
part of your agent. This page uses the [example tools](example-tools.md), copied into a
`tools.py` next to your script, with `Workspace(".")` as the root.

```python
import asyncio

from deepharness import Agent, OpenAI
from tools import edit_file, list_files, read_file, write_file

llm = OpenAI("gpt-4o-mini")
SYSTEM = "Work in the repository you are given. Read before you edit."

agent = Agent(llm, tools=[read_file, list_files, write_file, edit_file], system=SYSTEM)

state = asyncio.run(agent.arun("What does the executor do?"))
print(state.output)
```

That already works. Every path the tools receive goes through `Workspace.resolve()`, so a path
that climbs out of the root raises `OutsideWorkspace`. That reaches the model as the call's
result, so it corrects itself, and the read never happens.

The later snippets reuse `llm` and `SYSTEM`, and the ones that `await` belong inside an
`async def` run with `asyncio.run(...)` — the complete script at the end does exactly that.

Two things are wrong with it, and the rest of this page is fixing them: it can edit your files
with no say from you, and it will eventually send a request larger than the model's context
window.

## Read-only first

The strongest guarantee is not a rule, it is an absent tool. If the job is understanding rather
than changing, hand over only the read tools and there is nothing left to rule on:

```python
agent = Agent(llm, tools=[read_file, list_files])
```

The example `read_file` stops at a character limit and says how much it left out. Without that,
one read of a large file fills the window and every later turn pays for it again.

## A shell, and why it stops

```python
from tools import run_command

FILES = [read_file, list_files, write_file, edit_file]
agent = Agent(llm, tools=[*FILES, run_command], system=SYSTEM)
```

`run_command` runs through a shell — pipes and redirection are the point — with the workspace
as its working directory. It runs as your own process, so it is **not contained**: the workspace bounds where
a command starts, not what it can reach (to bound that, use a
[sandbox](tools.md#sandboxes-bounding-what-a-command-can-reach)). So the example gates it, as
it does `write_file` and `edit_file`: the run stops before the call and waits.

```python
state = await agent.arun("Add a docstring to Executor.run")

print(state.stop_reason)  # "paused"
print(state.paused[0].name)  # "edit_file"
print(state.paused[0].question)  # Run edit_file with {'path': ..., 'old': ...}?
```

Nothing else in that turn ran either. A turn half-applied while a human is deciding would be
worse than waiting, so the other calls are recorded as not run and the whole turn waits.

## Ruling on it

`approve()` runs the call with the arguments the model sent; `reject()` tells it no. Both
return the state, so resuming is one line:

```python
state = await agent.arun(state.approve())  # or state.reject()
```

A loop around that is the whole of a terminal coding agent:

```python
from deepharness import AgentState


async def drive(agent: Agent, prompt: str) -> AgentState:
    state = await agent.arun(prompt)
    while state.stop_reason == "paused":
        for call in state.paused:
            print(f"\n{call.question}")
            allowed = input("[y/N] ").strip().lower() == "y"
            state.approve(call.call_id) if allowed else state.reject(call.call_id)
        state = await agent.arun(state)
    return state
```

## Deciding without being asked every time

Being asked about every `run_command` gets old, and `requires_approval` cannot tell `git log`
from `git push` — they are the same tool. `Permissions` decides per call, from the arguments
the model actually sent:

```python
from deepharness.tools import Permissions, Rule

permissions = Permissions(
    allow=[
        read_file,
        list_files,
        Rule("run_command", {"command": "git log*"}),
        Rule("run_command", {"command": "pytest*"}),
    ],
    ask=[write_file, edit_file, run_command],
    deny=[
        Rule("run_command", {"command": "*rm -rf*"}),
        Rule("run_command", {"command": "*git push*"}),
    ],
)

agent = Agent(llm, tools=[*FILES, run_command], permissions=permissions)
```

`deny` beats `allow` beats `ask`, so widening a policy can never quietly override a refusal
already written down. A denied call never runs and the model is told, which lets it find
another way rather than retrying. A call no rule matches falls back to the tool's own
`requires_approval` — so a policy narrows what tools already declare instead of replacing it.

A deny or ask rule naming a tool the agent does not have is refused at construction: a
misspelled deny matches nothing, and a deny that matches nothing allows exactly what it was
written to stop.

## Letting it run unattended

Rules narrow what runs unasked, but a rule can't predict every harmful command. When the agent
should run without anyone watching, contain the commands instead of judging them: inside a
sandbox a command reaches only the workspace, so the gate can come off.

```python
from deepharness.tools import Bubblewrap
from tools import run_command_tool

async with Bubblewrap(".") as box:
    contained = run_command_tool(box)
    agent = Agent(
        llm,
        tools=[*FILES, contained],
        system=SYSTEM,
        permissions=Permissions(allow=[contained, write_file, edit_file]),
    )
    state = await agent.arun("Make the test suite pass")
```

An `allow` rule wins over a tool's own `requires_approval`, so nothing pauses. Commands and edits
still land in the real workspace on your disk, so commit or branch first.

## Keeping it inside the window

A coding agent reads files, and every result it reads is re-sent with every later turn.
`ContextPolicy` bounds both ends of that:

```python
from deepharness.agent import ContextPolicy

agent = Agent(
    llm,
    tools=[*FILES, run_command],
    permissions=permissions,
    context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
)
```

`tool_result_chars` truncates each result as it is recorded, eliding the middle and saying how
much went missing. `max_tokens` prunes the view the model is sent — oldest turns first, with a
note in their place — while `state.messages` keeps every message. Pruning never drops the
system prompt, never orphans a tool result, and always keeps the most recent turns.

## Watching it work

Most of a long run is tool calls the model never narrates, so text alone makes it look stalled:

```python
from deepharness import Finished, TextDelta
from deepharness.agent import StepStarted, ToolFinished, ToolStarted

async for event in agent.astream_events("Add a docstring to Executor.run"):
    match event:
        case StepStarted(step):
            print(f"\n— step {step}")
        case ToolStarted(name, arguments, _):
            print(f"{name}({arguments}) …")
        case ToolFinished(name, result, failed, _):
            print(f"{name} {'failed' if failed else 'ok'}: {result[:70]}")
        case TextDelta(text):
            print(text, end="", flush=True)
        case Finished(state):
            print(f"\nstopped because: {state.stop_reason}")
```

`ToolFinished.result` is the text the model will read, truncation included, so what you show
cannot drift from what it saw. A thinking model's reasoning arrives as `ThinkingDelta` and is
worth rendering differently — it is the model's working, not its answer.

## Stopping for the day

The pause is a returned state, not an exception, so the whole run survives a process boundary —
approval included:

```python
import json
from pathlib import Path

from deepharness import AgentState

Path("run.json").write_text(json.dumps(state.to_dict()))

# tomorrow, in another process
state = AgentState.from_dict(json.loads(Path("run.json").read_text()))
if state.stop_reason == "paused":
    state = await agent.arun(state.approve())
```

## All of it

```python
import asyncio
import json
from pathlib import Path

from deepharness import Agent, AgentState, OpenAI
from deepharness.agent import ContextPolicy
from deepharness.tools import Permissions, Rule
from tools import edit_file, list_files, read_file, run_command, write_file

SYSTEM = """You are a coding agent working inside one repository.

Read before you change anything: list_files to find your way,
read_file before an edit, edit_file for part of a file and write_file for a new
one, run_command for anything else. Keep prose short - the tool calls show your
work."""

SESSION = Path("run.json")


def build() -> Agent:
    return Agent(
        OpenAI("gpt-4o-mini"),
        tools=[read_file, list_files, write_file, edit_file, run_command],
        system=SYSTEM,
        context=ContextPolicy(max_tokens=100_000, tool_result_chars=8_000),
        permissions=Permissions(
            allow=[
                read_file,
                list_files,
                Rule("run_command", {"command": "git log*"}),
                Rule("run_command", {"command": "pytest*"}),
            ],
            ask=[write_file, edit_file, run_command],
            deny=[Rule("run_command", {"command": "*rm -rf*"})],
        ),
    )


async def drive(agent: Agent, state: AgentState) -> AgentState:
    """Run until it answers, asking about anything the policy gates."""
    state = await agent.arun(state)
    while state.stop_reason == "paused":
        for call in state.paused:
            print(f"\n{call.question}")
            if input("[y/N] ").strip().lower() == "y":
                state.approve(call.call_id)
            else:
                state.reject(call.call_id)
        state = await agent.arun(state)
    return state


async def main() -> None:
    agent = build()
    state = (
        AgentState.from_dict(json.loads(SESSION.read_text()))
        if SESSION.exists()
        else AgentState()
    )
    while (prompt := input("\n> ").strip()) not in ("", "quit"):
        state.messages.append({"role": "user", "content": prompt})
        state = await drive(agent, state)
        print(f"\n{state.output}")
        SESSION.write_text(json.dumps(state.to_dict()))


asyncio.run(main())
```

Sixty lines plus the tools you copied, and every decision in it is visible: which tools exist, what runs unasked, what
stops, how big a request may get, and where the conversation lives between runs.
