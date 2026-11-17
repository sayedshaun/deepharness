# Threat model

deepharness is a library, so it can't make an application safe on its own. It gives you
boundaries to put around an agent; what an agent can do depends on which you use and how you
combine them. This page lists what each boundary protects, what it doesn't, and what remains
yours to handle.

## Who and what to trust

| Party | Trust it? | Why |
| --- | --- | --- |
| You, the developer | Yes | You write the tools, choose the workspace, and set the sandbox options and permissions. None of these should ever be built from model output. |
| The model | No | It is steered by everything in its context, and it makes mistakes even when nothing is steering it. Treat every tool call as a request, not a decision. |
| What the model reads | No | Web pages, files in the repository, issues, emails, tool results. Any of it can contain instructions aimed at the model ([prompt injection](#prompt-injection)). |
| Your tools and MCP servers | As much as their code | They run in your process or as your user, with no sandbox. |
| The model provider | With your data | Everything in the context window is sent to it, including file contents a tool read. |

## What is worth protecting

- **Secrets:** API keys in your process and environment, credentials in your home directory,
  and anything secret in the workspace.
- **The host:** files outside the workspace, other processes, and the machine's memory and CPU.
- **The network:** both what the agent can reach and what it can send out.
- **Side effects:** whatever your tools can do in the world, such as pushing code, sending
  messages or spending money.

## What each boundary does

| Boundary | Protects against | Does not protect against |
| --- | --- | --- |
| `Workspace.resolve()` | A file tool reaching a path outside the root, including through `..` and symlinks | Anything your tool does without calling it; a symlink swapped in between the check and the open by a command running at the same time |
| `Bubblewrap` | A command writing outside the workspace, reading home directories, seeing your environment variables, using the network (unless `network=True`), outliving its timeout, or printing more than `max_output` | Reading the workspace and system directories such as `/etc`; anything run outside `box.run()`; kernel vulnerabilities, since namespaces are not a virtual machine |
| `max_memory_mb`, `max_processes` | A command exhausting memory or forking without end | Hosts without a systemd user session, where `start()` refuses instead |
| `Permissions`, `requires_approval` | Calls you never meant to allow, or want a human to see first | Arguments written to get past a glob pattern; a human approving something they didn't read |
| `Budget`, `max_steps`, timeouts | Runaway loops and token spend | Cost per step, which depends on the model and context size |
| Events, `on_event`, logging | Not knowing what happened | Anything you don't record or look at |

## Threats and what to do about them

### Prompt injection

Text the model reads can tell it to do something you didn't ask for: run a command, edit a
file, or put a secret in its answer. No library can reliably detect this. It can only be
contained:

- Assume anything the model can see, it can be steered to send somewhere. Keep secrets out of
  its reach, not just out of its instructions.
- Give an agent that reads untrusted content only the tools that task needs.
- Put writes and anything with an outside effect behind `ask`, and show the human the full
  arguments.
- Keep `network=False` for commands unless the task needs the network.

### Secrets leaking

- **Your process:** a `Bubblewrap` command starts with an empty environment, but your Python
  tools run with your full environment. A tool that returns `os.environ`, or reads `~/.aws`,
  hands that to the model and to the provider.
- **The workspace:** every sandboxed command can read the whole workspace. Keep `.env` files,
  keys and credentials out of a workspace an agent works in.
- **MCP servers:** `MCPServer.stdio(...)` passes your whole environment to the server process
  unless you pass `env=` with only what it needs.
- **Telemetry:** events carry tool arguments and results. Before exporting them to a tracing
  backend, check that what you record doesn't include secrets. Logged URLs have their query
  strings removed; nothing else is redacted for you.

### Commands doing damage

Run model-written commands through `Bubblewrap`, never `subprocess` on the host. Deny rules on a
command string are a convenience, not a boundary: `Rule("run_command", {"command": "*git
push*"})` doesn't match `git  push` or `sh -c 'git push'`. Let the sandbox decide what a command
can reach, and use rules to decide what deserves a human's attention.

### Exhausting the host

A timeout limits time only. Set `max_memory_mb` and `max_processes` when commands come from a
model, and keep `max_output` at a size your process can hold. `Budget` bounds steps and tokens
per run, but tokens only when you set `tokens=`; a graph's `max_steps` bounds how many waves it
runs.

### Data leaving the machine

`network=True` gives a command the network as this machine has it, not a list of allowed hosts,
so it can both download dependencies and upload your workspace. If a task needs packages,
install them before the agent runs, or with a command you write rather than one the model writes.
Your own tools that make HTTP requests are a second way out; give them fixed hosts, not URLs the
model chooses.

### Changes that run later on the host

A writable workspace means the agent's changes land on your machine. Anything you later run
from that directory without a sandbox runs code the agent may have written: test configuration,
`Makefile`s, git hooks, virtualenv scripts, editor tasks. Review the diff before running
anything from a workspace an agent changed, or mount it with `writable=False` when the task only
reads.

### Stored runs

`state.to_dict()` holds the whole conversation, including file contents and tool results, and a
paused run's pending call. Store it where only you can read and write it: whoever can edit a
stored state can change what the agent remembers and what is waiting for approval.

## Not covered

- **Isolating the whole agent.** The agent loop, your tools and MCP servers run in your process
  ([what runs where](tools.md#what-runs-where)). Isolating them needs a separate worker process
  with a protocol back to yours.
- **Network allowlists.** The network is all or nothing.
- **Platforms other than Linux.** There is no sandbox on macOS or Windows.
- **Serving many users.** Running strangers' prompts or code on shared infrastructure needs
  per-user isolation, quotas and a review of your own; deepharness gives you parts of that, not
  all of it.
- **An outside audit.** The sandbox is tested against real `bwrap` and systemd in CI, but no
  independent security review has been done.

## Before you deploy

- [ ] Commands go through `box.run()`, with `network=False` unless needed.
- [ ] `max_memory_mb` and `max_processes` are set where commands come from a model.
- [ ] The workspace holds no secrets, and is read-only when the task only reads.
- [ ] No tool returns environment variables or reads outside the workspace.
- [ ] MCP servers get an explicit `env=`.
- [ ] Writes and outside effects are behind `ask`, and approvals show the full arguments.
- [ ] Agents that read untrusted content have only the tools they need.
- [ ] Every agent has `Budget(tokens=...)`; by default only steps are limited, to 10.
- [ ] Stored states and exported events are kept somewhere only you can read.
- [ ] Changes in the workspace are reviewed before anything there runs unsandboxed.
