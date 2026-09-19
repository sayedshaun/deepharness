# Chat UI

A static page and a small FastAPI backend for driving a real agent against a
real model — the fastest way to see the harness work end to end: tools running
in a workspace, a permission gate stopping a write until you allow it, progress
events as they happen, and a conversation that keeps its state across turns.

```bash
pip install -e ".[examples]"
make chat                      # or: .venv/bin/python examples/chat/app.py
# open http://127.0.0.1:8765
```

`make chat` uses the project's virtualenv. Running `python examples/chat/app.py`
directly only works with that environment activated — otherwise the system
Python has neither `fastapi` nor `deepharness` on its path.

By default it points at a llama.cpp server over its OpenAI-compatible endpoint
and works in a **fresh temporary directory**, printed at startup. An agent with
a shell should not be aimed at a repository unless you say so:

```bash
python examples/chat/app.py \
  --base-url https://your-host/llamacpp/v1 \
  --model unsloth/gemma-4-E4B-it-GGUF \
  --workspace ./scratch \
  --port 8765
```

Every flag also reads from the environment (`DEEPHARNESS_CHAT_WORKSPACE`, …), so
`uvicorn examples.chat.app:app` works too.

## What it exercises

| Part | How you see it |
| --- | --- |
| `file_tools` / `shell_tool` | tool calls in the transcript; the workspace panel updates as files appear |
| `Workspace` | ask it to read `../../etc/passwd` and watch the call fail with `OutsideWorkspace` |
| `Permissions` | `write_file`, `edit_file` and `run_command` stop for a ruling; `read_file` and friends just run |
| deny rules | ask for `rm -rf .` — the call is refused and the model is told |
| progress events | `StepStarted` and `ToolStarted`/`ToolFinished` render as they arrive |
| reasoning | a thinking model's working streams into its own panel, open while it thinks and folded away once the answer starts |
| `ContextPolicy` | long tool results arrive truncated, with the marker visible |
| pausing and resuming | **Approve & run** replays the call with the model's own arguments; **Reject** tells it no |

Typing a new message while a gate is open counts as a refusal: the pending call
is recorded as denied rather than dropped, so the model learns it and the
transcript keeps a result for every call it asked for.

## Tests

```bash
node examples/chat/ui_test.mjs
```

The page has no build step, so this stubs the DOM calls it makes and exercises the
script directly — mostly guarding the path that matters: a paused run must put up
its approval gate even when nothing else renders. A slip in the file panel once
swallowed that gate, which looks exactly like a hung agent.

## Shape

- `app.py` — the backend. One `Session` owns the agent, the workspace and the
  conversation; each turn streams server-sent events. A paused state stays here
  rather than going to the browser, because an approval is a decision about what
  this process will run.
- `index.html` — the page. No build step, no CDN, no framework.

It is a development toy: one session, no authentication, and it hands a model a
shell. FastAPI and uvicorn are example dependencies — the library itself still
needs only `httpx`.
