# Changelog

Notable changes, newest first. This project is pre-1.0: while the major version is `0`, a
breaking change raises the minor version.

## Unreleased

### Breaking

- An approval pause no longer carries made-up wording: `PendingHumanInput.question` is empty for
  a `requires_approval` call, which is already described by `name` and `arguments`, and now
  defaults to `""`. A rejected call's result is "Not run: this call was rejected." rather than
  "Denied by the user.", since a policy or another model may be what rejected it.
- `deepharness.prebuilt` is removed, and with it `DeepResearch`, `ResearchResult`, `Finding`
  and the research events. Its prompts and limits were decisions about an application, and the
  shape is three agents in a three-node graph: [Deep research](docs/guide/research.md) now has
  it as an example to copy.
- The ready-made tools are removed: `file_tools`, `shell_tool`, `TavilySearch` (with
  `SearchResult` and `format_results`) and the `ToolName` enum. A tool's name, docstring and
  output are part of an agent, not of the library, so deepharness now ships only what tools are
  built from: `@tool`, `Workspace`, `Sandbox` and `Permissions`. The
  [example tools](docs/guide/example-tools.md) page has file, shell and web search tools to
  copy. A rule that used `ToolName.RUN_COMMAND` now names the tool itself or `"run_command"`.

### Added

- `deepharness.sandbox`, for running commands: `Bubblewrap` (Linux namespaces), used as `with`
  or `async with`, with `box.run(command, timeout)` called by the tools you write and by graph
  nodes directly. It makes the workspace the only writable host path (read-only with
  `writable=False`), keeps the network off unless `network=True`, caps each output stream at
  `max_output` bytes, checks on `start()` that a command can actually run, and discards
  everything outside the workspace when the block ends. `max_memory_mb` and `max_processes`
  cap each command and everything it starts, through a systemd user scope. Subclass `Sandbox`
  for another backend.
- Observability without a dependency. `StepFinished(step, usage, duration)` follows each model
  call, and `ToolFinished` carries the call's `duration`. `Executor.astream_events()` reports
  `NodeStarted`, `NodeFinished(name, duration)` and a final `GraphFinished(state)`; `run()` is
  that stream drained. HTTP retries are logged as warnings on the `deepharness` logger, which is
  silent until the application configures logging. See the new observability guide.
- `Agent(on_event=fn)` and `Graph.build(on_event=fn)` call `fn` with every event of a run, so
  `arun()` and `run()` can be observed without iterating a stream. `fn` may be async, except
  under the agent's sync `run()`.

### Fixed

- A graph node that raises now cancels the other nodes in its wave instead of leaving them
  running with no one to read their results.
- `astream_events()` reports each `ToolFinished` as soon as that tool ends, instead of holding
  every result in a turn until the slowest tool finished. The transcript still records results in
  call order.

## 0.3.0 — 2026-11-05

The release that makes an agent usable for long runs: tools that work in a directory, rules
about what they may do, a bounded context window, and progress you can watch.

### Breaking

- **Imports are layered.** `deepharness` now exports ~27 high-level names instead of 81;
  everything else comes from the package that owns it — `deepharness.agent`,
  `deepharness.graph`, `deepharness.providers`, `deepharness.tools`, `deepharness.prebuilt`,
  `deepharness.errors`. `Agent`, `Graph`, `tool`, `Message`, `Budget`, `Toolbox`, `Ctx`,
  `AgentState`, `Finished`, `TextDelta`, `Executor`, `DeepHarnessError` and all 15 providers
  are unchanged; `ContextPolicy`, `Permissions`, `file_tools`, `DeepResearch`,
  the content blocks, the progress events, the provider wrappers and the specific exceptions
  all moved one level down. No aliases are kept: an import that moved fails loudly rather than
  working for one more release.
- `save_session` and `load_session` are removed. Where a run is stored is the application's
  choice, so the library keeps only the serialization: `state.to_dict()` and
  `AgentState.from_dict(data)`. The equivalent of the old pair is
  `Path(p).write_text(json.dumps(state.to_dict()))` and
  `AgentState.from_dict(json.loads(Path(p).read_text()))`.
- Each tool result is truncated to 8000 characters as it is recorded
  (`ContextPolicy.tool_result_chars`). Pass `None` to keep the old unbounded behaviour.
- A turn that pauses for approval now records its unrun calls in the transcript. Without it, a
  resumed run sends a tool call no result answers, which vendors reject.
- `StreamAccumulator.feed()` returns a `StreamEvent` instead of `str | None`, so a payload can
  carry reasoning rather than prose. Only affects code implementing a provider by hand.
- `Toolbox.call` and `Toolbox.call_sync` take the tool's arguments as one dict:
  `toolbox.call("add", {"a": 1, "b": 2})` instead of `toolbox.call("add", a=1, b=2)`. As
  keywords they could collide with the method's own parameters, so a tool with a `name`
  argument could not be called, and an `ask` argument could switch off the approval gate.

### Added

- **Working in a directory.** `file_tools()` gives `read_file`, `list_files`, `search_files`,
  `write_file` and `edit_file`; `shell_tool()` gives `run_command`. Every path resolves through
  `Workspace`, which refuses anything outside its root — symlinks included — and raises
  `OutsideWorkspace`, which reaches the model as that call's result. Writers and `run_command`
  are gated by default.
- **Permissions.** `Permissions(allow=…, ask=…, deny=…)` with `Rule("run_command", {"command":
  "git *"})` decides per call, from the arguments the model actually sent. `deny` beats `allow`
  beats `ask`; a call no rule matches falls back to the tool's own `requires_approval`. A rule
  can name a tool by passing the tool itself or a `ToolName` member instead of a string, and a deny or ask rule naming an unregistered tool is refused at construction —
  a misspelled one would match nothing and allow what it was written to stop.
- **Context management.** `ContextPolicy` truncates each tool result and, with `max_tokens`
  set, prunes the view the model is sent while `state.messages` keeps every message. Pruning
  never drops the system prompt, never orphans a tool result and always keeps the last
  `keep_last` messages. `estimate_tokens()` is the heuristic behind it.
- **Progress events.** `astream_events()` now emits `StepStarted`, `ToolStarted`,
  `ToolFinished` and `ThinkingDelta` alongside `TextDelta`. `ToolFinished.result` is the text
  the model will read, truncation included.
- **Content blocks.** A message's content may be `Text`, `Image`, `Document` or `Thinking`
  blocks — `Message.human([Text("what changed?"), Image.from_path("ui.png")])`. Text-only
  content is still sent, and saved, as a plain string.
- **Reasoning round-trips.** Thinking arrives as `ThinkingDelta`, is available as
  `response.thinking`, is kept in the transcript, and is replayed to Anthropic with its
  signature — required after a tool call, and previously lost. OpenAI-compatible servers are
  read too: `reasoning_content` (llama.cpp, DeepSeek) and `reasoning` (OpenRouter), in both
  the response and the stream.
- **Provider wrappers.** `Fallback`, `Caching`, `RateLimited` and `Retrying` are themselves
  `LLM`s, so one wrapper covers the sync path, the async path and streaming:
  `Caching(RateLimited(Fallback(a, b), rps=2))`. `Wrapping` is the base for your own.
- **MCP client.** `MCPServer.stdio(...)` / `MCPServer.http(...)` turns a Model Context Protocol
  server's tools into ordinary toolbox entries, over stdio or streamable HTTP, with no new
  dependency. `Transport` is the extension point for anything those two do not reach.
- **Anthropic prompt caching.** `cache_prompt=True` puts a cache breakpoint on the system
  prompt and the last tool definition.
- **Cache counts in usage.** `TokenUsage` gains `cached_tokens` and `cache_write_tokens`, read
  from each vendor's own field, so a turn whose prompt was mostly a cache read is no longer
  indistinguishable from one that paid for all of it. Both are inside `prompt_tokens`;
  `Budget` still counts the full figure.
- **Concurrency cap.** `max_concurrency=` on any provider bounds its in-flight requests, held
  across retries and for a stream's whole body.
- **DeepResearch.** `DeepResearch` plans sub-questions, researches them in parallel and
  synthesizes one report, streaming its progress through `astream_events()`.
- **Web search.** `TavilySearch` wraps Tavily's API as a tool.
- **`LlamaCpp`** provider for llama.cpp's server.
- **Graph diagrams.** A built `Graph` renders as a terminal diagram.
- `Toolbox(approve=fn)` checks `fn(name, arguments)` before running a `requires_approval`
  call, wherever the toolbox is called from; a refusal raises `ToolDenied`. Without it an
  `Agent` pauses as before.

### Fixed

- A provider used in a second event loop no longer fails with "Event loop is closed": each
  loop gets its own async HTTP client and concurrency gate, so calling `asyncio.run(...)`
  twice with one provider works.
- A graph skips everything a skipped conditional branch alone leads to, rather than running
  it anyway, and settles such skips regardless of the order nodes were registered in.
- With `Toolbox(approve=...)`, `arun()` runs a turn holding a gated call in the model's order,
  so no later call runs while the approver is still deciding.
- A stdio MCP server that writes notifications before its reply is understood; the client reads
  past them to the message matching its request id.
- Gemini thinking tokens count as completion tokens, so the three usage numbers add up.
- A provider-truncated answer surfaces as `stop_reason == "truncated"` for OpenAI, Anthropic
  and Gemini rather than looking like a finished reply.
- A provider with no API key sends no auth header at all, which is the normal case for a local
  server.
- Four documentation snippets used names they never imported.

## 0.2.0 — 2026-09-04

- A graph run builds its own default state when none is given.
- A gateway can opt out of asking for streamed usage (`stream_usage=False`).
- Provider requests get a real timeout and retry transient failures with backoff.
- Gemini's API key moves from the query string to a header, keeping it out of error messages.
- Prompt tokens are kept in streamed usage; truncated tool arguments no longer kill a stream.
- Registering two different tools under one name is refused rather than silently shadowed.
- Graph state is validated, raising `ConfigurationError` instead of a bare `ValueError`.
- `OutputValidationError` is exported from the package root.

## 0.1.0 — 2026-09-01

First release: the agent think/act loop, `Toolbox` and `@tool`, the graph engine with parallel
branches and merging, typed dataclass state, and providers for OpenAI, Anthropic, Gemini and
the OpenAI-compatible gateways.
