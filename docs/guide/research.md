# Deep research

`DeepResearch` is the three-agent shape that keeps reappearing on top of `Agent`: split a
question into sub-questions, research each one independently, then write a single report from
the answers.

```python
import asyncio

from deepharness import OpenAI
from deepharness.prebuilt import DeepResearch
from deepharness.tools import TavilySearch

search = TavilySearch()  # reads TAVILY_API_KEY, or pass api_key=
research = DeepResearch(OpenAI("gpt-4o-mini"), tools=[search.as_tool()])

result = asyncio.run(research.arun("How do UK master's student visas work?"))
print(result.report)

for finding in result.findings:
    print(finding.question, "->", finding.answer)
```

```
╭══════╮      ╭──────────╮      ╭════════════╮
│ plan │ ───▶ │ research │ ───▶ │ synthesize │ ───▶ ResearchResult
╰══════╯      ╰──────────╯      ╰════════════╯
    │              │                  │
    │              │                  ╰─ one report, from every
    │              │                     finding and no tools
    │              ╰─ one Agent per sub-question, n_parallel at a
    │                 time, each with its own context and the tools
    ╰─ up to max_sub_questions, or the question itself as a fallback
```

The three agents differ only in prompt and tools: the planner returns a list of sub-questions
and holds no tools, each researcher is its own `Agent` — so one sub-question's evidence never
enters another's context — and the synthesizer sees every finding but no tools; it reports, it
does not research.

What `DeepResearch` does *not* decide is where the facts come from. The tools are injected and
the three prompts are replaceable, so the same workflow runs against a web search, an internal
document store, or a stub in tests. It is deliberately a plain sequence of three steps rather
than a `Graph` — a caller who needs a different shape (a loop back to more research, a review
step, branching) is better served composing `Agent`s into a `Graph` directly. See
[Agents as nodes](graph.md#agents-as-nodes).

## Result

`arun()` returns a `ResearchResult`:

| Field | What it holds |
| --- | --- |
| `query` | The original question. |
| `sub_questions` | What the planner split it into. |
| `findings` | One `Finding(question, answer)` per sub-question. |
| `report` | The synthesizer's final report. |
| `usage` | `TokenUsage` for the whole run — planner, every researcher, and the synthesizer. |

## Watching a run live

`astream_events()` is the actual driver — `arun()` just consumes it and keeps the last event, the
same relationship `Agent.astream_events()`/`arun()` have. Iterate it directly to show progress as
it happens, or read the synthesizer's report as it is written:

```python
from deepharness import TextDelta
from deepharness.prebuilt.research import (
    Planned,
    Planning,
    Researched,
    Researching,
    ResearchFinished,
    Synthesizing,
)

async for event in research.astream_events(query):
    match event:
        case Planning(q):
            print(f'planning: "{q}"')
        case Planned(sub_questions):
            print(f"{len(sub_questions)} sub-question(s)")
        case Researching(count):
            print(f"researching {count} at a time")
        case Researched(index, question):
            print(f"  [{index}] {question} done")
        case Synthesizing(findings):
            print(f"synthesizing from {findings} finding(s)")
        case TextDelta(text):
            print(text, end="", flush=True)
        case ResearchFinished(result):
            print(f"\n\n{len(result.findings)} findings")
```

`ResearchFinished` is deliberately not the same name as `Agent`'s `Finished`: the two wrap
different things (`result: ResearchResult` vs `state: AgentState`), and this stream and an
`Agent`'s are commonly read in the same file. Importing `Finished` from `deepharness` here
would be the wrong class — `case Finished(...):` would compile but never match, silently.

Nothing here prints on its own — the workflow only emits typed events, so it stays usable where
stdout is not free to write to (a server, a queued job). A caller that wants nothing at all just
calls `arun()` and never iterates the stream.

## Parallel, capped, or sequential

Left unset, `n_parallel` leaves the degree of parallelism to the model: the planner decides how
many sub-questions the question needs, and all of them run at once — one researcher for a single
factual question, several for a question with genuinely separate parts.

```python
DeepResearch(model, tools=[search.as_tool()], n_parallel=2)
```

Setting it to an `int` caps that instead, for limits the model cannot see: fanning six researchers
at a provider that allows two concurrent requests turns a fast run into a wall of 429s, and a
tool backed by one server may not survive six callers at once.

`sequential=True` runs one researcher at a time, each seeing every earlier answer — the only mode
that can do so, since parallel researchers structurally cannot: each starts before any other has
finished. Use it when later sub-questions build on earlier ones (find the things, then look up
details about the things found):

```python
DeepResearch(model, tools=[search.as_tool()], sequential=True)
```

`sequential=True` and `n_parallel=` are mutually exclusive — `sequential` already fixes the
degree of parallelism at one, so pass one or the other.

## Prompts and limits

```python
from deepharness import Budget

DeepResearch(
    model,
    tools=[search.as_tool()],
    max_sub_questions=4,
    researcher_system="Answer only from primary sources. Quote exactly.",
    budget=Budget(steps=6, tokens=200_000),
)
```

- `max_sub_questions` caps how many sub-questions the planner may return; a planner that returns
  nothing usable falls back to the original question as a single sub-question.
- `planner_system` / `researcher_system` / `synthesizer_system` replace the default prompt for
  each agent. All three reach their agent verbatim — the sub-question cap is appended to the
  planner's rather than interpolated into it, so a custom prompt containing braces is not
  mistaken for a format string.
- `budget` reaches every researcher (not the planner or synthesizer), the same [`Budget`](agents.md#token-usage-and-budgets)
  an `Agent` takes.
