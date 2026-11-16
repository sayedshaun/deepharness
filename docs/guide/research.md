# Deep research

A common multi-agent shape: split a question into sub-questions, research each one with its own
agent, then write one report from the answers. deepharness doesn't ship it as a class, because
its prompts, limits and output are the parts you will want to change. Here it is built from
`Agent` and `Graph`, ready to copy.

```
╭══════╮      ╭──────────╮      ╭════════════╮
│ plan │ ───▶ │ research │ ───▶ │ synthesize │ ───▶ report
╰══════╯      ╰──────────╯      ╰════════════╯
    │              │                  │
    │              │                  ╰─ one agent, every finding, no tools
    │              ╰─ one agent run per sub-question, in parallel,
    │                 each with its own context and the search tool
    ╰─ an agent with structured output: a list of sub-questions
```

```python
import asyncio
from dataclasses import dataclass, field

from deepharness import Agent, Graph, OpenAI

llm = OpenAI("gpt-4o-mini")
MAX_SUB_QUESTIONS = 4
PARALLEL = 2


@dataclass
class Plan:
    sub_questions: list[str]


planner = Agent(
    llm,
    output=Plan,
    system=(
        "Split the user's question into focused, independent sub-questions that, answered "
        f"together, answer it. Return at most {MAX_SUB_QUESTIONS}, and fewer if fewer will do."
    ),
)
researcher = Agent(
    llm,
    tools=[web_search],  # a @tool you write; see Example tools for one
    system="Research the question using your tools. Answer concisely and cite your sources.",
)
synthesizer = Agent(
    llm,
    system="Write one thorough report answering the question from the findings you are given.",
)


@dataclass
class Research:
    question: str = ""
    sub_questions: list[str] = field(default_factory=list)
    findings: list[tuple[str, str]] = field(default_factory=list)
    report: str = ""


graph = Graph(Research)


@graph.add(start=True)
async def plan(state: Research) -> Research:
    result = await planner.arun(state.question)
    planned = result.output.sub_questions if result.output else []
    state.sub_questions = planned[:MAX_SUB_QUESTIONS] or [state.question]
    return state


@graph.add()
async def research(state: Research) -> Research:
    limit = asyncio.Semaphore(PARALLEL)

    async def answer(question: str) -> tuple[str, str]:
        async with limit:
            result = await researcher.arun(question)
            return question, result.output

    state.findings = await asyncio.gather(*map(answer, state.sub_questions))
    return state


@graph.add(end=True)
async def synthesize(state: Research) -> Research:
    notes = "\n\n".join(f"## {q}\n{a}" for q, a in state.findings)
    result = await synthesizer.arun(
        f"Question: {state.question}\n\nFindings:\n\n{notes}"
    )
    state.report = result.output
    return state


graph.connect(plan, research)
graph.connect(research, synthesize)

result = asyncio.run(
    graph.build().run(Research("How do UK master's student visas work?"))
)
print(result.report)
```

Each `researcher.arun()` starts from a fresh state, so one sub-question's search results never
enter another's context. That is the point of splitting the work. The synthesizer sees every
finding but holds no tools: it reports, it doesn't research.

## Changing the shape

- **Parallelism.** `PARALLEL` caps how many researchers run at once. Set it to what your
  provider's rate limit and your search API can take: six researchers against a provider that
  allows two concurrent requests turns a fast run into a wall of 429s.
- **Sequential research.** When later sub-questions build on earlier answers (find the things,
  then look up details about them), replace `gather` with a loop that passes the findings so
  far into each prompt:

    ```python
    @graph.add()
    async def research(state: Research) -> Research:
        for question in state.sub_questions:
            known = "\n\n".join(f"## {q}\n{a}" for q, a in state.findings)
            result = await researcher.arun(f"{question}\n\nKnown so far:\n{known}")
            state.findings.append((question, result.output))
        return state
    ```

- **More research when the answers fall short.** Add a review node after `research` and a
  `loop=True` edge back to `plan`, conditioned on the review; see
  [loops](graph.md#loops).
- **Budgets.** Give the researcher a `Budget(steps=6, tokens=200_000)` so one stubborn
  sub-question can't consume the whole run.
- **Token usage.** Each agent keeps a running total in `agent.total_usage`; add the three
  together for the cost of a run.

## Watching it work

To show the report as it is written, drive the synthesizer with `astream_events()` instead of
`arun()`:

```python
from deepharness import Finished, TextDelta


@graph.add(end=True)
async def synthesize(state: Research) -> Research:
    notes = "\n\n".join(f"## {q}\n{a}" for q, a in state.findings)
    async for event in synthesizer.astream_events(
        f"Question: {state.question}\n\n{notes}"
    ):
        match event:
            case TextDelta(text):
                print(text, end="", flush=True)
            case Finished(final):
                state.report = final.output
    return state
```

The researchers' `astream_events()` streams work the same way: `ToolStarted` and
`ToolFinished` show each search as it happens. See [streaming](agents.md#streaming).
