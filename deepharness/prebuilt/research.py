"""A prebuilt deep-research workflow: plan, research in parallel, synthesize.

`DeepResearch` is the three-agent shape that keeps reappearing on top of
`Agent` - split a question into sub-questions, answer each one independently,
then write a single report from the answers - packaged so callers stop
rebuilding it:

    research = DeepResearch(model, tools=[search_tool], n_parallel=3)
    result = await research.arun("your research question")
    print(result.report)

What it does *not* decide is where the facts come from. The tools are
injected, the three prompts are replaceable, and nothing here knows about a
particular search API - so the same workflow runs against a web search, an
internal document store, or a stub in tests.

Deliberately a plain sequence of three steps rather than a Graph: the shape is
linear, and a caller who needs a different one (a loop back to more research,
a review step, branching) is better served composing `Agent`s into a `Graph`
themselves than configuring this class into something it is not.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

from ..agent import Agent, Budget, Toolbox
from ..agent import Finished as _AgentFinished
from ..errors import ConfigurationError
from ..providers.base import LLM, TextDelta, TokenUsage


@dataclass(slots=True)
class Planning:
    """A run has started on `query`."""

    query: str


@dataclass(slots=True)
class Planned:
    """The planner settled on these sub-questions."""

    sub_questions: list[str]


@dataclass(slots=True)
class Researching:
    """`count` researchers are about to run."""

    count: int


@dataclass(slots=True)
class Researched:
    """The researcher for `question` finished; `index` is its planned position."""

    index: int
    question: str


@dataclass(slots=True)
class Synthesizing:
    """Every finding is in; the report is being written."""

    findings: int


@dataclass(slots=True)
class ResearchFinished:
    """The run's final result, emitted once the report is complete.

    An async generator cannot return a value, so the result comes through as
    the final event - astream_events() yields this last, and arun() drains
    the stream and keeps it.

    Named distinctly from agent.Finished (which wraps an AgentState, not a
    ResearchResult) rather than reusing that name: this stream and an Agent's
    are commonly read side by side, and `from deepharness import Finished`
    would silently match nothing here - `case Finished(...):` never fires,
    with no error, if a caller happened to import the wrong one.
    """

    result: ResearchResult


ResearchEvent = (
    Planning
    | Planned
    | Researching
    | Researched
    | Synthesizing
    | TextDelta
    | ResearchFinished
)
"""What a run reports as it happens.

Typed rather than formatted prose, so a caller can count, route or record
events instead of matching substrings against a sentence meant for a human.
astream_events() is the only driver: iterate it directly to watch a run live,
or call arun() to drain it and get just the ResearchFinished result.
"""

DEFAULT_MAX_SUB_QUESTIONS = 5

PLANNER_SYSTEM = (
    "Split the user's research question into focused, independent "
    "sub-questions that, answered together, cover it. Use as few as the "
    "question actually needs: a single factual question needs exactly one, "
    "and only a question with genuinely separate parts needs several. Every "
    "extra sub-question costs another round of research, so do not pad the "
    "list to fill the allowance. Each sub-question is researched on its own, "
    "in parallel, and cannot see any other's answer - so if answering one "
    "part requires first knowing the answer to another (finding things, then "
    "looking up details about the things you found), keep both parts in a "
    "single sub-question rather than splitting them. If the question asks to "
    "find, list, name, or recommend specific things, at least one "
    "sub-question must explicitly ask "
    "to identify and name those specific things - not just describe general "
    "criteria about them. Do not answer the sub-questions yourself."
)

RESEARCHER_SYSTEM = (
    "Research the question you are given using the tools available to you. "
    "Gather evidence before answering rather than answering from memory, and "
    "look further if the first results do not settle it. Then write a "
    "detailed, well-organized answer, citing the sources you used."
)

SYNTHESIZER_SYSTEM = (
    "You are given a research question and answers to several sub-questions. "
    "Write one thorough, well-organized report answering the original "
    "question in full - directly naming and listing the specific things it "
    "asked for rather than only describing general criteria. Base the report "
    "only on the answers you were given; do not add facts of your own, and do "
    "not invent sources."
)


@dataclass
class _Plan:
    """The planner's structured output - the schema it must fill to answer."""

    sub_questions: list[str]


@dataclass(slots=True)
class _Researched:
    """One researcher's result, with the planned position Researched needs and
    the answer Finding needs - kept together so as_completed's arrival order
    doesn't have to be threaded back through two separate return values."""

    index: int
    question: str
    answer: str


@dataclass(slots=True)
class Finding:
    """One sub-question and the answer a researcher reached for it."""

    question: str
    answer: str


@dataclass(slots=True)
class ResearchResult:
    """A finished run: the report, plus the working behind it.

    `findings` is in the order the answers arrived: completion order when the
    researchers ran in parallel, planned order when they ran sequentially.

    `usage` is what this run cost across every agent it ran - the planner, all
    the researchers, and the synthesizer - since a fanned-out workflow's spend
    is otherwise impossible to recover from the outside.
    """

    query: str
    sub_questions: list[str]
    findings: list[Finding]
    report: str
    usage: TokenUsage


class DeepResearch:
    """Plan sub-questions, research them concurrently, synthesize one report.

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

    Give it a model and the tools its researchers should search with -
    `TavilySearch` reads TAVILY_API_KEY, the provider reads its own key:

    ```python
    import asyncio

    from deepharness import DeepResearch, OpenAI, TavilySearch

    search = TavilySearch()
    research = DeepResearch(
        OpenAI(model="gpt-4o-mini"),
        tools=[search.as_tool()],
        max_sub_questions=4,
        n_parallel=2,
    )

    result = asyncio.run(research.arun("How do UK master's student visas work?"))
    print(result.report)

    for finding in result.findings:
        print(finding.question, "->", finding.answer)
    ```

    Watch a run live by iterating astream_events() instead of arun() - it
    yields the same progress events arun() drains internally, plus TextDelta
    as the synthesizer writes the report and a final ResearchFinished with the result:

    ```python
    async for event in research.astream_events("your research question"):
        match event:
            case Planned(sub_questions):
                print(f"researching {len(sub_questions)} sub-question(s)")
            case Researched(index, question):
                print(f"  [{index}] {question}")
            case TextDelta(text):
                print(text, end="", flush=True)
            case ResearchFinished(result):
                print(f"\n\n{len(result.findings)} findings")
    ```

    Swap the prompts to change what the agents are asked for, and pass a
    `budget` to cap what each researcher may spend:

    ```python
    research = DeepResearch(
        model,
        tools=[search.as_tool()],
        researcher_system="Answer only from primary sources. Quote exactly.",
        budget=Budget(steps=6, tokens=200_000),
    )
    ```

    The three agents differ only in prompt and tools: the planner returns a
    list of sub-questions and holds no tools, each researcher is its own
    `Agent` so one sub-question's evidence never enters another's context, and
    the synthesizer sees the findings but no tools - it reports, it does not
    research.

    Left unset, n_parallel leaves the degree of parallelism to the model: the
    planner decides how many sub-questions the question needs, and all of them
    run at once - one researcher for a single factual question, several for a
    question with genuinely separate parts.

    Setting it to an int caps that instead, and exists for the limits the
    model cannot see: fanning six researchers at a provider that allows two
    concurrent requests turns a fast run into a wall of 429s, and a tool
    backed by one server may not survive six callers at once.

    Progress leaves through astream_events() - typed events out, rather than
    print, so the workflow is usable where stdout is not free to write to. A
    caller that only wants the result calls arun(), which drains the stream.

    All three prompts reach their agent verbatim: the sub-question cap is
    appended to the planner's rather than interpolated into it, since
    str.format would choke on a caller's prompt that happens to contain
    braces.
    """

    __slots__ = (
        "_budget",
        "_max_sub_questions",
        "_model",
        "_n_parallel",
        "_planner_system",
        "_researcher_system",
        "_sequential",
        "_synthesizer_system",
        "_tools",
    )

    def __init__(
        self,
        model: LLM,
        *,
        tools: Iterable[Callable[..., object]] | Toolbox = (),
        max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
        n_parallel: int | None = None,
        sequential: bool = False,
        planner_system: str = PLANNER_SYSTEM,
        researcher_system: str = RESEARCHER_SYSTEM,
        synthesizer_system: str = SYNTHESIZER_SYSTEM,
        budget: Budget | None = None,
    ):
        if max_sub_questions < 1:
            raise ConfigurationError(
                f"max_sub_questions must be at least 1, got {max_sub_questions}"
            )
        if n_parallel is not None and n_parallel < 1:
            raise ConfigurationError(
                f"n_parallel must be at least 1 when set, got {n_parallel}"
            )
        if sequential and n_parallel is not None:
            raise ConfigurationError(
                "sequential=True runs one researcher at a time, so n_parallel "
                "has nothing to cap - pass one or the other"
            )

        self._model = model
        self._tools = tools if isinstance(tools, Toolbox) else Toolbox(tools)
        self._max_sub_questions = max_sub_questions
        self._n_parallel = n_parallel
        self._sequential = sequential
        self._budget = budget
        self._researcher_system = researcher_system

        self._planner_system = (
            f"{planner_system}\n\nReturn at most {max_sub_questions} sub-questions."
        )
        self._synthesizer_system = synthesizer_system

    @property
    def tools(self) -> Toolbox:
        """The tools every researcher is given."""
        return self._tools

    @property
    def max_sub_questions(self) -> int:
        return self._max_sub_questions

    @property
    def sequential(self) -> bool:
        """Whether researchers run in order, each seeing the earlier answers."""
        return self._sequential

    @property
    def n_parallel(self) -> int | None:
        """The cap on concurrent researchers, or None to leave it to the model.

        None means every sub-question the planner produced runs at once, so
        the planner's own judgement sets the width of the fan-out.
        """
        return self._n_parallel

    async def arun(self, query: str) -> ResearchResult:
        """Research `query` end to end and return the report with its working.

        Drains astream_events() and keeps the ResearchFinished result, so a caller
        who does not want to watch a run live never has to think about
        events. Async only: the fan-out across sub-questions is the point of
        the workflow, so there is no synchronous path that would have to
        give it up. Drive it with `asyncio.run` from synchronous code.
        """
        async for event in self.astream_events(query):
            if isinstance(event, ResearchFinished):
                return event.result
        raise AssertionError(
            "a run always ends with ResearchFinished"
        )  # pragma: no cover

    async def astream_events(self, query: str) -> AsyncIterator[ResearchEvent]:
        """Drive one run, yielding progress and prose as they happen.

        The only driver: arun() consumes this and keeps the last event, so
        the workflow's mechanics - planning, fan-out, synthesis - exist once
        rather than once per public method.

        Every agent is built here rather than in __init__ so that two runs of
        the same instance cannot share one Agent's usage accounting or budget.
        """
        yield Planning(query)
        sub_questions, usage = await self._plan(query)
        yield Planned(sub_questions)

        yield Researching(len(sub_questions))
        findings: list[Finding] = []
        async for event, spent in self._research(sub_questions):
            findings.append(Finding(question=event.question, answer=event.answer))
            usage = usage + spent
            yield Researched(event.index, event.question)

        yield Synthesizing(len(findings))
        synthesizer = Agent(
            self._model, system=self._synthesizer_system, name="synthesizer"
        )
        async for event in synthesizer.astream_events(
            _synthesis_prompt(query, findings)
        ):
            if isinstance(event, TextDelta):
                yield event
            elif isinstance(event, _AgentFinished):
                usage = usage + synthesizer.total_usage
                yield ResearchFinished(
                    ResearchResult(
                        query=query,
                        sub_questions=sub_questions,
                        findings=findings,
                        report=event.state.output,
                        usage=usage,
                    )
                )

    async def _plan(self, query: str) -> tuple[list[str], TokenUsage]:
        """Split the query, falling back to the query itself.

        A planner that answers with nothing usable must not stall the run, and
        the original question is a serviceable single sub-question.
        """
        planner = Agent(
            self._model, system=self._planner_system, output=_Plan, name="planner"
        )
        result = await planner.arun(query)
        planned = result.output.sub_questions if result.answered else []
        sub_questions = planned[: self._max_sub_questions] or [query]
        return sub_questions, planner.total_usage

    async def _research(
        self, sub_questions: list[str]
    ) -> AsyncIterator[tuple[_Researched, TokenUsage]]:
        """Answer every sub-question, yielding each as it completes.

        Parallel is the default and uses as_completed, so a finished
        sub-answer lands when it is ready rather than when the slowest one
        catches up. Sequential exists for questions whose parts build on each
        other - find the things, then look up details about the things found -
        which parallel researchers structurally cannot do, since each starts
        before any other has answered.
        """
        if self._sequential:
            async for item in self._research_in_order(sub_questions):
                yield item
            return

        limit = asyncio.Semaphore(self._n_parallel) if self._n_parallel else None
        tasks = [
            asyncio.ensure_future(self._research_one(i, question, limit))
            for i, question in enumerate(sub_questions, 1)
        ]
        for task in asyncio.as_completed(tasks):
            yield await task

    async def _research_in_order(
        self, sub_questions: list[str]
    ) -> AsyncIterator[tuple[_Researched, TokenUsage]]:
        findings: list[Finding] = []
        for index, question in enumerate(sub_questions, 1):
            researched, usage = await self._research_one(
                index, question, None, findings
            )
            findings.append(
                Finding(question=researched.question, answer=researched.answer)
            )
            yield researched, usage

    async def _research_one(
        self,
        index: int,
        question: str,
        limit: asyncio.Semaphore | None,
        earlier: list[Finding] | None = None,
    ) -> tuple[_Researched, TokenUsage]:
        researcher = Agent(
            self._model,
            tools=self._tools,
            system=self._researcher_system,
            budget=self._budget or Budget(),
            name=f"researcher-{index}",
        )
        prompt = _with_earlier(question, earlier)
        if limit is None:
            result = await researcher.arun(prompt)
        else:
            async with limit:
                result = await researcher.arun(prompt)
        researched = _Researched(index=index, question=question, answer=result.output)
        return researched, researcher.total_usage


def _with_earlier(question: str, earlier: list[Finding] | None) -> str:
    """A sequential researcher's question, prefixed with what is known so far.

    Only the answers are carried, not the transcripts behind them: a
    researcher needs the facts an earlier one found, not the searches it ran
    to find them, and the context window is the binding limit here.
    """
    if not earlier:
        return question
    known = "\n\n".join(
        f"Sub-question: {finding.question}\nAnswer: {finding.answer}"
        for finding in earlier
    )
    return (
        f"Already established by earlier research:\n\n{known}\n\n"
        f"Now answer this, using what is above where it helps: {question}"
    )


def _synthesis_prompt(query: str, findings: list[Finding]) -> str:
    answers = "\n\n".join(
        f"Sub-question: {finding.question}\nAnswer: {finding.answer}"
        for finding in findings
    )
    return f"Research question: {query}\n\nFindings:\n\n{answers}"
