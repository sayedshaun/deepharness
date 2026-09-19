import asyncio
import json

import pytest

from deepharness.agent import Budget, tool
from deepharness.agent.output import FINAL_TOOL
from deepharness.errors import ConfigurationError
from deepharness.prebuilt.research import (
    RESEARCHER_SYSTEM,
    DeepResearch,
    Finding,
    Planned,
    Planning,
    Researched,
    ResearchEvent,
    ResearchFinished,
    Researching,
    Synthesizing,
)
from deepharness.providers.base import (
    LLM,
    CompletionResponse,
    TextDelta,
    TokenUsage,
    ToolCall,
)


class RoleProvider(LLM):
    """Answers by role rather than by call order.

    The researchers run concurrently, so a fixed response sequence would make
    assertions depend on scheduling. This keys off the system prompt instead:
    every call gets the reply its agent should get, whenever it arrives.
    """

    def __init__(
        self,
        sub_questions: list[str],
        report: str = "REPORT",
        usage: TokenUsage | None = None,
    ):
        self._sub_questions = sub_questions
        self._report = report
        self._usage = usage
        self.prompts: list[str] = []
        self.concurrent = 0
        self.peak_concurrent = 0

    def _reply(self, messages: list[dict]) -> CompletionResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        self.prompts.append(system)
        if "sub-questions that, answered together" in system:
            return CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_plan",
                        name=FINAL_TOOL,
                        arguments={"sub_questions": self._sub_questions},
                    )
                ],
                usage=self._usage,
            )
        if "Write one thorough" in system:
            return CompletionResponse(content=self._report, usage=self._usage)
        question = next((m["content"] for m in messages if m.get("role") == "user"), "")
        return CompletionResponse(content=f"answer to {question}", usage=self._usage)

    async def agenerate(self, messages, *, tools=None):
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)  # let siblings start before this one replies
            return self._reply(messages)
        finally:
            self.concurrent -= 1

    def generate(self, messages, *, tools=None):
        return self._reply(messages)


async def test_a_run_plans_researches_and_reports():
    model = RoleProvider(["what is A?", "what is B?"], report="final report")
    research = DeepResearch(model)

    result = await research.arun("A and B")

    assert result.query == "A and B"
    assert result.sub_questions == ["what is A?", "what is B?"]
    assert result.report == "final report"
    assert {finding.answer for finding in result.findings} == {
        "answer to what is A?",
        "answer to what is B?",
    }


async def test_the_planner_is_capped_at_max_sub_questions():
    model = RoleProvider(["one", "two", "three", "four"])
    research = DeepResearch(model, max_sub_questions=2)

    result = await research.arun("query")

    assert result.sub_questions == ["one", "two"]
    assert len(result.findings) == 2


async def test_a_planner_that_returns_nothing_falls_back_to_the_question():
    model = RoleProvider([])
    research = DeepResearch(model)

    result = await research.arun("the original question")

    assert result.sub_questions == ["the original question"]


async def test_n_parallel_caps_how_many_researchers_run_at_once():
    model = RoleProvider(["a", "b", "c", "d"])
    research = DeepResearch(model, n_parallel=2)

    result = await research.arun("query")

    assert len(result.findings) == 4
    # Planner and synthesizer run alone, so the peak is the researchers'.
    assert model.peak_concurrent <= 2


async def test_researchers_all_run_at_once_when_n_parallel_is_unset():
    model = RoleProvider(["a", "b", "c", "d"])
    research = DeepResearch(model)

    await research.arun("query")

    assert model.peak_concurrent == 4


async def test_the_tools_reach_every_researcher_but_not_the_other_agents():
    @tool
    def lookup(topic: str) -> str:
        """Look a topic up."""
        return f"{topic} looked up"

    schemas: list[list[str]] = []

    class ToolWatchingProvider(RoleProvider):
        async def agenerate(self, messages, *, tools=None):
            schemas.append([schema["name"] for schema in tools or []])
            return await super().agenerate(messages, tools=tools)

    model = ToolWatchingProvider(["a", "b"])
    research = DeepResearch(model, tools=[lookup])

    await research.arun("query")

    assert ["lookup"] in schemas, "researchers should be offered the tool"
    # The planner is offered only its output tool; the synthesizer, nothing.
    assert [FINAL_TOOL] in schemas
    assert [] in schemas
    assert "lookup" in research.tools


async def test_prompts_are_replaceable_and_used_verbatim():
    model = RoleProvider(["a"])
    research = DeepResearch(
        model,
        planner_system="PLAN {not-a-format-field}",
        researcher_system="RESEARCH",
        synthesizer_system="SYNTH",
    )

    # A custom planner prompt means the default's marker is gone, so the
    # stub falls through to its researcher branch - which is the point: the
    # prompt reached the model unchanged, braces and all.
    await research.arun("query")

    assert any(
        prompt.startswith("PLAN {not-a-format-field}") for prompt in model.prompts
    )
    assert "RESEARCH" in model.prompts
    assert "SYNTH" in model.prompts


async def test_the_sub_question_cap_is_appended_to_the_planner_prompt():
    model = RoleProvider(["a"])

    await DeepResearch(model, max_sub_questions=3).arun("query")

    assert any("Return at most 3 sub-questions." in prompt for prompt in model.prompts)


async def test_progress_is_reported_as_typed_events():
    """Events carry data, not prose, so a caller can count them exactly.

    The transcript of a run contains whatever the web and the model said, so
    counting by matching substrings against it miscounts on any run whose
    report happens to quote the words being matched.
    """
    model = RoleProvider(["a", "b"], report="the report")
    research = DeepResearch(model)

    events: list[ResearchEvent] = [e async for e in research.astream_events("query")]

    steps = [e for e in events if not isinstance(e, TextDelta)]
    assert [type(event) for event in steps] == [
        Planning,
        Planned,
        Researching,
        Researched,
        Researched,
        Synthesizing,
        ResearchFinished,
    ]
    assert steps[0].query == "query"
    assert steps[1].sub_questions == ["a", "b"]
    assert steps[2].count == 2
    assert {event.question for event in steps if isinstance(event, Researched)} == {
        "a",
        "b",
    }
    assert steps[-2].findings == 2
    assert steps[-1].result.report == "the report"

    text = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(text) == "the report"


async def test_a_result_reports_what_the_whole_run_cost():
    """Usage covers every agent, not just the last one to speak."""
    model = RoleProvider(["a", "b"], usage=TokenUsage(3, 2, 5))

    result = await DeepResearch(model).arun("query")

    # planner + two researchers + synthesizer = four agents, one call each.
    assert result.usage == TokenUsage(12, 8, 20)


async def test_usage_is_per_run_not_cumulative():
    """A second run reports its own cost, not the sum of both."""
    model = RoleProvider(["a"], usage=TokenUsage(1, 1, 2))
    research = DeepResearch(model)

    first = await research.arun("query")
    second = await research.arun("query")

    assert first.usage == second.usage == TokenUsage(3, 3, 6)


async def test_a_run_prints_nothing_on_its_own(capsys):
    """arun() drains the event stream itself; nothing reaches stdout unless
    the caller iterates astream_events() and prints it themselves."""
    model = RoleProvider(["a"])

    await DeepResearch(model).arun("query")

    assert capsys.readouterr().out == ""


async def test_the_budget_reaches_the_researchers():
    """A researcher that keeps calling tools must stop at the step budget.

    Asserted through behaviour rather than by reading the Agent's config: with
    steps=1 the loop gets one model call and no turn to react to the tool
    result, so the run ends without prose instead of looping forever.
    """

    @tool
    def lookup(topic: str) -> str:
        """Look a topic up."""
        return f"{topic} looked up"

    class ToolLoopingProvider(RoleProvider):
        """Every researcher turn asks for the tool again, never answering."""

        def _reply(self, messages):
            reply = super()._reply(messages)
            system = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            if system == RESEARCHER_SYSTEM:
                return CompletionResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="c", name="lookup", arguments={"topic": "x"})
                    ],
                )
            return reply

    model = ToolLoopingProvider(["a"])
    research = DeepResearch(model, tools=[lookup], budget=Budget(steps=1))

    result = await research.arun("query")

    assert result.findings[0].answer == ""
    assert len(result.findings) == 1


def test_bad_options_are_rejected_at_construction():
    model = RoleProvider(["a"])

    with pytest.raises(
        ConfigurationError, match="max_sub_questions must be at least 1"
    ):
        DeepResearch(model, max_sub_questions=0)

    with pytest.raises(ConfigurationError, match="n_parallel must be at least 1"):
        DeepResearch(model, n_parallel=0)


async def test_the_synthesizer_is_given_the_question_and_every_finding():
    model = RoleProvider(["what is A?", "what is B?"])
    seen: list[str] = []

    class PromptCapturingProvider(RoleProvider):
        async def agenerate(self, messages, *, tools=None):
            system = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            if "Write one thorough" in system:
                seen.append(
                    next(m["content"] for m in messages if m.get("role") == "user")
                )
            return await super().agenerate(messages, tools=tools)

    model = PromptCapturingProvider(["what is A?", "what is B?"])
    await DeepResearch(model).arun("A and B")

    assert len(seen) == 1
    prompt = seen[0]
    assert "A and B" in prompt
    assert "what is A?" in prompt
    assert "answer to what is B?" in prompt


def test_a_finding_is_a_typed_pair():
    finding = Finding("q", "a")

    assert (finding.question, finding.answer) == ("q", "a")
    with pytest.raises(AttributeError):
        finding.extra = json.dumps({})  # slots: no accidental attributes


async def test_sequential_researchers_see_the_earlier_answers():
    """The point of the mode: a later sub-question can use an earlier answer.

    Parallel researchers structurally cannot do this - each starts before any
    other has finished - so a question whose parts build on each other needs
    the answers threaded through.
    """
    seen: list[str] = []

    class PromptCapturingProvider(RoleProvider):
        async def agenerate(self, messages, *, tools=None):
            system = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            if system == RESEARCHER_SYSTEM:
                seen.append(
                    next(m["content"] for m in messages if m.get("role") == "user")
                )
            return await super().agenerate(messages, tools=tools)

    model = PromptCapturingProvider(["name the things", "detail each thing"])

    result = await DeepResearch(model, sequential=True).arun("query")

    assert len(seen) == 2
    assert seen[0] == "name the things"
    assert "Already established by earlier research" in seen[1]
    assert "answer to name the things" in seen[1]
    assert result.sub_questions == ["name the things", "detail each thing"]


async def test_sequential_keeps_the_planned_order():
    model = RoleProvider(["a", "b", "c"])

    result = await DeepResearch(model, sequential=True).arun("query")

    assert [finding.question for finding in result.findings] == ["a", "b", "c"]


async def test_sequential_runs_one_researcher_at_a_time():
    model = RoleProvider(["a", "b", "c"])

    await DeepResearch(model, sequential=True).arun("query")

    assert model.peak_concurrent == 1


async def test_parallel_researchers_do_not_see_each_other():
    """The default must stay isolated: no earlier answers in the prompt."""
    seen: list[str] = []

    class PromptCapturingProvider(RoleProvider):
        async def agenerate(self, messages, *, tools=None):
            system = next(
                (m["content"] for m in messages if m.get("role") == "system"), ""
            )
            if system == RESEARCHER_SYSTEM:
                seen.append(
                    next(m["content"] for m in messages if m.get("role") == "user")
                )
            return await super().agenerate(messages, tools=tools)

    model = PromptCapturingProvider(["a", "b"])

    await DeepResearch(model).arun("query")

    assert sorted(seen) == ["a", "b"]


def test_sequential_and_n_parallel_are_rejected_together():
    model = RoleProvider(["a"])

    with pytest.raises(ConfigurationError, match="n_parallel has nothing to cap"):
        DeepResearch(model, sequential=True, n_parallel=2)
