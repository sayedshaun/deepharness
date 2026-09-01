from dataclasses import dataclass

from deepharness.agent import Agent, Message, tool
from deepharness.providers.base import CompletionResponse, ToolCall
from deepharness.tools import Ctx, Toolbox

from .test_agent import ScriptedProvider


@dataclass
class Deps:
    tenant: str


def calls(*names_and_args):
    return CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=arguments)
            for i, (name, arguments) in enumerate(names_and_args)
        ],
    )


def test_a_ctx_parameter_is_hidden_from_the_model():
    @tool
    def remember(fact: str, ctx: Ctx) -> str:
        """Write a fact down."""
        return fact

    parameters = remember._tool_spec.parameters

    assert list(parameters["properties"]) == ["fact"]
    assert parameters["required"] == ["fact"]
    assert remember._tool_spec.ctx_params == ("ctx",)


def test_a_tool_without_a_ctx_parameter_declares_none():
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert add._tool_spec.ctx_params == ()


async def test_calling_a_tool_directly_gets_an_empty_context():
    @tool
    def peek(ctx: Ctx) -> str:
        """Report what the context holds."""
        return f"{ctx.state}|{ctx.deps}"

    assert await Toolbox([peek]).call("peek") == "None|None"


async def test_a_tool_reads_the_deps_the_run_was_given():
    seen: list[Deps] = []

    @tool
    def lookup(key: str, ctx: Ctx) -> str:
        """Look something up for the current tenant."""
        seen.append(ctx.deps)
        return f"{key} for {ctx.deps.tenant}"

    provider = ScriptedProvider(
        [calls(("lookup", {"key": "plan"})), CompletionResponse(content="done")]
    )
    agent = Agent(provider, tools=[lookup])

    state = await agent.arun("what plan?", deps=Deps(tenant="acme"))

    assert seen == [Deps(tenant="acme")]
    assert "plan for acme" in str(state.messages)


def test_deps_reach_a_sync_run_too():
    @tool
    def lookup(ctx: Ctx) -> str:
        """Report the tenant."""
        return ctx.deps.tenant

    provider = ScriptedProvider(
        [calls(("lookup", {})), CompletionResponse(content="done")]
    )
    agent = Agent(provider, tools=[lookup])

    state = agent.run("who?", deps=Deps(tenant="globex"))

    assert "globex" in str(state.messages)


async def test_a_tool_can_read_the_transcript_it_is_part_of():
    @tool
    def count_messages(ctx: Ctx) -> str:
        """Report how many messages the run has so far."""
        return str(len(ctx.state.messages))

    provider = ScriptedProvider(
        [calls(("count_messages", {})), CompletionResponse(content="done")]
    )
    agent = Agent(provider, tools=[count_messages])

    state = await agent.arun([Message.human("first"), Message.human("second")])

    assert any(message.get("content") == "2" for message in state.messages)


async def test_a_delegated_agent_inherits_the_callers_deps():
    seen: list[Deps] = []

    @tool
    def lookup(ctx: Ctx) -> str:
        """Report the tenant."""
        seen.append(ctx.deps)
        return ctx.deps.tenant

    sub_provider = ScriptedProvider(
        [calls(("lookup", {})), CompletionResponse(content="acme's answer")]
    )
    sub_agent = Agent(sub_provider, name="specialist", tools=[lookup])

    main_provider = ScriptedProvider(
        [
            calls(("specialist", {"input": "look it up"})),
            CompletionResponse(content="relayed"),
        ]
    )
    main_agent = Agent(main_provider, tools=[sub_agent.as_tool()])

    state = await main_agent.arun("delegate this", deps=Deps(tenant="acme"))

    assert seen == [Deps(tenant="acme")]
    assert state.output == "relayed"
