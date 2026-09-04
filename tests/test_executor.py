import asyncio
from dataclasses import dataclass, field

import pytest

from deepharness.errors import ConfigurationError
from deepharness.graph import (
    ConcurrentUpdateError,
    ExecutionError,
    Executor,
    Graph,
    StepLimitExceeded,
    concat,
)


@dataclass
class State:
    trace: list = field(default_factory=list)
    result_a: str = ""
    result_b: str = ""
    combined: str = ""


@dataclass
class MergedState:
    """Separate from State so the reducer only applies where it's under test."""

    trace: list = field(default_factory=list, metadata={"reducer": concat})
    rounds: int = 0


async def test_executes_single_start_end_node():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def only(state: State) -> State:
        state.trace.append("only")
        return state

    executor = graph.build()
    result = await executor.run(State())

    assert result.trace == ["only"]


async def test_executes_sequential_chain():
    graph = Graph(State)

    @graph.add(start=True)
    def first(state: State) -> State:
        state.trace.append("first")
        return state

    @graph.add(end=True)
    def second(state: State) -> State:
        state.trace.append("second")
        return state

    graph.connect(first, second)
    executor = graph.build()

    result = await executor.run(State())

    assert result.trace == ["first", "second"]


async def test_parallel_start_nodes_merge_before_join_node():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        state.result_a = "a"
        return state

    @graph.add(start=True)
    def func2(state: State) -> State:
        state.result_b = "b"
        return state

    @graph.add(end=True)
    def func3(state: State) -> State:
        state.combined = state.result_a + state.result_b
        return state

    graph.connect(func1, func3)
    graph.connect(func2, func3)
    executor = graph.build()

    result = await executor.run(State())

    assert result.result_a == "a"
    assert result.result_b == "b"
    assert result.combined == "ab"


async def test_conditional_routing_follows_matching_edge():
    graph = Graph(State)

    @graph.add(start=True)
    def router(state: State) -> State:
        state.result_a = "research"
        return state

    @graph.add()
    def researcher(state: State) -> State:
        state.trace.append("researcher")
        return state

    @graph.add()
    def writer(state: State) -> State:
        state.trace.append("writer")
        return state

    graph.connect(
        router, researcher, condition=lambda state: state.result_a == "research"
    )
    graph.connect(router, writer, condition=lambda state: state.result_a == "write")
    executor = graph.build()

    result = await executor.run(State())

    assert result.trace == ["researcher"]


async def test_conditional_routing_stops_when_no_edge_matches():
    graph = Graph(State)

    @graph.add(start=True)
    def router(state: State) -> State:
        state.result_a = "unknown"
        return state

    @graph.add()
    def researcher(state: State) -> State:
        state.trace.append("researcher")
        return state

    graph.connect(
        router, researcher, condition=lambda state: state.result_a == "research"
    )
    executor = graph.build()

    result = await executor.run(State())

    assert result.trace == []


async def test_node_failure_raises_execution_error():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def flaky(state: State) -> State:
        raise RuntimeError("boom")

    executor = graph.build()

    with pytest.raises(ExecutionError) as exc_info:
        await executor.run(State())

    assert exc_info.value.node_name == "flaky"
    assert isinstance(exc_info.value.original, RuntimeError)


async def test_supports_async_node_functions():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    async def async_node(state: State) -> State:
        await asyncio.sleep(0)
        state.trace.append("async_node")
        return state

    executor = graph.build()
    result = await executor.run(State())

    assert result.trace == ["async_node"]


async def test_run_does_not_mutate_the_input_state():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def only(state: State) -> State:
        state.trace.append("only")
        return state

    executor = graph.build()
    original = State()

    await executor.run(original)

    assert original.trace == []


async def test_reducer_keeps_both_concurrent_writes():
    graph = Graph(MergedState)

    @graph.add(start=True)
    def branch_a(state: MergedState) -> MergedState:
        state.trace.append("a")
        return state

    @graph.add(start=True)
    def branch_b(state: MergedState) -> MergedState:
        state.trace.append("b")
        return state

    @graph.add(end=True)
    def join(state: MergedState) -> MergedState:
        state.trace.append("join")
        return state

    graph.connect(branch_a, join)
    graph.connect(branch_b, join)

    result = await graph.build().run(MergedState())

    # without a reducer this used to keep only the last branch's append
    assert sorted(result.trace[:2]) == ["a", "b"]
    assert result.trace[2] == "join"


async def test_concurrent_write_without_reducer_raises():
    graph = Graph(State)

    @graph.add(start=True)
    def branch_a(state: State) -> State:
        state.trace.append("a")
        return state

    @graph.add(start=True)
    def branch_b(state: State) -> State:
        state.trace.append("b")
        return state

    @graph.add(end=True)
    def join(state: State) -> State:
        return state

    graph.connect(branch_a, join)
    graph.connect(branch_b, join)

    with pytest.raises(ConcurrentUpdateError) as exc_info:
        await graph.build().run(State())

    assert exc_info.value.field_name == "trace"
    assert exc_info.value.writer_count == 2


async def test_loop_edge_iterates_until_condition_flips():
    graph = Graph(MergedState)

    @graph.add(start=True)
    def agent(state: MergedState) -> MergedState:
        state.trace.append(f"agent{state.rounds}")
        state.rounds += 1
        return state

    @graph.add()
    def tools(state: MergedState) -> MergedState:
        state.trace.append("tools")
        return state

    graph.connect(agent, tools, condition=lambda s: s.rounds < 3)
    graph.connect(tools, agent, loop=True)

    result = await graph.build().run(MergedState())

    assert result.trace == ["agent0", "tools", "agent1", "tools", "agent2"]


async def test_node_upstream_of_a_loop_runs_once():
    graph = Graph(MergedState)

    @graph.add(start=True)
    def setup(state: MergedState) -> MergedState:
        state.trace.append("setup")
        return state

    @graph.add()
    def body(state: MergedState) -> MergedState:
        state.trace.append(f"body{state.rounds}")
        state.rounds += 1
        return state

    graph.connect(setup, body)
    graph.connect(body, body, loop=True, condition=lambda s: s.rounds < 3)

    result = await graph.build().run(MergedState())

    # setup is outside the loop body, so it keeps its result instead of re-running
    assert result.trace == ["setup", "body0", "body1", "body2"]


async def test_conditional_node_in_loop_body_runs_only_on_matching_passes():
    """A body branch that sits out one pass must still be reconsidered later."""
    graph = Graph(MergedState)

    @graph.add(start=True)
    def head(state: MergedState) -> MergedState:
        state.rounds += 1
        state.trace.append(f"head{state.rounds}")
        return state

    @graph.add()
    def always(state: MergedState) -> MergedState:
        state.trace.append("always")
        return state

    @graph.add()
    def odd_only(state: MergedState) -> MergedState:
        state.trace.append(f"odd{state.rounds}")
        return state

    graph.connect(head, always)
    graph.connect(head, odd_only, condition=lambda s: s.rounds % 2 == 1)
    graph.connect(always, head, loop=True, condition=lambda s: s.rounds < 3)

    result = await graph.build().run(MergedState())

    # skipped on round 2, then reconsidered and taken again on round 3
    assert [entry for entry in result.trace if entry.startswith("odd")] == [
        "odd1",
        "odd3",
    ]
    assert result.rounds == 3


async def test_step_limit_raises_with_partial_state():
    graph = Graph(MergedState)

    @graph.add(start=True)
    def forever(state: MergedState) -> MergedState:
        state.trace.append("tick")
        return state

    graph.connect(forever, forever, loop=True)

    with pytest.raises(StepLimitExceeded) as exc_info:
        await graph.build().run(MergedState(), max_steps=5)

    assert exc_info.value.limit == 5
    # the work done before the limit survives the exception
    assert exc_info.value.state.trace == ["tick"] * 5


async def test_conditional_fan_out_into_join_still_works():
    """Regression: a skipped branch must not wedge a downstream join."""
    graph = Graph(State)

    @graph.add(start=True)
    def router(state: State) -> State:
        state.result_a = "research"
        return state

    @graph.add()
    def researcher(state: State) -> State:
        state.trace.append("researcher")
        return state

    @graph.add()
    def writer(state: State) -> State:
        state.trace.append("writer")
        return state

    @graph.add(end=True)
    def join(state: State) -> State:
        state.trace.append("join")
        return state

    graph.connect(router, researcher, condition=lambda s: s.result_a == "research")
    graph.connect(router, writer, condition=lambda s: s.result_a == "write")
    graph.connect(researcher, join)
    graph.connect(writer, join)

    result = await graph.build().run(State())

    assert result.trace == ["researcher", "join"]


async def test_run_without_a_state_builds_the_declared_type():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def touch(state: State) -> State:
        state.combined = "built"
        return state

    result = await graph.build().run()

    assert isinstance(result, State)
    assert result.combined == "built"


async def test_run_without_a_state_and_without_a_declared_type_raises():
    executor = Executor(nodes={}, predecessors={}, state_type=None)

    with pytest.raises(ConfigurationError, match="needs a state"):
        await executor.run()


async def test_run_still_rejects_a_state_of_the_wrong_type():
    @dataclass
    class Other:
        value: str = ""

    graph = Graph(State)

    @graph.add(start=True, end=True)
    def touch(state: State) -> State:
        return state

    with pytest.raises(ConfigurationError, match="runs on State"):
        await graph.build().run(Other())
