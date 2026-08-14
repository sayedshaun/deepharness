import asyncio
from dataclasses import dataclass, field

import pytest

from subagents.graph import ExecutionError, Graph


@dataclass
class State:
    trace: list = field(default_factory=list)
    result_a: str = ""
    result_b: str = ""
    combined: str = ""


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
