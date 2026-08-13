import asyncio

import pytest

from subagents.agent import Agent
from subagents.executor import ExecutionError, Executor
from subagents.graph import Graph, Node


class RecordingAgent(Agent):
    """Appends its name to state['trace'] and returns state unchanged."""

    async def run(self, state):
        state.setdefault("trace", []).append(self.name)
        return state


class RoutingAgent(Agent):
    """Sets state['task_type'] to a fixed value so a router can branch on it."""

    def __init__(self, name: str, task_type: str):
        super().__init__(name)
        self.task_type = task_type

    async def run(self, state):
        state["task_type"] = self.task_type
        state.setdefault("trace", []).append(self.name)
        return state


class FlakyAgent(Agent):
    """Fails a fixed number of times before succeeding."""

    def __init__(self, name: str, failures: int):
        super().__init__(name)
        self.failures = failures
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"attempt {self.calls} failed")
        state.setdefault("trace", []).append(self.name)
        return state


class SlowAgent(Agent):
    async def run(self, state):
        await asyncio.sleep(10)
        return state


def build_graph(*nodes: Node) -> Graph:
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    return graph


async def test_executes_single_node():
    node = Node(RecordingAgent("only"))
    graph = build_graph(node)
    executor = Executor()

    result = await executor.run(graph, node, {})

    assert result["trace"] == ["only"]


async def test_executes_sequential_chain():
    first = Node(RecordingAgent("first"))
    second = Node(RecordingAgent("second"))
    graph = build_graph(first, second)
    graph.connect(first, second)
    executor = Executor()

    result = await executor.run(graph, first, {})

    assert result["trace"] == ["first", "second"]


async def test_conditional_routing_follows_matching_edge():
    router = Node(RoutingAgent("router", task_type="research"))
    researcher = Node(RecordingAgent("researcher"))
    writer = Node(RecordingAgent("writer"))
    graph = build_graph(router, researcher, writer)
    graph.connect(router, researcher, condition=lambda state: state["task_type"] == "research")
    graph.connect(router, writer, condition=lambda state: state["task_type"] == "write")
    executor = Executor()

    result = await executor.run(graph, router, {})

    assert result["trace"] == ["router", "researcher"]


async def test_conditional_routing_stops_when_no_edge_matches():
    router = Node(RoutingAgent("router", task_type="unknown"))
    researcher = Node(RecordingAgent("researcher"))
    graph = build_graph(router, researcher)
    graph.connect(router, researcher, condition=lambda state: state["task_type"] == "research")
    executor = Executor()

    result = await executor.run(graph, router, {})

    assert result["trace"] == ["router"]


async def test_retries_until_success():
    node = Node(FlakyAgent("flaky", failures=2), retry=2)
    graph = build_graph(node)
    executor = Executor()

    result = await executor.run(graph, node, {})

    assert result["trace"] == ["flaky"]
    assert node.agent.calls == 3


async def test_raises_execution_error_after_exhausting_retries():
    node = Node(FlakyAgent("flaky", failures=5), retry=2)
    graph = build_graph(node)
    executor = Executor()

    with pytest.raises(ExecutionError) as exc_info:
        await executor.run(graph, node, {})

    assert exc_info.value.node is node
    assert exc_info.value.agent_name == "flaky"
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.original, RuntimeError)


async def test_timeout_raises_execution_error():
    node = Node(SlowAgent("slow"), timeout=0.01)
    graph = build_graph(node)
    executor = Executor()

    with pytest.raises(ExecutionError) as exc_info:
        await executor.run(graph, node, {})

    assert isinstance(exc_info.value.original, asyncio.TimeoutError)


async def test_start_node_must_belong_to_graph():
    graph = Graph()
    outside_node = Node(RecordingAgent("outsider"))
    executor = Executor()

    with pytest.raises(ValueError):
        await executor.run(graph, outside_node, {})


async def test_state_propagates_across_nodes():
    first = Node(RecordingAgent("first"))
    second = Node(RecordingAgent("second"))
    graph = build_graph(first, second)
    graph.connect(first, second)
    executor = Executor()

    result = await executor.run(graph, first, {"input": "topic"})

    assert result["input"] == "topic"
    assert result["trace"] == ["first", "second"]
