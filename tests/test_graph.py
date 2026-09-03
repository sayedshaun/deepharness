from dataclasses import dataclass

import pytest

from deepharness.errors import ConfigurationError, DeepHarnessError
from deepharness.graph import Graph


@dataclass
class State:
    trace: list = None

    def __post_init__(self):
        if self.trace is None:
            self.trace = []


def test_add_registers_node_by_function_name():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    assert "func1" in graph.nodes
    assert graph.nodes["func1"].func is func1
    assert graph.nodes["func1"].start is True


def test_add_registers_node_by_explicit_name():
    graph = Graph(State)

    @graph.add(start=True, name="custom")
    def func1(state: State) -> State:
        return state

    assert "custom" in graph.nodes
    assert "func1" not in graph.nodes


def test_add_duplicate_name_raises():
    graph = Graph(State)

    @graph.add(start=True, name="dup")
    def func1(state: State) -> State:
        return state

    with pytest.raises(ValueError):

        @graph.add(name="dup")
        def func2(state: State) -> State:
            return state


def test_connect_by_function_reference():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    @graph.add(end=True)
    def func2(state: State) -> State:
        return state

    graph.connect(func1, func2)

    assert graph.edges["func1"] == [("func2", None, False)]


def test_connect_by_name():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    @graph.add(end=True)
    def func2(state: State) -> State:
        return state

    graph.connect("func1", "func2")

    assert graph.edges["func1"] == [("func2", None, False)]


def test_connect_with_condition():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    @graph.add(end=True)
    def func2(state: State) -> State:
        return state

    condition = lambda state: True
    graph.connect(func1, func2, condition=condition)

    assert graph.edges["func1"] == [("func2", condition, False)]


def test_connect_unregistered_function_raises():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def func1(state: State) -> State:
        return state

    def not_registered(state: State) -> State:
        return state

    with pytest.raises(ValueError):
        graph.connect(func1, not_registered)


def test_build_without_start_node_raises():
    graph = Graph(State)

    @graph.add(end=True)
    def func1(state: State) -> State:
        return state

    with pytest.raises(ValueError):
        graph.build()


def test_build_with_unreachable_node_raises():
    graph = Graph(State)

    @graph.add(start=True, end=True)
    def func1(state: State) -> State:
        return state

    @graph.add()
    def orphan(state: State) -> State:
        return state

    with pytest.raises(ValueError):
        graph.build()


def test_build_with_cycle_raises():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    @graph.add()
    def func2(state: State) -> State:
        return state

    graph.connect(func1, func2)
    graph.connect(func2, func1)

    with pytest.raises(ValueError, match="loop=True"):
        graph.build()


def test_build_with_declared_loop_succeeds():
    graph = Graph(State)

    @graph.add(start=True)
    def func1(state: State) -> State:
        return state

    @graph.add()
    def func2(state: State) -> State:
        return state

    graph.connect(func1, func2)
    graph.connect(func2, func1, loop=True)

    assert graph.build() is not None


def test_build_rejects_a_state_that_is_not_a_dataclass():
    class NotAState:
        pass

    graph = Graph(NotAState)

    @graph.add(start=True, end=True)
    def only(state):
        return state

    with pytest.raises(ConfigurationError, match="must be a dataclass"):
        graph.build()


async def test_run_rejects_a_state_of_the_wrong_type():
    """Otherwise the mismatch surfaces as an AttributeError from whichever node
    happened to touch a missing field first."""

    @dataclass
    class Other:
        value: int = 0

    graph = Graph(Other)

    @graph.add(start=True, end=True)
    def only(state):
        return state

    executor = graph.build()

    with pytest.raises(ConfigurationError, match="runs on Other, got int"):
        await executor.run(42)


def test_builder_errors_are_deepharness_errors():
    """They were bare ValueErrors, so ConfigurationError stays one too."""
    graph = Graph(State)

    with pytest.raises(DeepHarnessError):
        graph.connect("nope", "nowhere")
