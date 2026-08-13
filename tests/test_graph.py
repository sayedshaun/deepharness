import pytest

from subagents.agent import Agent
from subagents.graph import Graph, Node


def make_node(name: str) -> Node:
    return Node(Agent(name))


def test_add_node():
    graph = Graph()
    node = make_node("researcher")

    graph.add_node(node)

    assert graph.nodes["researcher"] is node


def test_add_duplicate_node_raises():
    graph = Graph()
    graph.add_node(make_node("researcher"))

    with pytest.raises(ValueError):
        graph.add_node(make_node("researcher"))


def test_connect_nodes():
    graph = Graph()
    source = make_node("router")
    target = make_node("researcher")
    graph.add_node(source)
    graph.add_node(target)

    graph.connect(source, target)

    assert source.edges == [(target, None)]


def test_connect_with_condition():
    graph = Graph()
    source = make_node("router")
    target = make_node("researcher")
    graph.add_node(source)
    graph.add_node(target)
    condition = lambda state: state["task_type"] == "research"

    graph.connect(source, target, condition=condition)

    assert source.edges == [(target, condition)]


def test_connect_unknown_source_raises():
    graph = Graph()
    target = make_node("researcher")
    graph.add_node(target)

    with pytest.raises(ValueError):
        graph.connect(make_node("router"), target)


def test_connect_unknown_target_raises():
    graph = Graph()
    source = make_node("router")
    graph.add_node(source)

    with pytest.raises(ValueError):
        graph.connect(source, make_node("researcher"))
