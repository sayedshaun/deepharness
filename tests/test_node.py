from subagents.agent import Agent
from subagents.graph import Node


def test_node_wraps_agent():
    agent = Agent("researcher")
    node = Node(agent)

    assert node.agent is agent


def test_node_default_configuration():
    node = Node(Agent("researcher"))

    assert node.retry == 0
    assert node.timeout is None
    assert node.edges == []


def test_node_custom_configuration():
    node = Node(Agent("researcher"), retry=3, timeout=5.0)

    assert node.retry == 3
    assert node.timeout == 5.0


def test_node_connect_without_condition():
    source = Node(Agent("router"))
    target = Node(Agent("researcher"))

    source.connect(target)

    assert source.edges == [(target, None)]


def test_node_connect_with_condition():
    source = Node(Agent("router"))
    target = Node(Agent("researcher"))
    condition = lambda state: True

    source.connect(target, condition=condition)

    assert source.edges == [(target, condition)]


def test_node_supports_multiple_edges():
    source = Node(Agent("router"))
    a = Node(Agent("researcher"))
    b = Node(Agent("writer"))

    source.connect(a)
    source.connect(b)

    assert source.edges == [(a, None), (b, None)]
