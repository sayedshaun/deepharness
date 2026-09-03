import enum
from typing import Any, Literal

import pytest

from deepharness.agent import Toolbox, tool
from deepharness.errors import ConfigurationError


def test_bare_decorator_builds_spec_from_signature_and_docstring():
    @tool
    def search(query: str, limit: int = 5) -> str:
        """Search the web for a query."""
        return query

    spec = search._tool_spec

    assert spec.name == "search"
    assert spec.description == "Search the web for a query."
    assert spec.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }


def test_decorator_with_overrides():
    @tool(name="web_search", description="Custom description")
    def search(query: str) -> str:
        """Ignored docstring."""
        return query

    spec = search._tool_spec

    assert spec.name == "web_search"
    assert spec.description == "Custom description"


def test_toolbox_register_and_schemas():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    toolbox = Toolbox()
    toolbox.register(add)

    assert toolbox.schemas() == [
        {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        }
    ]


def test_toolbox_register_without_decorator_still_works():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    toolbox = Toolbox()
    toolbox.register(multiply)

    assert toolbox.get("multiply").description == "Multiply two numbers."


async def test_toolbox_calls_sync_tool():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    toolbox = Toolbox()
    toolbox.register(add)

    result = await toolbox.call("add", a=1, b=2)

    assert result == 3


async def test_toolbox_calls_async_tool():
    @tool
    async def fetch(url: str) -> str:
        return f"content of {url}"

    toolbox = Toolbox()
    toolbox.register(fetch)

    result = await toolbox.call("fetch", url="example.com")

    assert result == "content of example.com"


def test_toolbox_get_unknown_tool_raises():
    toolbox = Toolbox()

    with pytest.raises(KeyError):
        toolbox.get("missing")


async def test_toolbox_call_unknown_tool_raises():
    toolbox = Toolbox()

    with pytest.raises(KeyError):
        await toolbox.call("missing")


def test_toolbox_registers_an_iterable_passed_to_the_constructor():
    @tool
    def first() -> str:
        """First."""
        return "1"

    def second() -> str:
        """Second, undecorated."""
        return "2"

    toolbox = Toolbox([first, second])

    assert len(toolbox) == 2
    assert {schema["name"] for schema in toolbox.schemas()} == {"first", "second"}


def test_empty_toolbox_is_falsy():
    assert not Toolbox()
    assert len(Toolbox()) == 0


def test_container_parameters_declare_their_item_type():
    @tool
    def fn(names: list[str], counts: list[int], mapping: dict[str, int]) -> str:
        """Doc."""
        return ""

    props = fn._tool_spec.parameters["properties"]

    assert props["names"] == {"type": "array", "items": {"type": "string"}}
    assert props["counts"] == {"type": "array", "items": {"type": "integer"}}
    assert props["mapping"] == {"type": "object"}


def test_literal_and_enum_parameters_become_enums():
    class Unit(enum.Enum):
        C = "c"
        F = "f"

    @tool
    def fn(units: Literal["c", "f"], unit: Unit) -> str:
        """Doc."""
        return ""

    props = fn._tool_spec.parameters["properties"]

    assert props["units"] == {"enum": ["c", "f"]}
    assert props["unit"] == {"enum": ["c", "f"]}


def test_optional_parameter_keeps_its_type_but_is_not_required():
    @tool
    def fn(city: str, limit: int | None = None) -> str:
        """Doc."""
        return ""

    spec = fn._tool_spec

    assert spec.parameters["properties"]["limit"] == {"type": "integer"}
    assert spec.parameters["required"] == ["city"]


def test_union_parameter_becomes_any_of():
    @tool
    def fn(value: int | str) -> str:
        """Doc."""
        return ""

    assert fn._tool_spec.parameters["properties"]["value"] == {
        "anyOf": [{"type": "integer"}, {"type": "string"}]
    }


def test_undescribable_parameters_stay_unconstrained():
    class Whatever:
        pass

    @tool
    def fn(a, b: Any, c: Whatever) -> str:
        """Doc."""
        return ""

    props = fn._tool_spec.parameters["properties"]

    assert props["a"] == {} and props["b"] == {} and props["c"] == {}


def test_register_refuses_to_shadow_a_different_tool_of_the_same_name():
    """The model picks a tool by name, so a collision makes one unreachable."""

    @tool(name="search")
    def search_web(query: str) -> str:
        return "web"

    @tool(name="search")
    def search_docs(query: str) -> str:
        return "docs"

    toolbox = Toolbox([search_web])

    with pytest.raises(ConfigurationError, match="already registered"):
        toolbox.register(search_docs)

    assert toolbox.get("search").func is search_web


def test_register_accepts_the_same_tool_twice():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    toolbox = Toolbox([add])
    toolbox.register(add)

    assert len(toolbox) == 1
