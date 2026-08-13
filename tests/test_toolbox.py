import pytest

from subagents.toolbox import Toolbox, tool


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
