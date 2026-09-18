"""An MCP server's tools, reached over each transport and used by an Agent."""

import json
import sys
import textwrap

import httpx
import pytest

from deepharness.agent import Agent, Toolbox
from deepharness.errors import MCPError
from deepharness.providers.base import CompletionResponse, ToolCall
from deepharness.tools import MCPServer, MCPTool, Transport

from .test_agent import ScriptedProvider

TOOLS = [
    {
        "name": "echo",
        "description": "Echo a message back.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "wipe",
        "description": "Delete everything.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class FakeTransport(Transport):
    """Answers the three methods a client uses, and records what it was asked."""

    def __init__(self, *, result="echoed", error=None, is_error=False):
        self.calls: list[tuple[str, dict]] = []
        self.notifications: list[str] = []
        self.closed = False
        self._result = result
        self._error = error
        self._is_error = is_error

    async def request(self, method, params):
        self.calls.append((method, params))
        if self._error is not None and method == "tools/call":
            raise MCPError(self._error)
        match method:
            case "initialize":
                return {"protocolVersion": "2025-06-18", "capabilities": {}}
            case "tools/list":
                return {"tools": TOOLS}
            case "tools/call":
                return {
                    "content": [{"type": "text", "text": self._result}],
                    "isError": self._is_error,
                }
        raise AssertionError(f"unexpected method {method}")

    async def notify(self, method, params):
        self.notifications.append(method)

    async def aclose(self):
        self.closed = True


async def test_connecting_handshakes_once():
    transport = FakeTransport()
    server = MCPServer(transport)

    await server.connect()
    await server.connect()

    assert [method for method, _ in transport.calls] == ["initialize"]
    assert transport.notifications == ["notifications/initialized"]


async def test_listing_tools_returns_typed_objects():
    tools = await MCPServer(FakeTransport()).list_tools()

    assert tools[0] == MCPTool(
        name="echo",
        description="Echo a message back.",
        input_schema=TOOLS[0]["inputSchema"],
        read_only=True,
    )


async def test_a_tool_keeps_the_server_s_own_schema():
    tools = await MCPServer(FakeTransport()).tools()
    box = Toolbox(tools)

    assert box.schemas()[0] == {
        "name": "echo",
        "description": "Echo a message back.",
        "parameters": TOOLS[0]["inputSchema"],
    }


async def test_only_a_tool_the_server_called_read_only_runs_unasked():
    tools = await MCPServer(FakeTransport()).tools()

    assert not tools[0]._tool_spec.requires_approval
    assert tools[1]._tool_spec.requires_approval


async def test_gating_can_be_turned_off_wholesale():
    tools = await MCPServer(FakeTransport(), requires_approval=False).tools()

    assert not any(tool._tool_spec.requires_approval for tool in tools)


async def test_calling_a_tool_sends_its_arguments():
    transport = FakeTransport(result="hi there")
    server = MCPServer(transport)

    assert await server.call("echo", {"message": "hi"}) == "hi there"
    assert transport.calls[-1] == (
        "tools/call",
        {"name": "echo", "arguments": {"message": "hi"}},
    )


async def test_a_tool_that_reports_failure_raises():
    server = MCPServer(FakeTransport(result="disk on fire", is_error=True))

    with pytest.raises(MCPError, match="disk on fire"):
        await server.call("wipe", {})


async def test_a_silent_tool_says_so_rather_than_returning_nothing():
    server = MCPServer(FakeTransport(result=""))

    assert await server.call("echo", {}) == "(no output)"


async def test_an_agent_runs_an_mcp_tool():
    server = MCPServer(FakeTransport(result="pong"), requires_approval=False)
    provider = ScriptedProvider(
        [
            CompletionResponse(
                content="",
                tool_calls=[
                    ToolCall(id="1", name="echo", arguments={"message": "ping"})
                ],
            ),
            CompletionResponse(content="it said pong"),
        ]
    )
    agent = Agent(provider, tools=await server.tools())

    state = await agent.arun("ping the server")

    assert state.answered
    assert any("pong" in m["content"] for m in state.messages if m["role"] == "tool")


async def test_closing_the_server_closes_its_transport():
    transport = FakeTransport()

    async with MCPServer(transport):
        pass

    assert transport.closed


# --- HTTP transport -----------------------------------------------------------


def _http_server(*, sse=False, session=None):
    """A handler answering JSON-RPC over JSON or SSE, whichever is asked for."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        seen.append({**message, "session": request.headers.get("Mcp-Session-Id")})
        if "id" not in message:
            return httpx.Response(202)
        result = {
            "initialize": {"protocolVersion": "2025-06-18"},
            "tools/list": {"tools": TOOLS},
            "tools/call": {"content": [{"type": "text", "text": "over http"}]},
        }[message["method"]]
        body = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        headers = {"Mcp-Session-Id": session} if session else {}
        if sse:
            return httpx.Response(
                200,
                headers={**headers, "content-type": "text/event-stream"},
                text=f"event: message\ndata: {json.dumps(body)}\n\n",
            )
        return httpx.Response(200, json=body, headers=headers)

    return seen, httpx.MockTransport(handler)


async def test_the_http_transport_reads_a_json_reply():
    _, transport = _http_server()
    client = httpx.AsyncClient(transport=transport)

    async with MCPServer.http("http://mcp.test/rpc", client=client) as server:
        assert await server.call("echo", {"message": "hi"}) == "over http"


async def test_the_http_transport_reads_an_sse_reply():
    _, transport = _http_server(sse=True)
    client = httpx.AsyncClient(transport=transport)

    async with MCPServer.http("http://mcp.test/rpc", client=client) as server:
        assert (await server.list_tools())[0].name == "echo"


async def test_a_session_id_is_carried_on_later_requests():
    seen, transport = _http_server(session="abc123")
    client = httpx.AsyncClient(transport=transport)

    async with MCPServer.http("http://mcp.test/rpc", client=client) as server:
        await server.list_tools()

    assert seen[0]["session"] is None
    assert seen[-1]["session"] == "abc123"


async def test_an_http_error_status_is_reported():
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    client = httpx.AsyncClient(transport=transport)
    server = MCPServer.http("http://mcp.test/rpc", client=client)

    with pytest.raises(MCPError, match="answered 500"):
        await server.connect()


async def test_a_json_rpc_error_is_reported_with_its_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "no such method"},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    server = MCPServer.http("http://mcp.test/rpc", client=client)

    with pytest.raises(MCPError, match="no such method"):
        await server.connect()


# --- stdio transport ----------------------------------------------------------

_SERVER_SOURCE = textwrap.dedent(
    """
    import json, sys

    TOOLS = json.loads(%r)

    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        else:
            args = message["params"]["arguments"]
            result = {"content": [{"type": "text", "text": "echo: " + args["message"]}]}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\\n")
        sys.stdout.flush()
    """
) % json.dumps(TOOLS)


async def test_a_stdio_server_runs_as_a_subprocess(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(_SERVER_SOURCE)

    async with MCPServer.stdio([sys.executable, str(script)]) as server:
        tools = await server.list_tools()
        result = await server.call("echo", {"message": "hi"})

    assert [tool.name for tool in tools] == ["echo", "wipe"]
    assert result == "echo: hi"


async def test_a_stdio_server_that_dies_is_reported(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("raise SystemExit(1)")
    server = MCPServer.stdio([sys.executable, str(script)])

    with pytest.raises(MCPError, match="closed the connection"):
        await server.connect()

    await server.aclose()


def test_a_stdio_server_needs_a_command():
    with pytest.raises(MCPError):
        MCPServer.stdio([])
