"""Tools that live in another process: a client for MCP servers.

The Model Context Protocol is JSON-RPC 2.0 over one of two transports, and both
are reachable with the standard library and the httpx client this project
already has - so an MCP server's tools become ordinary entries in a Toolbox
without a protocol SDK coming along with them.

Transport is an abstraction with two implementations rather than an interface
added on speculation: a local server is a subprocess speaking newline-delimited
JSON, a remote one is an HTTP endpoint answering with JSON or SSE, and nothing
above this module should have to know which it is talking to.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from ..errors import MCPError
from ..http import DEFAULT_TIMEOUT
from .toolbox import ToolSpec

_PROTOCOL_VERSION = "2025-06-18"
_CLIENT_INFO = {"name": "deepharness", "version": "0"}
_JSON_RPC = "2.0"


class Transport(ABC):
    """How one MCP server is reached."""

    __slots__ = ()

    @abstractmethod
    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a method and return its result, or raise MCPError."""

    @abstractmethod
    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a message the protocol expects no reply to."""

    @abstractmethod
    async def aclose(self) -> None:
        """Shut the transport down; safe to call more than once."""


@dataclass(slots=True)
class MCPTool:
    """One tool a server advertises, as data rather than a raw payload."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = False
    """Whether the server hinted the tool only reads. A hint, not a guarantee -
    it is used to decide what to ask a human about, never to skip a check."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MCPTool:
        annotations = data.get("annotations") or {}
        return cls(
            name=data.get("name", ""),
            description=data.get("description", "") or "",
            input_schema=data.get("inputSchema")
            or {"type": "object", "properties": {}},
            read_only=bool(annotations.get("readOnlyHint")),
        )


class MCPServer:
    """An MCP server, and its tools as callables an Agent can hold.

    The tools are async, so register them with an agent driven by arun() - the
    same rule any async tool follows here.

    A tool the server did not mark read-only is gated by default: it is code in
    another process, and asking is the right default for something this side
    cannot inspect. Widen that with Permissions rather than by turning it off,
    so what was allowed stays written down.
    """

    __slots__ = ("_initialized", "_requires_approval", "_transport")

    def __init__(self, transport: Transport, *, requires_approval: bool = True):
        self._transport = transport
        self._requires_approval = requires_approval
        self._initialized = False

    @classmethod
    def stdio(
        cls,
        command: str | Iterable[str],
        *,
        env: Mapping[str, str] | None = None,
        requires_approval: bool = True,
    ) -> MCPServer:
        """A server run as a local subprocess."""
        argv = [command] if isinstance(command, str) else list(command)
        return cls(StdioTransport(argv, env=env), requires_approval=requires_approval)

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        requires_approval: bool = True,
    ) -> MCPServer:
        """A server reached over streamable HTTP."""
        return cls(
            HTTPTransport(url, headers=headers, client=client),
            requires_approval=requires_approval,
        )

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def connect(self) -> None:
        """Perform the handshake, once, whoever asks first."""
        if self._initialized:
            return
        await self._transport.request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        )
        await self._transport.notify("notifications/initialized", {})
        self._initialized = True

    async def list_tools(self) -> list[MCPTool]:
        """What the server offers, as typed objects."""
        await self.connect()
        result = await self._transport.request("tools/list", {})
        return [MCPTool.from_json(tool) for tool in result.get("tools") or ()]

    async def tools(self) -> list[Callable[..., Any]]:
        """The server's tools, ready to hand to an Agent."""
        return [self.as_tool(tool) for tool in await self.list_tools()]

    def as_tool(self, tool: MCPTool) -> Callable[..., Any]:
        """One advertised tool as a callable carrying the server's own schema.

        The schema is passed through as the server published it: it is already a
        JSON Schema, and re-deriving one from a Python signature that does not
        exist would only lose fields.
        """

        async def call(**arguments: Any) -> str:
            return await self.call(tool.name, arguments)

        call.__name__ = tool.name
        call._tool_spec = ToolSpec(  # type: ignore[attr-defined]
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
            func=call,
            requires_approval=self._requires_approval and not tool.read_only,
        )
        return call

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool on the server and return its text result."""
        await self.connect()
        result = await self._transport.request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        text = "\n".join(
            block.get("text", "")
            for block in result.get("content") or ()
            if block.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPError(f"{name} failed: {text or 'no detail given'}")
        return text or "(no output)"

    async def aclose(self) -> None:
        await self._transport.aclose()


class StdioTransport(Transport):
    """A server as a subprocess exchanging newline-delimited JSON.

    Requests are serialized behind a lock: the protocol allows them to overlap,
    but a client that reads the next line as its own answer does not, and one
    lock is cheaper than correlating ids for a tool client that spends its time
    waiting on the model anyway.
    """

    __slots__ = ("_argv", "_env", "_lock", "_next_id", "_process")

    def __init__(self, argv: list[str], *, env: Mapping[str, str] | None = None):
        if not argv:
            raise MCPError("an MCP stdio server needs a command to run")
        self._argv = argv
        self._env = dict(env) if env else None
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0

    async def _started(self) -> asyncio.subprocess.Process:
        if self._process is None:
            self._process = await asyncio.create_subprocess_exec(
                *self._argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=self._env,
            )
        return self._process

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            process = await self._started()
            self._next_id += 1
            await self._write(
                process,
                {
                    "jsonrpc": _JSON_RPC,
                    "id": self._next_id,
                    "method": method,
                    "params": params,
                },
            )
            assert process.stdout is not None
            line = await process.stdout.readline()
            if not line:
                raise MCPError(f"MCP server closed the connection during {method}")
            return _result(json.loads(line), method)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        async with self._lock:
            process = await self._started()
            await self._write(
                process, {"jsonrpc": _JSON_RPC, "method": method, "params": params}
            )

    @staticmethod
    async def _write(
        process: asyncio.subprocess.Process, message: dict[str, Any]
    ) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(message).encode() + b"\n")
        await process.stdin.drain()

    async def aclose(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        await process.wait()


class HTTPTransport(Transport):
    """A server reached by POSTing JSON-RPC, answering with JSON or SSE.

    Both answer shapes are handled because both are allowed, and a server picks
    per response: a single result comes back as JSON, while a server that may
    stream notifications alongside it answers with an event stream.
    """

    __slots__ = ("_client", "_headers", "_next_id", "_owned", "_session", "_url")

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(dict(headers) if headers else {}),
        }
        self._owned = client is None
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._session: str | None = None
        self._next_id = 0

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        response = await self._post(
            {
                "jsonrpc": _JSON_RPC,
                "id": self._next_id,
                "method": method,
                "params": params,
            }
        )
        return _result(_payload(response, method), method)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._post({"jsonrpc": _JSON_RPC, "method": method, "params": params})

    async def _post(self, message: dict[str, Any]) -> httpx.Response:
        headers = dict(self._headers)
        if self._session is not None:
            headers["Mcp-Session-Id"] = self._session
        try:
            response = await self._client.post(self._url, json=message, headers=headers)
        except httpx.HTTPError as exc:
            raise MCPError(f"MCP request to {self._url} failed: {exc!r}") from exc
        if response.status_code >= 400:
            raise MCPError(
                f"MCP server answered {response.status_code} for "
                f"{message.get('method')}"
            )
        # Kept for the rest of the session: a server that issues one rejects
        # every later request that arrives without it.
        self._session = response.headers.get("Mcp-Session-Id", self._session)
        return response

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()


def _payload(response: httpx.Response, method: str) -> dict[str, Any]:
    """One JSON-RPC message out of a JSON body or an event stream."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        return response.json()
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data:
            continue
        message = json.loads(data)
        if "result" in message or "error" in message:
            return message
    raise MCPError(f"MCP server sent no result for {method}")


def _result(message: dict[str, Any], method: str) -> dict[str, Any]:
    """The result half of a JSON-RPC reply, or the error it carried instead."""
    if error := message.get("error"):
        raise MCPError(
            f"MCP server refused {method}: "
            f"{error.get('message', error)} (code {error.get('code', '?')})"
        )
    result = message.get("result")
    if not isinstance(result, dict):
        raise MCPError(f"MCP server returned no result for {method}")
    return result
