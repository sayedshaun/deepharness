"""A chat UI for driving a DeepHarness agent against a real model.

The page is static; this is what it talks to. A browser cannot import
deepharness, read a file or run a shell, and those are exactly the parts worth
exercising - the agent loop, the workspace tools, the permission policy, the
progress events, and a run that pauses until a human rules on it.

FastAPI is an example dependency, not a runtime one: the library still needs
only httpx. Install what this needs with `pip install -e ".[examples]"`.

    python examples/chat/app.py                      # a fresh temp workspace
    python examples/chat/app.py --workspace ./scratch

Single session, no authentication: a development toy. It gives the model a
shell inside its workspace, gated by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deepharness import Agent, AgentState, Finished, LlamaCpp, Message, TextDelta
from deepharness.agent import ContextPolicy, StepStarted, ToolFinished, ToolStarted
from deepharness.errors import DeepHarnessError
from deepharness.providers import ThinkingDelta
from deepharness.tools import (
    Permissions,
    Rule,
    ToolName,
    Workspace,
    file_tools,
    shell_tool,
)

SYSTEM = """You are a coding agent working inside one directory.

Use the tools rather than guessing: list_files and read_file before you change
anything, write_file to create a file, edit_file to change part of one,
run_command for anything else. Keep prose to a sentence or two - the tool calls
already show your work.
"""

PAGE = Path(__file__).with_name("index.html")
ENV_PREFIX = "DEEPHARNESS_CHAT_"


class Settings:
    """Read from the environment so `uvicorn app:app` works as well as __main__."""

    def __init__(self) -> None:
        env = os.environ.get
        self.base_url = env(
            f"{ENV_PREFIX}BASE_URL",
            "https://unraveled-cupbearer-outback.ngrok-free.dev/llamacpp/v1",
        )
        self.model = env(f"{ENV_PREFIX}MODEL", "unsloth/gemma-4-E4B-it-GGUF")
        self.context = int(env(f"{ENV_PREFIX}CONTEXT", "12000"))
        # A temp directory by default: an agent with a shell should not be
        # pointed at a repository unless someone says so out loud.
        self.workspace = env(f"{ENV_PREFIX}WORKSPACE") or tempfile.mkdtemp(
            prefix="deepharness-chat-"
        )


class Session:
    """One conversation, and the agent having it.

    A paused state stays here rather than going to the browser: an approval is a
    decision about what this process will run, and a client that could hand back
    an edited state would be making that decision instead of ruling on it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.workspace = Workspace(Path(settings.workspace).expanduser())
        self.model = LlamaCpp(settings.model, base_url=settings.base_url.rstrip("/"))
        self.agent = Agent(
            self.model,
            tools=[*file_tools(self.workspace), shell_tool(self.workspace)],
            system=SYSTEM,
            context=ContextPolicy(max_tokens=settings.context, tool_result_chars=4_000),
            permissions=Permissions(
                allow=[ToolName.READ_FILE, ToolName.LIST_FILES, ToolName.SEARCH_FILES],
                ask=[ToolName.WRITE_FILE, ToolName.EDIT_FILE, ToolName.RUN_COMMAND],
                deny=[Rule(ToolName.RUN_COMMAND, {"command": "*rm -rf*"})],
            ),
        )
        self._state: AgentState | None = None
        # One turn at a time: the agent and its transcript are shared state, and
        # two overlapping runs would interleave messages in one history.
        self._lock = threading.Lock()

    @property
    def pending(self) -> list[dict[str, Any]]:
        """The calls waiting on a human, as the page needs to show them."""
        if self._state is None:
            return []
        return [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments or {},
                "question": call.question,
            }
            for call in self._state.paused
        ]

    def send(self, message: str) -> Iterator[dict[str, Any]]:
        with self._lock:
            yield from self._drive(self._with_prompt(message))

    def rule(self, *, approve: bool, call_id: str | None) -> Iterator[dict[str, Any]]:
        with self._lock:
            if self._state is None or not self._state.paused:
                yield {"type": "error", "message": "nothing is waiting for a ruling"}
                return
            state = (
                self._state.approve(call_id) if approve else self._state.reject(call_id)
            )
            yield from self._drive(state)

    def reset(self) -> None:
        with self._lock:
            self._state = None

    def files(self) -> list[dict[str, Any]]:
        """What the workspace holds now, so the page can show what changed."""
        root = self.workspace.root
        return [
            {"path": self.workspace.relative(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        ]

    def read(self, relative: str) -> str:
        return self.workspace.resolve(relative).read_text(
            encoding="utf-8", errors="replace"
        )

    def _with_prompt(self, message: str) -> AgentState:
        """The conversation so far, plus what was just typed.

        Typing instead of ruling on an open gate is a refusal: the calls are
        rejected rather than dropped, so the model is told, and the transcript
        keeps a result for every call it asked for - which a vendor requires on
        the next request.
        """
        if self._state is None:
            return AgentState.of(message)
        for call in self._state.paused:
            # Recorded here rather than by resuming the run: resuming would
            # spend a model call on a turn nobody asked for, and its events
            # would have nowhere to go.
            self._state.messages.append(
                Message.tool(
                    "Denied by the user.", name=call.name, call_id=call.call_id
                ).to_dict()
            )
        self._state.paused.clear()
        self._state.messages.append(Message.human(message).to_dict())
        return self._state

    def _drive(self, state: AgentState) -> Iterator[dict[str, Any]]:
        """One run, as the events the page renders.

        Errors are reported as an event rather than raised: the response has
        already started streaming, so a raise would leave the page with a dead
        connection and no explanation.
        """
        try:
            for event in self.agent.stream_events(state):
                match event:
                    case StepStarted(step):
                        yield {"type": "step", "step": step}
                    case ToolStarted(name, arguments, _):
                        yield {
                            "type": "tool_start",
                            "name": name,
                            "arguments": arguments,
                        }
                    case ToolFinished(name, result, failed, _):
                        yield {
                            "type": "tool_end",
                            "name": name,
                            "result": result,
                            "failed": failed,
                        }
                    case ThinkingDelta(text):
                        yield {"type": "thinking", "text": text}
                    case TextDelta(text):
                        yield {"type": "text", "text": text}
                    case Finished(finished):
                        self._state = finished
                        yield self._summary(finished)
        except DeepHarnessError as exc:
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        except Exception:  # noqa: BLE001 - a toy server should show its stack
            yield {"type": "error", "message": traceback.format_exc(limit=3)}

    def _summary(self, state: AgentState) -> dict[str, Any]:
        return {
            "type": "done",
            "stop_reason": state.stop_reason,
            "answered": state.answered,
            "usage": {
                "prompt": state.usage.prompt_tokens,
                "completion": state.usage.completion_tokens,
                "total": state.usage.total_tokens,
            },
            "paused": self.pending,
            "files": self.files(),
        }


class Prompt(BaseModel):
    message: str


class Ruling(BaseModel):
    call_id: str | None = None


def sse(events: Iterator[dict[str, Any]]) -> Iterator[bytes]:
    """Server-sent events, one JSON object per event."""
    for event in events:
        yield f"data: {json.dumps(event)}\n\n".encode()


def create_app(session: Session | None = None) -> FastAPI:
    app = FastAPI(title="DeepHarness chat", docs_url=None, redoc_url=None)
    state = session or Session(Settings())

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/session")
    def describe() -> dict[str, Any]:
        """What the page shows in its header, and how it starts up mid-run."""
        return {
            "model": state.settings.model,
            "base_url": state.settings.base_url,
            "workspace": str(state.workspace.root),
            "tools": list(state.agent.tools.names()),
            "paused": state.pending,
            "files": state.files(),
        }

    @app.post("/chat")
    def chat(prompt: Prompt) -> StreamingResponse:
        message = prompt.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="empty message")
        return StreamingResponse(
            sse(state.send(message)), media_type="text/event-stream"
        )

    @app.post("/approve")
    def approve(ruling: Ruling) -> StreamingResponse:
        return StreamingResponse(
            sse(state.rule(approve=True, call_id=ruling.call_id)),
            media_type="text/event-stream",
        )

    @app.post("/reject")
    def reject(ruling: Ruling) -> StreamingResponse:
        return StreamingResponse(
            sse(state.rule(approve=False, call_id=ruling.call_id)),
            media_type="text/event-stream",
        )

    @app.post("/reset")
    def reset() -> dict[str, Any]:
        state.reset()
        return {"ok": True, "files": state.files()}

    @app.get("/files")
    def files() -> dict[str, Any]:
        return {"files": state.files()}

    @app.get("/file")
    def file(path: str) -> dict[str, Any]:
        try:
            return {"path": path, "content": state.read(path)}
        except (OSError, DeepHarnessError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", help="OpenAI-compatible llama.cpp endpoint")
    parser.add_argument("--model")
    parser.add_argument(
        "--workspace",
        help="the only directory the agent may touch; a temp dir by default",
    )
    parser.add_argument("--context", type=int)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    for name in ("base_url", "model", "workspace", "context"):
        if (value := getattr(args, name)) is not None:
            os.environ[f"{ENV_PREFIX}{name.upper()}"] = str(value)

    settings = Settings()
    Path(settings.workspace).expanduser().mkdir(parents=True, exist_ok=True)
    session = Session(settings)

    print(f"workspace : {session.workspace.root}")
    print(f"model     : {settings.model}")
    print(f"endpoint  : {settings.base_url}")
    print(f"open      : http://127.0.0.1:{args.port}\n")
    uvicorn.run(
        create_app(session), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
