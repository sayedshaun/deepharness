from __future__ import annotations

from typing import Any


class Message(dict):
    """A chat message, built via role-named constructors instead of a raw dict.

    Subclasses dict, so it's a drop-in replacement everywhere a plain
    {"role": ..., "content": ...} is expected today (Agent, Toolbox, every
    provider) - no other code needs to change to support it.
    """

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def human(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def ai(cls, content: str, *, tool_calls: list[dict[str, Any]] | None = None) -> Message:
        message = cls(role="assistant", content=content)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @classmethod
    def tool(cls, content: str, *, name: str, call_id: str | None = None) -> Message:
        message = cls(role="tool", name=name, content=content)
        if call_id is not None:
            message["tool_call_id"] = call_id
        return message
