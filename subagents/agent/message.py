from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Message:
    """A chat message, built via role-named constructors instead of a raw dict.

    A dataclass rather than a dict subclass: the wire format is a mapping, but a
    message is not a general-purpose mapping - nothing should be able to pop a
    role off one or update() it with arbitrary keys. to_dict() produces the wire
    form at the boundary, and Agent normalizes whatever it is handed, so a plain
    dict from a caller still works.
    """

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def human(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def ai(
        cls, content: str, *, tool_calls: list[dict[str, Any]] | None = None
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or None)

    @classmethod
    def tool(cls, content: str, *, name: str, call_id: str | None = None) -> Message:
        return cls(role="tool", content=content, name=name, tool_call_id=call_id)

    def to_dict(self) -> dict[str, Any]:
        """The wire form, without the fields this role does not use."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


def as_dict(message: Message | dict[str, Any]) -> dict[str, Any]:
    """One transcript entry as a plain dict, whoever built it."""
    return message.to_dict() if isinstance(message, Message) else dict(message)
