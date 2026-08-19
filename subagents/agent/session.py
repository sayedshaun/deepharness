from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .message import Message, as_dict


def save_session(path: str, messages: list[Message | dict[str, Any]]) -> None:
    """Write a message history to a JSON file, so a session can be resumed later."""
    Path(path).write_text(json.dumps([as_dict(m) for m in messages], indent=2))


def load_session(path: str) -> list[dict[str, Any]]:
    """Read a message history written by save_session().

    Returns [] when the file does not exist yet, so a first run needs no
    special case at the call site.
    """
    file = Path(path)
    if not file.exists():
        return []
    return json.loads(file.read_text())
