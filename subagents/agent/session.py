from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_session(path: str, messages: list[dict[str, Any]]) -> None:
    """Write a message history to a JSON file, so a session can be resumed later."""
    Path(path).write_text(json.dumps(messages, indent=2))


def load_session(path: str) -> list[dict[str, Any]]:
    """Read a message history written by save_session(), or [] if the file doesn't exist yet."""
    file = Path(path)
    if not file.exists():
        return []
    return json.loads(file.read_text())
