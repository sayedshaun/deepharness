"""Helpers every vendor's response parsing needs.

Each vendor's own wire types live beside its client - openai.py, anthropic.py,
gemini.py - so that everything one vendor knows about its own format is in one
place. Only what all three share is here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..errors import ProviderError


def require(data: dict[str, Any], key: str, where: str) -> Any:
    """A field the response must have, or a ProviderError naming it.

    The point of parsing responses at all: a vendor renaming or dropping a field
    should fail on the first response rather than quietly yielding an empty
    completion.
    """
    try:
        return data[key]
    except (KeyError, TypeError):
        raise ProviderError(
            f"{where} response is missing '{key}': {clip(data)}"
        ) from None


def clip(data: Any, limit: int = 200) -> str:
    """Response bodies can be long, and may carry keys we should not log."""
    text = str(data)
    return text if len(text) <= limit else f"{text[:limit]}..."


@dataclass(slots=True)
class Usage:
    """Token counts, named per vendor at the edges and normalized here."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def usage_from(
    usage: Any, *, prompt: str, completion: str, total: str | None = None
) -> Usage | None:
    """One vendor's token counts under its own key names, or None if it sent none.

    total is optional because Anthropic reports only the two halves; summing them
    here keeps that quirk out of the response types.
    """
    if not usage:
        return None
    prompt_tokens = usage.get(prompt, 0)
    completion_tokens = usage.get(completion, 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.get(total, 0)
        if total
        else prompt_tokens + completion_tokens,
    )


def load_arguments(raw: str) -> dict[str, Any]:
    """Argument JSON assembled from stream fragments, empty if unusable.

    A truncated stream can leave this unparseable; an empty dict lets the tool
    report the real problem (a missing argument) rather than the stream dying.
    """
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
