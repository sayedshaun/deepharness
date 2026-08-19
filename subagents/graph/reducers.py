"""Reducers for merging concurrent writes to one state field.

Declare one on the field itself, so the merge policy lives with the data:

    @dataclass
    class State:
        findings: list[str] = field(
            default_factory=list, metadata={"reducer": concat}
        )

A reducer takes the field's value at the start of the wave plus each branch's
value for it, and returns the merged value. Fields with no reducer may only be
written by one branch per wave - the executor raises rather than pick a winner.
"""

from __future__ import annotations

from typing import Any


def concat(base: list[Any], values: list[list[Any]]) -> list[Any]:
    """Append-only merge for sequences: keep base, then each branch's additions.

    Nodes here mutate and return the whole state, so a branch's value is the
    full list (base included) rather than just what it added - hence the
    suffix-vs-base comparison. A branch that replaced the list instead of
    appending to it contributes the whole thing.
    """
    merged = list(base)
    for value in values:
        if list(value[: len(base)]) == list(base):
            merged.extend(value[len(base) :])
        else:
            merged.extend(value)
    return merged


def merge_dicts(base: dict[Any, Any], values: list[dict[Any, Any]]) -> dict[Any, Any]:
    """Shallow dict merge; later branches win on a key two branches both set."""
    merged = dict(base)
    for value in values:
        merged.update(value)
    return merged
