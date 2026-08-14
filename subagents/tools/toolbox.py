from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolSpec:
    """Provider-agnostic description of a callable exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Mark a function as usable as an LLM tool.

    Can be used bare (`@tool`) or with overrides (`@tool(name=..., description=...)`).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._tool_spec = _build_spec(fn, name, description)  # type: ignore[attr-defined]
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def _build_spec(
    fn: Callable[..., Any],
    name: str | None,
    description: str | None,
) -> ToolSpec:
    signature = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        properties[param_name] = {"type": _json_type(hints.get(param_name))}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return ToolSpec(
        name=name or fn.__name__,
        description=description or inspect.getdoc(fn) or "",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        func=fn,
    )


def _json_type(annotation: Any) -> str:
    return _JSON_TYPES.get(annotation, "string")


class Toolbox:
    """Registry of tools that can be listed as schemas and invoked by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        spec = getattr(func, "_tool_spec", None) or _build_spec(func, None, None)
        self._tools[spec.name] = spec
        return func

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_schema() for spec in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> Any:
        result = self.get(name).func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def call_sync(self, name: str, **kwargs: Any) -> Any:
        result = self.get(name).func(**kwargs)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if close is not None:
                close()
            raise RuntimeError(
                f"Tool '{name}' is async; use Agent.arun() instead of Agent.run()"
            )
        return result
