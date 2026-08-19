from __future__ import annotations

import asyncio
import enum
import inspect
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ..errors import ConfigurationError, ToolNotFoundError

_NONE = type(None)
_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
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
    return ToolSpec(
        name=name or fn.__name__,
        description=description or inspect.getdoc(fn) or "",
        parameters=_parameters(fn),
        func=fn,
    )


def _parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """The JSON Schema object describing what the model may pass to fn."""
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = json_type(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


def json_type(annotation: Any) -> dict[str, Any]:
    """One annotation as JSON Schema. An empty dict means "unconstrained".

    An annotation this cannot describe stays unconstrained on purpose: saying
    nothing leaves the model to infer from the name and docstring, while
    defaulting it to "string" would tell the model something false.
    """
    origin = get_origin(annotation)

    if origin is Annotated:
        return json_type(get_args(annotation)[0])
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin is Union or origin is types.UnionType:
        variants = [a for a in get_args(annotation) if a is not _NONE]
        if len(variants) == 1:  # Optional[X] is just X, minus the requirement
            return json_type(variants[0])
        return {"anyOf": [json_type(a) for a in variants]}
    if origin in (list, set, frozenset, tuple):
        args = get_args(annotation)
        return {"type": "array", "items": json_type(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type):
        if annotation in _PRIMITIVES:
            return {"type": _PRIMITIVES[annotation]}
        if issubclass(annotation, enum.Enum):
            return {"enum": [member.value for member in annotation]}
        if annotation is dict:
            return {"type": "object"}
    return {}


class Toolbox:
    """Registry of tools that can be listed as schemas and invoked by name."""

    def __init__(self, tools: Iterable[Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for func in tools:
            self.register(func)

    def __len__(self) -> int:
        return len(self._tools)

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        spec = getattr(func, "_tool_spec", None) or _build_spec(func, None, None)
        self._tools[spec.name] = spec
        return func

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_schema() for spec in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a tool, running a sync one off the event loop.

        The concurrency arun() promises comes from gathering a turn's tool calls,
        and a blocking call inside one of those coroutines defeats it - so a
        plain def tool goes to a thread rather than stalling every other tool
        waiting alongside it.
        """
        func = self.get(name).func
        if inspect.iscoroutinefunction(inspect.unwrap(func)):
            return await func(**kwargs)
        result = await asyncio.to_thread(func, **kwargs)
        if inspect.isawaitable(result):  # a sync def that returns a coroutine
            return await result
        return result

    def call_sync(self, name: str, **kwargs: Any) -> Any:
        result = self.get(name).func(**kwargs)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if close is not None:
                close()
            raise ConfigurationError(
                f"Tool '{name}' is async; use Agent.arun() instead of Agent.run()"
            )
        return result
