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


@dataclass(slots=True)
class Ctx:
    """What a tool can see of the run it is part of.

    A tool asks for one by annotating a parameter `ctx: Ctx`; the runtime fills
    it in and hides the parameter from the model, which is how a tool reaches
    run state or injected dependencies without a module-level global.

    state is typed Any rather than AgentState so that tools/ stays independent
    of agent/ - the agent passes its AgentState, another caller may pass
    whatever it runs tools with.
    """

    state: Any = None
    deps: Any = None


@dataclass(slots=True)
class ToolSpec:
    """Provider-agnostic description of a callable exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    ctx_params: tuple[str, ...] = ()
    """Parameters the runtime fills with a Ctx, hidden from the model."""

    requires_approval: bool = False
    """Whether a human must allow each call before it runs."""

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
    requires_approval: bool = False,
) -> Callable[..., Any]:
    """Mark a function as usable as an LLM tool.

    Can be used bare (`@tool`) or with overrides (`@tool(name=..., description=...)`).

    requires_approval=True gates every call on a human: the agent pauses before
    running it and only runs it once approved. The gate lives on the tool rather
    than in the prompt so a model cannot skip it by not asking.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._tool_spec = _build_spec(fn, name, description, requires_approval)  # type: ignore[attr-defined]
        return fn

    if func is not None:
        return decorator(func)
    return decorator


def _build_spec(
    fn: Callable[..., Any],
    name: str | None,
    description: str | None,
    requires_approval: bool = False,
) -> ToolSpec:
    ctx_params = _ctx_params(fn)
    return ToolSpec(
        name=name or fn.__name__,
        description=description or inspect.getdoc(fn) or "",
        parameters=_parameters(fn, skip=ctx_params),
        func=fn,
        ctx_params=ctx_params,
        requires_approval=requires_approval,
    )


def _ctx_params(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Which parameters asked for a Ctx, by annotation."""
    hints = typing.get_type_hints(fn, include_extras=True)
    return tuple(
        name for name, hint in hints.items() if name != "return" and _wants_ctx(hint)
    )


def _wants_ctx(annotation: Any) -> bool:
    if annotation is Ctx:
        return True
    if get_origin(annotation) is Annotated:
        return any(arg is Ctx for arg in get_args(annotation))
    return False


def _parameters(
    fn: Callable[..., Any], *, skip: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The JSON Schema object describing what the model may pass to fn."""
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name == "self" or name in skip:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
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

    __slots__ = ("_tools",)

    def __init__(self, tools: Iterable[Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for func in tools:
            self.register(func)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """Whether a tool is registered, so callers can ask before get() raises."""
        return name in self._tools

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        spec = getattr(func, "_tool_spec", None) or _build_spec(func, None, None)
        self._tools[spec.name] = spec
        return func

    @staticmethod
    def _with_ctx(
        spec: ToolSpec, kwargs: dict[str, Any], ctx: Ctx | None
    ) -> dict[str, Any]:
        """Fill the tool's Ctx parameters, defaulting to an empty context.

        An empty Ctx rather than None so a tool can always read ctx.deps
        without guarding, even when called outside a run.
        """
        if not spec.ctx_params:
            return kwargs
        filled = dict(kwargs)
        for param in spec.ctx_params:
            filled[param] = ctx if ctx is not None else Ctx()
        return filled

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.to_schema() for spec in self._tools.values()]

    async def call(self, name: str, *, ctx: Ctx | None = None, **kwargs: Any) -> Any:
        """Invoke a tool, running a sync one off the event loop.

        The concurrency arun() promises comes from gathering a turn's tool calls,
        and a blocking call inside one of those coroutines defeats it - so a
        plain def tool goes to a thread rather than stalling every other tool
        waiting alongside it.
        """
        spec = self.get(name)
        func, kwargs = spec.func, self._with_ctx(spec, kwargs, ctx)
        if inspect.iscoroutinefunction(inspect.unwrap(func)):
            return await func(**kwargs)
        result = await asyncio.to_thread(func, **kwargs)
        if inspect.isawaitable(result):  # a sync def that returns a coroutine
            return await result
        return result

    def call_sync(self, name: str, *, ctx: Ctx | None = None, **kwargs: Any) -> Any:
        spec = self.get(name)
        result = spec.func(**self._with_ctx(spec, kwargs, ctx))
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if close is not None:
                close()
            raise ConfigurationError(
                f"Tool '{name}' is async; use Agent.arun() instead of Agent.run()"
            )
        return result
