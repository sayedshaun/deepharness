from __future__ import annotations

import dataclasses
import enum
import types
import typing
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ..errors import ConfigurationError, OutputValidationError
from ..tools import json_type

FINAL_TOOL = "final_answer"
"""Name of the synthetic tool an agent offers when output= is set.

Structured output is expressed as a tool rather than a provider-specific
response format so it works identically on every vendor: the model already
knows how to fill in a tool's arguments, and no provider needs a json-schema
mode for this to work.
"""

_NONE = type(None)


def final_tool_schema(output: type) -> dict[str, Any]:
    """The tool the model calls to deliver its answer as `output`."""
    return {
        "name": FINAL_TOOL,
        "description": (
            f"Deliver your final answer as {output.__name__}. "
            f"Call this once you have everything you need."
        ),
        "parameters": dataclass_schema(output),
    }


def find_final(response: Any) -> Any | None:
    """The FINAL_TOOL call in a response, if the model made one."""
    return next((call for call in response.tool_calls if call.name == FINAL_TOOL), None)


def dataclass_schema(output: type) -> dict[str, Any]:
    """JSON Schema for a dataclass, reusing the same converter tools use."""
    if not dataclasses.is_dataclass(output):
        raise ConfigurationError(
            f"output= must be a dataclass, got {output!r}. Its fields describe "
            f"the shape the model must return and are validated on the way back"
        )
    hints = typing.get_type_hints(output, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in dataclasses.fields(output):
        properties[field.name] = json_type(hints.get(field.name, Any))
        if field.default is dataclasses.MISSING and (
            field.default_factory is dataclasses.MISSING
        ):
            required.append(field.name)
    return {"type": "object", "properties": properties, "required": required}


def coerce(output: type, arguments: dict[str, Any]) -> Any:
    """Build an `output` instance from what the model sent.

    Every field is reported at once rather than one per round-trip, so a model
    that got two fields wrong learns both from a single reply.
    """
    hints = typing.get_type_hints(output, include_extras=True)
    values: dict[str, Any] = {}
    problems: list[str] = []

    for field in dataclasses.fields(output):
        if field.name not in arguments:
            if field.default is dataclasses.MISSING and (
                field.default_factory is dataclasses.MISSING
            ):
                problems.append(f"{field.name}: missing")
            continue
        try:
            values[field.name] = _convert(
                hints.get(field.name, Any), arguments[field.name]
            )
        except OutputValidationError as exc:
            problems.append(f"{field.name}: {exc}")

    if problems:
        raise OutputValidationError("; ".join(problems))
    return output(**values)


def _convert(annotation: Any, value: Any) -> Any:
    """Check one value against one annotation, converting where unambiguous."""
    if annotation is Any:
        return value

    origin = get_origin(annotation)

    if origin is Annotated:
        return _convert(get_args(annotation)[0], value)

    if origin is Union or origin is types.UnionType:
        variants = get_args(annotation)
        if value is None and _NONE in variants:
            return None
        for variant in variants:
            if variant is _NONE:
                continue
            try:
                return _convert(variant, value)
            except OutputValidationError:
                continue
        raise OutputValidationError(
            f"expected one of {_names(variants)}, got {value!r}"
        )

    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise OutputValidationError(f"expected one of {allowed}, got {value!r}")
        return value

    if origin in (list, set, frozenset, tuple):
        if not isinstance(value, list):
            raise OutputValidationError(f"expected a list, got {type(value).__name__}")
        args = get_args(annotation)
        items = [_convert(args[0], item) for item in value] if args else list(value)
        return origin(items)

    if origin is dict:
        if not isinstance(value, dict):
            raise OutputValidationError(
                f"expected an object, got {type(value).__name__}"
            )
        return value

    if isinstance(annotation, type):
        if dataclasses.is_dataclass(annotation):
            if not isinstance(value, dict):
                raise OutputValidationError(
                    f"expected an object, got {type(value).__name__}"
                )
            return coerce(annotation, value)
        if issubclass(annotation, enum.Enum):
            try:
                return annotation(value)
            except ValueError:
                allowed = [member.value for member in annotation]
                raise OutputValidationError(
                    f"expected one of {allowed}, got {value!r}"
                ) from None
        if annotation is bool:
            if not isinstance(value, bool):
                raise OutputValidationError(f"expected true or false, got {value!r}")
            return value
        if annotation is int and isinstance(value, bool):
            # bool is an int subclass in python, but a model sending true for a
            # count has misunderstood the field, not satisfied it.
            raise OutputValidationError("expected int, got bool")
        if (
            annotation is float
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return float(value)  # JSON has one number type; 22 is a fine float
        if not isinstance(value, annotation):
            raise OutputValidationError(
                f"expected {annotation.__name__}, got {type(value).__name__}"
            )
    return value


def _names(annotations: tuple[Any, ...]) -> str:
    return " or ".join(
        getattr(a, "__name__", str(a)) for a in annotations if a is not _NONE
    )
