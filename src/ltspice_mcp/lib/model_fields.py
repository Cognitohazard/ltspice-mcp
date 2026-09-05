"""Reading a Pydantic field the way a caller needs to see it.

Two surfaces describe the same models to a human or a model reading cold: the
Python API's ``api.reference()`` catalogue and the MCP ``inspect(kind="reference")``
lookup. They render differently — one a nested tree, the other a flat per-branch
table — but they must spell a *type*, a *default*, a *field name* and a
*constraint* identically, or the same argument reads as two different arguments
depending on which door was used.

So the reading lives here, once, and both renderers import it. Everything in
this module is a pure function of a model or a ``FieldInfo``: no rendering
decisions, no line wrapping, no I/O.
"""

from __future__ import annotations

import types as pytypes
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

__all__ = [
    "accepted_annotation",
    "constraint_label",
    "default_label",
    "describe_field",
    "field_name",
    "first_sentence",
    "item_model",
    "literal_values",
    "model_of",
    "non_null",
    "scalar_name",
    "strip_annotated",
    "type_label",
    "union_members",
]


def strip_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def union_members(annotation: Any) -> tuple[Any, ...] | None:
    origin = get_origin(annotation)
    if origin is Union or origin is pytypes.UnionType:
        return get_args(annotation)
    return None


def model_of(annotation: Any) -> type[BaseModel] | None:
    annotation = strip_annotated(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def scalar_name(annotation: Any) -> str:
    simple = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
        Any: "any",
    }
    if annotation in simple:
        return simple[annotation]
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return "object"
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def type_label(annotation: Any) -> str:
    """A one-line type, with enum members and union branches written out."""
    annotation = strip_annotated(annotation)

    if get_origin(annotation) is Literal:
        return "one of: " + " | ".join(repr(value) for value in get_args(annotation))

    members = union_members(annotation)
    if members is not None:
        rendered = [type_label(member) for member in members if member is not type(None)]
        label = " or ".join(dict.fromkeys(rendered))
        return f"{label} or null" if type(None) in members else label

    origin = get_origin(annotation)
    if origin in (list, Sequence):
        args = get_args(annotation)
        return f"list of {type_label(args[0])}" if args else "list"
    if origin is dict:
        args = get_args(annotation)
        return f"map of {type_label(args[1])}" if len(args) == 2 else "object"
    if origin is tuple:
        return "list"
    return scalar_name(annotation)


def accepted_annotation(field: FieldInfo) -> Any:
    """What the field accepts, which a coercing validator can widen.

    A ``BeforeValidator`` that takes ``True`` for a default policy declares that
    wider input for the JSON Schema; the catalogue reads the same declaration,
    so MCP and the Python API advertise one answer.
    """
    for meta in field.metadata:
        declared = getattr(meta, "json_schema_input_type", PydanticUndefined)
        if declared is not PydanticUndefined:
            return declared
    return field.annotation


def default_label(field: FieldInfo) -> str:
    if field.default_factory is not None:
        try:
            produced = field.default_factory()  # pyright: ignore[reportCallIssue]
        except TypeError:  # pragma: no cover - validated-data factories take an argument
            return "computed"
        if isinstance(produced, BaseModel):
            return "all defaults"
        if isinstance(produced, (list, dict, set)) and not produced:
            return "empty"
        return repr(produced)
    if field.default is PydanticUndefined:
        return "REQUIRED"
    return repr(field.default)


#: Numeric and length bounds a caller has to respect, and the comparison each
#: one spells. Read off the ``annotated_types`` markers pydantic stores in
#: ``FieldInfo.metadata`` by attribute name, so a marker class the library
#: renames does not silently drop the bound — it stops appearing, and the
#: constraint test for the surface fails.
_BOUND_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("ge", ">="),
    ("gt", ">"),
    ("le", "<="),
    ("lt", "<"),
    ("min_length", "min length"),
    ("max_length", "max length"),
)


def constraint_label(field: FieldInfo) -> str:
    """The field's bounds as one short phrase, or ``""`` when it has none.

    The units live in the description; the bounds do not, because repeating
    ``ge=1, le=20`` in prose is how the two drift apart. Reading them off the
    model keeps one statement of the range.
    """
    parts: list[str] = []
    for meta in field.metadata:
        for attribute, spelling in _BOUND_ATTRIBUTES:
            value = getattr(meta, attribute, None)
            if value is not None:
                parts.append(f"{spelling} {value}")
    return ", ".join(parts)


def field_name(model: type[BaseModel], name: str, field: FieldInfo) -> str:
    """The name a caller writes, which an alias may make different from the attribute."""
    alias = field.validation_alias
    if isinstance(alias, str):
        return alias
    choices = getattr(alias, "choices", None)
    if choices:
        first = choices[0]
        if isinstance(first, str):
            return first
    return field.alias if isinstance(field.alias, str) else name


def describe_field(field: FieldInfo) -> str:
    if field.description:
        return field.description
    # A leaf model documents itself in its docstring more often than in a
    # per-field description; use it rather than emit a bare line.
    nested = model_of(non_null(field.annotation))
    if nested is not None and nested.__doc__:
        return " ".join(nested.__doc__.split())
    return ""


#: Abbreviations that end in a period without ending a sentence. Without them a
#: description gets cut at "e.g." and the example — the useful half — is lost.
_ABBREVIATIONS = ("e.g", "i.e", "etc", "cf", "vs", "approx", "Fig")


def first_sentence(description: str, *, limit: int = 220) -> str:
    """Enough of a description to act on, inside a union's per-branch listing."""
    text = " ".join(description.split())
    if not text:
        return ""
    sentence = text
    start = 0
    while True:
        index = text.find(". ", start)
        if index == -1:
            break
        head = text[:index]
        if any(head.endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
            start = index + 2
            continue
        sentence = head + "."
        break
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def non_null(annotation: Any) -> Any:
    """The annotation with its ``| None`` branch dropped."""
    annotation = strip_annotated(annotation)
    members = union_members(annotation)
    if members is None:
        return annotation
    remaining = [member for member in members if member is not type(None)]
    if len(remaining) == 1:
        return strip_annotated(remaining[0])
    return annotation


def item_model(annotation: Any) -> Any:
    """What a ``list[...]`` field holds, or None when it is not a list."""
    annotation = non_null(annotation)
    if get_origin(annotation) in (list, Sequence):
        args = get_args(annotation)
        return strip_annotated(args[0]) if args else None
    return None


def literal_values(model: type[BaseModel], name: str) -> tuple[str, ...] | None:
    """The string members of a ``Literal`` field, or ``None`` if it is not one."""
    field = model.model_fields.get(name)
    if field is None:
        return None
    annotation = strip_annotated(field.annotation)
    if get_origin(annotation) is not Literal:
        return None
    values = get_args(annotation)
    if not values or not all(isinstance(value, str) for value in values):
        return None
    return tuple(str(value) for value in values)
