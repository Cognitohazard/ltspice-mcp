"""Leaving out the reply fields that say nothing.

Every reply the server sends, each tool's ``structuredContent`` and the Python
API's results alike, passes through :func:`omit_silent` once, on its way out.
It leaves out a field whose value is

- an empty list or an empty object,
- a null,
- a false flag, or
- a sub-answer that the overall answer beside it already gives
  (:data:`IMPLIED`): ``structurally_equivalent`` beside ``equivalent: true``.

So an absent field means "checked, and nothing to report", and a reply about
work that found nothing costs no tokens for saying so. An absent list reads as
empty and an absent flag as false.

A missing field never means "not checked". Whatever did not run, was skipped
or could not be read is reported by a field that has content (a
``checks_skipped`` row, a failure, a ``status``, a reason), so it is never left
out. A field whose empty, null or false value is itself the answer is named in
the replying tool's list of fields kept (:class:`Kept`, assembled in
``tools/_base.py``), with the reason it is kept.

The rule applies to the fields of objects, never to the items of a list: an
item's position is part of what it says, so a null sample in a series stays.

The tools layer names the replying tool's kept fields with
:func:`replying_with` for the length of each call, and every reply builder, the
response-budget measurement among them, reads them through
:func:`active_kept`.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Kept:
    """A reply field kept even when it is empty, null or false, and why."""

    reason: str
    data: bool = False
    """The field holds data rather than fields, such as a recipe's measured
    value: it is kept as it is, and nothing inside it is left out."""


#: A sub-answer and the overall answer that already gives it: the sub-answer is
#: left out of an object whose overall answer has that value. When the overall
#: answer has any other value, every sub-answer is kept, so the caller sees
#: which part held and which did not.
IMPLIED: Mapping[str, tuple[str, object]] = {
    # A circuit that is equivalent is structurally equivalent too.
    "structurally_equivalent": ("equivalent", True),
    # A simulator that is not available cannot be selected either; its
    # remediation or refusal says why.
    "selectable": ("available", False),
}


def says_nothing(value: Any) -> bool:
    """Whether a field holding ``value`` is left out: null, false, or empty."""
    if value is None or value is False:
        return True
    return isinstance(value, (dict, list, tuple)) and not value


_ACTIVE: contextvars.ContextVar[Mapping[str, Kept] | None] = contextvars.ContextVar(
    "reply_fields_kept", default=None
)


@contextlib.contextmanager
def replying_with(kept: Mapping[str, Kept]) -> Iterator[None]:
    """Name the kept fields of the tool whose reply is being built here."""
    token = _ACTIVE.set(kept)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_kept() -> Mapping[str, Kept] | None:
    """The kept fields named by the innermost :func:`replying_with`, or None."""
    return _ACTIVE.get()


def omit_silent(value: Any, kept: Mapping[str, Kept]) -> Any:
    """``value`` without the fields that say nothing.

    Copy-on-write: an object or list in which nothing was left out is returned
    as it is, so an unchanged reply costs no copies, and the input is never
    changed. An object that is left empty once its silent fields are gone says
    nothing either and is left out of its parent in turn.
    """
    if isinstance(value, dict):
        return _omit_in_object(value, kept)
    if isinstance(value, (list, tuple)):
        return _omit_in_list(value, kept)
    return value


def _omit_in_object(obj: dict[Any, Any], kept: Mapping[str, Kept]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    changed = False
    for key, item in obj.items():
        rule = kept.get(key) if isinstance(key, str) else None
        if rule is not None and rule.data:
            out[key] = item
            continue
        lean = omit_silent(item, kept)
        if rule is None and says_nothing(lean):
            changed = True
            continue
        changed = changed or lean is not item
        out[key] = lean
    for field, (overall, value) in IMPLIED.items():
        if field in out and overall in out and out[overall] is value:
            del out[field]
            changed = True
    return out if changed else obj


def _omit_in_list(items: list[Any] | tuple[Any, ...], kept: Mapping[str, Kept]) -> Any:
    out: list[Any] | None = None
    for index, item in enumerate(items):
        if not isinstance(item, (dict, list, tuple)):
            continue
        lean = omit_silent(item, kept)
        if lean is not item:
            if out is None:
                out = list(items)
            out[index] = lean
    return out if out is not None else items


def silent_fields(
    value: Any,
    kept: Mapping[str, Kept],
    *,
    allowed_nulls: frozenset[tuple[Any, ...]] = frozenset(),
    path: tuple[Any, ...] = (),
) -> Iterator[tuple[Any, ...]]:
    """The path of every field in ``value`` that :func:`omit_silent` leaves out.

    Empty for a reply that passed through it. ``allowed_nulls`` names the
    paths where a null stands for a number that is not finite: the reply
    builder writes those after the rule ran and says so in ``warnings``.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            rule = kept.get(key) if isinstance(key, str) else None
            at = (*path, key)
            if rule is not None and rule.data:
                continue
            if rule is None and says_nothing(item) and at not in allowed_nulls:
                yield at
                continue
            yield from silent_fields(item, kept, allowed_nulls=allowed_nulls, path=at)
        for field, (overall, implied) in IMPLIED.items():
            if field in value and value.get(overall) is implied:
                yield (*path, field)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from silent_fields(item, kept, allowed_nulls=allowed_nulls, path=(*path, index))


def may_say_nothing(schema: Any, kept: Mapping[str, Kept]) -> bool:
    """Whether a property with this schema can hold a value the rule leaves out.

    An object says something whenever one of its required properties always
    does (a finding's ``at`` always has its ``file``), and so does an array
    that has at least one item; anything the schema does not pin down may say
    nothing.
    """
    if not isinstance(schema, Mapping):
        return True
    if "const" in schema:
        return says_nothing(schema["const"])
    if "enum" in schema:
        return any(says_nothing(value) for value in schema["enum"])
    declared = schema.get("type")
    if declared is None:
        return True
    types = declared if isinstance(declared, list) else [declared]
    if "null" in types or "boolean" in types:
        return True
    if "array" in types and schema.get("minItems", 0) < 1:
        return True
    if "object" in types:
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        return not any(
            name in kept or (name in properties and not may_say_nothing(properties[name], kept))
            for name in schema.get("required", ())
        )
    return False


def relax_required(schema: Any, kept: Mapping[str, Kept]) -> Any:
    """Mark optional, in place, every property the rule may leave out.

    Walks the whole schema, ``$defs`` and branches included, and takes out of
    each object's ``required`` the properties that may hold an empty, null or
    false value and are not in ``kept``. Inner objects are relaxed first, so an
    object stays required when a property it always carries does. A kept
    property that holds data is not entered, since nothing inside it is left
    out. Returns ``schema``.
    """
    if isinstance(schema, dict):
        for key, item in schema.items():
            if key == "properties" and isinstance(item, dict):
                for name, sub in item.items():
                    rule = kept.get(name)
                    if rule is not None and rule.data:
                        continue
                    relax_required(sub, kept)
            else:
                relax_required(item, kept)
        properties = schema.get("properties")
        required = schema.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            schema["required"] = [
                name
                for name in required
                if name in kept
                or (name in properties and not may_say_nothing(properties[name], kept))
            ]
    elif isinstance(schema, list):
        for item in schema:
            relax_required(item, kept)
    return schema
