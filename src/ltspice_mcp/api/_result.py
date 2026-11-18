"""The result of a catalogued operation: the sent reply, read with its schema.

A reply leaves out every field that says nothing (``lib/reply_fields.py``), and
the Python API returns the reply's shape. :class:`SentResult` is that dict, as
sent, plus one answer: reading a field the operation's output schema declares,
and the reply left out, gives the value the rule left out
(``reply_fields.left_out_value``, which answers that question): an empty list
for an array, an empty object for an object, ``False`` for a flag, the implied
value of an implied sub-answer, ``None`` for anything else, a field whose
schema names no type among them. A key the schema does not declare, and a
field the tool keeps (``ALWAYS_PRESENT``), whose absence means something else,
raise ``KeyError`` as a plain dict's do, so a misspelt name still fails.

Everything else is the dict as sent: ``in``, ``len``, iteration, equality,
``repr`` and ``json.dumps`` see only the fields the reply carried, so ``in``
still tells whether a field was sent. ``dict(result)``, ``result.copy()`` and
``copy.copy`` give a plain dict at the top level; ``copy.deepcopy`` and pickle
give plain dicts throughout.

Nested objects and the rows of a list answer the same way, each by its own part
of the schema. Where a schema has branches (``oneOf``, ``anyOf``, ``allOf``), a
field declared in any branch counts as declared; a field the branches declare
with different kinds of value answers ``None``. A field kept as data
(``Kept.data``) is handed back as it is, since the rule left nothing out of it.

``_CataloguedMethod._run`` (``api/_methods.py``) is the one place a result is
wrapped; ``pyproject.toml`` bans the names everywhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ltspice_mcp.lib.reply_fields import Kept, left_out_value

_BRANCHES = ("oneOf", "anyOf", "allOf")
_NO_DEFAULT: Any = object()
_UNREAD: Any = object()


def _expanded(parts: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """``parts`` and every branch inside them, depth first."""
    out: list[Mapping[str, Any]] = []
    pending = list(parts)
    while pending:
        part = pending.pop(0)
        out.append(part)
        for keyword in _BRANCHES:
            pending.extend(b for b in part.get(keyword, ()) if isinstance(b, Mapping))
    return tuple(out)


class _Shape:
    """What one value of a result is declared as: the schema parts it answers by,
    and the replying tool's kept fields."""

    __slots__ = ("_extra", "_fields", "_items", "_parts", "kept")

    def __init__(self, parts: Iterable[Mapping[str, Any]], kept: Mapping[str, Kept]) -> None:
        self._parts = _expanded(parts)
        self.kept = kept
        self._fields: dict[str, _Shape] | None = None
        self._extra: Any = _UNREAD
        self._items: Any = _UNREAD

    def field(self, name: str) -> _Shape | None:
        """The shape of a declared field, or None where no part declares it."""
        if self._fields is None:
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for part in self._parts:
                properties = part.get("properties")
                if isinstance(properties, Mapping):
                    for key, sub in properties.items():
                        if isinstance(sub, Mapping):
                            grouped.setdefault(key, []).append(sub)
            self._fields = {key: _Shape(subs, self.kept) for key, subs in grouped.items()}
        return self._fields.get(name)

    def value_of(self, name: str) -> _Shape | None:
        """The shape of a present value: its declared field, else the map's values."""
        declared = self.field(name)
        if declared is not None:
            return declared
        if self._extra is _UNREAD:
            extra = [
                part["additionalProperties"]
                for part in self._parts
                if isinstance(part.get("additionalProperties"), Mapping)
            ]
            self._extra = _Shape(extra, self.kept) if extra else None
        return self._extra

    def items(self) -> _Shape | None:
        """The shape of a list's rows, or None where no part declares one."""
        if self._items is _UNREAD:
            rows = [
                part["items"] for part in self._parts if isinstance(part.get("items"), Mapping)
            ]
            self._items = _Shape(rows, self.kept) if rows else None
        return self._items

    def kinds(self) -> set[str]:
        """The JSON Schema types the parts declare."""
        kinds: set[str] = set()
        for part in self._parts:
            declared = part.get("type")
            if isinstance(declared, str):
                kinds.add(declared)
            elif isinstance(declared, list):
                kinds.update(kind for kind in declared if isinstance(kind, str))
        return kinds

    def shaped(self, value: Any) -> Any:
        """``value`` with each object in it answering by its part of the schema."""
        if isinstance(value, dict):
            return SentResult(value, self)
        if isinstance(value, list):
            rows = self.items()
            if rows is not None:
                return [rows.shaped(item) for item in value]
        return value


class SentResult(dict[str, Any]):
    """A sent reply that answers a declared field it left out with the value
    the reply rule left out."""

    __slots__ = ("_shape",)

    def __init__(self, value: Mapping[str, Any], shape: _Shape) -> None:
        super().__init__()
        self._shape = shape
        for key, item in value.items():
            rule = shape.kept.get(key) if isinstance(key, str) else None
            inner = None if rule is not None and rule.data else shape.value_of(key)
            dict.__setitem__(self, key, item if inner is None else inner.shaped(item))

    def _absent(self, key: Any) -> Any:
        field = self._shape.field(key) if isinstance(key, str) else None
        if field is None:
            raise KeyError(key)
        value = left_out_value(key, field.kinds(), self, self._shape.kept)
        return SentResult(value, field) if isinstance(value, dict) else value

    def __missing__(self, key: Any) -> Any:
        return self._absent(key)

    def get(self, key: Any, default: Any = _NO_DEFAULT) -> Any:  # type: ignore[override]
        if key in self:
            return dict.__getitem__(self, key)
        if default is not _NO_DEFAULT:
            return default
        try:
            return self._absent(key)
        except KeyError:
            return None

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickled and deep-copied as the plain dict it holds: the schema it
        # reads by stays with the operation that sent it.
        return (dict, (dict(self),))


_SHAPES: dict[str, _Shape] = {}


def sent_result(value: Any, tool: str) -> Any:
    """``value``, a reply ``tool`` sent, answering by ``tool``'s output schema.

    Anything but a dict is returned as it is."""
    if not isinstance(value, dict):
        return value
    shape = _SHAPES.get(tool)
    if shape is None:
        from ltspice_mcp.tools import get_tools
        from ltspice_mcp.tools._base import ALWAYS_PRESENT

        schema = get_tools()[1][tool].definition.output_schema
        if schema is None:
            raise LookupError(f"{tool!r} declares no output schema")
        shape = _SHAPES[tool] = _Shape([schema], ALWAYS_PRESENT[tool])
    return SentResult(value, shape)
