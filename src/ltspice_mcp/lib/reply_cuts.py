"""How each list in a reply may be cut when the reply is too long to send whole.

Every list an output schema declares says, under the extension key :data:`CUT`,
what may happen to it when its reply does not fit:

- :data:`NEVER`: it is sent whole. Leaving an entry out would hide a fact the
  caller must see, such as a failure, a warning or what the simulator said
  about a run. What keeps such a list short is where it is made: a cap, one
  entry per code and channel, or one per op in the call.
- :data:`ITEMS`: entries may be left out from the end, and the caller is told
  how to read the rest.
- :func:`rows`: each entry stands on its own and is named by its key fields.
  A cut leaves out whole rows, never part of one, so a row's fields never part
  from each other. The key fields are required in the row unless a caller's
  argument projects the row (``projected_by``), in which case they are the
  keys of the row before that projection.
- :func:`series`: samples thinned together with every other list of the reply
  that declares the same series name, such as a waveform's axis and its
  values, so the lists stay aligned.

The declaration is data in the schema, so one declaration on a shared schema
fragment covers every reply that uses it. JSON Schema validators ignore the
key. :func:`schema_lists` is the one walk over a schema that finds every list
and reads its declaration.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

#: The extension key a list's schema declares its cut under.
CUT: Final = "x-cut"

#: Sent whole: what bounds the list is where it is made.
NEVER: Final = "never"

#: Entries may be left out from the end.
ITEMS: Final = "items"

CutKind = Literal["never", "items", "rows", "series"]


def rows(*keys: str, projected_by: str | None = None) -> dict[str, Any]:
    """The declaration of a list of rows named by ``keys``, cut whole.

    ``projected_by`` names the argument that lets a caller project a row down
    to some of its fields, so the keys are those of the row before projection
    and the reply schema cannot require them.
    """
    if not keys:
        raise ValueError("a list of rows names at least one key field")
    declared: dict[str, Any] = {"rows": list(keys)}
    if projected_by is not None:
        declared["projected_by"] = projected_by
    return declared


def series(name: str) -> dict[str, Any]:
    """The declaration of a list of samples thinned together with every other
    list of the reply that declares the series ``name``."""
    if not name:
        raise ValueError("a series is named")
    return {"series": name}


@dataclass(frozen=True)
class ReplyList:
    """One list in an output schema, with how it may be cut.

    ``path`` names where the list sits in the reply: properties joined by
    ``.``, ``[]`` for an entry of a list, ``.*`` for a value of a map, and
    ``<oneOf0>`` for a branch of a union. ``kind`` is None when the list
    declares nothing.
    """

    path: str
    schema: Mapping[str, Any]
    kind: CutKind | None
    keys: tuple[str, ...] = ()
    projected_by: str | None = None
    series: str | None = None
    #: The object schema whose property the list is; None for a list that is
    #: an entry of another list, a value of a map, or the whole schema.
    parent: Mapping[str, Any] | None = None

    @property
    def row(self) -> Mapping[str, Any]:
        """The schema of one entry of the list."""
        item = self.schema.get("items")
        return item if isinstance(item, Mapping) else {}


def is_list(schema: Mapping[str, Any]) -> bool:
    """Whether ``schema`` admits a JSON array."""
    declared = schema.get("type")
    return declared == "array" or (isinstance(declared, list) and "array" in declared)


def read_cut(
    path: str, schema: Mapping[str, Any], parent: Mapping[str, Any] | None = None
) -> ReplyList:
    """The list at ``path`` with its declaration read. A declaration that is
    not one of the four forms raises ``ValueError`` naming the path."""
    declared = schema.get(CUT)
    if declared is None:
        return ReplyList(path, schema, None, parent=parent)
    if declared in (NEVER, ITEMS):
        return ReplyList(path, schema, declared, parent=parent)
    if isinstance(declared, Mapping):
        keys = declared.get("rows")
        if (
            isinstance(keys, list)
            and keys
            and all(isinstance(key, str) and key for key in keys)
            and set(declared) <= {"rows", "projected_by"}
        ):
            projected_by = declared.get("projected_by")
            if projected_by is None or (isinstance(projected_by, str) and projected_by):
                return ReplyList(path, schema, "rows", tuple(keys), projected_by, parent=parent)
        name = declared.get("series")
        if set(declared) == {"series"} and isinstance(name, str) and name:
            return ReplyList(path, schema, "series", series=name, parent=parent)
    raise ValueError(
        f"{path or '<root>'}: {CUT} is {declared!r}; declare {NEVER!r}, {ITEMS!r}, "
        "rows(key, ...) or series(name)"
    )


def schema_lists(schema: Mapping[str, Any]) -> Iterator[ReplyList]:
    """Every list ``schema`` declares, outermost first, with its declaration.

    The walk enters properties, a list's entries, a map's values, the branches
    of ``anyOf``, ``oneOf`` and ``allOf``, and local ``$ref``s into ``$defs``.
    A fragment shared by reference is met once per place it is used.
    """
    definitions = schema.get("$defs", {})

    def walk(
        node: Any,
        path: str,
        resolving: frozenset[str],
        parent: Mapping[str, Any] | None = None,
    ) -> Iterator[ReplyList]:
        if not isinstance(node, Mapping):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            if name in resolving:
                return
            resolving = resolving | {name}
            node = definitions[name]
        if is_list(node):
            yield read_cut(path, node, parent)
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            for name, child in properties.items():
                yield from walk(child, f"{path}.{name}" if path else name, resolving, node)
        if isinstance(node.get("items"), Mapping):
            yield from walk(node["items"], f"{path}[]", resolving)
        if isinstance(node.get("additionalProperties"), Mapping):
            yield from walk(node["additionalProperties"], f"{path}.*", resolving)
        for key in ("anyOf", "oneOf", "allOf"):
            for index, branch in enumerate(node.get(key, ())):
                yield from walk(branch, f"{path}<{key}{index}>", resolving)

    yield from walk(schema, "", frozenset())
