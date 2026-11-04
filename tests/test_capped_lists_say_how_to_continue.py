"""A list cut short says how to read the rest.

Design principle "Safe edits" (``docs/DESIGN.md``): a list cut short says how
to read the rest. And "A short reply, the data behind an id": a result too
large to read whole should be readable in part.

Every output schema a tool or a delegated handler declares is walked. An object
that declares a ``truncated`` flag is a list that can be cut short, and it must
declare the way on: a cursor (``next_cursor``, ``cursor``, ``continue`` or
``next``) in that object or in an object around it within the same reply item,
or a ``hint`` beside the flag that says what to do instead (``run_code``, whose
dropped output cannot be paged).

Counts of omitted items (``*_omitted``, ``*_truncated``) are not held to this
here: ``inspect``'s open-window ``difference_omitted`` has no way to read the
rest yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

import ltspice_mcp.tools as tools_pkg
from ltspice_mcp.tools._base import page_schema

#: Fields that continue a cut list from where it stopped.
CURSORS = frozenset({"next_cursor", "cursor", "continue", "next"})


def _declared_schemas() -> dict[str, dict[str, Any]]:
    """Every handler's declared output schema, registered or delegated, by name."""
    schemas: dict[str, dict[str, Any]] = {}
    for module in vars(tools_pkg).values():
        if not isinstance(module, ModuleType) or not module.__name__.startswith(
            "ltspice_mcp.tools."
        ):
            continue
        for obj in vars(module).values():
            schema = getattr(obj, "__output_schema__", None)
            if schema is not None and callable(obj):
                schemas[f"{module.__name__.rsplit('.', 1)[-1]}.{obj.__name__}"] = schema
    return schemas


def _cut_lists(
    root: dict[str, Any], node: Any, path: str, around: tuple[frozenset[str], ...]
) -> Iterator[tuple[str, frozenset[str], tuple[frozenset[str], ...]]]:
    """Each object with a ``truncated`` flag: its path, its fields, and the
    fields of the objects around it up to the reply item that holds it."""
    if not isinstance(node, dict):
        return
    if "$ref" in node and node["$ref"].startswith("#/$defs/"):
        node = root["$defs"][node["$ref"].split("/")[-1]]
    properties = node.get("properties")
    inside = around
    if isinstance(properties, dict):
        fields = frozenset(properties)
        if "truncated" in fields:
            yield path or "<root>", fields, around
        inside = (*around, fields)
        for name, child in properties.items():
            yield from _cut_lists(root, child, f"{path}.{name}" if path else name, inside)
    if isinstance(node.get("items"), dict):
        # A list's item is a reply item of its own: the cursors around the
        # list do not page what is inside one item.
        yield from _cut_lists(root, node["items"], f"{path}[]", ())
    if isinstance(node.get("additionalProperties"), dict):
        yield from _cut_lists(root, node["additionalProperties"], f"{path}.*", inside)
    for key in ("anyOf", "oneOf", "allOf"):
        for index, branch in enumerate(node.get(key, ())):
            yield from _cut_lists(root, branch, f"{path}<{key}{index}>", around)


def _continued(fields: frozenset[str], around: tuple[frozenset[str], ...]) -> bool:
    return bool(CURSORS & fields) or "hint" in fields or any(CURSORS & outer for outer in around)


def test_the_walk_finds_the_cut_lists():
    # A walk that found nothing would pass the check below.
    found = {
        (name, path)
        for name, schema in _declared_schemas().items()
        for path, _, _ in _cut_lists(schema, schema, "", ())
    }
    for expected in (
        ("jobs.handle_jobs", "<oneOf0>.runs"),
        ("experiments.handle_run_experiments", "runs"),
        ("run_code.handle_run_code", "<root>"),
        ("inspect_tools.handle_inspect", "results[].page.collections.*"),
        ("schematic_edit.handle_edit_schematic", "views.preexisting"),
        ("analyze.handle_analyze_results", "results.*.per_run"),
    ):
        assert expected in found, sorted(found)


def test_every_list_cut_short_says_how_to_read_the_rest():
    stranded = [
        f"{name}: {path}"
        for name, schema in _declared_schemas().items()
        for path, fields, around in _cut_lists(schema, schema, "", ())
        if not _continued(fields, around)
    ]
    assert not stranded, (
        "These lists can be cut short and declare no way to read the rest: a "
        f"next_cursor beside the flag, or a hint that says what to do: {stranded}"
    )


def test_a_page_without_its_cursor_is_caught():
    # The shared page shape, without its cursor, fails the check above.
    page = page_schema()
    del page["properties"]["next_cursor"]
    schema = {"type": "object", "properties": {"rows": page}}
    (found,) = list(_cut_lists(schema, schema, "", ()))
    assert not _continued(found[1], found[2])
