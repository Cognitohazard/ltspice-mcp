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
from typing import Any

from ltspice_mcp.tools._base import page_schema
from tests.conftest import CONTINUATIONS, declared_output_schemas


def _declared_schemas() -> dict[str, dict[str, Any]]:
    """Every handler's declared output schema, registered or delegated, by
    ``module.handler``."""
    return {
        f"{module.__name__.rsplit('.', 1)[-1]}.{handler.__name__}": schema
        for module, handler, schema in declared_output_schemas()
    }


def _cut_lists(schema: dict[str, Any]) -> Iterator[tuple[str, bool]]:
    """Each object in an output schema with a ``truncated`` flag: its path, and
    whether it declares a way to read the rest (a cursor in it or around it
    within its reply item, or a hint beside the flag)."""

    def walk(node: Any, path: str, cursor_around: bool) -> Iterator[tuple[str, bool]]:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref", "")
        if ref.startswith("#/$defs/"):
            node = schema["$defs"][ref.split("/")[-1]]
        properties = node.get("properties")
        inside = cursor_around
        if isinstance(properties, dict):
            inside = cursor_around or bool(CONTINUATIONS & properties.keys())
            if "truncated" in properties:
                yield path or "<root>", inside or "hint" in properties
            for name, child in properties.items():
                yield from walk(child, f"{path}.{name}" if path else name, inside)
        if isinstance(node.get("items"), dict):
            # A list's item is a reply item of its own: the cursors around the
            # list do not page what is inside one item.
            yield from walk(node["items"], f"{path}[]", False)
        if isinstance(node.get("additionalProperties"), dict):
            yield from walk(node["additionalProperties"], f"{path}.*", inside)
        for key in ("anyOf", "oneOf", "allOf"):
            for index, branch in enumerate(node.get(key, ())):
                yield from walk(branch, f"{path}<{key}{index}>", cursor_around)

    yield from walk(schema, "", False)


def test_the_walk_finds_the_cut_lists():
    # A walk that found nothing would pass the check below.
    found = {
        (name, path)
        for name, schema in _declared_schemas().items()
        for path, _ in _cut_lists(schema)
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
        for path, continued in _cut_lists(schema)
        if not continued
    ]
    assert not stranded, (
        "These lists can be cut short and declare no way to read the rest: a "
        f"next_cursor beside the flag, or a hint that says what to do: {stranded}"
    )


def test_a_page_without_its_cursor_is_caught():
    # The shared page shape, without its cursor, fails the check above.
    page = page_schema()
    del page["properties"]["next_cursor"]
    assert list(_cut_lists({"type": "object", "properties": {"rows": page}})) == [("rows", False)]
