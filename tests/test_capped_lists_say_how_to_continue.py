"""A list cut short says how to read the rest.

Design principle "Safe edits" (``docs/DESIGN.md``): a list cut short says how
to read the rest. And "A short reply, the data behind an id": a result too
large to read whole should be readable in part.

Every output schema a tool or a delegated handler declares is walked. An object
that declares a ``truncated`` flag, or a count of the items a list left out
(an integer ``omitted``, ``*_omitted`` or ``*_truncated``), holds a list that
can be cut short, and it must declare the way on: a cursor (``next_cursor``,
``cursor``, ``continue`` or ``next``) in that object or in an object around it
within the same reply item, or a ``hint`` beside the flag or the count that
says what to do instead (``run_code``, whose dropped output cannot be paged).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ltspice_mcp.tools._base import page_schema
from tests.conftest import (
    CONTINUATIONS,
    counts_what_was_left_out,
    cut_without_continuation,
    declared_output_schemas,
)


def _declared_schemas() -> dict[str, dict[str, Any]]:
    """Every handler's declared output schema, registered or delegated, by
    ``module.handler``."""
    return {
        f"{module.__name__.rsplit('.', 1)[-1]}.{handler.__name__}": schema
        for module, handler, schema in declared_output_schemas()
    }


def _counts_what_it_left_out(name: str, schema: Any) -> bool:
    """Whether a property is a count of the items a list left out: an integer
    by a name the reply checker reads too."""
    return (
        counts_what_was_left_out(name)
        and isinstance(schema, dict)
        and schema.get("type") == "integer"
    )


def _cut_lists(schema: dict[str, Any]) -> Iterator[tuple[str, bool]]:
    """Each object in an output schema with a ``truncated`` flag or a count of
    what a list left out: its path (with the count's name), and whether it
    declares a way to read the rest (a cursor in it or around it within its
    reply item, or a hint beside the flag or the count)."""

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
                if _counts_what_it_left_out(name, child):
                    yield f"{path or '<root>'}:{name}", inside or "hint" in properties
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
        ("inspect_tools.handle_inspect", "results[].data.designs[]:difference_omitted"),
        ("analysis.handle_plot_waveform", "<root>:traces_truncated"),
        ("verify.handle_verify_circuit", "findings_omitted.*:omitted"),
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


def test_a_count_of_what_was_left_out_without_a_way_on_is_caught():
    # A list and the count of what it left out, with nothing that reads the rest.
    rows = {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": {"type": "string"}},
            "rows_omitted": {"type": "integer"},
        },
    }
    assert list(_cut_lists(rows)) == [("<root>:rows_omitted", False)]
    rows["properties"]["hint"] = {"type": "string"}
    assert list(_cut_lists(rows)) == [("<root>:rows_omitted", True)]


# The reply checker the conformance hook runs on every reply in the suite
# (``tests/conftest.py``) holds a reply's counts to the rule the schemas above
# are held to: a count of what a list left out, above zero, needs a cursor in
# its reply item or a hint beside it.


def test_the_reply_checker_catches_a_count_with_no_way_on():
    reply = {"rows": ["a", "b"], "rows_truncated": 5, "page": {"items": [1], "omitted": 3}}
    assert list(cut_without_continuation(reply)) == [("rows_truncated",), ("page", "omitted")]


def test_the_reply_checker_passes_a_count_with_a_hint_beside_it():
    reply = {"rows": ["a", "b"], "rows_truncated": 5, "hint": "rows(offset=2) reads the rest."}
    assert list(cut_without_continuation(reply)) == []


def test_the_reply_checker_passes_a_count_with_a_cursor_in_its_reply_item():
    beside = {"items": [1], "omitted": 3, "next_cursor": "c2"}
    around = {"next_cursor": "c2", "page": {"items": [1], "omitted": 3}}
    assert list(cut_without_continuation(beside)) == []
    assert list(cut_without_continuation(around)) == []


def test_a_cursor_in_another_reply_item_does_not_continue_a_count():
    # Each item of a list is a reply item of its own: one item's cursor does
    # not page what another left out, nor does a cursor around the list.
    reply = {
        "next_cursor": "c2",
        "designs": [{"next_cursor": "d2"}, {"difference": ["x"], "difference_omitted": 4}],
    }
    assert list(cut_without_continuation(reply)) == [("designs", 1, "difference_omitted")]


def test_a_count_of_nothing_left_out_is_no_cut():
    assert list(cut_without_continuation({"rows": ["a"], "rows_omitted": 0})) == []
