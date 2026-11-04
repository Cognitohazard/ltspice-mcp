"""A list cut short says how to read the rest.

Design principle "Safe edits" (``docs/DESIGN.md``): a list cut short says how
to read the rest. And "A short reply, the data behind an id": a result too
large to read whole should be readable in part.

Every output schema a tool or a delegated handler declares is walked. A field
that says items were left out of a list marks a list that can be cut short: a
``truncated`` flag, or a count of omitted items (``omitted``, ``*_omitted``,
``*_truncated``). It must declare the way on: a cursor (``next_cursor``,
``cursor``, ``continue`` or ``next``) in the object that declares it or in an
object around it within the same reply item, or a ``hint`` beside it that says
what to do instead (``run_code``, whose dropped output cannot be paged).

``NO_WAY_ON_YET`` names the fields that declare no way on yet, each with the
reason. A test fails once one of them gains a way on, so the list only shrinks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ltspice_mcp.tools._base import page_schema
from tests.conftest import CONTINUATIONS, declared_output_schemas

#: Cut lists with no way to read the rest yet, by ``(handler, path)``, each
#: with the reason. Remove an entry when its list gains a way on.
NO_WAY_ON_YET: dict[tuple[str, str], str] = {
    ("inspect_tools.handle_inspect", "results[].data.designs[].difference_omitted"): (
        "an open LTspice window's sheet lists where it differs from the file up to "
        "a number of entries and counts the rest; no call lists the rest"
    ),
}


def _declared_schemas() -> dict[str, dict[str, Any]]:
    """Every handler's declared output schema, registered or delegated, by
    ``module.handler``."""
    return {
        f"{module.__name__.rsplit('.', 1)[-1]}.{handler.__name__}": schema
        for module, handler, schema in declared_output_schemas()
    }


def _counts_omitted(name: str) -> bool:
    """Whether a field named ``name`` counts items left out of a list."""
    return name == "omitted" or name.endswith(("_omitted", "_truncated"))


def _cut_lists(schema: dict[str, Any]) -> Iterator[tuple[str, bool]]:
    """Each place in an output schema that says a list was cut short, and
    whether it declares a way to read the rest (a cursor in its object or
    around it within its reply item, or a hint beside it).

    A ``truncated`` flag is reported at the object that declares it, which is
    the list's page; a count of omitted items at the count itself.
    """

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
            continued = inside or "hint" in properties
            if "truncated" in properties:
                yield path or "<root>", continued
            for name in properties:
                if _counts_omitted(name):
                    yield f"{path}.{name}" if path else name, continued
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


def _every_cut_list() -> dict[tuple[str, str], bool]:
    return {
        (name, path): continued
        for name, schema in _declared_schemas().items()
        for path, continued in _cut_lists(schema)
    }


def test_the_walk_finds_the_cut_lists():
    # A walk that found nothing would pass the check below.
    found = set(_every_cut_list())
    for expected in (
        ("jobs.handle_jobs", "<oneOf0>.runs"),
        ("experiments.handle_run_experiments", "runs"),
        ("run_code.handle_run_code", "<root>"),
        ("inspect_tools.handle_inspect", "results[].page.collections.*"),
        ("schematic_edit.handle_edit_schematic", "views.preexisting"),
        ("analyze.handle_analyze_results", "results.*.per_run"),
        # Counts of omitted items.
        ("verify.handle_verify_circuit", "findings_omitted.*.omitted"),
        ("analysis.handle_plot_waveform", "traces_truncated"),
        *NO_WAY_ON_YET,
    ):
        assert expected in found, sorted(found)


def test_every_list_cut_short_says_how_to_read_the_rest():
    stranded = [
        f"{name}: {path}"
        for (name, path), continued in _every_cut_list().items()
        if not continued and (name, path) not in NO_WAY_ON_YET
    ]
    assert not stranded, (
        "These lists can be cut short and declare no way to read the rest: a "
        f"next_cursor beside the flag or count, or a hint that says what to do: {stranded}"
    )


def test_the_lists_with_no_way_on_yet_still_have_none():
    # The exemptions only shrink: one that has gained a way on is removed.
    found = _every_cut_list()
    continued = [place for place in NO_WAY_ON_YET if found.get(place)]
    assert not continued, (
        f"These lists now say how to continue; remove them from NO_WAY_ON_YET: {continued}"
    )


def test_a_page_without_its_cursor_is_caught():
    # The shared page shape, without its cursor, fails the check above.
    page = page_schema()
    del page["properties"]["next_cursor"]
    assert list(_cut_lists({"type": "object", "properties": {"rows": page}})) == [("rows", False)]


def test_a_count_of_omitted_items_without_a_way_on_is_caught():
    rows = {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": {"type": "object"}},
            "rows_omitted": {"type": "integer"},
        },
    }
    assert list(_cut_lists(rows)) == [("rows_omitted", False)]
    rows["properties"]["next_cursor"] = {"type": "string"}
    assert list(_cut_lists(rows)) == [("rows_omitted", True)]
