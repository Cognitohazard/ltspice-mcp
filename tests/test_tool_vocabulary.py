"""One name, one meaning across the tools.

A caller carries an argument from one tool to the next: a job and case from a
receipt into ``plot_waveform``, an attached ``run_experiments.analyze`` block
into ``analyze_results``. These tests read the published input schemas and
hold each shared name to one shape wherever it appears, so a value one tool
accepts is not refused by another under the same name.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from collections.abc import Set
from typing import Any

import pytest

from ltspice_mcp.tools import get_tools

# Names that address a result or size a reply. Each must have one schema in
# every tool that takes it.
SHARED_NAMES = ("job_id", "case_id", "run_index", "plot_index", "dialect", "other_raw", "budget")

# Schema keywords a shared argument still differs by, and why, until a change
# of its own settles them.
KNOWN_DIFFERENCES = {
    "job_id": (
        {"minLength"},
        "jobs refuses an empty job_id as it validates the call; plot_waveform, "
        "analyze_results' sources and inspect take an empty one as absent",
    ),
}

# analyze_results arguments the attached block does not take, and why.
NOT_ATTACHED = {
    "sources": "the job is the source",
    "budget": "the attached result is rendered on the receipt, under run_experiments' budget",
    "continue": "the attached result is rendered on the receipt, under run_experiments' budget",
}

_PROSE = frozenset({"description", "title", "examples"})


def _resolved(node: Any, defs: dict[str, Any]) -> Any:
    """``node`` with every ``$ref`` inlined and its prose removed."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _resolved(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        return {
            key: (
                {name: _resolved(value, defs) for name, value in child.items()}
                if key == "properties"
                else _resolved(child, defs)
            )
            for key, child in node.items()
            if key not in _PROSE and key != "$defs"
        }
    if isinstance(node, list):
        return [_resolved(item, defs) for item in node]
    return node


@functools.cache
def _published() -> dict[str, dict[str, Any]]:
    tools, _ = get_tools()
    return {
        tool.name: _resolved(tool.input_schema, tool.input_schema.get("$defs", {}))
        for tool in tools
    }


def _branches(node: dict[str, Any]) -> list[dict[str, Any]]:
    """The object schemas a node accepts: itself, or its union's members, with
    the parts a per-action schema keeps its arguments in (``allOf``, ``then``)."""
    members = node.get("anyOf") or node.get("oneOf")
    if members:
        return [m for b in members for m in _branches(b)]
    parts = [*node.get("allOf", ()), node.get("then"), node.get("else")]
    return [node, *(m for part in parts if isinstance(part, dict) for m in _branches(part))]


def _arguments(node: dict[str, Any], path: str, found: dict[str, dict[str, Any]]) -> None:
    """Every named argument under ``node``, as ``found[name][path] = schema``."""
    for branch in _branches(node):
        for name, schema in (branch.get("properties") or {}).items():
            found[name][f"{path}.{name}"] = schema
            _arguments(schema, f"{path}.{name}", found)
        if isinstance(branch.get("items"), dict):
            _arguments(branch["items"], f"{path}[]", found)


def _shared_arguments() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, schema in _published().items():
        _arguments(schema, name, found)
    return found


def _shapes(uses: dict[str, Any], ignored: Set[str] = frozenset()) -> dict[str, list[str]]:
    shapes: dict[str, list[str]] = defaultdict(list)
    for path, schema in uses.items():
        kept = {key: value for key, value in schema.items() if key not in ignored}
        shapes[json.dumps(kept, sort_keys=True)].append(path)
    return shapes


def _object(node: dict[str, Any]) -> dict[str, Any]:
    (branch,) = [b for b in _branches(node) if b.get("type") == "object"]
    return branch


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_a_shared_argument_has_one_schema(name: str):
    uses = _shared_arguments()[name]
    assert len({path.split(".", 1)[0] for path in uses}) > 1, f"{name} is no longer shared"
    ignored, _ = KNOWN_DIFFERENCES.get(name, (set(), ""))
    shapes = _shapes(uses, ignored)
    assert len(shapes) == 1, f"{name} differs between tools: {list(shapes.values())}"
    if ignored:
        # A known difference that has gone must leave this table too.
        assert len(_shapes(uses)) > 1, f"{name} no longer differs; drop it here"


def test_a_step_is_chosen_by_the_same_selector_everywhere():
    uses = _shared_arguments()["step"]
    selectors = {path: _object(schema) for path, schema in uses.items()}
    assert set(selectors) == {
        "analyze_results.step",
        "run_experiments.analyze.step",
        "plot_waveform.step",
    }
    assert len(_shapes(selectors)) == 1


class TestTheAttachedAnalyzeBlockReadsLikeAnalyzeResults:
    @staticmethod
    def _schemas() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        published = _published()
        results = published["analyze_results"]["properties"]
        source = results["sources"]["items"]["properties"]
        attached = _object(published["run_experiments"]["properties"]["analyze"])["properties"]
        return results, source, attached

    def test_every_attached_argument_is_spelled_as_analyze_results_spells_it(self):
        results, source, attached = self._schemas()
        for name in set(attached) - {"recipes", "include"}:
            assert name in results or name in source, name
            assert attached[name] == results.get(name, source.get(name)), name

    def test_the_attached_block_leaves_out_only_what_the_job_answers(self):
        results, _, attached = self._schemas()
        assert set(results) - set(attached) == set(NOT_ATTACHED)

    def test_include_differs_only_by_provenance(self):
        # run_experiments takes provenance as an argument of its own.
        results, _, attached = self._schemas()
        theirs = json.loads(json.dumps(results["include"]))
        del _object(theirs)["properties"]["provenance"]
        assert attached["include"] == theirs
