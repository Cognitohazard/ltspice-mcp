"""The rule that leaves the fields that say nothing out of every reply.

``lib/reply_fields.py`` holds the rule; ``tools/_base.py`` holds each tool's
fields kept anyway (``ALWAYS_PRESENT``) and applies the rule to every reply.
These tests pin the rule itself, the per-tool lists, the schemas the rule
relaxes, and that the MCP reply and the Python API's result have one shape.
The conformance hook in ``conftest.py`` holds every reply the suite makes to
the same rule.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.reply_fields import (
    IMPLIED,
    Kept,
    may_say_nothing,
    omit_silent,
    relax_required,
    says_nothing,
    silent_fields,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import ALL_KEPT, ALWAYS_PRESENT, registry, reply_payload
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit
from tests.conftest import SyncApi

NONE: dict[str, Kept] = {}


class TestTheRule:
    def test_it_leaves_out_an_empty_list_an_empty_object_a_null_and_a_false_flag(self):
        reply = {"failures": [], "views": {}, "next_cursor": None, "truncated": False, "ok": True}
        assert omit_silent(reply, NONE) == {"ok": True}

    def test_it_keeps_a_zero_an_empty_string_and_true(self):
        """Only what the rule names says nothing: a count of zero is an answer."""
        reply = {"total": 0, "stdout": "", "shown": True, "items": [1]}
        assert omit_silent(reply, NONE) == reply

    def test_it_never_leaves_out_an_item_of_a_list(self):
        """A position in a list is part of what an item says, so a null sample stays."""
        reply = {"y": [1.0, None, 3.0], "rows": [{"a": None}, {"a": 1}], "pairs": [[], [1]]}
        assert omit_silent(reply, NONE) == {
            "y": [1.0, None, 3.0],
            "rows": [{}, {"a": 1}],
            "pairs": [[], [1]],
        }

    def test_an_object_left_empty_is_left_out_in_turn(self):
        reply = {"render": {"note": None, "scale": None}, "wiring": {"pins_total": 0}}
        assert omit_silent(reply, NONE) == {"wiring": {"pins_total": 0}}

    def test_a_kept_field_stays_empty_null_or_false(self):
        kept = {"ok": Kept("a verdict"), "equivalent": Kept("a verdict")}
        reply = {"ok": False, "equivalent": None, "error": None}
        assert omit_silent(reply, kept) == {"ok": False, "equivalent": None}

    def test_a_field_that_holds_data_is_kept_as_it_is(self):
        kept = {"value": Kept("measured data", data=True)}
        reply = {"value": {"cutoff_low_hz": None, "exact_match": False, "warnings": []}, "x": None}
        assert omit_silent(reply, kept) == {"value": reply["value"]}

    @pytest.mark.parametrize(("overall", "kept_too"), [(True, False), (False, True), (None, True)])
    def test_a_sub_answer_goes_only_where_the_overall_answer_gives_it(self, overall, kept_too):
        kept = {"equivalent": Kept(""), "structurally_equivalent": Kept("")}
        reply = {"equivalent": overall, "structurally_equivalent": True}
        assert ("structurally_equivalent" in omit_silent(reply, kept)) is kept_too

    def test_every_implied_sub_answer_names_an_overall_answer_kept_by_some_tool(self):
        for field, (overall, _value) in IMPLIED.items():
            assert field in ALL_KEPT and overall in ALL_KEPT, field

    def test_an_unchanged_reply_is_returned_as_it_is_and_no_input_is_changed(self):
        lean = {"a": 1, "b": [{"c": "x"}]}
        assert omit_silent(lean, NONE) is lean
        fat = {"a": 1, "b": [{"c": None}], "d": []}
        before = copy.deepcopy(fat)
        omit_silent(fat, NONE)
        assert fat == before

    def test_silent_fields_names_what_the_rule_would_leave_out(self):
        reply = {"a": None, "b": {"c": [], "d": 1}, "e": [{"f": False}]}
        assert sorted(silent_fields(reply, NONE)) == [("a",), ("b", "c"), ("e", 0, "f")]
        assert list(silent_fields(omit_silent(reply, NONE), NONE)) == []
        assert list(silent_fields(reply, NONE, allowed_nulls=frozenset({("a",)}))) == [
            ("b", "c"),
            ("e", 0, "f"),
        ]

    def test_a_null_written_for_a_number_that_is_not_finite_is_kept_and_said(self):
        """The rule runs first, so the substitution stays visible where it was made."""
        sent = reply_payload({"min": float("nan"), "note": None}, NONE)
        assert sent["min"] is None
        assert "note" not in sent
        assert any("min" in warning for warning in sent["warnings"])


class TestTheKeptFields:
    def test_every_registered_tool_lists_its_kept_fields(self):
        names = {rt.definition.name for rt in registry._registered}
        assert names <= set(ALWAYS_PRESENT)

    def test_every_kept_field_says_why(self):
        for tool, fields in ALWAYS_PRESENT.items():
            for name, kept in fields.items():
                assert kept.reason.strip(), f"{tool}.{name} gives no reason"


class TestTheSchemas:
    @pytest.mark.parametrize("name", sorted(ALWAYS_PRESENT))
    def test_no_output_schema_requires_a_field_a_reply_may_leave_out(self, name: str):
        """A reply leaves a silent field out, so its tool's schema may not require it."""
        (registered,) = [rt for rt in registry._registered if rt.definition.name == name]
        schema = registered.definition.output_schema
        assert schema is not None
        assert relax_required(copy.deepcopy(schema), ALWAYS_PRESENT[name]) == schema

    def test_relaxing_keeps_what_always_says_something(self):
        kept = {"ok": Kept("")}
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "ok": {"type": "boolean"},
                "failures": {"type": "array"},
                "cursor": {"type": ["string", "null"]},
                "at": {
                    "type": "object",
                    "properties": {"file": {"type": "string"}, "line": {"type": "integer"}},
                    "required": ["file"],
                },
                "render": {"type": "object", "properties": {"note": {"type": "string"}}},
            },
            "required": ["path", "ok", "failures", "cursor", "at", "render"],
        }
        assert relax_required(schema, kept)["required"] == ["path", "ok", "at"]

    @pytest.mark.parametrize(
        ("schema", "silent"),
        [
            ({"type": "integer"}, False),
            ({"type": "string", "enum": ["a", "b"]}, False),
            ({"type": "array", "minItems": 1}, False),
            ({"type": "boolean"}, True),
            ({"type": ["integer", "null"]}, True),
            ({}, True),
        ],
    )
    def test_what_may_say_nothing(self, schema: dict[str, Any], silent: bool):
        assert may_say_nothing(schema, NONE) is silent


def test_says_nothing_is_the_rule():
    assert [says_nothing(v) for v in (None, False, [], {}, (), 0, "", True, [0])] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


async def test_the_api_result_has_the_mcp_replys_shape(state_no_sim: SessionState, work_dir: Path):
    """The Python API's complete result leaves out the same fields, so a script
    and a tool call read one shape."""
    deck = work_dir / "rc.cir"
    deck.write_text(
        "* rc\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n", encoding="utf-8"
    )
    args = {"path": str(deck), "checks": ["syntax", "quality"], "export_to": "managed"}
    reply = await handle_verify_circuit(VerifyCircuitInput.model_validate(args), state_no_sim)
    assert reply.structured_content is not None
    result = await _api_verify(state_no_sim, args)
    assert list(silent_fields(result, ALWAYS_PRESENT["verify_circuit"])) == []
    assert set(result) == set(reply.structured_content)
    for absent in ("failures", "warnings", "comparison", "export", "render", "scene"):
        assert absent not in result


async def _api_verify(state: SessionState, args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(lambda: SyncApi(state).verify_circuit(**args))
