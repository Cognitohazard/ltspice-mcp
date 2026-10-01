"""Tests for edit_schematic — the transactional, revision-guarded .asc author tool.

Covers the revision guard (sha match / mismatch / missing), the commit protocol
(staged write + rename-last, with crash injection before and after the rename),
a base:"blank" build reaching the same sheet as the same ops applied to an
existing one, each op's own facts under ``results``, the wiring metric and the
paginated touched / pin_legend / label_only_pins views, the refusal of every spelling of a render this tool no
longer has, the post-commit compare stage (success / mismatch / export
failure), and an archetype-scale blank build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import TypeAdapter, ValidationError
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError, PathSecurityError
from ltspice_mcp.lib.schematic_ops import (
    OP_RESULT_FACTS,
    build_on_wire_predicate,
    collect_component_geometry,
    get_asc_editor,
    post_op_warnings,
    run_op_batch,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.schematic_edit import (
    EditSchematicInput,
    complete_edit_schematic_data,
    evaluate_edit_schematic,
    handle_edit_schematic,
)
from tests import _fake_netlister as fake_netlister
from tests._asc_ops import apply_ops
from tests.test_api_reference import _op_kinds

# Validates a raw op dict into the tool's own op union, so the control path
# below builds exactly the op objects the tool would have built.
_OPS_ADAPTER = TypeAdapter(list[se.ConsolidatedOp])


def _assert_schema(result) -> dict:
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(se._OUTPUT_SCHEMA).validate(data)
    return data


def _edit_input(**kw) -> EditSchematicInput:
    """Build the input from kwargs via model_validate so op dicts validate at
    runtime without tripping the static arg-type check on the op union."""
    return EditSchematicInput.model_validate(kw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


# Two resistors side by side; R1.1<->R2.1 wired over the top, R1.2 labeled.
_DIVIDER_OPS: list[dict] = [
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 400, "y": 300},
    {"op": "add_component", "reference": "R2", "symbol": "res", "x": 700, "y": 300},
    {
        "op": "wire_pins",
        "from_pin": "R1.1",
        "to_pin": "R2.1",
        "waypoints": [{"x": 400, "y": 200}, {"x": 700, "y": 200}],
    },
    {"op": "add_net_label", "net": "vout", "pin": "R1.2"},
]


async def _build_blank(state: SessionState, name: str, ops: list[dict], **kw) -> dict:
    result = await handle_edit_schematic(
        _edit_input(target=f"{name}.asc", base="blank", ops=ops, **kw), state
    )
    return _assert_schema(result)


# ---------------------------------------------------------------------------
# Blank-build parity + basic commit
# ---------------------------------------------------------------------------


async def test_blank_build_parity_with_the_shared_op_runner(asc_state, work_dir):
    """base:"blank" writes the same .asc the shared op runner produces directly.

    The tool wraps the runner in a revision guard, a staged write and a view
    pass; none of that may alter the geometry the ops describe. Driving the
    runner against the same blank template is the control.
    """
    data = await _build_blank(asc_state, "parity_edit", _DIVIDER_OPS)
    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"

    control = work_dir / "parity_runner.asc"
    control.write_text(se._BLANK_TEMPLATE, newline="\n")
    editor = get_asc_editor(control, asc_state)
    ops = _OPS_ADAPTER.validate_python(_DIVIDER_OPS)
    _, abort = run_op_batch(editor, ops, control, stop_on_error=True)
    assert abort is None
    editor.save_netlist(control)

    assert (work_dir / "parity_edit.asc").read_text() == control.read_text()


async def test_a_t_junction_endpoint_commits_through_the_whole_transaction(asc_state):
    """A wire_pins endpoint given as {x, y} on a wire's interior is carried
    through the tool's own validation, commit and touched view, which reads each
    op's pins by name and must pass over one that has none."""
    ops = [
        *_DIVIDER_OPS,
        {"op": "add_component", "reference": "R3", "symbol": "res", "x": 550, "y": 100},
        {"op": "wire_pins", "from_pin": "R3.2", "to_pin": {"x": 550, "y": 200}},
    ]
    data = await _build_blank(asc_state, "tee_edit", ops)
    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"
    touched = {row["ref"] for row in data["views"]["touched"]["items"]}
    assert "R3" in touched


# The divider's route over the top leaves this rail along y=200; the res
# fixture's pins sit 48 above and below its origin, so R1.2 and R2.2 are at
# y=348.
_RAIL = {"from": _DIVIDER_OPS[2]["waypoints"][0], "to": _DIVIDER_OPS[2]["waypoints"][1]}
_BOTTOM = {"from": {"x": 400, "y": 348}, "to": {"x": 700, "y": 348}}

_FACT_OPS: list[dict] = [
    *_DIVIDER_OPS,
    {"op": "add_component", "reference": "R3", "symbol": "res", "x": 550, "y": 100},
    # 5: a stem from R3 ending on the rail's interior
    {"op": "wire_pins", "from_pin": "R3.2", "to_pin": {"x": 550, "y": 200}},
    # 6 and 7: the same straight run twice
    {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "R2.2"},
    {"op": "wire_pins", "from_pin": "R1.2", "to_pin": "R2.2"},
]

_FACT_RESULTS: list[dict] = [
    {
        "index": 5,
        "op": "wire_pins",
        "junctions": [{"x": 550, "y": 200, "via": "endpoint", "wire": _RAIL}],
    },
    {"index": 7, "op": "wire_pins", "already_present": [_BOTTOM]},
]


@pytest.mark.parametrize("dry_run", [False, True], ids=["commit", "dry_run"])
async def test_the_response_carries_each_ops_own_facts(asc_state, dry_run: bool):
    """What an op found on the sheet reaches the caller as a field, not text.

    A T onto a wire's interior names the wire it joined; a route already on
    the sheet names the segments it did not redraw. Both used to stop at the
    op runner, so the caller could learn of neither from the response. The
    ops that found nothing, the plain routes included, have no entry.
    """
    data = await _build_blank(asc_state, "facts", _FACT_OPS, dry_run=dry_run)

    assert data["outcome"] == "complete"
    assert data["results"] == _FACT_RESULTS


def test_every_op_declaring_result_facts_is_an_op_this_tool_takes():
    # A misspelt key would relay nothing for that op, without an error.
    assert set(OP_RESULT_FACTS) <= _op_kinds()


async def test_repeated_op_warnings_arrive_once_with_a_count(asc_state):
    """A batch is where the same advisory repeats.

    Every add_net_label of an already-labelled net reports the same duplicate
    advisory, so a converter-scale build would spend hundreds of identical
    lines saying one thing — and dropping them all instead (what this surface
    did) means the caller never learns the ops warned at all. One entry, with
    how many ops it covers.
    """
    ops = [
        {"op": "add_net_label", "net": "vout", "x": 400 + 64 * index, "y": 300}
        for index in range(4)
    ]

    data = await _build_blank(asc_state, "dupwarn", ops)

    duplicates = [w for w in data["warnings"] if "already labels a net" in w]
    assert len(duplicates) == 1, duplicates
    assert "3 ops" in duplicates[0]
    assert "collapsed" in duplicates[0]


async def test_committed_sha_matches_file(asc_state, work_dir):
    data = await _build_blank(asc_state, "shacheck", _DIVIDER_OPS)
    assert data["sha256"] == _sha(work_dir / "shacheck.asc")


# ---------------------------------------------------------------------------
# Revision guard
# ---------------------------------------------------------------------------


async def test_existing_edit_requires_and_honors_sha(asc_state, work_dir):
    first = await _build_blank(asc_state, "rev", _DIVIDER_OPS)
    sha0 = first["sha256"]

    add_r3 = [{"op": "add_component", "reference": "R3", "symbol": "res", "x": 1000, "y": 300}]
    ok = _assert_schema(
        await handle_edit_schematic(
            _edit_input(target="rev.asc", expected_sha256=sha0, ops=add_r3), asc_state
        )
    )
    assert ok["outcome"] == "complete"
    assert ok["sha256"] != sha0
    assert "R3" in (work_dir / "rev.asc").read_text()


async def test_stale_sha_returns_revision_conflict(asc_state, work_dir):
    first = await _build_blank(asc_state, "conflict", _DIVIDER_OPS)
    sha0 = first["sha256"]
    # A peer commits, moving the file off sha0.
    await handle_edit_schematic(
        _edit_input(
            target="conflict.asc",
            expected_sha256=sha0,
            ops=[{"op": "add_component", "reference": "R3", "symbol": "res", "x": 1000, "y": 300}],
        ),
        asc_state,
    )
    sha1 = _sha(work_dir / "conflict.asc")

    loser = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="conflict.asc",
                expected_sha256=sha0,  # stale
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R4",
                        "symbol": "res",
                        "x": 1300,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert loser["outcome"] == "failed"
    assert loser["error"]["code"] == "revision_conflict"
    # Full error envelope: a revision conflict is retryable, nothing was written.
    assert loser["error"]["stage"] == "revision_check"
    assert loser["error"]["retryable"] is True
    assert loser["error"]["commit_state"] == "not_started"
    assert loser["commit_state"] == "not_committed"
    # Nothing written: the file is unchanged and R4 never landed.
    assert _sha(work_dir / "conflict.asc") == sha1
    assert "R4" not in (work_dir / "conflict.asc").read_text()


async def test_revision_conflict_payload_carries_current_sha(asc_state, work_dir):
    """A conflict names the revision now in force, so the retry needs no re-read.

    Without it the caller learns only that its token is stale and has to go
    fetch the current one — the error would report the problem while withholding
    the handle that fixes it.
    """
    first = await _build_blank(asc_state, "conflictsha", _DIVIDER_OPS)
    add = [{"op": "add_component", "reference": "R3", "symbol": "res", "x": 1000, "y": 300}]
    await handle_edit_schematic(
        _edit_input(target="conflictsha.asc", expected_sha256=first["sha256"], ops=add),
        asc_state,
    )
    current = _sha(work_dir / "conflictsha.asc")

    loser = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="conflictsha.asc",
                expected_sha256=first["sha256"],  # stale
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R4",
                        "symbol": "res",
                        "x": 1300,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert loser["error"]["code"] == "revision_conflict"
    assert loser["sha256"] == current


async def test_first_edit_commits_with_the_digest_inspect_reported(asc_state, work_dir):
    """A fresh session must be able to obtain its first edit token through the
    product: read the sheet, edit it, one call each. Before inspect reported the
    digest, no read tool did — the only way to get one was to provoke an error
    or hash the file outside the server."""
    await _build_blank(asc_state, "firstedit", _DIVIDER_OPS)
    (found,) = (
        await handle_inspect(
            InspectInput.model_validate(
                {"queries": [{"kind": "components", "path": "firstedit.asc"}]}
            ),
            asc_state,
        )
    ).structured_content["results"]

    committed = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="firstedit.asc",
                expected_sha256=found["data"]["sha256"],
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert committed["outcome"] == "complete"
    assert "R3" in (work_dir / "firstedit.asc").read_text()


async def test_missing_expected_sha_refusal_hands_back_the_current_digest(asc_state, work_dir):
    """The guard stands — nothing is written without the token — but the
    refusal is what the caller reads next, so it carries the digest they need
    instead of sending them off to fetch it. Otherwise the first edit of every
    session pays a refused call plus a read before anything can commit."""
    await _build_blank(asc_state, "needsha", _DIVIDER_OPS)
    current = _sha(work_dir / "needsha.asc")
    before = (work_dir / "needsha.asc").read_bytes()

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="needsha.asc",
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )

    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    assert data["error"]["code"] == "expected_sha256_required"
    assert data["sha256"] == current
    assert current in data["error"]["message"]
    assert (work_dir / "needsha.asc").read_bytes() == before


async def test_dry_run_needs_no_expected_sha_and_hands_back_the_digest(asc_state, work_dir):
    """The token prevents a lost update, which only a write can cause. A dry run
    used to be refused without it, so validating an edit cost a read first."""
    await _build_blank(asc_state, "drysha", _DIVIDER_OPS)
    target = work_dir / "drysha.asc"
    current = _sha(target)
    before = _fingerprint(work_dir)

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="drysha.asc",
                dry_run=True,
                ops=[{"op": "set_component_value", "reference": "R2", "value": "4k7"}],
            ),
            asc_state,
        )
    )

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    assert "error" not in data
    assert data["sha256"] == current
    assert current in data["hint"]
    assert data["observations"] == []
    assert _fingerprint(work_dir) == before


async def test_op_less_read_needs_no_expected_sha(asc_state, work_dir):
    """Paging the preexisting view is an op-less read; it writes nothing."""
    await _build_blank(asc_state, "readsha", _DIVIDER_OPS)
    current = _sha(work_dir / "readsha.asc")

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(target="readsha.asc", ops=[], return_views=["pin_legend"]),
            asc_state,
        )
    )

    assert data["outcome"] == "complete"
    assert {row["ref"] for row in data["views"]["pin_legend"]["items"]} == {"R1", "R2"}
    assert data["sha256"] == current
    assert _sha(work_dir / "readsha.asc") == current


async def test_dry_run_reports_a_stale_expected_sha_as_a_fact(asc_state, work_dir):
    await _build_blank(asc_state, "stalesha", _DIVIDER_OPS)
    current = _sha(work_dir / "stalesha.asc")
    stale = "0" * 64

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="stalesha.asc",
                expected_sha256=stale,
                dry_run=True,
                ops=[{"op": "set_component_value", "reference": "R2", "value": "4k7"}],
            ),
            asc_state,
        )
    )

    assert data["outcome"] == "complete"
    assert "error" not in data
    assert data["sha256"] == current
    (note,) = data["observations"]
    assert stale in note and current in note and "revision_conflict" in note
    assert {"stage": "revision_check", "ok": False, "error": "sha mismatch"} in data["stages"]
    # A commit quoting the same stale token is still refused.
    committed = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="stalesha.asc",
                expected_sha256=stale,
                ops=[{"op": "set_component_value", "reference": "R2", "value": "4k7"}],
            ),
            asc_state,
        )
    )
    assert committed["error"]["code"] == "revision_conflict"
    assert _sha(work_dir / "stalesha.asc") == current


async def test_parallel_session_revision_race(config, work_dir, asc_symbols):
    """Two sessions over one file: one commits, the stale one gets revision_conflict.

    Same-directory sessions coordinate through the shared edit guard + file lock;
    the sha is rechecked inside the guard, so the second committer loses.
    """
    session_a = SessionState.create(config, available={})
    session_b = SessionState.create(config, available={})

    first = _assert_schema(
        await handle_edit_schematic(
            _edit_input(target="race.asc", base="blank", ops=_DIVIDER_OPS), session_a
        )
    )
    sha0 = first["sha256"]

    a = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="race.asc",
                expected_sha256=sha0,
                ops=[
                    {
                        "op": "add_component",
                        "reference": "RA",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            session_a,
        )
    )
    assert a["outcome"] == "complete"

    b = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="race.asc",
                expected_sha256=sha0,  # stale — A already moved it
                ops=[
                    {
                        "op": "add_component",
                        "reference": "RB",
                        "symbol": "res",
                        "x": 1300,
                        "y": 300,
                    }
                ],
            ),
            session_b,
        )
    )
    assert b["outcome"] == "failed"
    assert b["error"]["code"] == "revision_conflict"
    assert "RB" not in (work_dir / "race.asc").read_text()


# ---------------------------------------------------------------------------
# Commit-protocol crash injection
# ---------------------------------------------------------------------------


async def test_crash_before_rename_leaves_the_target_and_writes_nothing_else(
    asc_state, work_dir, monkeypatch
):
    first = await _build_blank(asc_state, "crash", _DIVIDER_OPS)
    sha0 = first["sha256"]

    def boom(_tmp, _target):
        raise RuntimeError("injected rename failure")

    monkeypatch.setattr(se, "_commit_rename", boom)

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="crash.asc",
                expected_sha256=sha0,
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    # Full error envelope on the commit-failure path: a transient I/O fault is
    # retryable and the failed commit phase is named from the stage bookkeeping.
    assert data["error"]["code"] == "commit_failed"
    assert data["error"]["retryable"] is True
    assert data["error"]["commit_state"] == "not_started"
    assert data["error"]["stage"] in {"stage_asc", "rename"}
    assert data["build_id"]  # echoed
    # Target untouched.
    assert _sha(work_dir / "crash.asc") == sha0
    # Nothing else is written either: no staging temp, and no quarantined
    # draft beside the target. The caller still holds its own ops, and the
    # response names the stage that failed.
    assert not list(work_dir.glob("crash.asc.staging-*"))
    assert not list(work_dir.glob("crash.draft-*.asc"))


async def test_crash_after_rename_stays_committed(asc_state, work_dir, monkeypatch):
    (work_dir / "ref.cir").write_text("Vin in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n")

    async def boom_export(_copy, _state):
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(se, "_export_asc_to_netlist", boom_export)

    data = await _build_blank(
        asc_state, "aftercommit", _DIVIDER_OPS, compare={"reference": "ref.cir"}
    )
    # The rename succeeded, so the sheet is committed even though a post-rename
    # (reference-export) stage failed. The failed stage is the shortfall that
    # keeps the call off "complete".
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"
    assert (work_dir / "aftercommit.asc").is_file()
    assert data["verification"]["export_error"]
    assert data["verification"]["equivalent"] is None


# ---------------------------------------------------------------------------
# The former "connect" spelling of wire_pins is not accepted
# ---------------------------------------------------------------------------


def test_connect_op_rejected_by_model():
    with pytest.raises(ValidationError):
        _edit_input(
            target="x.asc",
            ops=[{"op": "connect", "from_pin": "R1.1", "to_pin": "R2.1"}],
        )


def test_connect_absent_from_input_schema():
    from ltspice_mcp.tools._base import registry

    reg = next(r for r in registry._registered if r.definition.name == "edit_schematic")
    import json

    schema_text = json.dumps(reg.definition.input_schema)
    # No op literal "connect" anywhere in the schema (description prose aside).
    assert '"connect"' not in schema_text
    assert "wire_pins" in schema_text


# ---------------------------------------------------------------------------
# dry_run leaves everything untouched
# ---------------------------------------------------------------------------


async def test_dry_run_writes_nothing(asc_state, work_dir):
    first = await _build_blank(asc_state, "dry", _DIVIDER_OPS)
    sha0 = first["sha256"]

    before = _fingerprint(work_dir)

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="dry.asc",
                expected_sha256=sha0,
                dry_run=True,
                return_views=["pin_legend"],
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    # Every op validated, geometry computed, nothing written.
    assert data["wiring"]["pins_total"] == 6  # R1, R2, R3
    assert {row["ref"] for row in data["views"]["pin_legend"]["items"]} == {"R1", "R2", "R3"}
    # Directory fingerprint and target sha both unchanged.
    assert _fingerprint(work_dir) == before
    assert _sha(work_dir / "dry.asc") == sha0
    assert not (work_dir / ".ltspice-mcp" / "renders").exists()


async def test_empty_ops_with_views_reads_the_sheet(asc_state, work_dir):
    """The whole-sheet pin table is only reachable through this tool, so a batch
    with no ops and a requested view is a read: nothing written, views returned."""
    first = await _build_blank(asc_state, "readonly", _DIVIDER_OPS)
    sha0 = first["sha256"]
    before = _fingerprint(work_dir)
    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="readonly.asc",
                expected_sha256=sha0,
                ops=[],
                return_views=["pin_legend"],
            ),
            asc_state,
        )
    )
    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    assert {row["ref"] for row in data["views"]["pin_legend"]["items"]} >= {"R1", "R2"}
    assert _fingerprint(work_dir) == before
    assert _sha(work_dir / "readonly.asc") == sha0


async def test_empty_ops_without_views_is_rejected(asc_state):
    await _build_blank(asc_state, "noop", _DIVIDER_OPS)
    with pytest.raises(NetlistError, match="return_views"):
        await handle_edit_schematic(
            _edit_input(
                target="noop.asc",
                expected_sha256=_sha(Path(asc_state.working_dir) / "noop.asc"),
                ops=[],
                return_views=[],
            ),
            asc_state,
        )


async def test_dry_run_surfaces_all_op_failures(asc_state):
    await _build_blank(asc_state, "dryfail", _DIVIDER_OPS)
    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="dryfail.asc",
                expected_sha256=_sha(Path(asc_state.working_dir) / "dryfail.asc"),
                dry_run=True,
                ops=[
                    {"op": "set_component_value", "reference": "NOPE1", "value": "1k"},
                    {"op": "set_component_value", "reference": "NOPE2", "value": "2k"},
                ],
            ),
            asc_state,
        )
    )
    # Both bad ops surface at once (dry run does not stop on the first), and a
    # validation pass in which every op failed is not a complete call.
    assert len(data["failures"]) == 2
    assert data["outcome"] == "partial"


async def test_default_view_covers_only_the_refs_the_batch_touched(asc_state, work_dir):
    """An ack-shaped edit returns the geometry of what it edited, not the sheet.

    R8 adds one net label and used to be handed every pin on the sheet. The
    whole-sheet table is still one explicit ``return_views`` away.
    """
    built = await _build_blank(asc_state, "touched-scope", _DIVIDER_OPS)
    assert {row["ref"] for row in built["views"]["touched"]["items"]} == {"R1", "R2"}
    assert "pin_legend" not in built["views"]

    follow_up = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="touched-scope.asc",
                expected_sha256=_sha(work_dir / "touched-scope.asc"),
                ops=[{"op": "set_component_value", "reference": "R2", "value": "4k7"}],
            ),
            asc_state,
        )
    )

    assert [row["ref"] for row in follow_up["views"]["touched"]["items"]] == ["R2"]
    # Nothing was lost — the sheet still has both, on request.
    whole_sheet = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="touched-scope.asc",
                expected_sha256=_sha(work_dir / "touched-scope.asc"),
                return_views=["pin_legend"],
                ops=[{"op": "set_component_value", "reference": "R2", "value": "5k"}],
            ),
            asc_state,
        )
    )
    assert {row["ref"] for row in whole_sheet["views"]["pin_legend"]["items"]} == {"R1", "R2"}


async def test_dry_run_seam_keeps_full_views_while_mcp_pages_them(asc_state, work_dir):
    args = _edit_input(
        target="dry-full.asc",
        base="blank",
        dry_run=True,
        return_views=["pin_legend"],
        view_limit=2,
        ops=_many_labeled_ops(5),
    )
    neutral = await evaluate_edit_schematic(args, asc_state)
    assert neutral.views is not None
    assert len(neutral.views.pin_legend) == 5
    assert len(neutral.views.label_only_pins) == 5
    complete = complete_edit_schematic_data(neutral, args)
    assert complete["views"]["pin_legend"]["items"] == list(neutral.views.pin_legend)
    assert complete["wiring"]["label_only_pins"]["items"] == list(neutral.views.label_only_pins)
    assert complete["views"]["pin_legend"]["truncated"] is False
    assert not (work_dir / "dry-full.asc").exists()

    mcp = _assert_schema(await handle_edit_schematic(args, asc_state))
    page = mcp["views"]["pin_legend"]
    assert page["items"] == list(neutral.views.pin_legend[:2])
    assert (page["total"], page["returned"], page["truncated"]) == (5, 2, True)
    assert page["total"] - page["returned"] == len(neutral.views.pin_legend) - 2


async def test_views_are_bound_to_the_committed_bytes_not_a_later_file_revision(
    asc_state,
    work_dir,
    monkeypatch,
):
    original_commit = se._commit_asc

    def commit_then_peer_revision(text, target, build_id, encoding):
        outcome = original_commit(text, target, build_id, encoding)
        if outcome.renamed:
            target.write_text("Version 4.1\nSHEET 1 880 680\nTEXT 32 32 Left 2 ;peer revision\n")
        return outcome

    monkeypatch.setattr(se, "_commit_asc", commit_then_peer_revision)
    args = _edit_input(
        target="interleaved.asc",
        base="blank",
        return_views=["pin_legend"],
        ops=_DIVIDER_OPS,
    )
    neutral = await evaluate_edit_schematic(args, asc_state)
    assert neutral.views is not None
    assert neutral.data["sha256"] == neutral.views.sha256
    assert neutral.views.sha256 != _sha(work_dir / "interleaved.asc")
    # The legend describes the bytes this transaction committed, not the
    # peer's revision that landed on the target a moment later.
    assert {row["ref"] for row in neutral.views.pin_legend} == {"R1", "R2"}

    complete = complete_edit_schematic_data(neutral, args)
    assert complete["sha256"] == neutral.views.sha256
    assert {row["ref"] for row in complete["views"]["pin_legend"]["items"]} == {
        "R1",
        "R2",
    }


# ---------------------------------------------------------------------------
# Wiring metric
# ---------------------------------------------------------------------------


async def test_wiring_metric_and_label_only(asc_state):
    data = await _build_blank(asc_state, "wiring", _DIVIDER_OPS)
    w = data["wiring"]
    assert w["pins_total"] == 4
    assert w["pins_wired"] == 2  # R1.1 and R2.1 on the wire
    assert w["pins_label_only"] == 1  # R1.2 carries "vout", no wire
    page = w["label_only_pins"]
    assert page["total"] == 1
    only = page["items"][0]
    assert only["pin"] == "R1.2"
    assert only["net"] == "vout"


# ---------------------------------------------------------------------------
# Findings scoped to the edit
# ---------------------------------------------------------------------------

# A sheet that already has problems before anyone edits it: four resistors with
# no wires (every free pin floats), two of them tied to their nets by a label
# alone, and a label sitting on nothing. Seven sheet findings, two label-only
# pins.
_UNTIDY_OPS: list[dict] = [
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 300},
    {"op": "add_component", "reference": "R2", "symbol": "res", "x": 300, "y": 300},
    {"op": "add_component", "reference": "R3", "symbol": "res", "x": 500, "y": 300},
    {"op": "add_component", "reference": "R4", "symbol": "res", "x": 700, "y": 300},
    {"op": "add_net_label", "net": "n3", "pin": "R3.1"},
    {"op": "add_net_label", "net": "n4", "pin": "R4.1"},
    {"op": "add_net_label", "net": "orphan", "x": 900, "y": 900},
]
_UNTIDY_FINDINGS = 7
_UNTIDY_LABEL_ONLY = 2


def _sheet_findings(data: dict) -> list[str]:
    """The sheet's own findings in the warnings channel; op advisories are
    attributed to their op and prefixed with it."""
    return [w for w in data["warnings"] if not w.startswith("op ")]


def _whole_sheet(state: SessionState, path: Path) -> tuple[list[str], list[str]]:
    """Every finding message and label-only pin on the sheet as committed.

    Read through a fresh parse of the file with the same validation pass the
    tool runs, so the control shares nothing with the envelope under test.
    """
    state.editors.invalidate(path)
    editor = get_asc_editor(path, state)
    labels = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}
    wired = build_on_wire_predicate(
        [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    )
    label_only = [
        f"{comp['ref']}.{pin['name']}"
        for comp in collect_component_geometry(editor)
        for pin in comp["pins"]
        if not wired((pin["x"], pin["y"])) and (pin["x"], pin["y"]) in labels
    ]
    return [w["message"] for w in post_op_warnings(editor)], label_only


async def _untidy_sheet(state: SessionState, name: str) -> Path:
    built = await _build_blank(state, name, _UNTIDY_OPS)
    assert built["commit_state"] == "committed"
    return Path(state.working_dir) / f"{name}.asc"


async def test_a_blank_build_withholds_nothing(asc_state):
    """Everything on a sheet built from blank is new, so everything is reported."""
    data = await _build_blank(asc_state, "untidy-blank", _UNTIDY_OPS)
    assert len(_sheet_findings(data)) == _UNTIDY_FINDINGS
    assert data["wiring"]["label_only_pins"]["total"] == _UNTIDY_LABEL_ONLY
    assert data["preexisting"] == {
        "count": 0,
        "findings": 0,
        "label_only_pins": 0,
        "cursor": None,
    }


async def test_an_edit_reports_what_it_introduced_and_counts_the_rest(asc_state):
    """Adding one unwired part reports that part's two floating pins, not the
    seven findings the sheet already had; those are counted, not listed."""
    sheet = await _untidy_sheet(asc_state, "untidy-add")

    data = await apply_ops(
        asc_state,
        sheet,
        [{"op": "add_component", "reference": "R5", "symbol": "res", "x": 1100, "y": 300}],
    )

    assert data["commit_state"] == "committed"
    assert _sheet_findings(data) == [
        "Floating pin: R5.1 at (1100,252)",
        "Floating pin: R5.2 at (1100,348)",
    ]
    assert data["preexisting"]["findings"] == _UNTIDY_FINDINGS
    assert data["preexisting"]["label_only_pins"] == _UNTIDY_LABEL_ONLY
    assert data["preexisting"]["count"] == _UNTIDY_FINDINGS + _UNTIDY_LABEL_ONLY
    assert data["preexisting"]["cursor"]
    # structuredContent is all a structured-aware client reads, so the route
    # to the rest rides in the hint as well as in the schema.
    assert "predate" in data["hint"]
    assert "view_cursors.preexisting" in data["hint"]
    # Reported plus withheld is the whole sheet: nothing was dropped.
    whole, _ = _whole_sheet(asc_state, sheet)
    assert len(whole) == len(_sheet_findings(data)) + data["preexisting"]["findings"]


async def test_a_finding_on_a_reference_the_batch_named_is_reported(asc_state):
    """R1's floating pins predate the edit, but the edit is about R1."""
    sheet = await _untidy_sheet(asc_state, "untidy-touch")

    data = await apply_ops(
        asc_state, sheet, [{"op": "set_component_value", "reference": "R1", "value": "2k"}]
    )

    assert _sheet_findings(data) == [
        "Floating pin: R1.1 at (100,252)",
        "Floating pin: R1.2 at (100,348)",
    ]
    assert data["preexisting"]["findings"] == _UNTIDY_FINDINGS - 2


async def test_a_finding_at_a_coordinate_the_batch_named_is_reported(asc_state):
    """The orphan label predates the edit; the directive is placed on it."""
    sheet = await _untidy_sheet(asc_state, "untidy-coord")

    data = await apply_ops(
        asc_state,
        sheet,
        [{"op": "add_directive", "instruction": ".op", "x": 900, "y": 900}],
    )

    assert _sheet_findings(data) == ["Dangling label 'orphan' at (900,900)"]
    assert data["preexisting"]["findings"] == _UNTIDY_FINDINGS - 1


async def test_a_finding_at_a_routes_coordinate_endpoint_is_reported(asc_state, work_dir):
    """The duplicated stub predates the edit; the route ends on its free end,
    naming that point as surely as a waypoint or a label's x, y would."""
    await _build_blank(
        asc_state,
        "dup-stub",
        [
            {"op": "add_component", "reference": "R1", "symbol": "res", "x": 100, "y": 300},
            {"op": "add_component", "reference": "R2", "symbol": "res", "x": 300, "y": 300},
        ],
    )
    sheet = work_dir / "dup-stub.asc"
    # edit_schematic never draws a segment twice, so the duplicate is written raw.
    sheet.write_bytes(sheet.read_bytes() + b"WIRE 100 252 100 200\n" * 2)

    data = await apply_ops(
        asc_state,
        sheet,
        [
            {
                "op": "wire_pins",
                "from_pin": "R2.1",
                "to_pin": {"x": 100, "y": 200},
                "waypoints": [{"x": 300, "y": 200}],
            }
        ],
    )

    assert data["commit_state"] == "committed"
    assert any(w.startswith("Duplicate wire (2×)") for w in _sheet_findings(data))


async def test_label_only_pins_are_scoped_and_reconcile_with_the_sheet_totals(asc_state):
    sheet = await _untidy_sheet(asc_state, "untidy-labels")

    data = await apply_ops(
        asc_state,
        sheet,
        [
            {"op": "add_component", "reference": "R6", "symbol": "res", "x": 1300, "y": 300},
            {"op": "add_net_label", "net": "n6", "pin": "R6.1"},
        ],
    )

    wiring = data["wiring"]
    assert [row["pin"] for row in wiring["label_only_pins"]["items"]] == ["R6.1"]
    # The metric stays whole-sheet; the list is what this edit did, and the
    # difference is counted rather than dropped.
    assert wiring["pins_label_only"] == _UNTIDY_LABEL_ONLY + 1
    assert (
        wiring["label_only_pins"]["total"] + data["preexisting"]["label_only_pins"]
        == wiring["pins_label_only"]
    )
    _, label_only = _whole_sheet(asc_state, sheet)
    assert len(label_only) == wiring["pins_label_only"]


async def test_asking_for_the_preexisting_view_returns_the_rest_in_the_same_call(asc_state):
    sheet = await _untidy_sheet(asc_state, "untidy-ask")

    data = await apply_ops(
        asc_state,
        sheet,
        [{"op": "add_component", "reference": "R5", "symbol": "res", "x": 1100, "y": 300}],
        return_views=["touched", "preexisting"],
    )

    page = data["views"]["preexisting"]
    assert page["total"] == data["preexisting"]["count"]
    assert "listed in views.preexisting" in data["hint"]
    assert page["next_cursor"] is None
    findings = [row["message"] for row in page["items"] if row["kind"] != "label_only_pin"]
    pins = [row["pin"] for row in page["items"] if row["kind"] == "label_only_pin"]
    assert sorted(pins) == ["R3.1", "R4.1"]
    whole, _ = _whole_sheet(asc_state, sheet)
    assert sorted(findings + _sheet_findings(data)) == sorted(whole)


async def test_the_preexisting_cursor_pages_every_withheld_row(asc_state):
    """Echoing preexisting.cursor is the request: no return_views entry needed.
    An op-less read reports nothing as new, so its pages cover the whole sheet."""
    sheet = await _untidy_sheet(asc_state, "untidy-page")
    edit = await apply_ops(
        asc_state,
        sheet,
        [{"op": "add_component", "reference": "R5", "symbol": "res", "x": 1100, "y": 300}],
    )

    rows: list[dict] = []
    cursor = edit["preexisting"]["cursor"]
    for _ in range(10):
        page = await apply_ops(
            asc_state, sheet, [], view_cursors={"preexisting": cursor}, view_limit=3
        )
        assert page["commit_state"] == "not_committed"
        view = page["views"]["preexisting"]
        assert view["returned"] <= 3
        rows += view["items"]
        cursor = view["next_cursor"]
        if cursor is None:
            break

    whole, label_only = _whole_sheet(asc_state, sheet)
    assert sorted(r["message"] for r in rows if r["kind"] != "label_only_pin") == sorted(whole)
    assert sorted(r["pin"] for r in rows if r["kind"] == "label_only_pin") == sorted(label_only)
    assert len(rows) == len(whole) + len(label_only)


async def test_an_echoed_cursor_returns_its_view_without_a_return_views_entry(asc_state):
    """The cursor is the request. A pin_legend cursor sent with the default
    return_views used to be validated and then ignored, so the page it asked
    for never came back."""
    built = await _build_blank(
        asc_state,
        "cursor-implies",
        _many_labeled_ops(5),
        return_views=["pin_legend"],
        view_limit=2,
    )
    sheet = Path(asc_state.working_dir) / "cursor-implies.asc"

    page = await apply_ops(
        asc_state,
        sheet,
        [],
        view_cursors={"pin_legend": built["views"]["pin_legend"]["next_cursor"]},
        view_limit=2,
    )

    assert [row["ref"] for row in page["views"]["pin_legend"]["items"]] == ["R2", "R3"]


async def test_a_label_only_pin_is_named_by_the_labels_at_its_coordinate(asc_state):
    """A label-only pin is on no wire, so its net holds only what sits at its
    coordinate: the label-only list names it from the labels there, without
    tracing the sheet, and must agree with the traced whole-sheet legend."""
    await _build_blank(
        asc_state,
        "label-only-nets",
        # The same name on the wired net as on the label-only pin: it may not
        # leak from one into the other's net name.
        [*_DIVIDER_OPS, {"op": "add_net_label", "net": "vout", "x": 550, "y": 200}],
    )
    sheet = Path(asc_state.working_dir) / "label-only-nets.asc"
    # A second name on the label-only pin. The tool refuses to write one (it
    # shorts two nets), so only a hand-written sheet carries it.
    with sheet.open("a", newline="\n") as handle:
        handle.write("FLAG 400 348 alias\n")

    data = await apply_ops(asc_state, sheet, [], return_views=["pin_legend", "preexisting"])

    traced = {
        f"{row['ref']}.{pin['name']}": pin["net"]
        for row in data["views"]["pin_legend"]["items"]
        for pin in row["pins"]
    }
    rows = [r for r in data["views"]["preexisting"]["items"] if r["kind"] == "label_only_pin"]
    assert [(row["pin"], row["net"]) for row in rows] == [("R1.2", "alias/vout")]
    assert all(row["net"] == traced[row["pin"]] for row in rows)


async def test_malformed_preexisting_cursor_is_rejected_before_the_sheet_is_written(asc_state):
    sheet = await _untidy_sheet(asc_state, "untidy-badcursor")
    before = sheet.read_bytes()
    with pytest.raises(NetlistError, match="preexisting"):
        await apply_ops(
            asc_state,
            sheet,
            [{"op": "set_component_value", "reference": "R1", "value": "2k"}],
            view_cursors={"preexisting": "tampered"},
        )
    assert sheet.read_bytes() == before


async def test_the_python_seam_scopes_the_same_way_and_completes_the_view(asc_state):
    """One evaluator: the API gets the MCP's scoped lists, and the withheld rows
    complete rather than paged."""
    sheet = await _untidy_sheet(asc_state, "untidy-api")
    args = _edit_input(
        target=sheet.name,
        expected_sha256=_sha(sheet),
        dry_run=True,
        return_views=["preexisting"],
        view_limit=1,
        ops=[{"op": "add_component", "reference": "R5", "symbol": "res", "x": 1100, "y": 300}],
    )

    complete = complete_edit_schematic_data(await evaluate_edit_schematic(args, asc_state), args)

    assert _sheet_findings(complete) == [
        "Floating pin: R5.1 at (1100,252)",
        "Floating pin: R5.2 at (1100,348)",
    ]
    view = complete["views"]["preexisting"]
    assert view["total"] == view["returned"] == _UNTIDY_FINDINGS + _UNTIDY_LABEL_ONLY
    assert view["truncated"] is False
    assert complete["preexisting"]["count"] == view["total"]


# ---------------------------------------------------------------------------
# Paginated views + cursor resumption
# ---------------------------------------------------------------------------


def _many_labeled_ops(n: int) -> list[dict]:
    ops: list[dict] = []
    for i in range(n):
        ref = f"R{i}"
        ops.append(
            {
                "op": "add_component",
                "reference": ref,
                "symbol": "res",
                "x": 200 + 200 * i,
                "y": 300,
            }
        )
        ops.append({"op": "add_net_label", "net": f"n{i}", "pin": f"{ref}.1"})
    return ops


async def test_view_pagination_cursor_resumption(asc_state):
    data = await _build_blank(
        asc_state, "pages", _many_labeled_ops(5), return_views=["pin_legend"], view_limit=2
    )
    # label_only_pins page 1
    lo1 = data["wiring"]["label_only_pins"]
    assert lo1["total"] == 5
    assert lo1["returned"] == 2
    assert lo1["truncated"] is True
    assert lo1["next_cursor"]
    # pin_legend page 1
    pl1 = data["views"]["pin_legend"]
    assert pl1["total"] == 5
    assert pl1["returned"] == 2

    seen_refs = [it["ref"] for it in pl1["items"]]
    seen_pins = [it["pin"] for it in lo1["items"]]
    cursor_lo = lo1["next_cursor"]
    cursor_pl = pl1["next_cursor"]
    # Walk the rest through cursor resumption.
    for _ in range(5):
        page = _assert_schema(
            await handle_edit_schematic(
                _edit_input(
                    target="pages.asc",
                    expected_sha256=_sha(Path(asc_state.working_dir) / "pages.asc"),
                    dry_run=True,
                    return_views=["pin_legend"],
                    view_limit=2,
                    view_cursors={"label_only_pins": cursor_lo, "pin_legend": cursor_pl},
                    # Ops that name every part without moving one keep the paged
                    # list identical across resume calls: add_net_label would change
                    # label_only membership, and an op naming none of the parts
                    # would leave their label-only pins counted under preexisting.
                    ops=[
                        {"op": "set_component_value", "reference": f"R{i}", "value": "1k"}
                        for i in range(5)
                    ],
                ),
                asc_state,
            )
        )
        seen_pins += [it["pin"] for it in page["wiring"]["label_only_pins"]["items"]]
        seen_refs += [it["ref"] for it in page["views"]["pin_legend"]["items"]]
        cursor_lo = page["wiring"]["label_only_pins"]["next_cursor"]
        cursor_pl = page["views"]["pin_legend"]["next_cursor"]
        if cursor_lo is None and cursor_pl is None:
            break
    # Full coverage, no duplicates.
    assert len(seen_pins) == 5 and len(set(seen_pins)) == 5
    assert len(seen_refs) == 5 and len(set(seen_refs)) == 5


async def test_cross_view_cursor_rejected(asc_state):
    data = await _build_blank(asc_state, "xcursor", _many_labeled_ops(5), view_limit=2)
    legend_cursor = data["wiring"]["label_only_pins"]["next_cursor"]
    # Feeding a label_only_pins cursor to pin_legend is rejected up front.
    with pytest.raises(NetlistError, match="pin_legend"):
        await handle_edit_schematic(
            _edit_input(
                target="xcursor.asc",
                expected_sha256=_sha(Path(asc_state.working_dir) / "xcursor.asc"),
                dry_run=True,
                view_cursors={"pin_legend": legend_cursor},
                ops=[{"op": "add_net_label", "net": "spare", "pin": "R0.2"}],
            ),
            asc_state,
        )


async def test_malformed_touched_cursor_is_rejected_before_the_sheet_is_written(asc_state):
    """A bad views.touched cursor must be caught with the other arguments.

    The touched view is paged after the commit, so a cursor only checked there
    rejects the call with the sheet already rewritten — the exact failure the
    up-front check exists to prevent.
    """
    await _build_blank(asc_state, "touchedcursor", _DIVIDER_OPS)
    sheet = Path(asc_state.working_dir) / "touchedcursor.asc"
    before = sheet.read_bytes()

    with pytest.raises(NetlistError, match="touched"):
        await handle_edit_schematic(
            _edit_input(
                target="touchedcursor.asc",
                expected_sha256=_sha(sheet),
                return_views=["touched"],
                view_cursors={"touched": "tampered"},
                ops=[{"op": "add_net_label", "net": "spare", "pin": "R2.2"}],
            ),
            asc_state,
        )
    assert sheet.read_bytes() == before


# ---------------------------------------------------------------------------
# Views this tool does not have
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["render", "occupancy"])
async def test_a_view_this_tool_does_not_have_is_rejected(asc_state, view: str):
    """Strict validation rejects the variant at the schema, so no dead stub
    path can exist behind it.

    'render' is verify_circuit's — its policy carries a pixel cap, a delivery
    channel and a render-only mode this one never had — and 'occupancy' was
    measured against the plain views and produced no placement defect either
    way, so it was never shipped."""
    with pytest.raises(ValidationError):
        await _build_blank(asc_state, f"noview-{view}", _DIVIDER_OPS, return_views=[view])


async def test_the_edit_response_carries_no_render_block(asc_state):
    data = await _build_blank(asc_state, "norender", _DIVIDER_OPS)
    assert "render" not in data["views"]
    # Rendering was this tool's only artifact producer, so the key went with it.
    assert "artifacts" not in data


@pytest.mark.parametrize(
    "payload",
    [
        {"render": True},
        {"render_format": "svg"},
        {"render_scale": 2.0},
        {"reference": "ref.cir"},
        {"write_failed_draft": True},
        {"format": "json"},
    ],
    ids=["render", "render-format", "render-scale", "flat-reference", "draft", "format"],
)
def test_the_arguments_this_tool_no_longer_takes(payload: dict):
    """Each of these had somewhere better to be, or nothing to do.

    Drawing belongs to verify_circuit; the flat 'reference' was a second
    spelling of 'compare'; a failed batch writes nothing by design, so there
    was no draft to quarantine that the caller's own ops did not already
    describe; and 'format' chose a text rendering that structured-aware
    clients drop, which the other six tools stopped advertising.
    """
    with pytest.raises(ValidationError):
        _edit_input(target="gone.asc", base="blank", ops=_DIVIDER_OPS, **payload)


# ---------------------------------------------------------------------------
# Post-commit reference stage
# ---------------------------------------------------------------------------

_REF_DECK = "Vin in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n"
_REF_DECK_DIFFERENT = "Vin in 0 5\nR1 in out 1k\nR2 out mid 2k\nR3 mid 0 3k\n.end\n"


async def test_reference_success(asc_state, work_dir, monkeypatch):
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def fake_export(_copy, _state):
        return _REF_DECK

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(asc_state, "refok", _DIVIDER_OPS, compare={"reference": "ref.cir"})
    assert data["commit_state"] == "committed"
    assert data["verification"]["equivalent"] is True
    # The verdict is the answer; the exported deck only confirms it.
    assert "netlist" not in data
    assert data["stages"][-1] == {"stage": "reference", "ok": True}


async def test_an_equivalent_comparison_leaves_the_netlist_out(asc_state, monkeypatch):
    """A confirmed match does not echo the exported deck.

    Every compare used to return the committed sheet's whole netlist, even
    when the verdict was ``equivalent: true``. That deck is equivalent to the
    reference the caller supplied, so it carried no fact the verdict did not,
    and every later turn re-read it.
    """

    async def fake_export(_copy, _state):
        return _REF_DECK

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(asc_state, "reflean", _DIVIDER_OPS, compare={"reference": _REF_DECK})
    assert data["verification"]["equivalent"] is True
    assert data["outcome"] == "complete"
    assert "netlist" not in data


async def test_reference_may_be_netlist_text(asc_state, monkeypatch):
    """The shared compare spec reads a multi-line reference as netlist text;
    this tool honours that the same way verify_circuit does."""

    async def fake_export(_copy, _state):
        return _REF_DECK

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(asc_state, "reftext", _DIVIDER_OPS, compare={"reference": _REF_DECK})
    assert data["commit_state"] == "committed"
    assert data["verification"]["equivalent"] is True
    assert data["verification"]["reference"] == "inline netlist"


async def test_reference_outside_sandbox_names_the_text_alternative(asc_state):
    """A reference written outside allowed paths is rejected before anything is
    written, and the rejection says the deck's text can be passed instead."""
    with pytest.raises(PathSecurityError, match="netlist text itself"):
        await _build_blank(
            asc_state, "refout", _DIVIDER_OPS, compare={"reference": "/outside/ref.cir"}
        )


async def test_reference_structural_diff_reports_the_delta(asc_state, monkeypatch):
    """The compare argument is verify_circuit's: 'structural_diff' returns the
    added/removed/changed delta with the verdict derived from it."""

    async def fake_export(_copy, _state):
        return _REF_DECK_DIFFERENT

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(
        asc_state,
        "refdiff",
        _DIVIDER_OPS,
        compare={"reference": _REF_DECK, "mode": "structural_diff"},
    )
    assert data["commit_state"] == "committed"
    comparison = data["verification"]["comparison"]
    assert comparison["mode"] == "structural_diff"
    assert comparison["components_added"] == ["R3"]
    # A node rewire is not a value change: structural_diff lists it under neither
    # 'changed' nor 'removed'; equivalence mode is the one that catches it.
    assert comparison["components_removed"] == []
    assert data["verification"]["equivalent"] is False
    assert data["outcome"] == "partial"


async def test_reference_anchors_are_honoured(asc_state, monkeypatch):
    async def fake_export(_copy, _state):
        return _REF_DECK

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(
        asc_state, "refanchor", _DIVIDER_OPS, compare={"reference": _REF_DECK, "anchors": ["out"]}
    )
    assert data["verification"]["comparison"]["mode"] == "equivalence"
    assert data["verification"]["equivalent"] is True


async def test_reference_mismatch_stays_committed(asc_state, work_dir, monkeypatch):
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def fake_export(_copy, _state):
        return _REF_DECK_DIFFERENT

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(asc_state, "refbad", _DIVIDER_OPS, compare={"reference": "ref.cir"})
    assert data["commit_state"] == "committed"
    assert data["verification"]["equivalent"] is False
    # A difference is data, not a failure: the sheet stays committed.
    assert (work_dir / "refbad.asc").is_file()
    # The sheet's exported deck is the side of the difference to diagnose from.
    assert data["netlist"] == _REF_DECK_DIFFERENT


async def test_reference_mismatch_is_a_partial_outcome(asc_state, work_dir, monkeypatch):
    """A committed sheet that does not match its reference is not a clean call.

    verify_circuit already reports a non-equivalent comparison as ``partial``;
    the same comparison reached through edit_schematic's reference stage must
    say the same thing, or the caller is told the same mismatch is a shortfall
    on one tool and a clean result on the other.
    """
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def fake_export(_copy, _state):
        return _REF_DECK_DIFFERENT

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(
        asc_state, "refpartial", _DIVIDER_OPS, compare={"reference": "ref.cir"}
    )
    assert data["verification"]["equivalent"] is False
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"


async def test_reference_export_failure_is_a_partial_outcome(asc_state, work_dir, monkeypatch):
    """An unexportable sheet leaves the comparison with no verdict at all.

    ``equivalent: null`` is the absence of a result, so it keeps the call off
    ``complete`` the same way a real difference does.
    """
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def boom_export(_copy, _state):
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(se, "_export_asc_to_netlist", boom_export)
    data = await _build_blank(
        asc_state, "refnoverdict", _DIVIDER_OPS, compare={"reference": "ref.cir"}
    )
    assert data["verification"]["equivalent"] is None
    assert data["verification"]["export_error"]
    # Nothing exported, so there is no deck to return.
    assert "netlist" not in data
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"


async def test_rejected_reference_path_refuses_before_committing(asc_state, work_dir):
    """A reference outside allowed_paths is an argument fault, caught pre-commit.

    Resolving it in the post-commit stage instead wrote the sheet and then
    aborted with a bare raise, so the caller was told only "error" and never
    learned the new sha of the file it had just committed.
    """
    with pytest.raises(PathSecurityError):
        await _build_blank(
            asc_state,
            "denied_ref",
            _DIVIDER_OPS,
            compare={"reference": "/etc/ltspice-mcp-not-allowed.cir"},
        )
    # Nothing was written: the rejection lands before the commit protocol runs.
    assert not (work_dir / "denied_ref.asc").exists()
    assert not list(work_dir.glob("denied_ref.asc.staging-*"))


_ADD_R2 = [{"op": "add_component", "reference": "R2", "symbol": "res", "x": 400, "y": 96}]


@pytest.mark.parametrize("mode", ["equivalence", "structural_diff"])
async def test_asc_reference_is_exported_like_the_committed_sheet(
    asc_state, work_dir, monkeypatch, mode
):
    """An additive edit compared with the sheet it started from reports the
    addition and nothing else.

    The committed sheet reaches the comparison as its LTspice export, so the
    reference .asc has to as well: read through the schematic editor instead,
    every SpiceLine read as a changed component and the exporter's boilerplate
    as added directives, and the graph engine lexed the schematic as SPICE.
    """
    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_netlister.export_asc_to_netlist)
    sheet = work_dir / "amp.asc"
    sheet.write_text(fake_netlister.amp_asc(), newline="\n")
    original = work_dir / "amp_orig.asc"
    original.write_text(fake_netlister.amp_asc(), newline="\n")

    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="amp.asc",
                expected_sha256=_sha(sheet),
                ops=_ADD_R2,
                compare={"reference": "amp_orig.asc", "mode": mode},
            ),
            asc_state,
        )
    )

    assert data["commit_state"] == "committed"
    verification = data["verification"]
    assert verification.get("compare_error") is None
    comparison = verification["comparison"]
    if mode == "structural_diff":
        assert comparison["components_added"] == ["R2"]
        assert comparison["components_removed"] == []
        assert comparison["components_changed"] == []
        assert comparison["directives_added"] == []
        assert comparison["directives_removed"] == []
    else:
        assert [c["ref"] for c in comparison["added"]] == ["R2"]
        assert comparison["removed"] == []
        assert comparison["value_mismatches"] == []
    assert verification["equivalent"] is False
    # The reference was exported from a copy; nothing was written beside it.
    assert not (work_dir / "amp_orig.net").exists()


async def test_unexportable_asc_reference_is_a_compare_error(asc_state, work_dir, monkeypatch):
    """The committed sheet exported, the reference did not: nothing to compare
    against, so no verdict — and the sheet stays committed."""
    sheet = work_dir / "amp.asc"
    sheet.write_text(fake_netlister.amp_asc(), newline="\n")
    (work_dir / "amp_orig.asc").write_text(fake_netlister.amp_asc(), newline="\n")

    async def export_only_the_committed_sheet(asc_copy: Path, state: SessionState) -> str:
        netlist = await fake_netlister.export_asc_to_netlist(asc_copy, state)
        if "R2" not in netlist:  # the reference: the sheet before R2 was added
            raise RuntimeError("injected reference export failure")
        return netlist

    monkeypatch.setattr(se, "_export_asc_to_netlist", export_only_the_committed_sheet)
    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="amp.asc",
                expected_sha256=_sha(sheet),
                ops=_ADD_R2,
                compare={"reference": "amp_orig.asc", "mode": "structural_diff"},
            ),
            asc_state,
        )
    )

    assert data["commit_state"] == "committed"
    verification = data["verification"]
    assert verification.get("export_error") is None
    assert "injected reference export failure" in verification["compare_error"]
    assert verification["equivalent"] is None
    assert data["outcome"] == "partial"
    # No verdict is not a match: the sheet's deck is the half the caller has.
    assert "R2" in data["netlist"]


# ---------------------------------------------------------------------------
# Post-commit escapes still return a committed envelope
# ---------------------------------------------------------------------------


async def test_post_commit_reference_error_returns_committed_envelope(
    asc_state, work_dir, monkeypatch
):
    """An exception in the reference stage reports the commit, not a bare raise."""
    (work_dir / "ref.cir").write_text(_REF_DECK)

    def boom(*_a, **_kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(se, "_write_export_copy", boom)

    data = await _build_blank(asc_state, "refboom", _DIVIDER_OPS, compare={"reference": "ref.cir"})
    committed = work_dir / "refboom.asc"
    assert committed.is_file()
    # The caller learns the commit happened AND the sha it must edit against next.
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"
    assert data["sha256"] == _sha(committed)
    assert data["error"]["code"] == "post_commit_failed"
    assert data["error"]["stage"] == "reference"
    assert data["error"]["commit_state"] == "committed"
    assert "no space left on device" in data["error"]["message"]
    assert data["stages"][-1] == {
        "stage": "reference",
        "ok": False,
        "error": "no space left on device",
    }
    # And that sha is usable: the follow-up edit does not hit revision_conflict.
    nxt = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="refboom.asc",
                expected_sha256=data["sha256"],
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R3",
                        "symbol": "res",
                        "x": 1000,
                        "y": 300,
                    }
                ],
            ),
            asc_state,
        )
    )
    assert nxt["outcome"] == "complete"


async def test_failure_after_a_completed_stage_is_not_reported_against_it(
    asc_state, work_dir, monkeypatch
):
    """One stage, one verdict: a later failure must not overwrite an earlier ok.

    The reference stage records its own outcome. An exception raised after it —
    in hint or envelope assembly — used to append a SECOND 'reference' entry with
    ok=false, leaving two entries for one stage with opposite verdicts and no way
    to tell the caller which of the two actually happened.
    """
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def fake_export(_copy, _state):
        return _REF_DECK

    def boom(*_a, **_kw):
        raise RuntimeError("hint assembly blew up")

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    monkeypatch.setattr(se, "_commit_hint", boom)

    data = await _build_blank(
        asc_state, "hintboom", _DIVIDER_OPS, compare={"reference": "ref.cir"}
    )
    assert data["commit_state"] == "committed"
    assert data["outcome"] == "partial"
    assert [s for s in data["stages"] if s["stage"] == "reference"] == [
        {"stage": "reference", "ok": True}
    ]
    assert data["stages"][-1] == {
        "stage": "response",
        "ok": False,
        "error": "hint assembly blew up",
    }
    assert data["error"]["stage"] == "response"


async def test_post_commit_view_error_returns_committed_envelope(asc_state, work_dir, monkeypatch):
    """A view-assembly exception is post-commit too, and reports the same way."""

    def boom(*_a, **_kw):
        raise RuntimeError("legend blew up")

    monkeypatch.setattr(se, "paginate_view", boom)

    result = await handle_edit_schematic(
        _edit_input(target="viewboom.asc", base="blank", ops=_DIVIDER_OPS), asc_state
    )
    data = _assert_schema(result)
    committed = work_dir / "viewboom.asc"
    assert committed.is_file()
    assert data["commit_state"] == "committed"
    assert data["sha256"] == _sha(committed)
    assert data["error"]["stage"] == "views"
    assert "legend blew up" in data["error"]["message"]


# ---------------------------------------------------------------------------
# Op-failure transaction abort
# ---------------------------------------------------------------------------


async def test_bad_op_aborts_transaction_nothing_written(asc_state, work_dir):
    data = _assert_schema(
        await handle_edit_schematic(
            _edit_input(
                target="abort.asc",
                base="blank",
                ops=[
                    {
                        "op": "add_component",
                        "reference": "R1",
                        "symbol": "res",
                        "x": 400,
                        "y": 300,
                    },
                    {"op": "set_component_value", "reference": "GHOST", "value": "1k"},
                ],
            ),
            asc_state,
        )
    )
    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    assert data["failures"][0]["op"] == "set_component_value"
    # Full error envelope: an op-application failure is a validation fault, so it
    # is not retryable, and nothing was written.
    assert data["error"]["code"] == "op_failed"
    assert data["error"]["stage"] == "apply_ops"
    assert data["error"]["retryable"] is False
    assert data["error"]["commit_state"] == "not_started"
    assert not (work_dir / "abort.asc").exists()


# ---------------------------------------------------------------------------
# Archetype-scale blank build (ops-at-scale through base:"blank")
# ---------------------------------------------------------------------------

_ARCHETYPE_OPS: list[dict] = [
    {"op": "add_component", "reference": "M1", "symbol": "nmos", "x": 400, "y": 300},
    {"op": "add_component", "reference": "D1", "symbol": "diode", "x": 700, "y": 300},
    {"op": "add_component", "reference": "E1", "symbol": "e", "x": 1000, "y": 300},
    {"op": "add_component", "reference": "G1", "symbol": "g", "x": 1300, "y": 300},
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 400, "y": 600},
    {"op": "add_component", "reference": "R2", "symbol": "res", "x": 700, "y": 600},
    {"op": "add_component", "reference": "C1", "symbol": "cap", "x": 1000, "y": 600},
]


async def test_archetype_scale_blank_build(asc_state, work_dir):
    """The >2-pin classes (MOSFET, controlled sources) build through base:"blank"."""
    data = await _build_blank(
        asc_state, "arch", _ARCHETYPE_OPS, return_views=["pin_legend"], view_limit=50
    )
    assert data["outcome"] == "complete"
    legend = {
        e["ref"]: {p["name"] for p in e["pins"]} for e in data["views"]["pin_legend"]["items"]
    }
    assert legend["M1"] == {"D", "G", "S"}
    assert legend["D1"] == {"A", "K"}
    assert legend["E1"] == {"+", "-", "P", "N"}
    assert legend["G1"] == {"+", "-", "NC+", "NC-"}
    # All seven landed in the committed file.
    text = (work_dir / "arch.asc").read_text()
    for ref in ("M1", "D1", "E1", "G1", "R1", "R2", "C1"):
        assert ref in text


class TestRenderAndCompareArguments:
    """What this tool accepts for `compare`, and that it accepts no render.

    Rendering belongs to verify_circuit alone: its policy has a pixel cap, a
    delivery channel and a render-only mode, so an edit that also wants a
    picture is one call away from the better tool. Asserted on the RESOLVED
    value, not on the raw payload.
    """

    @staticmethod
    def _edit(**kwargs: Any) -> EditSchematicInput:
        return EditSchematicInput.model_validate({"target": "sheet.asc", "ops": [], **kwargs})

    @pytest.mark.parametrize(
        "payload",
        [{"render": True}, {"render": {"format": "svg"}}, {"return_views": ["render"]}],
        ids=["bool", "policy", "view-name"],
    )
    def test_no_render_spelling_is_accepted(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            self._edit(**payload)

    def test_no_compare_argument_means_no_comparison(self):
        assert self._edit().compare is None

    def test_the_object_names_the_reference(self):
        spec = self._edit(compare={"reference": "golden.cir"}).compare
        assert spec is not None
        assert spec.reference == "golden.cir"
        # The tolerance and anchors keep the graph engine's own defaults.
        assert spec.rtol == 1e-6
        assert spec.anchors is None

    def test_the_object_carries_the_comparison_controls(self):
        spec = self._edit(compare={"reference": "g.cir", "rtol": 1e-3, "anchors": ["out"]}).compare
        assert spec is not None
        assert (spec.rtol, spec.anchors) == (1e-3, ["out"])

    @pytest.mark.parametrize(
        "payload",
        [{"compare": {"rtol": 1e-3}}, {"compare": {"reference": "g.cir", "mode": "x"}}],
        ids=["no-reference", "verify-only-mode"],
    )
    def test_refused_compare_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            self._edit(**payload)


class TestHierarchicalPortPreservation:
    @staticmethod
    def _sheet(work_dir: Path) -> Path:
        path = work_dir / "ports.asc"
        path.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "FLAG 0 0 IN\nIOPIN 0 0 In\n"
            "FLAG 160 0 OUT\nIOPIN 160 0 Out\n"
            "FLAG 320 0 IN\nIOPIN 320 0 BiDir\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    async def test_an_ordinary_edit_keeps_ordered_ports(self, state_no_sim, work_dir):
        path = self._sheet(work_dir)
        result = await handle_edit_schematic(
            _edit_input(
                target=str(path),
                expected_sha256=_sha(path),
                ops=[{"op": "add_directive", "instruction": ".param marker=1"}],
            ),
            state_no_sim,
        )
        assert _assert_schema(result)["commit_state"] == "committed"
        reopened = AscEditor(path)
        assert [(p.text.text, p.text.coord.X, p.direction) for p in reopened.ports] == [
            ("IN", 0, "In"),
            ("OUT", 160, "Out"),
            ("IN", 320, "BiDir"),
        ]
        assert ".param marker=1" in path.read_text()

    async def test_removing_a_port_label_refuses_before_write(self, state_no_sim, work_dir):
        path = self._sheet(work_dir)
        original = path.read_bytes()
        with pytest.raises(NetlistError, match="port"):
            await handle_edit_schematic(
                _edit_input(
                    target=str(path),
                    expected_sha256=_sha(path),
                    ops=[{"op": "remove_net_label", "x": 0, "y": 0}],
                ),
                state_no_sim,
            )
        assert path.read_bytes() == original

    def test_ports_follow_their_label_objects(self, work_dir):
        editor = AscEditor(self._sheet(work_dir))
        editor.labels[0].text = "RENAMED"
        editor.labels[0].coord.X = 64
        before = [(id(p.text), p.text.text, p.text.coord.X, p.direction) for p in editor.ports]
        rendered = se._render_editor_text(editor)
        assert "FLAG 64 0 RENAMED\nIOPIN 64 0 In\n" in rendered
        assert "FLAG 320 0 IN\nIOPIN 320 0 BiDir\n" in rendered
        assert [
            (id(p.text), p.text.text, p.text.coord.X, p.direction) for p in editor.ports
        ] == before

    @pytest.mark.parametrize("invalid", ["duplicate_label", "duplicate_port", "orphan"])
    def test_ambiguous_port_associations_refuse(self, work_dir, invalid):
        editor = AscEditor(self._sheet(work_dir))
        if invalid == "duplicate_label":
            editor.labels.append(editor.labels[0])
        elif invalid == "duplicate_port":
            editor.ports.append(editor.ports[0])
        else:
            editor.labels.pop(0)
        with pytest.raises(NetlistError, match="port"):
            se._render_editor_text(editor)

    @pytest.mark.parametrize("dry_run", [False, True])
    @pytest.mark.parametrize("pending_level", [None, "child", "grandchild"])
    async def test_pending_child_changes_never_write_through_parent(
        self,
        asc_state,
        work_dir,
        dry_run,
        pending_level,
    ):
        child = self._sheet(work_dir)
        (work_dir / "ports.asy").write_text(
            "Version 4\nSymbolType BLOCK\nRECTANGLE Normal -32 -32 32 32\n"
            "PIN -32 0 LEFT 8\nPINATTR PinName IN\nPINATTR SpiceOrder 1\n"
            "PIN 32 0 RIGHT 8\nPINATTR PinName OUT\nPINATTR SpiceOrder 2\n",
            encoding="utf-8",
            newline="\n",
        )
        outer = work_dir / "outer.asc"
        outer.write_text(
            "Version 4\nSHEET 1 880 680\n"
            "FLAG -32 0 IN\nIOPIN -32 0 In\n"
            "FLAG 32 0 OUT\nIOPIN 32 0 Out\n"
            "SYMBOL ports 0 0 R0\nSYMATTR InstName X2\n",
            encoding="utf-8",
            newline="\n",
        )
        (work_dir / "outer.asy").write_bytes((work_dir / "ports.asy").read_bytes())
        parent = work_dir / "parent.asc"
        parent.write_text(
            "Version 4\nSHEET 1 880 680\nSYMBOL outer 0 0 R0\nSYMATTR InstName X1\n",
            encoding="utf-8",
            newline="\n",
        )
        editor = get_asc_editor(parent, asc_state)
        loaded_child = editor.get_subcircuit("X1")
        grandchild = loaded_child.get_subcircuit("X2")
        changed = loaded_child if pending_level == "child" else grandchild
        if pending_level is not None:
            changed.set_parameter("child_change", 1)
            assert changed.updated
        if pending_level != "child":
            assert not loaded_child.updated
        before = {p: p.read_bytes() for p in (parent, outer, child)}
        request = _edit_input(
            target=str(parent),
            expected_sha256=_sha(parent),
            dry_run=dry_run,
            ops=[{"op": "add_directive", "instruction": ".param parent_change=1"}],
        )
        if pending_level is None:
            data = _assert_schema(await handle_edit_schematic(request, asc_state))
            assert data["commit_state"] == ("not_committed" if dry_run else "committed")
            assert outer.read_bytes() == before[outer]
            assert child.read_bytes() == before[child]
            if dry_run:
                assert parent.read_bytes() == before[parent]
            assert not loaded_child.updated and not grandchild.updated
            return
        with pytest.raises(NetlistError, match=r"child|descendant"):
            await handle_edit_schematic(request, asc_state)
        assert {p: p.read_bytes() for p in (parent, outer, child)} == before
        assert changed.updated
