"""Tests for edit_schematic — the transactional, revision-guarded .asc author tool.

Covers the revision guard (sha match / mismatch / missing), the commit protocol
(staged write + rename-last, with crash injection before and after the rename),
the parity of a base:"blank" build against create_schematic + apply_schematic_ops,
the wiring metric + paginated pin_legend / label_only_pins views, the SVG/PNG
render view (present and forced-absent raster), the post-commit reference stage
(success / mismatch / export failure), and an archetype-scale blank build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jsonschema
import pytest
from pydantic import TypeAdapter, ValidationError

from ltspice_mcp.errors import NetlistError, PathSecurityError
from ltspice_mcp.lib import raster
from ltspice_mcp.lib.schematic_ops import (
    get_asc_editor,
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
    control.write_text(se._BLANK_TEMPLATE)
    editor = get_asc_editor(control, asc_state)
    ops = _OPS_ADAPTER.validate_python(_DIVIDER_OPS)
    _, abort = run_op_batch(editor, ops, control, stop_on_error=True)
    assert abort is None
    editor.save_netlist(control)

    assert (work_dir / "parity_edit.asc").read_text() == control.read_text()


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


async def test_crash_before_rename_leaves_target_and_writes_draft(
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
                write_failed_draft=True,
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
    # Quarantine draft written and named for the build.
    draft = work_dir / f"crash.draft-{data['build_id']}.asc"
    assert draft.is_file()
    assert "R3" in draft.read_text()
    # No staging temp left behind.
    assert not list(work_dir.glob("crash.asc.staging-*"))


async def test_crash_after_rename_stays_committed(asc_state, work_dir, monkeypatch):
    (work_dir / "ref.cir").write_text("Vin in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n")

    async def boom_export(_copy, _state):
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(se, "_export_asc_to_netlist", boom_export)

    data = await _build_blank(asc_state, "aftercommit", _DIVIDER_OPS, reference="ref.cir")
    # The rename succeeded, so the sheet is committed even though a post-rename
    # (reference-export) stage failed. The failed stage is the shortfall that
    # keeps the call off "complete".
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"
    assert (work_dir / "aftercommit.asc").is_file()
    assert data["verification"]["export_error"]
    assert data["verification"]["equivalent"] is None


# ---------------------------------------------------------------------------
# connect exclusion (ID-15)
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
                return_views=["pin_legend", "render"],
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
    assert data["views"]["render"]["status"] == "dry_run"
    # Directory fingerprint and target sha both unchanged.
    assert _fingerprint(work_dir) == before
    assert _sha(work_dir / "dry.asc") == sha0
    assert not (work_dir / ".ltspice-mcp" / "renders").exists()


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
    rendered_sources: list[str] = []
    original_render = se._render_view

    def commit_then_peer_revision(text, target, build_id, encoding):
        outcome = original_commit(text, target, build_id, encoding)
        if outcome.renamed:
            target.write_text("Version 4.1\nSHEET 1 880 680\nTEXT 32 32 Left 2 ;peer revision\n")
        return outcome

    def record_render_source(path, *args, **kwargs):
        rendered_sources.append(path.read_text())
        return original_render(path, *args, **kwargs)

    monkeypatch.setattr(se, "_commit_asc", commit_then_peer_revision)
    monkeypatch.setattr(se, "_render_view", record_render_source)
    args = _edit_input(
        target="interleaved.asc",
        base="blank",
        return_views=["pin_legend", "render"],
        render_format="svg",
        ops=_DIVIDER_OPS,
    )
    neutral = await evaluate_edit_schematic(args, asc_state)
    assert neutral.views is not None
    assert neutral.data["sha256"] == neutral.views.sha256
    assert neutral.views.sha256 != _sha(work_dir / "interleaved.asc")
    assert {row["ref"] for row in neutral.views.pin_legend} == {"R1", "R2"}
    assert rendered_sources and "SYMATTR InstName R1" in rendered_sources[0]
    assert "peer revision" not in rendered_sources[0]

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
                    # A no-op-on-geometry op keeps the paged list identical across
                    # resume calls (add_net_label would change label_only membership).
                    ops=[{"op": "add_directive", "instruction": ".op"}],
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


# ---------------------------------------------------------------------------
# Render view
# ---------------------------------------------------------------------------


async def test_render_svg(asc_state, work_dir):
    data = await _build_blank(
        asc_state, "rsvg", _DIVIDER_OPS, return_views=["render"], render_format="svg"
    )
    render = data["views"]["render"]
    assert render["image_format"] == "svg"
    assert render["path"].endswith(".svg")
    assert Path(render["path"]).is_file()  # noqa: ASYNC240
    assert data["artifacts"][0]["kind"] == "render"


@pytest.mark.skipif(not raster.raster_available(), reason="cairosvg not installed")
async def test_render_png_when_raster_available(asc_state):
    data = await _build_blank(
        asc_state, "rpng", _DIVIDER_OPS, return_views=["render"], render_format="png"
    )
    render = data["views"]["render"]
    assert render["image_format"] == "png"
    assert render["path"].endswith(".png")
    assert render["width"] and render["height"]


async def test_render_png_falls_back_to_svg_without_raster(asc_state, monkeypatch):
    # Force the optional dependency absent regardless of the environment.
    monkeypatch.setattr(raster, "_load_cairosvg", lambda: None)
    data = await _build_blank(
        asc_state, "rfallback", _DIVIDER_OPS, return_views=["render"], render_format="png"
    )
    render = data["views"]["render"]
    assert render["image_format"] == "svg"
    assert "raster" in (render["note"] or "")


async def test_occupancy_view_variant_is_rejected(asc_state):
    """'occupancy' is not a return view; strict validation rejects the
    variant at the schema so no dead stub path can exist behind it."""
    with pytest.raises(ValidationError):
        await _build_blank(asc_state, "occ", _DIVIDER_OPS, return_views=["occupancy"])


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
    data = await _build_blank(asc_state, "refok", _DIVIDER_OPS, reference="ref.cir")
    assert data["commit_state"] == "committed"
    assert data["verification"]["equivalent"] is True
    assert data["netlist"] == _REF_DECK
    assert data["stages"][-1] == {"stage": "reference", "ok": True}


async def test_reference_mismatch_stays_committed(asc_state, work_dir, monkeypatch):
    (work_dir / "ref.cir").write_text(_REF_DECK)

    async def fake_export(_copy, _state):
        return _REF_DECK_DIFFERENT

    monkeypatch.setattr(se, "_export_asc_to_netlist", fake_export)
    data = await _build_blank(asc_state, "refbad", _DIVIDER_OPS, reference="ref.cir")
    assert data["commit_state"] == "committed"
    assert data["verification"]["equivalent"] is False
    # A difference is data, not a failure: the sheet stays committed.
    assert (work_dir / "refbad.asc").is_file()


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
    data = await _build_blank(asc_state, "refpartial", _DIVIDER_OPS, reference="ref.cir")
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
    data = await _build_blank(asc_state, "refnoverdict", _DIVIDER_OPS, reference="ref.cir")
    assert data["verification"]["equivalent"] is None
    assert data["verification"]["export_error"]
    assert data["outcome"] == "partial"
    assert data["commit_state"] == "committed"


async def test_render_view_failure_is_a_partial_outcome(asc_state, monkeypatch):
    """A view that failed is recorded in ``failures``, so the call is partial.

    The commit itself stands; what the caller asked for and did not get is the
    render, and the outcome has to say so rather than reporting ``complete``
    beside a populated failures channel.
    """

    def boom(*_a, **_kw):
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(se, "_render_committed_text", boom)
    data = await _build_blank(asc_state, "renderfail", _DIVIDER_OPS, return_views=["render"])
    assert [f["stage"] for f in data["failures"]] == ["render"]
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
            asc_state, "denied_ref", _DIVIDER_OPS, reference="/etc/ltspice-mcp-not-allowed.cir"
        )
    # Nothing was written: the rejection lands before the commit protocol runs.
    assert not (work_dir / "denied_ref.asc").exists()
    assert not list(work_dir.glob("denied_ref.asc.staging-*"))


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

    data = await _build_blank(asc_state, "refboom", _DIVIDER_OPS, reference="ref.cir")
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

    data = await _build_blank(asc_state, "hintboom", _DIVIDER_OPS, reference="ref.cir")
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
