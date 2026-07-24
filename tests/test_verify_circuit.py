"""Tests for verify_circuit — the non-mutating AUTHOR gate over any circuit file.

Covers check-subset selection and netlist-vs-.asc applicability, both compare
modes (equivalence via the connectivity graph, structural_diff via the shipped
diff internals) over identical / reordered / value-changed / topology-changed
pairs, the ID-28 safe_path include gate (an escaping in-deck include denied and
never read, proved with a canary outside the roots), the render policy (SVG,
PNG-present, and PNG forced-absent), the sidecar export (in-place .net + diff
against the prior), .sp dispatch, findings at+subject completeness, and the
quality checks firing on label-island and text-overlap fixtures while staying
silent on a clean sheet.
"""

from __future__ import annotations

import typing
from pathlib import Path

import jsonschema
import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import raster
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import verify as vc
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit


class FakeSim:
    """Stub LTspice class; the real exporter is monkeypatched per test."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/LTspice.exe"]


def _assert_schema(result) -> dict:
    data = result.structuredContent
    assert data is not None
    jsonschema.Draft202012Validator(vc._OUTPUT_SCHEMA).validate(data)
    return data


async def _run(state: SessionState, **kw) -> dict:
    result = await handle_verify_circuit(VerifyCircuitInput.model_validate(kw), state)
    return _assert_schema(result)


def _with_ltspice(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"ltspice": FakeSim})


# ---------------------------------------------------------------------------
# Test decks
# ---------------------------------------------------------------------------

_BASE = "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n"
_REORDERED = "* divider\nR2 out 0 2k\nR1 in out 1k\nV1 in 0 5\n.end\n"
_VALUE_CHANGED = "* divider\nV1 in 0 5\nR1 in out 4k7\nR2 out 0 2k\n.end\n"
# Same component set and values, different wiring: R2 now sits across in-0
# instead of out-0. Isomorphism breaks (structural), but the component/value
# signatures are unchanged (a node-blind diff cannot see it).
_TOPOLOGY_CHANGED = "* divider\nV1 in 0 5\nR1 in out 1k\nR2 in 0 2k\n.end\n"

_RES_ASC = (
    "Version 4.1\nSHEET 1 880 680\nSYMBOL res 400 300 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\n"
)


def _write(work_dir: Path, name: str, text: str) -> Path:
    p = work_dir / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# Check-subset selection + applicability
# ---------------------------------------------------------------------------


async def test_netlist_default_runs_syntax_only(state_no_sim, work_dir):
    deck = _write(work_dir, "d.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck))
    assert data["kind"] == "netlist"
    assert data["checks_run"] == ["syntax"]
    skipped = {s["check"]: s["reason"] for s in data["checks_skipped"]}
    for check in ("symbols", "export", "layout", "quality"):
        assert "not applicable to a netlist file" in skipped[check]
    assert "no reference supplied" in skipped["compare"]


async def test_netlist_rejects_asc_only_check(state_no_sim, work_dir):
    deck = _write(work_dir, "d.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck), checks=["symbols"])
    assert data["checks_run"] == []
    reasons = {s["check"]: s["reason"] for s in data["checks_skipped"]}
    assert "not applicable to a netlist file" in reasons["symbols"]


async def test_asc_default_without_ltspice(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "s.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc))
    assert data["kind"] == "asc"
    assert set(data["checks_run"]) == {"symbols", "layout", "quality"}
    reasons = {s["check"]: s["reason"] for s in data["checks_skipped"]}
    assert "LTspice not detected" in reasons["export"]


async def test_asc_rejects_syntax_check(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "s.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), checks=["syntax"])
    reasons = {s["check"]: s["reason"] for s in data["checks_skipped"]}
    assert "not applicable to a .asc file" in reasons["syntax"]
    assert data["checks_run"] == []


async def test_sp_dispatch(state_no_sim, work_dir):
    deck = _write(work_dir, "d.sp", _BASE)
    data = await _run(state_no_sim, path=str(deck))
    assert data["kind"] == "netlist"
    assert "syntax" in data["checks_run"]
    assert data["outcome"] == "complete"


async def test_unsupported_kind_is_error(state_no_sim, work_dir):
    bad = _write(work_dir, "notes.txt", "hello")
    result = await handle_verify_circuit(VerifyCircuitInput(path=str(bad)), state_no_sim)
    data = _assert_schema(result)
    assert result.isError is True
    assert data["outcome"] == "failed"


async def test_path_denied_is_error(state_no_sim):
    result = await handle_verify_circuit(VerifyCircuitInput(path="/etc/passwd"), state_no_sim)
    data = _assert_schema(result)
    assert result.isError is True
    assert data["findings"][0]["rule_id"] == "path_denied"


# ---------------------------------------------------------------------------
# syntax findings
# ---------------------------------------------------------------------------


async def test_syntax_finding_shape(state_no_sim, work_dir):
    # R with only one node is an element-arity fault the validator catches.
    deck = _write(work_dir, "bad.cir", "* bad\nR1 a 1k\n.end\n")
    data = await _run(state_no_sim, path=str(deck))
    assert data["outcome"] == "partial"
    assert data["findings"], "a one-node resistor should yield an arity finding"
    for f in data["findings"]:
        assert f["at"]["file"] == str(deck)
        assert f["subject"]
        assert f["severity"] == "error"


# ---------------------------------------------------------------------------
# compare — equivalence
# ---------------------------------------------------------------------------


async def test_equivalence_identical_text_reference(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck), reference=_BASE)
    assert "compare" in data["checks_run"]
    assert data["comparison"]["mode"] == "equivalence"
    assert data["comparison"]["equivalent"] is True
    assert data["outcome"] == "complete"


async def test_equivalence_reordered(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _REORDERED)
    data = await _run(state_no_sim, path=str(deck), reference=str(ref))
    assert data["comparison"]["equivalent"] is True


async def test_equivalence_value_changed(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _VALUE_CHANGED)
    data = await _run(state_no_sim, path=str(deck), reference=str(ref))
    assert data["comparison"]["equivalent"] is False
    assert data["outcome"] == "partial"


async def test_equivalence_topology_changed(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _TOPOLOGY_CHANGED)
    data = await _run(state_no_sim, path=str(deck), reference=str(ref))
    assert data["comparison"]["structurally_equivalent"] is False
    assert data["comparison"]["equivalent"] is False


# ---------------------------------------------------------------------------
# compare — structural_diff
# ---------------------------------------------------------------------------


async def test_structural_identical(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _BASE)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    assert data["comparison"]["mode"] == "structural_diff"
    assert data["comparison"]["equivalent"] is True
    assert data["outcome"] == "complete"


async def test_structural_reordered(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _REORDERED)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    assert data["comparison"]["equivalent"] is True


async def test_structural_value_changed(state_no_sim, work_dir):
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _VALUE_CHANGED)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    assert data["comparison"]["equivalent"] is False
    changed = {c["reference"] for c in data["comparison"]["components_changed"]}
    assert "R1" in changed


async def test_structural_topology_change_is_node_blind(state_no_sim, work_dir):
    """A pure re-wire (same components/values) reads as equivalent under a
    node-blind structural diff — the mode difference from equivalence, on purpose."""
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _TOPOLOGY_CHANGED)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    assert data["comparison"]["equivalent"] is True


# ---------------------------------------------------------------------------
# ID-28: escaping include denied and never read
# ---------------------------------------------------------------------------


async def test_escaping_include_denied_no_read(state_no_sim, work_dir):
    # Canary lives OUTSIDE the single allowed root (work_dir).
    outside = work_dir.parent / "outside_roots"
    outside.mkdir(exist_ok=True)
    canary = outside / "canary.lib"
    canary.write_text(".subckt CANARY 1 2\nR9 1 2 1\n.ends\n")

    deck = _write(
        work_dir,
        "cand.cir",
        f"* c\nX1 in out CANARY\nR1 in out 1k\n.include {canary}\n.end\n",
    )
    ref = _write(work_dir, "ref.cir", "* r\nR1 in out 1k\n.end\n")

    data = await _run(state_no_sim, path=str(deck), reference=str(ref), checks=["compare"])
    denied = [f for f in data["findings"] if f["rule_id"] == "path_denied"]
    assert denied, "the escaping include must surface a path_denied finding"
    assert str(canary) in denied[0]["subject"]
    assert denied[0]["at"]["file"] == str(deck)
    # NO read: the CANARY subckt was never loaded, so it stays unresolved.
    unresolved = {u["name"].upper() for u in data["comparison"]["unresolved_subckts"]}
    assert "CANARY" in unresolved


# ---------------------------------------------------------------------------
# quality checks (label-island / text-overlap / clean sheet)
# ---------------------------------------------------------------------------

_LABEL_ISLAND_ASC = "Version 4.1\nSHEET 1 880 680\nFLAG 100 100 sig\nFLAG 300 100 sig\n"
_CLEAN_ASC = (
    "Version 4.1\nSHEET 1 880 680\nFLAG 100 100 sig\nWIRE 100 100 300 100\nFLAG 300 100 sig\n"
)
_TEXT_OVERLAP_ASC = (
    "Version 4.1\nSHEET 1 880 680\n"
    "SYMBOL res 400 300 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\n"
    "TEXT 400 300 Left 2 ;note\n"
)


async def test_quality_fires_on_label_island(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "island.asc", _LABEL_ISLAND_ASC)
    data = await _run(state_no_sim, path=str(asc), checks=["quality"])
    islands = [f for f in data["findings"] if f["rule_id"] == "label_island"]
    assert len(islands) == 1
    assert islands[0]["subject"] == "sig"
    assert islands[0]["evidence"]["stub_count"] == 2
    assert islands[0]["at"]["file"] == str(asc)
    assert "x" in islands[0]["at"] and "y" in islands[0]["at"]


async def test_quality_silent_on_clean_sheet(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "clean.asc", _CLEAN_ASC)
    data = await _run(state_no_sim, path=str(asc), checks=["quality"])
    assert data["findings"] == []
    assert data["outcome"] == "complete"


async def test_quality_fires_on_text_overlap(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "overlap.asc", _TEXT_OVERLAP_ASC)
    data = await _run(state_no_sim, path=str(asc), checks=["quality"])
    overlaps = [f for f in data["findings"] if f["rule_id"] == "text_in_symbol_body"]
    assert overlaps, "text anchored inside a symbol body should fire"
    for f in overlaps:
        assert f["at"]["file"] == str(asc)
        assert f["subject"]


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


async def test_render_svg(state_no_sim, work_dir, asc_symbols, monkeypatch):
    asc = _write(work_dir, "r.asc", _RES_ASC)
    # render.mode="only" skips every check, so the O(n²) layout scan must not run.
    calls = {"n": 0}
    real_layout_issues = vc.layout_issues

    def _counting(scene):
        calls["n"] += 1
        return real_layout_issues(scene)

    monkeypatch.setattr(vc, "layout_issues", _counting)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "svg"})
    render = data["render"]
    assert render["image_format"] == "svg"
    assert render["path"].endswith(".svg")
    assert Path(render["path"]).is_file()  # noqa: ASYNC240
    assert render["sha256"]
    assert render["downscaled"] is False
    # mode="only" skips every check.
    assert data["checks_run"] == []
    assert calls["n"] == 0, "render.mode='only' must not run the layout_issues scan"


@pytest.mark.skipif(not raster.raster_available(), reason="cairosvg not installed")
async def test_render_png_present(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "rp.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "png"})
    render = data["render"]
    assert render["image_format"] == "png"
    assert render["path"].endswith(".png")
    assert render["width"] and render["height"]


async def test_render_png_forced_absent_declares_extra(
    state_no_sim, work_dir, asc_symbols, monkeypatch
):
    monkeypatch.setattr(raster, "_load_cairosvg", lambda: None)
    asc = _write(work_dir, "rf.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "png"})
    render = data["render"]
    assert render["image_format"] == "svg"
    assert "raster" in (render["note"] or "")
    render_failures = [f for f in data["failures"] if f["stage"] == "render"]
    assert render_failures, "a PNG request without the extra must declare a per-item failure"
    assert "raster" in render_failures[0]["error"]


async def test_render_max_pixels_downscales(state_no_sim, work_dir, asc_symbols):
    if not raster.raster_available():
        pytest.skip("cairosvg not installed")
    asc = _write(work_dir, "rd.asc", _RES_ASC)
    data = await _run(
        state_no_sim,
        path=str(asc),
        render={"mode": "only", "format": "png", "scale": 3.0, "max_pixels": 100},
    )
    render = data["render"]
    assert render["downscaled"] is True
    assert render["width"] * render["height"] <= 100 * 4  # bounded near the cap


# ---------------------------------------------------------------------------
# export — managed (non-destructive) and sidecar (in-place + diff)
# ---------------------------------------------------------------------------

_NEW_NET = "* exported\nR1 in out 1k\nR2 out 0 2k\nV1 in 0 5\n.end\n"
_PRIOR_NET = "* exported\nR1 in out 1k\nV1 in 0 5\n.end\n"


def _fake_exporter(_cls, asc_path, timeout=0):
    net = Path(asc_path).with_suffix(".net")
    net.write_text(_NEW_NET)
    return net


async def test_managed_export_is_non_destructive(config, work_dir, asc_symbols, monkeypatch):
    monkeypatch.setattr(vc, "_create_netlist", _fake_exporter)
    state = _with_ltspice(config)
    asc = _write(work_dir, "m.asc", _RES_ASC)
    data = await _run(state, path=str(asc), checks=["export"])
    assert data["export"]["ok"] is True
    assert data["export"]["destination"] == "managed"
    assert data["export"]["sha256"]
    # The caller's own sidecar .net is untouched by a managed export.
    assert not (work_dir / "m.net").exists()
    assert ".ltspice-mcp" in data["export"]["netlist"]


async def test_sidecar_export_writes_in_place_with_diff(
    config, work_dir, asc_symbols, monkeypatch
):
    monkeypatch.setattr(vc, "_create_netlist", _fake_exporter)
    state = _with_ltspice(config)
    asc = _write(work_dir, "s.asc", _RES_ASC)
    prior = _write(work_dir, "s.net", _PRIOR_NET)  # a prior sidecar to diff against

    data = await _run(state, path=str(asc), checks=["export"], export_to="sidecar")
    assert data["export"]["destination"] == "sidecar"
    assert data["export"]["netlist"] == str(prior)  # overwrote the sidecar in place
    assert prior.read_text() == _NEW_NET
    diff = data["export"]["diff_vs_prior"]
    assert diff is not None
    assert "R2" in diff["components_added"]
