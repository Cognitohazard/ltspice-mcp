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

Every response in this module is validated against verify_circuit's own declared
output_schema *closed* — additionalProperties:false injected into every object
that declares properties — so a key the handler emits but the schema never
declared fails here. The plain schema cannot catch that: JSON Schema admits
undeclared keys by default, which is how ``comparison`` came to spread sixteen
undeclared keys past both the suite and the session-wide conformance hook.
"""

from __future__ import annotations

import hashlib
import typing
from pathlib import Path
from types import NoneType
from typing import Any, get_args

import jsonschema
import pytest
from pydantic import ValidationError

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import compact_validation_error
from ltspice_mcp.lib import raster
from ltspice_mcp.lib.schematic_scene import LayoutIssue, Scene
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import verify as vc
from ltspice_mcp.tools._base import CompareSpec, RenderPolicy
from ltspice_mcp.tools.schematic_edit import EditSchematicInput
from ltspice_mcp.tools.verify import (
    STRUCTURAL_DELTA_PROPS,
    VerifyCircuitInput,
    evaluate_verify_circuit,
    handle_verify_circuit,
)


class FakeSim:
    """Stub LTspice class; the real exporter is monkeypatched per test."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/LTspice.exe"]


def _model_of(input_model: type[Any], field: str) -> type:
    """The model class behind an ``X | None`` field annotation."""
    annotation = input_model.model_fields[field].annotation
    return next(
        arg for arg in get_args(annotation) if isinstance(arg, type) and arg is not NoneType
    )


def _closed(node: Any) -> Any:
    """The schema with additionalProperties:false wherever it declares properties.

    An object with no ``properties`` (a finding's free-form ``evidence``) is left
    open, because it genuinely carries caller-defined keys.
    """
    if isinstance(node, dict):
        closed = {key: _closed(value) for key, value in node.items()}
        if isinstance(node.get("properties"), dict):
            closed["additionalProperties"] = False
        return closed
    if isinstance(node, list):
        return [_closed(item) for item in node]
    return node


_CLOSED_OUTPUT_SCHEMA = _closed(vc._OUTPUT_SCHEMA)


def _assert_schema(result) -> dict:
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(_CLOSED_OUTPUT_SCHEMA).validate(data)
    return data


#: Comparison controls this file writes as loose keywords, packed into the one
#: ``compare`` object the tool takes. What the argument models accept is
#: TestRenderAndCompareArguments below; here the comparison BEHAVIOUR is, so
#: the call sites stay readable.
_COMPARE_KEYS = {"reference": "reference", "compare_mode": "mode", "anchors": "anchors"}


async def _run(state: SessionState, **kw) -> dict:
    compare = {key: kw.pop(flat) for flat, key in _COMPARE_KEYS.items() if flat in kw}
    if compare:
        kw["compare"] = compare
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


async def test_netlist_default_runs_the_text_deck_checks(state_no_sim, work_dir):
    deck = _write(work_dir, "d.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck))
    assert data["kind"] == "netlist"
    assert data["checks_run"] == ["syntax", "quality"]
    skipped = {s["check"]: s["reason"] for s in data["checks_skipped"]}
    for check in ("symbols", "export", "layout"):
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
    assert result.is_error is True
    assert data["outcome"] == "failed"


async def test_path_denied_is_error(state_no_sim):
    result = await handle_verify_circuit(VerifyCircuitInput(path="/etc/passwd"), state_no_sim)
    data = _assert_schema(result)
    assert result.is_error is True
    assert data["findings"][0]["rule_id"] == "path_denied"


# ---------------------------------------------------------------------------
# syntax findings
# ---------------------------------------------------------------------------


async def test_syntax_finding_shape(state_no_sim, work_dir):
    # R with only one node is an element-arity fault the validator catches.
    deck = _write(work_dir, "bad.cir", "* bad\nR1 a 1k\n.end\n")
    data = await _run(state_no_sim, path=str(deck), checks=["syntax"])
    assert data["outcome"] == "partial"
    assert data["findings"], "a one-node resistor should yield an arity finding"
    for f in data["findings"]:
        assert f["at"]["file"] == str(deck)
        assert f["subject"]
        assert f["severity"] == "error"


async def test_lexer_warnings_reach_the_observations(state_no_sim, work_dir):
    """The lexer already reports what it had to guess about — an unclosed
    .SUBCKT, a mismatched .ENDS, a stray continuation. The syntax check lexes
    the deck and then read only its cards, so those notes were computed and
    dropped: the caller was told the deck is clean when the lexer had said the
    subcircuit never closes."""
    deck = _write(
        work_dir,
        "unclosed.cir",
        "* unclosed\n.subckt AMP a b\nR1 a b 1k\nV1 a 0 1\n.end\n",
    )

    data = await _run(state_no_sim, path=str(deck))

    assert any("unclosed .SUBCKT" in note for note in data["observations"]), data["observations"]


async def test_neutral_findings_are_uncapped_and_mcp_reapplies_rule_cap(
    state_no_sim,
    work_dir,
    monkeypatch,
):
    asc = _write(work_dir, "crowded.asc", "Version 4.1\nSHEET 1 880 680\n")
    issues = [
        LayoutIssue(
            kind="floating_pin",
            refs=(f"R{index}.1",),
            coords=((index * 16, 0),),
            detail=f"floating pin {index}",
        )
        for index in range(vc.FINDING_RULE_CAP + 7)
    ]

    def crowded_scene(path, _state, *, compute_issues):
        assert compute_issues is True
        return Scene(source=path), issues

    monkeypatch.setattr(vc, "_analyze_scene", crowded_scene)
    args = VerifyCircuitInput(path=str(asc), checks=["layout"])
    neutral = await evaluate_verify_circuit(args, state_no_sim)
    full = [f for f in neutral.data["findings"] if f["rule_id"] == "floating_pin"]
    assert len(full) == vc.FINDING_RULE_CAP + 7
    assert not any("showing" in note for note in neutral.data["observations"])

    mcp = await handle_verify_circuit(args, state_no_sim)
    data = _assert_schema(mcp)
    shown = [finding for finding in data["findings"] if finding["rule_id"] == "floating_pin"]
    assert shown == full[: vc.FINDING_RULE_CAP]
    assert len(full) - len(shown) == 7
    assert any(
        note == (f"floating_pin: showing {vc.FINDING_RULE_CAP} of {len(full)} findings")
        for note in data["observations"]
    )


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


async def test_reference_outside_sandbox_names_the_text_alternative(state_no_sim, work_dir):
    """A reference outside the sandbox is a path_denied finding, and the finding
    names the alternative a caller can act on without widening the sandbox."""
    deck = _write(work_dir, "cand.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck), reference="/outside/ref.cir")
    finding = next(f for f in data["findings"] if f["rule_id"] == "path_denied")
    assert "netlist text itself" in finding["evidence"]["detail"]


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
# Schema conformance: every emitted key is a declared key
# ---------------------------------------------------------------------------

# Differs from _BASE in all four ways at once so both comparison modes populate
# every list they can: R1's value changed, R2 rewired (isomorphism break), C1
# absent, L1 extra.
_MULTI_DIFF = "* divider\nV1 in 0 5\nR1 in out 4k7\nR2 in 0 2k\nL1 out 0 1u\n.end\n"
_MULTI_BASE = "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\nC1 out 0 1n\n.end\n"


def _declared_comparison_keys() -> set[str]:
    return set(vc.COMPARISON_SCHEMA["properties"])


async def test_equivalence_comparison_declares_every_key_it_emits(state_no_sim, work_dir):
    """The equivalence payload is ``{"mode": ...} | GraphComparison.as_dict()``;
    every one of those keys must be declared, not just the verdict booleans."""
    deck = _write(work_dir, "cand.cir", _MULTI_BASE)
    ref = _write(work_dir, "ref.cir", _MULTI_DIFF)
    data = await _run(state_no_sim, path=str(deck), reference=str(ref))
    comparison = data["comparison"]
    undeclared = set(comparison) - _declared_comparison_keys()
    assert not undeclared, f"equivalence emits undeclared comparison keys: {sorted(undeclared)}"
    # The difference detail a caller acts on actually arrived, so this is a real
    # payload and not a degenerate all-empty one that would pass vacuously.
    assert {c["ref"] for c in comparison["added"]} == {"C1"}
    assert {c["ref"] for c in comparison["removed"]} == {"L1"}
    assert {v["ref"] for v in comparison["value_mismatches"]} == {"R1"}
    assert comparison["node_partition_mismatches"]


async def test_structural_comparison_declares_every_key_it_emits(state_no_sim, work_dir):
    """Same for the structural_diff payload, whose keys are disjoint from the
    equivalence ones — one flat schema has to cover both."""
    deck = _write(work_dir, "cand.cir", _MULTI_BASE)
    ref = _write(work_dir, "ref.cir", _MULTI_DIFF)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    comparison = data["comparison"]
    undeclared = set(comparison) - _declared_comparison_keys()
    assert not undeclared, (
        f"structural_diff emits undeclared comparison keys: {sorted(undeclared)}"
    )
    assert comparison["components_added"] == ["C1"]
    assert comparison["components_removed"] == ["L1"]
    assert {c["reference"] for c in comparison["components_changed"]} == {"R1"}


# ---------------------------------------------------------------------------
# Parse warnings have exactly one home: the top-level warnings channel
# ---------------------------------------------------------------------------


async def test_unparseable_reference_warns_at_top_level(state_no_sim, work_dir):
    """A deck that could not be parsed is diffed as empty, so everything on the
    other side reads as added/removed. That caveat must reach the declared
    top-level channel — inside ``comparison`` it was invisible to the schema, and
    a structured-only client would have read the bogus delta as fact."""
    deck = _write(work_dir, "cand.cir", _BASE)
    missing = work_dir / "never_written.cir"
    data = await _run(
        state_no_sim, path=str(deck), reference=str(missing), compare_mode="structural_diff"
    )
    assert "warnings" not in data["comparison"], "parse warnings must not ride inside comparison"
    assert any("could not be parsed" in w for w in data["warnings"])
    assert any("never_written.cir" in w for w in data["warnings"])
    # The delta is the bogus one the warning is about — the unparseable reference
    # read as empty, so this circuit's whole component set looks newly added.
    assert set(data["comparison"]["components_added"]) == {"V1", "R1", "R2"}
    # Presentation mirrors it: without this the hint reads "No problems found".
    assert "could not be parsed" in data["hint"]


async def test_unparseable_deck_reaches_no_verdict(state_no_sim, work_dir):
    """A delta measured against a deck nothing could read is not evidence of a
    difference any more than of a match — the side it was measured against was
    fabricated. So the verdict is null, and the outcome stays off 'complete'."""
    deck = _write(work_dir, "cand.cir", _BASE)
    data = await _run(
        state_no_sim,
        path=str(deck),
        reference=str(work_dir / "never_written.cir"),
        compare_mode="structural_diff",
    )
    assert data["comparison"]["equivalent"] is None
    assert data["outcome"] == "partial"
    assert "no verdict" in data["hint"]


def test_two_unparseable_decks_are_not_equivalent(work_dir):
    """The defect this closes: two unread decks both diff as EMPTY circuits, so
    the delta is empty — and an empty delta otherwise means 'these match'. That
    combination reported equivalent/complete for a comparison that compared
    nothing.

    Driven at ``compare_structural`` because the handler cannot currently reach
    the pairing: it gates on ``path.is_file()``, and ``extract_netlist_info``
    raises only on a missing file (malformed content surfaces as lexer warnings
    and ``<unparseable>`` values, never an exception), so the candidate side of a
    netlist comparison always parses. The rule is pinned here anyway — it should
    hold by construction, not by whichever inputs happen to be reachable today.
    """
    comparison, _findings, failure, warnings = vc.compare_structural(
        work_dir / "no_such_reference.cir", work_dir / "no_such_candidate.cir"
    )
    assert failure is None
    assert comparison is not None
    assert not any(comparison[key] for key in STRUCTURAL_DELTA_PROPS), (
        "precondition: the delta really is empty, the shape that read as a match"
    )
    assert comparison["equivalent"] is None
    assert len(warnings) == 3, "the caveat plus one message per unread deck"
    # The null verdict has to survive into the outcome, or the fix stops at the
    # payload and the call still reports success.
    assert vc._outcome([], [], comparison) == "partial"


async def test_unreadable_reference_contract_differs_by_mode_as_documented(state_no_sim, work_dir):
    """The two modes answer an unreadable reference differently, on purpose, and
    ``compare.mode``'s description promises exactly this. Equivalence cannot build
    a comparison at all — isomorphism is undefined without both graphs — so it
    reports a compare failure and no comparison. structural_diff diffs the missing
    side as empty, so the delta survives with a null verdict and a warning. Pinned
    because a documented asymmetry that drifts is worse than an undocumented one.
    """
    deck = _write(work_dir, "cand.cir", _BASE)
    missing = str(work_dir / "never_written.cir")

    equiv = await _run(state_no_sim, path=str(deck), reference=missing)
    assert equiv["comparison"] is None
    assert [f["stage"] for f in equiv["failures"]] == ["compare"]
    assert "compare" not in equiv["checks_run"]

    structural = await _run(
        state_no_sim, path=str(deck), reference=missing, compare_mode="structural_diff"
    )
    assert structural["comparison"]["equivalent"] is None
    assert structural["failures"] == []
    assert structural["warnings"]
    assert "compare" in structural["checks_run"]

    # What the caller CAN rely on across both modes: not a clean result, and the
    # reason is somewhere in the response rather than inferred from silence.
    assert equiv["outcome"] == structural["outcome"] == "partial"


async def test_clean_comparison_emits_no_warnings(state_no_sim, work_dir):
    """The channel stays empty when nothing was assumed — so a populated
    ``warnings`` always means something, and is never decorative."""
    deck = _write(work_dir, "cand.cir", _BASE)
    ref = _write(work_dir, "ref.cir", _BASE)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    assert data["warnings"] == []


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


async def test_simulator_library_include_is_read_though_the_sandbox_denies_it(
    config, work_dir, monkeypatch
):
    """The run path and the verify path agree on the simulator's own library.

    Staging accepts a reference into the detected install's library so a MOSFET
    sheet can run at all (LTspice's netlister appends that ``.lib`` itself). If
    only staging accepts it, verify_circuit reports the schematic's own library
    as an unusable include on the one request that asks whether the schematic
    still matches the circuit being simulated.
    """
    install = work_dir.parent / "fake_ltspice_install" / "lib" / "cmp"
    install.mkdir(parents=True, exist_ok=True)
    shipped = install / "standard.mos"
    shipped.write_text(".subckt SHIPPED 1 2\nR9 1 2 1\n.ends\n")
    # The detected install reports its own library dirs; that discovery is the
    # environment, the acceptance of what it reports is what is under test.
    monkeypatch.setattr(
        FakeSim,
        "get_default_library_paths",
        classmethod(lambda _cls: [str(install)]),
        raising=False,
    )
    state = _with_ltspice(config)

    body = f"* c\nX1 in out SHIPPED\nR1 in out 1k\n.lib {shipped}\n.end\n"
    deck = _write(work_dir, "cand.cir", body)
    ref = _write(work_dir, "ref.cir", body)

    data = await _run(state, path=str(deck), reference=str(ref), checks=["compare"])

    assert not [f for f in data["findings"] if f["rule_id"] == "path_denied"]
    assert data["comparison"]["unresolved_subckts"] == []
    assert data["comparison"]["equivalent"] is True


# ---------------------------------------------------------------------------
# netlist quality checks (connectivity)
# ---------------------------------------------------------------------------

# 'out' is wired to R1 and to nothing else.
_DANGLING_NODE_DECK = "* stub\nV1 in 0 5\nR1 in out 1k\n.op\n.end\n"
# The .meas names a node the deck never declares — the classic unlabelled-net
# export, where the directive silently measures nothing.
_UNDEFINED_REF_DECK = (
    "* probe\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\n"
    ".meas tran vx FIND V(vref) AT 1m\n.tran 1m\n.end\n"
)
# 'mid' has two terminals but both are capacitor plates, so it reaches ground
# through no DC-conductive element and its operating point is undefined.
_FLOATING_NET_DECK = "* ac coupled\nV1 in 0 5\nC1 in mid 1u\nC2 mid 0 1u\n.op\n.end\n"


async def test_netlist_quality_reports_a_node_with_one_terminal(state_no_sim, work_dir):
    deck = _write(work_dir, "dangling.cir", _DANGLING_NODE_DECK)
    data = await _run(state_no_sim, path=str(deck), checks=["quality"])
    assert data["checks_run"] == ["quality"]
    dangling = [f for f in data["findings"] if f["rule_id"] == "dangling_node"]
    assert len(dangling) == 1, data["findings"]
    assert "out" in dangling[0]["evidence"]["detail"]
    assert dangling[0]["at"]["file"] == str(deck)
    # Legal SPICE — a deliberately unterminated fragment is a fact the caller
    # weighs, not a fault, so it must not turn the call partial.
    assert dangling[0]["severity"] == "observation"
    assert data["outcome"] == "complete"


async def test_netlist_quality_reports_a_directive_naming_nothing(state_no_sim, work_dir):
    deck = _write(work_dir, "probe.cir", _UNDEFINED_REF_DECK)
    data = await _run(state_no_sim, path=str(deck), checks=["quality"])
    missing = [f for f in data["findings"] if f["rule_id"] == "undefined_reference"]
    assert len(missing) == 1, data["findings"]
    assert "vref" in missing[0]["evidence"]["detail"]
    assert missing[0]["severity"] == "warning"
    assert data["outcome"] == "partial"


async def test_netlist_quality_reports_a_net_with_no_dc_path_to_ground(state_no_sim, work_dir):
    deck = _write(work_dir, "floating.cir", _FLOATING_NET_DECK)
    data = await _run(state_no_sim, path=str(deck), checks=["quality"])
    floating = [f for f in data["findings"] if f["rule_id"] == "floating_net"]
    assert len(floating) == 1, data["findings"]
    assert "mid" in floating[0]["evidence"]["detail"]
    assert floating[0]["severity"] == "warning"
    assert data["outcome"] == "partial"


async def test_netlist_quality_silent_on_a_clean_deck(state_no_sim, work_dir):
    deck = _write(work_dir, "clean.cir", _BASE)
    data = await _run(state_no_sim, path=str(deck), checks=["quality"])
    assert data["checks_run"] == ["quality"]
    assert data["findings"] == []
    assert data["outcome"] == "complete"


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


async def test_render_reports_the_digest_of_the_sheet_it_drew(state_no_sim, work_dir, asc_symbols):
    """Rendering reads the file on disk, and a peer may commit between an edit
    returning and this call running.

    Without the sheet's own digest beside the image, a picture of a revision
    the caller never wrote is indistinguishable from a picture of theirs: the
    render block's 'sha256' is the image's, and the export block's is the
    netlist's. 'source_sha256' is the one a caller compares with the sha256
    edit_schematic handed back.
    """
    asc = _write(work_dir, "digest.asc", _RES_ASC)
    on_disk = hashlib.sha256(asc.read_bytes()).hexdigest()

    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "svg"})
    render = data["render"]
    assert render["source_sha256"] == on_disk
    # Not the image's digest, and not the exported netlist's.
    assert render["source_sha256"] != render["sha256"]

    # A peer's commit changes it, which is the whole point.
    asc.write_text(_RES_ASC.replace("1k", "2k"), encoding="utf-8")
    again = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "svg"})
    assert again["render"]["source_sha256"] != on_disk


async def test_the_digest_names_the_bytes_that_were_drawn(state_no_sim, work_dir, asc_symbols):
    """The provenance rides with the scene, not with a second read of the path.

    A peer committing between the parse and the render is the race the field
    exists to expose. Hashing the file again afterwards reported a revision
    that was never drawn, and reported it as if nothing had happened.
    """
    asc = _write(work_dir, "drawn.asc", _RES_ASC)
    drawn = hashlib.sha256(asc.read_bytes()).hexdigest()
    scene, _ = vc._analyze_scene(asc, state_no_sim, compute_issues=False)

    asc.write_text(_RES_ASC.replace("1k", "2k"), encoding="utf-8")
    assert hashlib.sha256(asc.read_bytes()).hexdigest() != drawn

    payload, _, failures, _ = await vc._do_render(
        scene, "asc", vc.VerifyRenderPolicy(format="svg"), asc, state_no_sim
    )
    assert not failures
    assert payload is not None
    assert payload["source_sha256"] == drawn


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


# ---------------------------------------------------------------------------
# render argument spellings
# ---------------------------------------------------------------------------


def test_render_true_is_the_default_policy():
    args = VerifyCircuitInput.model_validate({"path": "x.asc", "render": True})
    assert args.render is not None
    assert args.render.mode == "with_checks"
    assert args.render.format == "png"
    assert args.render.delivery == "artifact"


def test_advertised_delivery_states_the_inline_cost():
    """The advertised copy of render.delivery must say what an inline image costs.

    The choice is paid on every later turn (the image stays in context), and
    the bare enum gave a caller no reason to prefer the default: one caller
    asked for 'both' on every post-edit verify, and the three images came to
    86% of everything that session read back.
    """
    from ltspice_mcp.tools import registry

    defs, _ = registry.get_tools()
    schema = next(d for d in defs if d.name == "verify_circuit").input_schema
    delivery = schema["$defs"]["VerifyRenderPolicy"]["properties"]["delivery"]
    text = delivery.get("description") or ""
    assert "tokens" in text and "artifact" in text, text


def test_render_false_and_none_skip_the_drawing():
    assert VerifyCircuitInput.model_validate({"path": "x.asc", "render": False}).render is None
    assert VerifyCircuitInput.model_validate({"path": "x.asc", "render": None}).render is None
    assert VerifyCircuitInput.model_validate({"path": "x.asc"}).render is None


def test_render_object_still_validates_and_overrides():
    args = VerifyCircuitInput.model_validate(
        {"path": "x.asc", "render": {"mode": "only", "format": "svg"}}
    )
    assert args.render is not None
    assert args.render.mode == "only"
    assert args.render.format == "svg"


def test_render_bad_mode_error_enumerates_the_modes():
    with pytest.raises(ValidationError) as excinfo:
        VerifyCircuitInput.model_validate({"path": "x.asc", "render": {"mode": "always"}})
    detail = compact_validation_error(excinfo.value)
    assert "with_checks" in detail
    assert "only" in detail


def test_render_scalar_refusal_names_the_accepted_spellings():
    with pytest.raises(ValidationError) as excinfo:
        VerifyCircuitInput.model_validate({"path": "x.asc", "render": "png"})
    detail = compact_validation_error(excinfo.value)
    assert "with_checks" in detail
    assert "only" in detail
    assert "true" in detail


def test_render_boolean_is_advertised_in_the_json_schema():
    schema = VerifyCircuitInput.model_json_schema()
    advertised = repr(schema["properties"]["render"]) + repr(schema.get("$defs", {}))
    assert "boolean" in advertised


def test_api_types_exports_the_models_errors_name():
    from ltspice_mcp.api import types as api_types

    # Asserted against the classes the two tools' own fields validate against;
    # an identity check against the module the export came from would hold no
    # matter which class had been re-exported there.
    assert api_types.VerifyRenderPolicy is _model_of(VerifyCircuitInput, "render")
    assert api_types.VerifyCompareSpec is _model_of(VerifyCircuitInput, "compare")
    # Both tools take the same compare spec: one comparison engine, one shape.
    assert api_types.VerifyCompareSpec is _model_of(EditSchematicInput, "compare")
    # RenderPolicy is exported as the base VerifyRenderPolicy subclasses; only
    # verify_circuit takes a render, so no tool field validates against it.
    assert issubclass(api_types.VerifyRenderPolicy, api_types.RenderPolicy)
    for name in api_types.__all__:
        assert getattr(api_types, name, None) is not None, name


def test_the_exported_policy_models_are_accepted_by_their_tool():
    """An exported argument model must validate on the field it is exported for.

    verify_circuit takes VerifyRenderPolicy, a SUBCLASS of the shared
    RenderPolicy, and a parent instance is not a child instance — so passing a
    base RenderPolicy is rejected by pydantic. Exporting only the base under
    the verify_circuit heading pointed callers at the one type that tool
    cannot take, and never exported the one it can.
    """
    from ltspice_mcp.api import types as api_types

    verify = api_types.VerifyCircuitInput(
        path="divider.asc",
        render=api_types.VerifyRenderPolicy(format="svg"),
        compare=api_types.VerifyCompareSpec(reference="ref.cir"),
    )
    assert verify.render is not None and verify.render.format == "svg"
    assert verify.compare is not None and verify.compare.reference == "ref.cir"

    edit = api_types.EditSchematicInput.model_validate(
        {
            "target": "divider.asc",
            "ops": [{"op": "add_net_label", "net": "vout", "pin": "R1.2"}],
            "compare": api_types.VerifyCompareSpec(reference="ref.cir"),
        }
    )
    assert edit.compare is not None and edit.compare.reference == "ref.cir"


class TestRenderAndCompareArguments:
    """What this tool accepts for `render` and `compare`, on the resolved value.

    Both tools that compare take the same model — literally the same class, so
    they cannot drift — and a tool that needs more subclasses it, which is why
    only this one advertises the two comparison modes and the render policy's
    delivery and mode.
    """

    @staticmethod
    def _verify(**kwargs: Any) -> VerifyCircuitInput:
        return VerifyCircuitInput.model_validate({"path": "deck.cir", **kwargs})

    def test_no_arguments_render_nothing_and_compare_nothing(self):
        args = self._verify()
        assert args.render is None
        assert args.compare is None

    @pytest.mark.parametrize(
        ("payload", "field", "expected"),
        [
            ({"render": True}, "format", "png"),
            ({"render": {"format": "svg"}}, "format", "svg"),
            ({"render": {"scale": 2.0}}, "scale", 2.0),
            ({"render": {"max_pixels": 100_000}}, "max_pixels", 100_000),
            ({"render": {"mode": "only"}}, "mode", "only"),
            ({"render": {"delivery": "inline"}}, "delivery", "inline"),
        ],
    )
    def test_render_policy_spellings(self, payload: dict[str, Any], field: str, expected: Any):
        policy = self._verify(**payload).render
        assert policy is not None
        assert getattr(policy, field) == expected

    def test_false_and_omitted_both_draw_nothing(self):
        assert self._verify(render=False).render is None
        assert self._verify().render is None

    def test_the_compare_object_carries_every_control(self):
        spec = self._verify(
            compare={
                "reference": "golden.cir",
                "mode": "structural_diff",
                "anchors": ["out", "vdd"],
                "rtol": 1e-3,
            }
        ).compare
        assert spec is not None
        assert spec.reference == "golden.cir"
        assert spec.mode == "structural_diff"
        assert spec.anchors == ["out", "vdd"]
        assert spec.rtol == 1e-3

    @pytest.mark.parametrize(
        "payload",
        [
            {"render": "yes"},
            {"render": {"mode": "sideways"}},
            {"compare": {"rtol": 1e-3}},
        ],
        ids=["not-a-policy", "unknown-mode", "no-reference"],
    )
    def test_refused_spellings(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            self._verify(**payload)

    @pytest.mark.parametrize(
        ("tool", "field", "shared"),
        [
            (EditSchematicInput, "compare", CompareSpec),
            (VerifyCircuitInput, "compare", CompareSpec),
            (VerifyCircuitInput, "render", RenderPolicy),
        ],
    )
    def test_the_two_tools_share_one_argument_model(
        self, tool: type[Any], field: str, shared: type
    ):
        """Not "the same fields" — literally the same class, so they cannot drift.

        A tool that needs more subclasses the shared model, so the shared half
        stays one declaration and the extra half is visibly that tool's own.
        """
        assert issubclass(_model_of(tool, field), shared)
