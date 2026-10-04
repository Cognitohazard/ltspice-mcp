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
import sys
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
from tests import _fake_netlister as fake_netlister
from tests.conftest import needs_raster


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
    p.write_text(text, newline="\n")
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


@pytest.mark.parametrize("suffix", [".sp", ".spice"])
async def test_netlist_suffix_dispatch(state_no_sim, work_dir, suffix):
    """``.spice`` is the extension xschem and the sky130 testbenches write."""
    deck = _write(work_dir, f"d{suffix}", _BASE)
    data = await _run(state_no_sim, path=str(deck))
    assert data["kind"] == "netlist"
    assert "syntax" in data["checks_run"]
    assert data["outcome"] == "complete"


async def test_a_path_through_a_parent_segment_inside_the_sandbox_is_checked(
    state_no_sim, work_dir
):
    deck = _write(work_dir, "d.cir", _BASE)
    (work_dir / "sub").mkdir()
    data = await _run(state_no_sim, path="sub/../d.cir")
    assert data["kind"] == "netlist"
    assert Path(data["path"]) == deck.resolve()
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


@pytest.mark.parametrize("codec", ["utf-8", "cp1252"])
async def test_micro_signs_are_one_observation_per_file_when_no_misreading_reader_is_known(
    state_no_sim, work_dir, codec
):
    """A µ is micro to a reader that decodes the file in the encoding it was
    written in, which LTspice 24 and later do for the UTF-8 they write. With no
    LTspice XVII known to the session, a deck full of them is one fact about the
    file, with the count and lines, and the outcome stays complete."""
    deck = work_dir / "rc.net"
    deck.write_bytes("* rc\nR1 in out 1k\nC1 out 0 23µ\nC2 out 0 4.7µF\n.end\n".encode(codec))

    data = await _run(state_no_sim, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "observation"
    assert finding["at"] == {"file": str(deck), "line": 3}
    assert finding["subject"] == "rc.net"
    evidence = finding["evidence"]
    assert evidence["count"] == 2
    assert evidence["lines"] == [3, 4]
    assert evidence["tokens"] == ["23µ", "4.7µF"]
    assert evidence["encoding"] == codec
    assert "'u' is micro in every encoding" in evidence["reason"]
    assert data["outcome"] == "complete"


class FakeXVII(FakeSim):
    """An LTspice XVII install: the executable name is the build's identity."""

    spice_exe: typing.ClassVar[list[str]] = ["C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe"]


@pytest.mark.parametrize("codec", ["utf-8", "utf-8-sig", "utf-16"])
async def test_micro_sign_is_a_warning_per_value_when_the_sessions_ltspice_is_xvii(
    config, work_dir, codec
):
    """LTspice XVII decodes a deck as cp1252, so a µ in any other encoding
    reaches it as other characters and the value runs without its scale."""
    state = SessionState.create(config, available={"ltspice": FakeXVII})
    deck = work_dir / "rc.net"
    deck.write_bytes("* rc\nR1 in out 1k\nC1 out 0 23µ\n.end\n".encode(codec))

    data = await _run(state, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "warning"
    assert finding["at"] == {"file": str(deck), "line": 3}
    assert finding["subject"] == "23µ"
    assert finding["evidence"]["ascii_spelling"] == "23u"
    assert finding["evidence"]["suffix"] == "U+00B5"
    assert finding["evidence"]["reader"] == FakeXVII.spice_exe[0]
    assert finding["evidence"]["card"] == "C1 out 0 23µ"
    assert data["outcome"] == "partial"


async def test_a_cp1252_micro_sign_read_by_xvii_stays_an_observation(config, work_dir):
    """In cp1252 a µ is the one byte XVII reads as micro: nothing is misread."""
    state = SessionState.create(config, available={"ltspice": FakeXVII})
    deck = work_dir / "rc.net"
    deck.write_bytes("* rc\nC1 out 0 23µ\n.end\n".encode("cp1252"))

    data = await _run(state, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "observation"
    assert data["outcome"] == "complete"


async def test_a_deck_that_names_xvii_as_its_writer_warns_when_stored_as_utf8(
    state_no_sim, work_dir
):
    """The deck's own header names the reader it was written for."""
    deck = work_dir / "rc.net"
    deck.write_bytes("* rc\n* Generated by LTspice XVII\nC1 out 0 23µ\n.end\n".encode())

    data = await _run(state_no_sim, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "warning"
    assert finding["evidence"]["reader"] == "LTspice XVII"


async def test_a_build_an_xvii_run_reported_is_a_known_reader(config, work_dir):
    """An executable not named like XVII is still known to be one once a run on
    it reported that build in its own output, as capabilities reports it."""
    from ltspice_mcp.lib.experiment_types import Completeness, ExperimentCase, ExperimentJob
    from ltspice_mcp.lib.simulator_build import SimulatorExecutable, executable_path
    from ltspice_mcp.lib.store import Store

    class Renamed(FakeSim):
        spice_exe: typing.ClassVar[list[str]] = [str(work_dir / "tools" / "ltspice.exe")]

    # Recorded as the server records it, in the platform's own spelling.
    program = executable_path(Renamed)
    assert program is not None
    state = SessionState.create(config, available={"ltspice": Renamed})
    deck = work_dir / "rc.net"
    deck.write_bytes("* rc\nC1 out 0 23µ\n.end\n".encode())
    state.add_experiment_job(
        ExperimentJob(
            job_id="exp_xvii_0001",
            request_id="xvii",
            fingerprint="f" * 64,
            canonicalizer_version=1,
            control_token="token",
            store_path=Store(work_dir).job_record("exp_xvii_0001"),
            cases=[
                ExperimentCase(
                    case_id="case_0000",
                    run_index=0,
                    circuit="rc",
                    circuit_path=deck,
                    staged_deck=deck,
                    deck_sha256="deck-sha",
                    simulator_version="Linear Technology Corporation LTspice XVII",
                )
            ],
            sources=[],
            simulator="Renamed",
            completeness=Completeness(declared=1, expanded=1),
            simulator_executable=SimulatorExecutable(
                path=program, sha256=None, bytes=None, modified=None
            ),
        ),
        already_persisted=True,
    )

    data = await _run(state, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "warning"
    assert finding["evidence"]["reader"] == (
        f"Linear Technology Corporation LTspice XVII ({program})"
    )


async def test_micro_sign_warnings_are_capped_like_every_repeating_rule(config, work_dir):
    """60 values spelled with µ are 25 findings and a note counting the rest."""
    state = SessionState.create(config, available={"ltspice": FakeXVII})
    cards = "".join(f"C{i} n{i} 0 {i + 1}µ\n" for i in range(60))
    deck = work_dir / "many.net"
    deck.write_bytes(f"* many\n{cards}.end\n".encode())

    data = await _run(state, path=str(deck), checks=["syntax"])

    shown = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert len(shown) == vc.FINDING_RULE_CAP
    assert (
        f"value_suffix_micro_sign: showing {vc.FINDING_RULE_CAP} of 60 findings"
        in (data["observations"])
    )


async def test_syntax_flags_greek_mu_like_the_micro_sign(state_no_sim, work_dir):
    deck = work_dir / "rc.cir"
    deck.write_bytes("* rc\n.param tau=10μs\n.end\n".encode())

    data = await _run(state_no_sim, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["evidence"]["tokens"] == ["10μs"]
    assert finding["evidence"]["lines"] == [2]


async def test_syntax_blocks_a_mis_decoded_micro_suffix(state_no_sim, work_dir):
    # UTF-8 on purpose: written in the platform default, cp1252 on Windows,
    # 'Âµ' becomes C2 B5 and reads back as a genuine UTF-8 micro sign.
    deck = work_dir / "rc.cir"
    deck.write_bytes("* rc\nR1 in out 1k\nC1 out 0 23Âµ\n.end\n".encode())

    data = await _run(state_no_sim, path=str(deck), checks=["syntax"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_nonascii"]
    assert finding["severity"] == "error"
    assert finding["evidence"]["reads_as"] == "23"
    assert finding["evidence"]["likely_intended"] == "23u"


async def test_export_stage_reports_micro_signs_in_the_exported_netlist(
    config, work_dir, asc_symbols, monkeypatch
):
    """LTspice 24 and later write the exported .net as UTF-8, µ as C2 B5, and
    read it back as micro. That netlist is what a user hands to another LTspice,
    so the export stage reports its micro signs as a fact about the file; the
    schematic itself is left exactly as it was."""

    def exporter(_cls, asc_path, timeout=0):
        net = Path(asc_path).with_suffix(".net")
        net.write_bytes(
            "* rc.asc\n* Generated by LTspice 24.1.9 for Windows.\nC1 out 0 23µ\n.end\n".encode()
        )
        return net

    monkeypatch.setattr(vc, "_create_netlist", exporter)
    state = _with_ltspice(config)
    asc = work_dir / "rc.asc"
    asc.write_bytes(_RES_ASC.replace("Value 1k", "Value 23µ").encode("utf-8"))
    before = asc.read_bytes()

    data = await _run(state, path=str(asc), checks=["export"])

    (finding,) = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
    assert finding["severity"] == "observation"
    assert finding["at"] == {"file": data["export"]["netlist"], "line": 3}
    assert finding["evidence"]["encoding"] == "utf-8"
    assert finding["evidence"]["generated_by"] == "LTspice 24.1.9 for Windows."
    assert data["outcome"] == "complete"
    assert asc.read_bytes() == before


async def test_managed_export_leaves_the_schematics_folder_untouched(exporting_state, project_dir):
    """The default export mode's contract: nothing is written beside the caller's file.

    The staged copy is exported inside the store, but the copy is taken under
    the schematic's cross-process lock, and that lock used to live in a
    ``.ltspice-mcp/locks/`` directory created beside the schematic.
    """
    sheet = _write(project_dir, "amp.asc", fake_netlister.amp_asc())

    data = await _run(exporting_state, path=str(sheet), checks=["export"])

    assert data["export"]["ok"] is True
    assert data["export"]["destination"] == "managed"
    assert sorted(p.name for p in project_dir.iterdir()) == ["amp.asc"]


async def test_an_exports_relative_include_resolves_beside_the_schematic(
    exporting_state, work_dir, project_dir, tmp_path_factory, monkeypatch
):
    """With the store outside the sandbox, a managed export is still compared whole.

    The export sits in the store's scratch; its ``.include models.inc`` names the
    file beside the schematic, which is inside the sandbox. Resolved against the
    scratch copy instead, the include is refused as a path outside the allowed
    roots and the comparison reports the author's own library as denied.
    """
    monkeypatch.setenv("LTSPICE_MCP_STORE_DIR", str(tmp_path_factory.mktemp("stores")))
    _write(project_dir, "models.inc", ".param rload=10k\n")
    sheet = _write(
        project_dir,
        "amp.asc",
        _RES_ASC.replace("Value 1k", "Value {rload}") + "TEXT 0 0 Left 2 !.include models.inc\n",
    )
    reference = _write(
        project_dir, "ref.net", "* ref\n.include models.inc\nR1 a b {rload}\n.end\n"
    )
    assert not exporting_state.store.root.is_relative_to(work_dir)

    data = await _run(
        exporting_state, path=str(sheet), checks=["export", "compare"], reference=str(reference)
    )

    assert data["failures"] == []
    assert [f for f in data["findings"] if f["rule_id"] == "path_denied"] == []
    assert data["comparison"]["equivalent"] is True
    assert sorted(p.name for p in project_dir.iterdir()) == ["amp.asc", "models.inc", "ref.net"]


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
# compare against LTspice exports: an .asc reference, continuation lines, and
# the exporter's own boilerplate
# ---------------------------------------------------------------------------

# The exported amp written by hand: the model on one line with its parameters
# reordered and comma-separated, the MOSFET's parameters reordered, different
# case and spacing, and none of the exporter's boilerplate.
_AMP_BY_HAND = (
    "* amp, by hand\n"
    "V1 V1_1 V1_2 5\n"
    "R1 R1_1 R1_2 10k\n"
    "M1 M1_1 M1_2 M1_3 M1_4 NMOS w=10u l=1u\n"
    ".MODEL MYN NMOS (KP=100u, VTO = 0.7)\n"
    ".param rload = 10k\n"
    ".tran 1m\n"
    ".end\n"
)


def _empty_delta() -> dict[str, list]:
    return {key: [] for key in STRUCTURAL_DELTA_PROPS}


def _delta(comparison: dict) -> dict[str, list]:
    return {key: comparison[key] for key in STRUCTURAL_DELTA_PROPS}


@pytest.fixture
def exporting_state(config, asc_symbols, monkeypatch) -> SessionState:
    """A session whose LTspice export is the fake netlister (real boilerplate)."""
    monkeypatch.setattr(vc, "_create_netlist", fake_netlister.create_netlist)
    return _with_ltspice(config)


async def _diff_exported(state: SessionState, sheet: Path, reference: Path) -> dict:
    """structural_diff of ``sheet``'s export against ``reference``."""
    return await _run(
        state,
        path=str(sheet),
        checks=["export", "compare"],
        reference=str(reference),
        compare_mode="structural_diff",
    )


async def test_unchanged_sheet_against_its_own_asc_reports_no_changes(exporting_state, work_dir):
    """A sheet compared with itself is the same circuit, so the delta is empty.

    The reference .asc used to be read through the schematic editor while the
    sheet under test was read from its LTspice export, so the two sides never
    had the same representation: every component carrying a SpiceLine read as
    changed, the multi-line TEXT block read as one directive removed and three
    added, and the exporter's .backanno / standard.mos / default-model lines read
    as added.
    """
    sheet = _write(work_dir, "amp.asc", fake_netlister.amp_asc())

    data = await _diff_exported(exporting_state, sheet, sheet)

    assert data["failures"] == []
    assert _delta(data["comparison"]) == _empty_delta()
    assert data["comparison"]["equivalent"] is True
    assert data["warnings"] == []
    assert data["outcome"] == "complete"
    # Exporting the reference staged a copy; nothing was written beside it.
    assert not (work_dir / "amp.net").exists()


async def test_additive_edit_against_the_original_asc_reports_only_the_addition(
    exporting_state, work_dir
):
    original = _write(work_dir, "amp_orig.asc", fake_netlister.amp_asc())
    edited = _write(work_dir, "amp.asc", fake_netlister.amp_asc(fake_netlister.R2_PART))

    data = await _diff_exported(exporting_state, edited, original)

    assert _delta(data["comparison"]) == {**_empty_delta(), "components_added": ["R2"]}
    assert data["comparison"]["equivalent"] is False


async def test_equivalence_exports_an_asc_reference(exporting_state, work_dir):
    """The graph engine lexes whatever it is handed as SPICE, so an .asc
    reference has to arrive as its export, not as the schematic text."""
    sheet = _write(work_dir, "amp.asc", fake_netlister.amp_asc())

    data = await _run(
        exporting_state, path=str(sheet), checks=["export", "compare"], reference=str(sheet)
    )

    assert data["failures"] == []
    assert data["comparison"]["mode"] == "equivalence"
    assert data["comparison"]["equivalent"] is True
    assert not (work_dir / "amp.net").exists()


async def test_hand_written_reference_matches_the_export(exporting_state, work_dir):
    """Continuation lines, spacing, case, parameter order and the exporter's
    boilerplate are spelling, not circuit: none of them is a difference."""
    sheet = _write(work_dir, "amp.asc", fake_netlister.amp_asc())
    ref = _write(work_dir, "amp_ref.cir", _AMP_BY_HAND)

    data = await _diff_exported(exporting_state, sheet, ref)

    assert _delta(data["comparison"]) == _empty_delta()
    assert data["comparison"]["equivalent"] is True


async def test_default_model_line_counts_when_the_reference_declares_that_model(
    exporting_state, work_dir
):
    """The exporter's parameterless ``.model NMOS NMOS`` is boilerplate only
    while the reference declares no model of that name. Here the reference
    defines NMOS itself, so the export's default replaces it — a real change.
    PMOS is still undeclared, so its default line stays out of the delta."""
    sheet = _write(work_dir, "amp.asc", fake_netlister.amp_asc())
    ref = _write(
        work_dir,
        "amp_ref.cir",
        _AMP_BY_HAND.replace(".end\n", ".model NMOS NMOS(KP=50u)\n.end\n"),
    )

    data = await _diff_exported(exporting_state, sheet, ref)

    comparison = data["comparison"]
    assert comparison["directives_added"] == [".model NMOS NMOS"]
    assert comparison["directives_removed"] == [".model NMOS NMOS(KP=50u)"]
    assert comparison["components_changed"] == []


async def test_a_directive_is_compared_whole_across_continuation_lines(state_no_sim, work_dir):
    """A continuation line belongs to the card above it. Read line by line, the
    ``+`` line was dropped and the card was compared on its first line alone,
    so a change on the continuation line went unseen."""
    ref = _write(
        work_dir, "ref.cir", "* m\nR1 a 0 1k\n.model MYN NMOS(VTO=0.7\n+ KP=100u)\n.end\n"
    )
    deck = _write(
        work_dir, "cand.cir", "* m\nR1 a 0 1k\n.model MYN NMOS(VTO=0.7\n+ KP=200u)\n.end\n"
    )

    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )

    comparison = data["comparison"]
    assert comparison["directives_removed"] == [".model MYN NMOS(VTO=0.7 KP=100u)"]
    assert comparison["directives_added"] == [".model MYN NMOS(VTO=0.7 KP=200u)"]


async def test_instance_parameter_change_is_a_component_change(state_no_sim, work_dir):
    """An instance's parameters are part of what it is: a MOSFET widened from
    10u to 20u is changed, though its model name is the same."""
    ref = _write(work_dir, "ref.cir", "* m\nM1 d g 0 0 NMOS l=1u w=10u\n.end\n")
    deck = _write(work_dir, "cand.cir", "* m\nM1 d g 0 0 NMOS l=1u w=20u\n.end\n")

    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )

    assert data["comparison"]["components_changed"] == [
        {"reference": "M1", "before": "NMOS l=1u w=10u", "after": "NMOS l=1u w=20u"}
    ]


async def test_only_the_exporter_boilerplate_is_normalized(state_no_sim, work_dir):
    """``.backanno`` and the install's own ``standard.*`` library are dropped;
    a library the author added is a directive like any other."""
    ref = _write(work_dir, "ref.cir", "* m\nR1 a 0 1k\n.end\n")
    deck = _write(
        work_dir,
        "cand.cir",
        "* m\nR1 a 0 1k\n"
        ".lib C:\\Program Files\\ADI\\LTspice\\lib\\cmp\\standard.bjt\n"
        ".lib models/opamps.lib\n.backanno\n.end\n",
    )

    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )

    assert data["comparison"]["directives_added"] == [".lib models/opamps.lib"]
    assert data["comparison"]["directives_removed"] == []


@pytest.mark.parametrize("mode", ["equivalence", "structural_diff"])
async def test_asc_reference_without_ltspice_fails_the_compare(
    state_no_sim, work_dir, asc_symbols, mode
):
    """With no exporter there is no netlist to compare against. The compare
    stage fails and says why, rather than diffing a schematic's attributes
    against a netlist's cards."""
    deck = _write(work_dir, "cand.cir", _AMP_BY_HAND)
    ref = _write(work_dir, "amp.asc", fake_netlister.amp_asc())

    data = await _run(state_no_sim, path=str(deck), reference=str(ref), compare_mode=mode)

    assert data["comparison"] is None
    assert [f["stage"] for f in data["failures"]] == ["compare"]
    assert "LTspice" in data["failures"][0]["error"]
    assert data["outcome"] == "partial"


async def test_asc_text_reference_is_refused(exporting_state, work_dir):
    """Schematic text pasted as the reference has no path to export from, and
    lexed as SPICE it is nonsense; the compare fails and names the fix."""
    deck = _write(work_dir, "cand.cir", _AMP_BY_HAND)

    data = await _run(
        exporting_state,
        path=str(deck),
        reference=fake_netlister.amp_asc(),
        compare_mode="structural_diff",
    )

    assert data["comparison"] is None
    assert [f["stage"] for f in data["failures"]] == ["compare"]
    assert ".asc" in data["failures"][0]["error"]


def test_structural_compare_refuses_a_schematic(work_dir):
    """The shared compare entry point takes netlists only; a schematic handed
    to it directly is a failure, never a diff of mismatched representations."""
    ref = _write(work_dir, "amp.asc", fake_netlister.amp_asc())
    deck = _write(work_dir, "cand.cir", _AMP_BY_HAND)

    comparison, _findings, failure, _warnings = vc.compare_structural(ref, deck)

    assert comparison is None
    assert failure is not None and "amp.asc" in failure["error"]


# ---------------------------------------------------------------------------
# ID-28: escaping include denied and never read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["absolute", "parent-relative"])
async def test_escaping_include_denied_no_read(state_no_sim, work_dir, spelling):
    """An include outside the single allowed root (work_dir) is denied and never
    read, whether it names the file outright or climbs to it through ``..``:
    a path is judged by where it lands, not by how it is spelled."""
    outside = work_dir.parent / "outside_roots"
    outside.mkdir(exist_ok=True)
    canary = outside / "canary.lib"
    canary.write_text(".subckt CANARY 1 2\nR9 1 2 1\n.ends\n")
    include = str(canary) if spelling == "absolute" else f"../{outside.name}/canary.lib"

    deck = _write(
        work_dir,
        "cand.cir",
        f"* c\nX1 in out CANARY\nR1 in out 1k\n.include {include}\n.end\n",
    )
    ref = _write(work_dir, "ref.cir", "* r\nR1 in out 1k\n.end\n")

    data = await _run(state_no_sim, path=str(deck), reference=str(ref), checks=["compare"])
    denied = [f for f in data["findings"] if f["rule_id"] == "path_denied"]
    assert denied, "the escaping include must surface a path_denied finding"
    assert include in denied[0]["subject"]
    assert denied[0]["at"]["file"] == str(deck)
    # NO read: the CANARY subckt was never loaded, so it stays unresolved.
    unresolved = {u["name"].upper() for u in data["comparison"]["unresolved_subckts"]}
    assert "CANARY" in unresolved


@pytest.mark.parametrize("separator", ["/", "\\"], ids=["slash", "backslash"])
async def test_parent_relative_include_inside_the_roots_is_read(state_no_sim, work_dir, separator):
    """``.include ../models/x.lib`` from a deck in a subfolder names a file
    inside the sandbox; the run path stages it, so the compare must read it
    rather than report it as resolving outside the allowed roots."""
    models = work_dir / "models"
    models.mkdir()
    (models / "parts.lib").write_text(".subckt SHARED 1 2\nR9 1 2 1\n.ends\n")
    decks = work_dir / "decks"
    decks.mkdir()
    include = separator.join(["..", "models", "parts.lib"])
    body = f"* c\nX1 in out SHARED\nR1 in out 1k\n.include {include}\n.end\n"
    deck = _write(decks, "cand.cir", body)
    ref = _write(decks, "ref.cir", body)

    data = await _run(state_no_sim, path=str(deck), reference=str(ref), checks=["compare"])

    assert not [f for f in data["findings"] if f["rule_id"] == "path_denied"], data["findings"]
    assert data["comparison"]["unresolved_subckts"] == []
    assert data["comparison"]["equivalent"] is True


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


@needs_raster
async def test_render_png_present(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "rp.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "png"})
    render = data["render"]
    assert render["image_format"] == "png"
    assert render["path"].endswith(".png")
    assert render["width"] and render["height"]


async def test_render_png_forced_absent_declares_extra(
    state_no_sim, work_dir, asc_symbols, raster_extra_missing
):
    asc = _write(work_dir, "rf.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "png"})
    render = data["render"]
    assert render["image_format"] == "svg"
    assert "raster" in (render["note"] or "")
    render_failures = [f for f in data["failures"] if f["stage"] == "render"]
    assert render_failures, "a PNG request without the extra must declare a per-item failure"
    assert "'raster' extra" in render_failures[0]["error"]
    assert "ltspice-mcp[raster]" in (render_failures[0]["remedy"] or "")


async def test_render_png_without_native_cairo_blames_the_library(
    state_no_sim, work_dir, asc_symbols, raster_native_missing
):
    """With the extra installed and libcairo absent, "install the extra" is
    advice the caller already followed. The failure names the native library,
    and the remedy is the one for the platform the server runs on."""
    asc = _write(work_dir, "rn.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "png"})
    assert data["render"]["image_format"] == "svg"
    (failure,) = [f for f in data["failures"] if f["stage"] == "render"]
    assert "Cairo" in failure["error"]
    assert "not installed" not in failure["error"]
    assert failure["remedy"] == raster.native_library_remedy(sys.platform)
    assert "Cairo" in data["render"]["note"]
    # The headline a structured-only client reads carries it too.
    assert "Cairo" in data["hint"]


async def test_inline_svg_request_says_why_nothing_was_inlined(
    state_no_sim, work_dir, asc_symbols
):
    """Inline delivery is PNG only. A caller that asked for it and got an SVG
    used to learn only 'returned_inline: false', with nothing saying why or
    that the drawing is on disk."""
    asc = _write(work_dir, "ri.asc", _RES_ASC)
    result = await handle_verify_circuit(
        VerifyCircuitInput.model_validate(
            {
                "path": str(asc),
                "render": {"mode": "only", "format": "svg", "delivery": "inline"},
            }
        ),
        state_no_sim,
    )
    data = _assert_schema(result)
    render = data["render"]
    assert render["returned_inline"] is False
    assert render["inline_skipped"] == "svg_requested"
    note = render["note"] or ""
    assert "PNG only" in note
    assert render["path"] in note
    # The caller chose SVG, so nothing failed: this is an explanation, not a
    # shortfall, and it still reaches the headline.
    assert data["failures"] == []
    assert data["outcome"] == "complete"
    assert "not returned inline" in data["hint"]
    assert render["path"] in data["hint"]
    assert not [c for c in result.content if c.type == "image"]


async def test_inline_png_request_without_raster_says_why_nothing_was_inlined(
    state_no_sim, work_dir, asc_symbols, raster_extra_missing
):
    asc = _write(work_dir, "rj.asc", _RES_ASC)
    data = await _run(
        state_no_sim,
        path=str(asc),
        render={"mode": "only", "format": "png", "delivery": "both"},
    )
    render = data["render"]
    assert render["returned_inline"] is False
    assert render["inline_skipped"] == "png_unavailable"
    note = render["note"] or ""
    # Why the PNG could not be made, then where the SVG went instead.
    assert "'raster' extra" in note
    assert render["path"] in note


async def test_artifact_delivery_skips_nothing(state_no_sim, work_dir, asc_symbols):
    # Nothing inline was asked for, so nothing inline was skipped.
    asc = _write(work_dir, "ra.asc", _RES_ASC)
    data = await _run(state_no_sim, path=str(asc), render={"mode": "only", "format": "svg"})
    assert data["render"]["inline_skipped"] is None
    assert data["render"]["note"] is None


@needs_raster
async def test_inline_png_is_delivered(state_no_sim, work_dir, asc_symbols):
    asc = _write(work_dir, "rk.asc", _RES_ASC)
    result = await handle_verify_circuit(
        VerifyCircuitInput.model_validate(
            {"path": str(asc), "render": {"mode": "only", "delivery": "inline"}}
        ),
        state_no_sim,
    )
    data = _assert_schema(result)
    assert data["render"]["returned_inline"] is True
    assert data["render"]["inline_skipped"] is None
    assert [c.mime_type for c in result.content if c.type == "image"] == ["image/png"]


@needs_raster
async def test_render_max_pixels_downscales(state_no_sim, work_dir, asc_symbols):
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
