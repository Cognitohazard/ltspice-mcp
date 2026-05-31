"""Tests for the 2026-05-29 open-followups batch.

Covers the new tools (schematic_from_netlist §3a, trace_net §3b) and the
polish items: query_value magnitude_linear (V7-IMP-9), edge_metrics level
override (V7-P2-1, see test_signal_analysis), step_get snap-warning +
default-at label (V7-P2-2 / V7-FR-5), and the add_component floating-pin
filtering / O(1) wire index (V7-FR-9).
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import NetlistError, ResultError
from ltspice_mcp.lib.raw_parser import sample_to_dict
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    BodeMetricsInput,
    QueryValueInput,
    handle_bode_metrics,
    handle_query_value,
)
from ltspice_mcp.tools.circuit import (
    AddComponentInput,
    CircuitReadInput,
    MoveComponentInput,
    ResetSchematicInput,
    SchematicFromNetlistInput,
    StepGetInput,
    TraceNetInput,
    _build_on_wire_predicate,
    _parse_netlist_for_synth,
    _point_on_segment,
    handle_add_component,
    handle_move_component,
    handle_read_circuit,
    handle_reset_schematic,
    handle_schematic_from_netlist,
    handle_step_get,
    handle_trace_net,
)

RC_NETLIST = (
    "* RC low-pass filter\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.ac dec 10 1 100k\n.end\n"
)


def _inject_raw(state: SessionState, path: Path, raw: MagicMock) -> None:
    path.write_bytes(b"placeholder")
    state.results.set(path, raw)


def _read_bytes(p: Path) -> bytes:
    """Sync file read (keeps blocking pathlib I/O out of async test bodies)."""
    return p.read_bytes()


# ---------------------------------------------------------------------------
# _parse_netlist_for_synth (pure)
# ---------------------------------------------------------------------------


class TestParseNetlistForSynth:
    def test_basic_rc(self):
        instances, directives, skipped, _warnings = _parse_netlist_for_synth(RC_NETLIST)
        refs = {i.ref for i in instances}
        assert refs == {"V1", "R1", "C1"}
        assert not skipped
        assert any(d.lower().startswith(".ac") for d in directives)
        # Title line dropped, .end dropped.
        assert all(not d.lower().startswith(".end") for d in directives)

    def test_symbol_and_nodes_mapping(self):
        instances, *_ = _parse_netlist_for_synth(RC_NETLIST)
        by_ref = {i.ref: i for i in instances}
        assert by_ref["R1"].symbol == "res"
        assert by_ref["C1"].symbol == "cap"
        assert by_ref["V1"].symbol == "voltage"
        assert by_ref["R1"].nodes == ("in", "out")
        assert by_ref["R1"].value == "1k"

    def test_multi_token_source_value_preserved(self):
        instances, *_ = _parse_netlist_for_synth(
            "* t\nV1 in 0 SINE(0 1 1k) AC 1\nR1 in 0 1k\n.end\n"
        )
        v1 = next(i for i in instances if i.ref == "V1")
        assert v1.nodes == ("in", "0")
        assert v1.value == "SINE(0 1 1k) AC 1"

    def test_unsupported_element_skipped(self):
        instances, _, skipped, _ = _parse_netlist_for_synth(
            "* t\nM1 d g s b NMOS\nR1 a b 1k\n.end\n"
        )
        assert {i.ref for i in instances} == {"R1"}
        assert any(s["ref"] == "M1" for s in skipped)

    def test_subckt_def_warns(self):
        _, _, _, warnings = _parse_netlist_for_synth(
            "* t\nR1 a b 1k\n.subckt amp in out\nR2 in out 1k\n.ends\n.end\n"
        )
        assert any("ubcircuit" in w for w in warnings)

    def test_malformed_body_skipped_not_raised(self):
        # "R1 net(a b 1k" lexes as an instance (R prefix) but tokenize_body
        # raises on the unbalanced paren — it must be skipped, not crash.
        instances, _, skipped, _ = _parse_netlist_for_synth(
            "* t\nR1 net(a b 1k\nC1 a 0 1u\n.end\n"
        )
        assert {i.ref for i in instances} == {"C1"}
        assert any(s["ref"] == "R1" and "tokenize" in s["reason"] for s in skipped)

    def test_no_title_keeps_first_instance(self):
        # F2: a bare netlist fragment (no '*' title) must NOT silently drop its
        # first card — that used to delete the source (V1) and leave a dead
        # circuit with no feedback.
        instances, _directives, skipped, warnings = _parse_netlist_for_synth(
            "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1u\n.ac dec 10 1 100k\n.end\n"
        )
        assert {i.ref for i in instances} == {"V1", "R1", "C1"}
        assert not skipped
        assert any("title" in w.lower() for w in warnings)

    def test_title_comment_dropped_without_warning(self):
        # RC_NETLIST has a leading '* RC low-pass filter' comment: it is the
        # conventional deck title, dropped silently, all instances kept.
        instances, _directives, _skipped, warnings = _parse_netlist_for_synth(RC_NETLIST)
        assert {i.ref for i in instances} == {"V1", "R1", "C1"}
        assert not any("title" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# schematic_from_netlist (§3a)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSchematicFromNetlist:
    async def test_roundtrip_through_read_circuit(self, asc_state: SessionState, work_dir: Path):
        res = await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_rc", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["placed"] == 3
        assert set(sc["nets"]) == {"in", "out", "0"}
        assert sc["directive_count"] == 1
        assert not sc["skipped"]

        read = await handle_read_circuit(CircuitReadInput(path="synth_rc.asc"), asc_state)
        rsc = read.structuredContent
        assert rsc["type"] == "asc"
        refs = {c["reference"] for c in rsc["components"]}
        assert refs == {"R1", "C1", "V1"}
        values = {c["reference"]: c["value"] for c in rsc["components"]}
        assert values["R1"] == "1k"
        assert values["C1"] == "1u"
        assert values["V1"] == "AC 1"
        label_texts = {lbl["text"] for lbl in rsc["labels"]}
        assert {"in", "out", "0"} <= label_texts
        assert any(d.lower().startswith(".ac") for d in rsc["directives"])

    async def test_overwrite_after_read_uses_fresh_stub(
        self, asc_state: SessionState, work_dir: Path
    ):
        # Regression (Codex review): the overwrite path must populate the fresh
        # blank stub, not an editor cached from a prior read of the old content.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="ow_read", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        await handle_read_circuit(CircuitReadInput(path="ow_read.asc"), asc_state)  # caches editor
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(
                name="ow_read", content="* t\nR9 a b 2k\nC9 b 0 2u\n.end\n", overwrite=True
            ),
            asc_state,
        )
        read = await handle_read_circuit(CircuitReadInput(path="ow_read.asc"), asc_state)
        assert read.structuredContent is not None
        refs = {c["reference"] for c in read.structuredContent["components"]}
        assert refs == {"R9", "C9"}  # only the new content, not R1/C1 from the first synth

    async def test_reports_skipped_unsupported(self, asc_state: SessionState):
        content = "* t\nM1 d g s NMOS\nR1 in out 1k\nC1 out 0 1u\n.end\n"
        res = await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_skip", content=content, overwrite=True),
            asc_state,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["placed"] == 2  # R1, C1
        assert any(s["ref"] == "M1" for s in sc["skipped"])

    async def test_refuses_overwrite_by_default(self, asc_state: SessionState, work_dir: Path):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_dup", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        with pytest.raises(NetlistError, match="already exists"):
            await handle_schematic_from_netlist(
                SchematicFromNetlistInput(name="synth_dup", content=RC_NETLIST),
                asc_state,
            )

    async def test_nothing_to_place_raises(self, asc_state: SessionState):
        with pytest.raises(NetlistError, match="Nothing to place"):
            await handle_schematic_from_netlist(
                SchematicFromNetlistInput(name="synth_empty", content="* just a title\n"),
                asc_state,
            )


# ---------------------------------------------------------------------------
# trace_net (§3b)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTraceNet:
    async def test_name_based_net_on_synth_output(self, asc_state: SessionState):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_rc", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        # R1.1 is on node "in" (SpiceOrder 1). V1.+ is also on "in" — they are
        # at different coordinates connected only by the shared label name.
        res = await handle_trace_net(TraceNetInput(path="trace_rc.asc", pin="R1.1"), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        refs = {p["reference"] for p in sc["pins"]}
        assert refs == {"R1", "V1"}
        assert sc["is_shorted"] is False

    async def test_trace_by_net_name(self, asc_state: SessionState):
        # net:in matches one FLAG per pin (V1.+ and R1.1) — _resolve_pin would
        # refuse the ambiguity, but trace_net seeds from a match and name-merges.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_byname", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        res = await handle_trace_net(
            TraceNetInput(path="trace_byname.asc", pin="net:in"), asc_state
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["labels"] == ["in"]
        assert {p["reference"] for p in sc["pins"]} == {"R1", "V1"}

    async def test_trace_by_missing_net_name_raises(self, asc_state: SessionState):
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="trace_miss", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        with pytest.raises(NetlistError, match="not found"):
            await handle_trace_net(
                TraceNetInput(path="trace_miss.asc", pin="net:nonexistent"), asc_state
            )

    async def test_short_detection(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "short.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nWIRE 0 0 100 0\nFLAG 0 0 a\nFLAG 100 0 b\n")
        res = await handle_trace_net(TraceNetInput(path="short.asc", x=0, y=0), asc_state)
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["is_shorted"] is True
        assert set(sc["labels"]) == {"a", "b"}

    async def test_empty_coordinate_raises(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "empty.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\nFLAG 0 0 a\n")
        with pytest.raises(NetlistError, match="Nothing found"):
            await handle_trace_net(TraceNetInput(path="empty.asc", x=500, y=500), asc_state)


# ---------------------------------------------------------------------------
# _build_on_wire_predicate (V7-FR-9 O(1) wire index)
# ---------------------------------------------------------------------------


class TestOnWirePredicate:
    def test_matches_point_on_segment(self):
        segments = [((0, 0), (100, 0)), ((100, 0), (100, 80)), ((50, 50), (50, 50))]
        on_wire = _build_on_wire_predicate(segments)
        probes = [(0, 0), (50, 0), (100, 0), (100, 40), (100, 80), (50, 50), (10, 10), (200, 0)]
        for p in probes:
            expected = any(_point_on_segment(p, v1, v2) for v1, v2 in segments)
            assert on_wire(p) == expected, p

    def test_endpoints_and_spans(self):
        on_wire = _build_on_wire_predicate([((0, 0), (0, 100))])
        assert on_wire((0, 0))
        assert on_wire((0, 50))
        assert on_wire((0, 100))
        assert not on_wire((10, 50))


# ---------------------------------------------------------------------------
# add_component floating-pin filtering (V7-FR-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAddComponentFloatingFilter:
    async def test_only_new_component_floating_pins(self, asc_state: SessionState, work_dir: Path):
        asc = work_dir / "build.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        # First component: both pins float.
        await handle_add_component(
            AddComponentInput(path="build.asc", reference="R1", symbol="res", x=100, y=100),
            asc_state,
        )
        # Second component placed far away: its warnings must NOT re-list R1's
        # floating pins (the O(n^2) spam this fix removes).
        res = await handle_add_component(
            AddComponentInput(path="build.asc", reference="R2", symbol="res", x=400, y=100),
            asc_state,
        )
        vw = res.structuredContent.get("validation_warnings", [])
        refs = {w["ref"] for w in vw}
        assert refs <= {"R2"}
        assert "R1" not in refs


# ---------------------------------------------------------------------------
# sample_to_dict magnitude_linear (V7-IMP-9)
# ---------------------------------------------------------------------------


class TestSampleToDict:
    def test_complex_sample_has_magnitude_linear(self):
        d = sample_to_dict(complex(0.0, 1.0))
        assert d["magnitude_linear"] == pytest.approx(1.0)
        assert d["magnitude_db"] == pytest.approx(0.0, abs=1e-9)
        assert d["phase_deg"] == pytest.approx(90.0)

    def test_real_sample_unchanged(self):
        d = sample_to_dict(3.5)
        assert d == {"value": 3.5}
        assert "magnitude_linear" not in d


# ---------------------------------------------------------------------------
# query_value magnitude_linear on AC (V7-IMP-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestQueryValueMagnitudeLinear:
    async def test_ac_returns_magnitude_linear(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        freq = np.array([10.0, 100.0, 1000.0])
        volt = np.array([1 + 0j, 0.7 + 0.7j, 0.1 + 0j])
        raw.get_axis.return_value = freq
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: volt
        path = work_dir / "ac.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_query_value(
            QueryValueInput(raw_file="ac.raw", signal="V(out)", at="100"), state_no_sim
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["magnitude_linear"] == pytest.approx(abs(0.7 + 0.7j))
        assert "magnitude_db" in sc


# ---------------------------------------------------------------------------
# step_get snap-warning + default-at (V7-P2-2 / V7-FR-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStepGet:
    async def test_raw_axis_snap_warning(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["Rval", "V(out)"]
        raw.get_axis.return_value = np.array([500.0, 1000.0, 2000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0, 3.0])
        path = work_dir / "dc.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dc.raw", axis="Rval", value="99999", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["exact_match"] is False
        assert sc["actual_value"] == 2000.0
        assert sc.get("warnings")

    async def test_raw_axis_exact_match_no_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["Rval", "V(out)"]
        raw.get_axis.return_value = np.array([500.0, 1000.0, 2000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0, 3.0])
        path = work_dir / "dc2.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dc2.raw", axis="Rval", value="1k", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["exact_match"] is True
        assert sc["actual_value"] == 1000.0
        assert not sc.get("warnings")

    async def test_raw_axis_complex_ac_keeps_magnitude(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # axis name "frequency" == trace 0 → raw-axis branch on an AC raw.
        # The complex sample must survive as magnitude/phase, not float()'d.
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        raw.get_axis.return_value = np.array([10.0, 100.0, 1000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1 + 0j, 0.7 + 0.7j, 0.1 + 0j])
        path = work_dir / "acaxis.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="acaxis.raw", axis="frequency", value="100", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert "magnitude_linear" in sc
        assert "magnitude_db" in sc
        assert "value" not in sc  # complex sample, not a real scalar
        assert sc["magnitude_linear"] == pytest.approx(abs(0.7 + 0.7j))
        assert not sc.get("warnings")

    async def test_raw_axis_interior_offgrid_no_clamp_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Dense continuous axis: an interior off-grid request is a normal
        # nearest-neighbour lookup, NOT an out-of-range clamp — no warning.
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["v1", "V(out)"]
        raw.get_axis.return_value = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        path = work_dir / "dense.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dense.raw", axis="v1", value="1.01", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["actual_value"] == 1.0
        assert sc["exact_match"] is False  # off-grid, honest
        assert not sc.get("warnings")  # but interior → not "clamped"

    async def test_step_lookup_inside_range_snap_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # temp=50 sits between discrete steps {27, 85}: nearest-step used, not
        # "clamped" (it is inside the swept range).
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"temp": 27.0}, {"temp": 85.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "tempstep.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="tempstep.raw", axis="temp", value="50", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["actual_value"] == 27.0
        assert sc["exact_match"] is False
        assert any("nearest step" in w for w in sc.get("warnings", []))
        assert all("clamped" not in w for w in sc.get("warnings", []))

    async def test_step_lookup_default_at_label(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        # Axis name != requested axis → falls to the .step parameter lookup.
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}, {"Rval": 2000.0}]
        raw.get_axis.return_value = np.array([10.0, 100.0, 1000.0])
        raw.get_wave = lambda name, step=0: np.array([0.5, 0.6, 0.7])
        path = work_dir / "step.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="step.raw", axis="Rval", value="1000", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["step_index"] == 1
        assert sc["exact_match"] is True
        assert sc["actual_at"] == 10.0
        assert any("No 'at' given" in w for w in sc.get("warnings", []))


# ---------------------------------------------------------------------------
# bode_metrics consolidation (§2a) + query_value step absorb (§2b)
# ---------------------------------------------------------------------------


def _ac_raw() -> MagicMock:
    """An AC raw mock: real frequency axis + complex first-order-LPF response."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "AC Analysis"
    raw.get_trace_names.return_value = ["frequency", "V(out)"]
    freq = np.logspace(0, 5, 200)  # 1 Hz .. 100 kHz
    fc = 1591.5
    H = 1.0 / (1.0 + 1j * (freq / fc))
    raw.get_axis.return_value = freq
    raw.get_steps.return_value = [0]
    raw.get_wave = lambda name, step=0: H
    return raw


@pytest.mark.asyncio
class TestBodeMetrics:
    async def test_point_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode.raw", signal="V(out)", mode="point", frequencies=["1k"]
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert "points" in res.structuredContent

    async def test_crossing_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode2.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode2.raw",
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        cs = res.structuredContent["crossings"]
        assert cs and abs(cs[0]["frequency_hz"] - 1591.5) / 1591.5 < 0.05

    async def test_slope_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode3.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode3.raw", signal="V(out)", mode="slope", f_low="10k", f_high="100k"
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        # First-order LPF stopband ≈ -20 dB/decade.
        assert res.structuredContent["slope_db_per_decade"] < -15

    async def test_filter_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode4.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        res = await handle_bode_metrics(
            BodeMetricsInput(raw_file="bode4.raw", signal="V(out)", mode="filter"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert "filter_type" in res.structuredContent

    async def test_crossing_requires_quantity_and_level(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = work_dir / "bode5.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        with pytest.raises(ResultError, match="requires 'quantity' and 'level'"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="bode5.raw", signal="V(out)", mode="crossing"),
                state_no_sim,
            )

    async def test_slope_requires_bounds(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode6.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        with pytest.raises(ResultError, match="requires 'f_low' and 'f_high'"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="bode6.raw", signal="V(out)", mode="slope"),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestQueryValueStepAbsorb:
    async def test_step_axis_dispatches_to_step_lookup(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}, {"Rval": 2000.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "qstep.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_query_value(
            QueryValueInput(
                raw_file="qstep.raw", signal="V(out)", step_axis="Rval", step_value="1000"
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert res.structuredContent["step_index"] == 1
        assert res.structuredContent["exact_match"] is True

    async def test_step_axis_requires_step_value(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "qstep2.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        with pytest.raises(ResultError, match="step_value"):
            await handle_query_value(
                QueryValueInput(raw_file="qstep2.raw", signal="V(out)", step_axis="Rval"),
                state_no_sim,
            )

    async def test_requires_at_without_step_axis(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "qstep3.raw"
        _inject_raw(state_no_sim, path, _ac_raw())
        with pytest.raises(ResultError, match="'at' is required"):
            await handle_query_value(
                QueryValueInput(raw_file="qstep3.raw", signal="V(out)"), state_no_sim
            )


# ---------------------------------------------------------------------------
# §4b — bode_metrics(all_steps=True): per-step AC metrics in one call
# ---------------------------------------------------------------------------


def _stepped_ac_raw(fcs: list[float]) -> MagicMock:
    """A stepped AC raw: one first-order-LPF response per cutoff in ``fcs``."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "AC Analysis"
    raw.get_trace_names.return_value = ["frequency", "V(out)"]
    freq = np.logspace(0, 5, 200)
    responses = [1.0 / (1.0 + 1j * (freq / fc)) for fc in fcs]
    raw.get_axis.return_value = freq
    raw.get_steps.return_value = [{"fc": fc} for fc in fcs]
    raw.get_wave = lambda name, step=0: responses[step]
    return raw


@pytest.mark.asyncio
class TestBodeMetricsAllSteps:
    async def test_crossing_per_step(self, state_no_sim: SessionState, work_dir: Path):
        fcs = [500.0, 5000.0]
        path = work_dir / "stepped.raw"
        _inject_raw(state_no_sim, path, _stepped_ac_raw(fcs))
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="stepped.raw",
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
                all_steps=True,
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["all_steps"] is True
        assert sc["step_count"] == 2
        steps = sc["steps"]
        assert [s["step"] for s in steps] == [0, 1]
        # The -3 dB crossing of each step tracks that step's cutoff.
        for i, fc in enumerate(fcs):
            cs = steps[i]["crossings"]
            assert cs and abs(cs[0]["frequency_hz"] - fc) / fc < 0.05

    async def test_single_step_warns(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "onestep.raw"
        _inject_raw(state_no_sim, path, _ac_raw())  # get_steps == [0]
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="onestep.raw", signal="V(out)", mode="filter", all_steps=True
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["step_count"] == 1
        assert len(sc["steps"]) == 1
        assert any("not stepped" in w for w in sc.get("warnings", []))

    async def test_all_steps_still_validates_mode_args(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # all_steps must enforce the same per-mode required args as single-step.
        path = work_dir / "stepped2.raw"
        _inject_raw(state_no_sim, path, _stepped_ac_raw([500.0, 5000.0]))
        with pytest.raises(ResultError, match="requires 'f_low' and 'f_high'"):
            await handle_bode_metrics(
                BodeMetricsInput(
                    raw_file="stepped2.raw", signal="V(out)", mode="slope", all_steps=True
                ),
                state_no_sim,
            )

    async def test_all_steps_total_failure_raises(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Regression: a non-AC raw makes every step fail. all_steps must surface
        # a real error, not a "success" full of buried per-step errors.
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [0]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "tran.raw"
        _inject_raw(state_no_sim, path, raw)
        with pytest.raises(ResultError, match="AC analysis"):
            await handle_bode_metrics(
                BodeMetricsInput(
                    raw_file="tran.raw", signal="V(out)", mode="filter", all_steps=True
                ),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# §3d — reset_schematic: revert an .asc to its pre-first-edit state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResetSchematic:
    async def test_revert_after_edit(self, asc_state: SessionState, asc_file: Path):
        original = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200),
            asc_state,
        )
        assert _read_bytes(asc_file) != original  # edit landed
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is True
        assert _read_bytes(asc_file) == original  # byte-exact restore

    async def test_nothing_to_revert(self, asc_state: SessionState, asc_file: Path):
        # No in-session edit captured → reverted=False, not an error.
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False
        assert res.structuredContent["bytes"] is None

    async def test_snapshot_is_pre_first_edit(self, asc_state: SessionState, asc_file: Path):
        # Two edits, then reset → restores the state before the FIRST edit.
        original = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200), asc_state
        )
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=400, y=400), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert _read_bytes(asc_file) == original

    async def test_reset_then_reedit_resnapshots(self, asc_state: SessionState, asc_file: Path):
        # After a reset the snapshot is dropped; a new edit establishes a fresh
        # restore point, and a reset with no new edit finds nothing.
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=300, y=200), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        res = await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False
        after_reset = _read_bytes(asc_file)
        await handle_move_component(
            MoveComponentInput(path=asc_file.name, reference="R1", x=500, y=500), asc_state
        )
        await handle_reset_schematic(ResetSchematicInput(path=asc_file.name), asc_state)
        assert _read_bytes(asc_file) == after_reset

    async def test_requires_asc(self, state_no_sim: SessionState, work_dir: Path):
        cir = work_dir / "x.cir"
        cir.write_text("* t\nR1 a b 1k\n.end\n")
        with pytest.raises(NetlistError, match=r"requires an \.asc"):
            await handle_reset_schematic(ResetSchematicInput(path="x.cir"), state_no_sim)

    async def test_synth_new_file_not_revertible_to_stub(self, asc_state: SessionState):
        # Regression: schematic_from_netlist writes a blank stub then edits it.
        # reset_schematic must NOT restore that 30-byte stub for a NEW file —
        # there's no pre-session state, so it reports reverted=False.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_reset", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        res = await handle_reset_schematic(ResetSchematicInput(path="synth_reset.asc"), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is False

    async def test_synth_overwrite_reverts_to_original(
        self, asc_state: SessionState, work_dir: Path
    ):
        # overwrite=true synth over an existing file → reset restores the
        # ORIGINAL bytes (captured before the overwrite), not the blank stub.
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(name="synth_ow", content=RC_NETLIST, overwrite=True),
            asc_state,
        )
        original = _read_bytes(work_dir / "synth_ow.asc")
        await handle_schematic_from_netlist(
            SchematicFromNetlistInput(
                name="synth_ow", content="* t\nR9 a b 2k\nC9 b 0 2u\n.end\n", overwrite=True
            ),
            asc_state,
        )
        assert _read_bytes(work_dir / "synth_ow.asc") != original
        res = await handle_reset_schematic(ResetSchematicInput(path="synth_ow.asc"), asc_state)
        assert res.structuredContent is not None
        assert res.structuredContent["reverted"] is True
        assert _read_bytes(work_dir / "synth_ow.asc") == original
