"""Tests for the result-observation surfacer (lib/result_observations.py).

Covers the pure surfacing functions and the build_simulation_summary wiring.
The surfacer surfaces facts (it does not judge), so the assertions check that
the right facts appear with the right ``kind``/``severity`` — and that benign
results surface nothing.
"""

# The tests below index Observation TypedDict keys they construct (a missing key
# fails the test loudly), so the not-required-access check adds no value here.
# pyright: reportTypedDictNotRequiredAccess=false
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.lib.raw_parser import build_simulation_summary
from ltspice_mcp.lib.result_observations import (
    parse_requested_outputs,
    reconciliation_observations,
    relay_observations,
    surface_observations,
    value_observations,
)
from ltspice_mcp.tools._base import format_observations
from tests.conftest import LTSPICE_TRAN_RC_VFINAL

FIXTURES = Path(__file__).parent / "fixtures"


def _make_raw_mock(trace_names, axis, waves, plotname="Transient Analysis"):
    raw = MagicMock()
    raw.get_raw_property.return_value = plotname
    raw.get_trace_names.return_value = trace_names
    raw.get_steps.return_value = [0]
    raw.get_axis.return_value = axis
    raw.get_wave = lambda name, step=0: waves[name]
    return raw


class TestParseRequestedOutputs:
    def test_meas_with_type_keyword(self):
        out = parse_requested_outputs(".meas TRAN vpp PP V(out)\n.MEAS AC gain MAX V(o)")
        assert out["meas"] == ["vpp", "gain"]

    def test_meas_without_type_keyword(self):
        # ``.meas <name> ...`` with no analysis-type token — name is token 1.
        out = parse_requested_outputs(".meas myval FIND V(x) AT 1m")
        assert out["meas"] == ["myval"]

    def test_four_signals(self):
        out = parse_requested_outputs(".four 1k V(out) V(in)")
        assert out["four"] == ["V(out)", "V(in)"]

    def test_ignores_non_directives(self):
        out = parse_requested_outputs("R1 n1 n2 1k\nV1 in 0 1\n.tran 1m")
        assert out["meas"] == [] and out["four"] == []


class TestRelayObservations:
    def test_errors_become_relay_observations(self):
        obs = relay_observations({"errors": ["singular matrix"]})
        assert len(obs) == 1
        assert obs[0]["code"] == "log_error"
        assert obs[0]["kind"] == "relay"
        assert obs[0]["severity"] == "error"
        assert obs[0]["evidence"]["log"] == "singular matrix"

    def test_multiline_error_uses_first_line_as_detail(self):
        obs = relay_observations({"errors": ["file(3): bad\n.meas x\n^"]})
        assert obs[0]["detail"] == "file(3): bad"
        assert "^" in obs[0]["evidence"]["log"]

    def test_duplicate_errors_deduped(self):
        obs = relay_observations({"errors": ["singular matrix", "singular matrix"]})
        assert len(obs) == 1

    def test_meas_errors_surfaced(self):
        obs = relay_observations({"meas_errors": [{"directive": ".meas x vdb(o)"}]})
        assert obs[0]["code"] == "meas_parse_error"
        assert obs[0]["severity"] == "error"

    def test_clean_summary_no_relay_observations(self):
        assert relay_observations({"measurements": {"vpp": 1.0}}) == []


class TestReconciliationObservations:
    def test_missing_measurement(self):
        obs = reconciliation_observations({"measurements": {}}, {"meas": ["vpp"]})
        assert obs[0]["code"] == "unmet_request"
        assert obs[0]["kind"] == "reconciliation"
        assert obs[0]["evidence"]["reason"] == "missing"

    def test_produced_measurement_not_flagged(self):
        obs = reconciliation_observations({"measurements": {"vpp": 1.0}}, {"meas": ["vpp"]})
        assert obs == []

    def test_case_insensitive_match(self):
        obs = reconciliation_observations({"measurements": {"VPP": 1.0}}, {"meas": ["vpp"]})
        assert obs == []

    def test_failed_measurement_classified_failed(self):
        obs = reconciliation_observations(
            {"measurements": {}, "failed_measurements": ["vpp"]}, {"meas": ["vpp"]}
        )
        assert obs[0]["evidence"]["reason"] == "failed"

    def test_failed_meas_present_as_null_key_still_classified_failed(self):
        # Real run shape: a FAILED .meas appears BOTH as a null-valued key in
        # ``measurements`` AND in ``failed_measurements`` — the empty-measurements
        # shape the test above uses never actually occurs. The null key must not
        # shadow the "failed" classification (regression: it did, so a failed
        # measurement surfaced nowhere in ``observations``).
        obs = reconciliation_observations(
            {"measurements": {"vpp": {"values": [None]}}, "failed_measurements": ["vpp"]},
            {"meas": ["vpp"]},
        )
        assert len(obs) == 1
        assert obs[0]["code"] == "unmet_request"
        assert obs[0]["evidence"]["reason"] == "failed"

    def test_ngspice_batch_skip_classified(self):
        summary = {
            "measurements": {},
            "warnings": [
                "ngspice does not evaluate .meas in batch mode when a rawfile is set (-b -r). Skipped: vpp"
            ],
        }
        obs = reconciliation_observations(summary, {"meas": ["vpp"]})
        assert obs[0]["evidence"]["reason"] == "skipped_in_batch_mode"

    def test_four_missing_when_no_fourier(self):
        obs = reconciliation_observations({"fourier": []}, {"four": ["V(out)"]})
        assert obs[0]["evidence"]["request_kind"] == "four"
        assert obs[0]["evidence"]["reason"] == "missing"


class TestValueObservations:
    def test_extreme_value_surfaced(self):
        obs = value_observations({"V(n2)": np.array([1e30, 1e30])})
        assert len(obs) == 1
        assert obs[0]["code"] == "extreme_value"
        assert obs[0]["kind"] == "value"
        assert "severity" not in obs[0]  # value facts carry no invented severity
        assert obs[0]["evidence"]["peak_abs"] == 1e30

    def test_non_finite_surfaced(self):
        obs = value_observations({"V(x)": np.array([1.0, np.nan, np.inf])})
        codes = {o["code"] for o in obs}
        assert "non_finite" in codes
        nf = next(o for o in obs if o["code"] == "non_finite")
        assert nf["evidence"]["non_finite_count"] == 2

    def test_normal_values_surface_nothing(self):
        assert value_observations({"V(out)": np.array([0.0, 3.3, 5.0])}) == []

    def test_complex_ac_uses_magnitude(self):
        # |3+4j| = 5, well under the salience threshold → nothing surfaced.
        assert value_observations({"V(o)": np.array([3 + 4j])}) == []

    def test_self_relative_blowup_surfaced_regardless_of_median_regime(self):
        # A diverged buck node sits under the 1e8 absolute floor but dwarfs its
        # own operating level. Detection must not hinge on whether the median
        # lands on the idle level or the rail, so test the rail-dominant shape
        # (~12 V for 80% of samples, then a ring to ~11 MV) — the case that the
        # adversarial review broke when the ratio gate was 1e6.
        rail_dominant = np.concatenate([np.full(800, 12.0), np.full(200, 0.05), [1.09e7]])
        obs = value_observations({"V(sw)": rail_dominant})
        assert len(obs) == 1
        assert obs[0]["code"] == "extreme_value"
        assert "severity" not in obs[0]  # surfaced fact, no invented severity
        assert obs[0]["evidence"]["peak_abs"] == pytest.approx(1.09e7)
        # median is the rail (~12 V); ratio ≈ 9e5 still clears the 1e5 gate.
        assert obs[0]["evidence"]["median_abs"] == pytest.approx(12.0)

    def test_self_relative_blowup_caught_when_body_is_uniform(self):
        # The motivating comment case: a trace flat at 12 V with one divergent
        # sample. median = 12, ratio ≈ 8.3e5 ⇒ surfaced.
        body = np.full(1000, 12.0)
        body[-1] = 1e7
        obs = value_observations({"V(n)": body})
        assert len(obs) == 1
        assert obs[0]["code"] == "extreme_value"

    def test_self_relative_blowup_caught_from_zero_baseline(self):
        # A grounded/idle node sitting at exactly 0 V that explodes late. The
        # median over all samples is 0 (most are 0), so the ratio is undefined —
        # a peak past the floor from a flat-zero baseline is itself the
        # divergence signature and must surface (the >0-filter would have hidden
        # it: dropping the zeros leaves only the spike, center == peak).
        body = np.zeros(1000)
        body[-1] = 1e7
        obs = value_observations({"V(float)": body})
        assert len(obs) == 1
        assert obs[0]["code"] == "extreme_value"
        assert obs[0]["evidence"]["peak_abs"] == pytest.approx(1e7)
        assert obs[0]["evidence"]["median_abs"] == 0.0

    def test_small_spike_from_zero_baseline_not_flagged(self):
        # Same shape but a benign peak under the floor (a logic node toggling
        # 0→3.3 V) must not trip — the magnitude gate protects it.
        body = np.zeros(1000)
        body[-1] = 3.3
        assert value_observations({"V(d)": body}) == []

    def test_legitimate_large_step_not_flagged(self):
        # A clean 0→12 V step: peak 12 V is under the magnitude gate, so the
        # ratio is never even consulted. No false positive.
        assert value_observations({"V(out)": np.array([0.0, 6.0, 12.0])}) == []

    def test_large_but_proportionate_not_flagged(self):
        # A node genuinely operating at ~50 kV (peak above the gate) whose
        # values are all the same order: ratio is ~1, so no blow-up signature.
        assert value_observations({"V(hv)": np.array([4.9e4, 5.0e4, 5.1e4])}) == []


class TestSurfaceObservations:
    def test_value_scan_off_no_value_or_coverage(self):
        obs = surface_observations({"errors": []}, value_scan="off")
        assert obs == []

    def test_skipped_large_emits_coverage(self):
        obs = surface_observations({"point_count": 500000}, value_scan="skipped_large")
        assert len(obs) == 1
        assert obs[0]["code"] == "value_scan_skipped"
        assert obs[0]["kind"] == "coverage"

    def test_combines_all_kinds(self):
        summary = {"errors": ["singular matrix"], "measurements": {}}
        obs = surface_observations(
            summary,
            requested={"meas": ["vpp"], "four": []},
            value_traces={"V(n2)": np.array([1e30])},
            value_scan="scan",
        )
        kinds = {o["kind"] for o in obs}
        assert kinds == {"relay", "reconciliation", "value"}


class TestFormatObservations:
    def test_omits_relay_renders_others(self):
        obs = [
            {"code": "log_error", "kind": "relay", "detail": "singular matrix"},
            {"code": "unmet_request", "kind": "reconciliation", "detail": ".meas vpp missing"},
            {"code": "extreme_value", "kind": "value", "detail": "V(n2) reaches |1e+30|"},
        ]
        lines = format_observations(obs)
        # Relay omitted (it prints as an Error elsewhere); the others render.
        assert not any("singular matrix" in ln for ln in lines)
        assert any("reconciliation" in ln and "vpp" in ln for ln in lines)
        assert any("value" in ln for ln in lines)

    def test_empty_and_all_relay_return_empty(self):
        assert format_observations([]) == []
        assert format_observations([{"code": "log_error", "kind": "relay", "detail": "x"}]) == []


class TestBuildSummaryWiring:
    def test_observations_always_present(self):
        raw = _make_raw_mock(
            ["time", "V(out)"], np.array([0.0, 1.0]), {"V(out)": np.array([0.0, 1.0])}
        )
        summary = build_simulation_summary(raw, None)
        assert summary["observations"] == []

    def test_scan_surfaces_extreme_value(self):
        raw = _make_raw_mock(
            ["time", "V(n2)"],
            np.array([0.0]),
            {"V(n2)": np.array([1e30])},
            plotname="Operating Point",
        )
        summary = build_simulation_summary(raw, None, value_scan="scan")
        codes = {o["code"] for o in summary["observations"]}
        assert "extreme_value" in codes

    def test_skipped_large_records_coverage(self):
        raw = _make_raw_mock(
            ["time", "V(out)"], np.array([0.0, 1.0]), {"V(out)": np.array([0.0, 1.0])}
        )
        summary = build_simulation_summary(raw, None, value_scan="skipped_large")
        assert any(o["code"] == "value_scan_skipped" for o in summary["observations"])


class TestOperatingPointValueScan:
    """An operating-point raw has no sweep axis, so its FIRST trace is a real
    node and must be scanned.

    Regression: the value scan unconditionally treated ``trace_names[0]`` as the
    sweep axis and skipped it — correct for .tran/.ac/.dc, but an .op has no axis,
    so a degenerate node that sorts first was silently never scanned. Uses a
    recorded real LTspice .op raw whose extreme node (V(hot)=1e9) is trace 0 —
    the no-axis shape ``_make_raw_mock`` (always axis-first) never constructs.
    """

    def test_extreme_first_trace_is_surfaced(self):
        from spicelib import RawRead

        raw = RawRead(str(FIXTURES / "op_extreme_node.raw"))
        # Sanity: the extreme node really is the first trace (a non-axis signal).
        assert raw.get_trace_names()[0].lower() == "v(hot)"
        summary = build_simulation_summary(raw, None, value_scan="scan")
        codes = {o["code"] for o in summary["observations"]}
        assert "extreme_value" in codes


# An observation is a surfaced FACT: only these keys may appear, and
# ``severity`` only when relayed from the simulator itself (kind == "relay").
# No verdict fields (confidence/unreliable/...) — the consumer judges.
_OBSERVATION_FACT_KEYS = {"code", "kind", "detail", "severity", "evidence"}


def _assert_observations_are_facts(observations: list) -> None:
    assert isinstance(observations, list)
    for obs in observations:
        assert set(obs) <= _OBSERVATION_FACT_KEYS, obs
        if obs["kind"] != "relay":
            assert "severity" not in obs, obs


class TestBuildSummaryRealLogPairs:
    """build_simulation_summary against recorded LTspice .raw/.log pairs.

    Everything above drives the surfacer with hand-built summary dicts and
    mocked RawRead instances; these tests run the REAL log-reading branch
    (measurement parsing, diagnostics extraction, requested-vs-produced
    reconciliation) on logs LTspice actually wrote, so the summary shapes the
    observations see are the ones the parser genuinely produces.
    """

    @staticmethod
    def _summarize(name: str, requested: dict[str, list[str]]) -> dict:
        from spicelib import RawRead

        raw = RawRead(str(FIXTURES / f"{name}.raw"))
        return build_simulation_summary(
            raw, FIXTURES / f"{name}.log", requested=requested, value_scan="scan"
        )

    def test_tran_meas_parsed_from_real_log_and_reconciled_clean(self):
        # The recorded deck carried ``.meas tran vfinal FIND V(out) AT=0.9m``;
        # the log holds ``vfinal: V(out) =0.999876166042 at 0.0009``.
        requested = parse_requested_outputs(
            ".tran 0 1m 0 5u\n.meas tran vfinal FIND V(out) AT=0.9m"
        )
        assert requested["meas"] == ["vfinal"]

        summary = self._summarize("ltspice_tran_rc", requested)

        vfinal = summary["measurements"]["vfinal"]
        assert vfinal["values"] == [pytest.approx(LTSPICE_TRAN_RC_VFINAL, rel=1e-9)]
        assert vfinal["at"] == pytest.approx(0.9e-3, rel=1e-6)
        assert "failed_measurements" not in summary
        assert "errors" not in summary
        # Requested .meas was produced, values are healthy, nothing relayed:
        # a clean run surfaces NO observations (empty list != verified, but
        # nothing tripped a check here).
        assert summary["observations"] == []

    def test_ac_log_without_meas_surfaces_nothing(self):
        # The AC deck requested no .meas/.four, and its real log carries none:
        # the log branch must not invent measurements or observations.
        requested = parse_requested_outputs(
            "V1 in 0 AC 1\nR1 in out 1k\nC1 out 0 159.15n\n.ac dec 20 10 100k"
        )
        assert requested == {"meas": [], "four": []}

        summary = self._summarize("ltspice_ac_rc", requested)

        assert "measurements" not in summary
        assert "failed_measurements" not in summary
        assert "errors" not in summary
        assert summary["observations"] == []

    def test_requested_meas_absent_from_real_log_reconciled_as_missing(self):
        # Deck asks for a .meas, but the recorded DC log carries no
        # measurement at all — the real parse-then-reconcile chain must
        # surface exactly one unmet_request fact naming it.
        requested = parse_requested_outputs(".dc V1 0 5 0.5\n.meas dc vhalf FIND V(out) AT 2.5")
        assert requested["meas"] == ["vhalf"]

        summary = self._summarize("ltspice_dc_div", requested)

        assert "measurements" not in summary
        unmet = [o for o in summary["observations"] if o["code"] == "unmet_request"]
        assert len(unmet) == 1
        assert unmet[0]["kind"] == "reconciliation"
        assert unmet[0]["evidence"] == {
            "name": "vhalf",
            "request_kind": "meas",
            "reason": "missing",
        }
        _assert_observations_are_facts(summary["observations"])


class TestValueScanPointBudget:
    """The value-scan gate keys off the ESTIMATED total sample count (axis points
    × non-axis traces), not single-vs-multi-point and not axis points alone.

    A normal multi-point run is fully scanned (value facts surfaced, no
    ``value_scan_skipped``); only a run whose total samples exceed
    ``_VALUE_SCAN_SAMPLE_BUDGET`` skips the scan and records the coverage gap.
    The tests drive the real ``parse_success_summary`` gate against a recorded
    221-point LTspice .tran raw, monkeypatching the budget so the SAME raw
    crosses the boundary — pinning the gate without a multi-million-sample
    fixture.
    """

    def test_small_multipoint_run_is_scanned_not_skipped(self):
        from ltspice_mcp.lib import log_parser

        summary = log_parser.parse_success_summary(
            FIXTURES / "ltspice_tran_rc.raw", FIXTURES / "ltspice_tran_rc.log", 0.0
        )
        # 221 points × a few traces is far under the 5M-sample budget: it is
        # scanned, so the skip-coverage observation must be absent (a benign run
        # surfaces no value facts here either).
        assert not any(o["code"] == "value_scan_skipped" for o in summary["observations"])

    def test_large_run_surfaces_skipped_scan(self, monkeypatch):
        from ltspice_mcp.lib import log_parser

        # Drop the budget below the fixture's sample count so the SAME multi-point
        # raw now exceeds it — exercises the gate's >budget branch.
        monkeypatch.setattr(log_parser, "_VALUE_SCAN_SAMPLE_BUDGET", 1)

        summary = log_parser.parse_success_summary(
            FIXTURES / "ltspice_tran_rc.raw", FIXTURES / "ltspice_tran_rc.log", 0.0
        )
        skipped = [o for o in summary["observations"] if o["code"] == "value_scan_skipped"]
        assert len(skipped) == 1
        assert skipped[0]["kind"] == "coverage"
        assert skipped[0]["evidence"]["point_count"] == 221

    def test_value_scan_gate_counts_traces_not_just_points(self, monkeypatch):
        """A wide result (moderate points, many traces) skips even when the
        point count alone is under budget — the gate must multiply by trace
        count, or a wide node dump would load every trace on completion."""
        from spicelib import RawRead

        from ltspice_mcp.lib import log_parser

        header = RawRead(str(FIXTURES / "ltspice_tran_rc.raw"), traces_to_read=None)
        non_axis = max(0, len(header.get_trace_names()) - 1)
        assert non_axis >= 2, "fixture needs >=2 non-axis traces to exercise trace gating"

        # Budget = the point count (221). Under a points-only gate this scans
        # (221 <= 221); under the real total-sample gate it skips because
        # 221 * non_axis > 221.
        monkeypatch.setattr(log_parser, "_VALUE_SCAN_SAMPLE_BUDGET", 221)
        summary = log_parser.parse_success_summary(
            FIXTURES / "ltspice_tran_rc.raw", FIXTURES / "ltspice_tran_rc.log", 0.0
        )
        skipped = [o for o in summary["observations"] if o["code"] == "value_scan_skipped"]
        assert len(skipped) == 1, "wide-but-few-points run must skip the value scan"
