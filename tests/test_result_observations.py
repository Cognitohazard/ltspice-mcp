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

from ltspice_mcp.lib.raw_parser import build_simulation_summary
from ltspice_mcp.lib.result_observations import (
    parse_requested_outputs,
    reconciliation_observations,
    relay_observations,
    surface_observations,
    value_observations,
)
from ltspice_mcp.tools._base import format_observations

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
            "warnings": ["ngspice cannot evaluate .meas in batch mode. skipped: vpp"],
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
