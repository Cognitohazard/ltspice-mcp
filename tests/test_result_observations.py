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
    parse_source_amplitudes,
    reconciliation_observations,
    relay_observations,
    surface_observations,
    value_observations,
)
from ltspice_mcp.tools._base import format_observations
from tests.conftest import LTSPICE_TRAN_RC_VFINAL

FIXTURES = Path(__file__).parent / "fixtures"

# The source-relative trigger's canonical runaway: ±850 V oscillation, far
# below the absolute extreme-value floor but huge next to any small drive.
RUNAWAY_WAVE = 850.0 * np.sin(np.linspace(0, 30, 500))


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


class TestParseSourceAmplitudes:
    """Best-effort V-card drive-amplitude extraction for the source-relative
    extreme-value trigger. Line 1 of every deck is the title."""

    def test_dc_forms(self):
        text = "title\nV1 in 0 5\nVdd vdd 0 DC 3.3\nV2 a 0 10m\nR1 in 0 1k\n"
        amps = parse_source_amplitudes(text)
        assert amps == {
            "V1": pytest.approx(5.0),
            "Vdd": pytest.approx(3.3),
            "V2": pytest.approx(0.01),
        }

    def test_function_specs(self):
        text = (
            "title\n"
            "V1 a 0 SINE(2 1 1k)\n"  # offset+amp → 3
            "V2 b 0 PULSE(0 -5 0 1n 1n 1u 2u)\n"  # max |level| → 5
            "V3 c 0 PWL(0 0 1m 12 2m -15)\n"  # max |v| → 15
            "V4 d 0 EXP(0.5 4 0 1u 2u 1u)\n"  # max of the two levels → 4
        )
        amps = parse_source_amplitudes(text)
        assert amps["V1"] == pytest.approx(3.0)
        assert amps["V2"] == pytest.approx(5.0)
        assert amps["V3"] == pytest.approx(15.0)
        assert amps["V4"] == pytest.approx(4.0)

    def test_continuation_lines_joined(self):
        text = "title\nV1 in 0 PWL(0 0\n+ 1m 12\n+ 2m -20)\n"
        assert parse_source_amplitudes(text)["V1"] == pytest.approx(20.0)

    def test_ac_only_source_contributes_nothing(self):
        # An AC-only card drives nothing in .tran/.op; its mag (and optional
        # phase) must not be misread as a DC level — and it is PARSED (known
        # zero drive), so it must not disarm the deck either.
        text = "title\nV1 in 0 AC 1 90\nV2 a 0 5\n"
        assert parse_source_amplitudes(text) == {"V2": pytest.approx(5.0)}

    def test_ac_spec_after_dc_ignored(self):
        assert parse_source_amplitudes("title\nV1 in 0 5 AC 1\n")["V1"] == pytest.approx(5.0)

    def test_unbounded_source_disarms_whole_deck(self):
        # An unresolvable {expr} rail might be the deck's LARGEST source; a
        # max() over the remaining cards would state a false "largest
        # independent voltage source" fact. All-or-nothing: return {}.
        text = "title\nVdd vdd 0 {vin*2}\nVsig in 0 SINE(0 10m 1k)\n"
        assert parse_source_amplitudes(text) == {}

    def test_partial_function_spec_disarms(self):
        # A PULSE cut short by a non-numeric arg leaves the second (often
        # larger) level unknown — the card can't be bounded, so the deck
        # disarms rather than contributing the small first level.
        text = "title\nV1 in 0 PULSE(0.5 {vhi} 0 1n 1n 1u 2u)\nV2 a 0 5\n"
        assert parse_source_amplitudes(text) == {}

    def test_key_value_tokens_ignored_on_parseable_card(self):
        text = "title\nV2 a 0 5 Rser=0.1\n"
        assert parse_source_amplitudes(text) == {"V2": pytest.approx(5.0)}

    def test_inline_comment_trail_stripped(self):
        # A ``; comment`` trail must not read as an unparseable token and
        # falsely disarm the deck.
        text = "title\nV1 a 0 5 ; main supply\n"
        assert parse_source_amplitudes(text) == {"V1": pytest.approx(5.0)}

    def test_spaced_key_value_tolerated(self):
        # ``Rser = 0.1`` tokenizes as three words without the '=' collapse —
        # the bare 'Rser' would read as unparseable and falsely disarm.
        text = "title\nV1 a 0 5 Rser = 0.1\n"
        assert parse_source_amplitudes(text) == {"V1": pytest.approx(5.0)}

    def test_param_reference_resolved(self):
        # The common parameterized-rail idiom must not disarm the trigger.
        text = "title\n.param VDD=12\nVdd vdd 0 {vdd}\nVsig in 0 SINE(0 10m 1k)\n"
        amps = parse_source_amplitudes(text)
        assert amps["Vdd"] == pytest.approx(12.0)
        assert amps["Vsig"] == pytest.approx(0.01)

    def test_bare_voltage_unit_suffix_parses(self):
        # LTspice accepts '5V'; parse_spice_value alone raises on it (no scale
        # suffix, unit tail only) — dropping such a rail would collapse the
        # reference onto a small signal source.
        amps = parse_source_amplitudes("title\nVdd vdd 0 5V\nV2 a 0 DC 3.3V\n")
        assert amps == {"Vdd": pytest.approx(5.0), "V2": pytest.approx(3.3)}

    def test_negative_dc_uses_absolute_value(self):
        # abs() is load-bearing: without it a -15 V rail fails the amp>0 gate
        # and silently vanishes from the reference.
        amps = parse_source_amplitudes("title\nVneg a 0 -15\nVee b 0 DC -12\n")
        assert amps == {"Vneg": pytest.approx(15.0), "Vee": pytest.approx(12.0)}

    def test_current_sources_excluded(self):
        # A 5 A I-card is not a 5 V reference — only V-cards bound voltages.
        amps = parse_source_amplitudes("title\nI1 a 0 5\nV1 in 0 0.1\n")
        assert amps == {"V1": pytest.approx(0.1)}

    def test_zero_sense_source_is_parsed_not_a_drop(self):
        # A 0 V current-sense source contributes nothing but is fully parsed —
        # it must not disarm the deck.
        amps = parse_source_amplitudes("title\nVsense a b 0\nV1 in 0 5\n")
        assert amps == {"V1": pytest.approx(5.0)}

    def test_title_line_never_a_source(self):
        assert parse_source_amplitudes("Voltage reg startup 5 10\nR1 a 0 1k\n") == {}


class TestSourceRelativeTrigger:
    """Third extreme_value trigger: a voltage trace dwarfing every V-source in
    the deck (the undamped-LC class — far below the absolute floor, invisible
    to the self-relative gate because the whole trace is large)."""

    def test_fires_on_moderate_magnitude_divergence(self):
        # ±850 V oscillation from a 0.1 V drive: 8500× ≥ the 100× salience.
        wave = RUNAWAY_WAVE
        obs = value_observations({"V(n2)": wave}, source_reference=("V1", 0.1))
        assert len(obs) == 1
        assert obs[0]["code"] == "extreme_value"
        assert "severity" not in obs[0]
        assert obs[0]["evidence"]["source_name"] == "V1"
        assert obs[0]["evidence"]["source_amplitude"] == pytest.approx(0.1)
        assert obs[0]["evidence"]["peak_abs"] == pytest.approx(850.0, rel=1e-3)

    def test_silent_below_salience(self):
        # A boost/ring at 14× the source: not lifted into view.
        wave = 170.0 * np.sin(np.linspace(0, 30, 500))
        assert value_observations({"V(sw)": wave}, source_reference=("Vin", 12.0)) == []

    def test_fires_against_supply_rail_reference(self):
        # The rail (largest source) is the reference: an 850 V runaway on a
        # 12 V-railed deck is ~71× — must fire even though the small stimulus
        # that seeded it is no longer the comparison point.
        wave = RUNAWAY_WAVE
        obs = value_observations({"V(n2)": wave}, source_reference=("Vdd", 12.0))
        assert len(obs) == 1
        assert obs[0]["evidence"]["source_name"] == "Vdd"

    def test_current_traces_not_compared(self):
        # A V-source amplitude bounds voltages, not currents — no unit-crossing.
        wave = np.full(100, 850.0)
        assert value_observations({"I(L1)": wave}, source_reference=("V1", 0.1)) == []

    def test_not_doubled_when_absolute_gate_fired(self):
        # One extreme_value per trace: the absolute gate already surfaced it.
        obs = value_observations({"V(n)": np.array([1e30])}, source_reference=("V1", 0.1))
        assert len(obs) == 1
        assert "source_name" not in obs[0]["evidence"]

    def test_disarmed_without_reference(self):
        wave = RUNAWAY_WAVE
        assert value_observations({"V(n2)": wave}) == []


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

    def test_source_reference_armed_only_for_real_valued_analyses(self):
        wave = RUNAWAY_WAVE
        amps = {"V1": 0.1}

        def run(sim_type):
            return surface_observations(
                {"sim_type": sim_type},
                value_traces={"V(n2)": wave},
                value_scan="scan",
                source_amplitudes=amps,
            )

        assert any(o["code"] == "extreme_value" for o in run("Transient Analysis"))
        assert any(o["code"] == "extreme_value" for o in run("Operating Point"))
        # AC/noise traces are per-frequency small-signal gains; a .dc sweep's
        # source range comes from the .dc line, not the V-card. All disarmed.
        assert run("AC Analysis") == []
        assert run("Noise Spectral Density") == []
        assert run("DC transfer characteristic") == []
        # An unparseable raw defaults sim_type to 'Unknown' — the allow-list
        # gate (not an AC/noise exclusion list) must keep those disarmed too.
        assert run("Unknown") == []
        assert (
            surface_observations(
                {},  # no sim_type key at all
                value_traces={"V(n2)": wave},
                value_scan="scan",
                source_amplitudes=amps,
            )
            == []
        )

    def test_largest_source_is_the_reference(self):
        # The 12 V rail bounds the scale, not the 10 mV signal input — a
        # gain-of-100 amplifier output (1 V) stays quiet.
        obs = surface_observations(
            {"sim_type": "Transient Analysis"},
            value_traces={"V(out)": np.array([0.0, 1.0])},
            value_scan="scan",
            source_amplitudes={"Vsig": 0.01, "Vdd": 12.0},
        )
        assert obs == []

    def test_meas_batch_abort_links_parse_error_to_missing(self):
        summary = {
            "meas_errors": [{"directive": ".meas tran bad MAX I(Qnope)"}],
            "measurements": {},
        }
        obs = surface_observations(
            summary, requested={"meas": ["aaa_good", "bad", "zzz_good"], "four": []}
        )
        abort = [o for o in obs if o["code"] == "meas_batch_abort"]
        assert len(abort) == 1
        assert abort[0]["kind"] == "reconciliation"
        assert "earlier" in abort[0]["detail"]
        assert ".meas tran bad MAX I(Qnope)" in abort[0]["evidence"]["failed_directives"]
        assert set(abort[0]["evidence"]["missing"]) >= {"aaa_good", "zzz_good"}

    def test_no_abort_link_without_parse_error(self):
        obs = surface_observations({"measurements": {}}, requested={"meas": ["vpp"], "four": []})
        assert all(o["code"] != "meas_batch_abort" for o in obs)

    def test_four_misses_excluded_from_abort_link(self):
        # A missing .four also reconciles with reason="missing", but the
        # Fourier pipeline is unrelated to the .meas batch abort — it must not
        # inflate the count or appear in evidence.missing.
        summary = {"meas_errors": [{"directive": ".meas tran bad MAX I(Qnope)"}]}
        obs = surface_observations(summary, requested={"meas": ["good"], "four": ["V(out)"]})
        abort = next(o for o in obs if o["code"] == "meas_batch_abort")
        assert abort["evidence"]["missing"] == ["good"]
        assert abort["detail"].startswith("1 .meas requested")

    def test_no_abort_link_for_ngspice_batch_skip(self):
        # ngspice batch-mode misses are classified skipped_in_batch_mode, a
        # different mechanism — a coincident parse error must not produce the
        # LTspice batch-abort causal claim.
        summary = {
            "meas_errors": [{"directive": ".meas tran bad MAX I(Qnope)"}],
            "warnings": ["ngspice batch mode skips .meas evaluation"],
        }
        obs = surface_observations(summary, requested={"meas": ["vpp"], "four": []})
        assert all(o["code"] != "meas_batch_abort" for o in obs)

    def test_no_abort_link_when_misses_are_failed(self):
        # A FAIL'ed measurement ran and didn't trigger — that's not the abort
        # cascade, so no causal link.
        summary = {
            "meas_errors": [{"directive": ".meas tran bad MAX I(Qnope)"}],
            "measurements": {"vpp": None},
            "failed_measurements": ["vpp"],
        }
        obs = surface_observations(summary, requested={"meas": ["vpp"], "four": []})
        assert all(o["code"] != "meas_batch_abort" for o in obs)


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

    def test_source_amplitudes_kwarg_arms_the_trigger(self):
        # Pins the build_simulation_summary -> surface_observations hand-off:
        # dropping the kwarg leaves every direct-function test green while the
        # feature goes silently dead.
        raw = _make_raw_mock(
            ["time", "V(n2)"],
            np.array([0.0, 1.0]),
            {"V(n2)": np.array([0.0, 850.0])},
        )
        summary = build_simulation_summary(
            raw, None, value_scan="scan", source_amplitudes={"V1": 0.1}
        )
        ev = next(o for o in summary["observations"] if o["code"] == "extreme_value")
        assert ev["evidence"]["source_name"] == "V1"

    def test_success_summary_threads_deck_sources_end_to_end(self, tmp_path: Path):
        # Full wiring: parse_success_summary reads the deck, parses the tiny
        # SINE drive, and the recorded ~volt-scale RC output fires the
        # source-relative trigger — the hand-off chain a refactor could drop
        # at three places with all direct-function tests still green.
        from ltspice_mcp.lib import log_parser

        deck = tmp_path / "rc.cir"
        deck.write_text(
            "rc lowpass\nV1 in 0 SINE(0 1u 1k)\nR1 in out 1k\nC1 out 0 100n\n.tran 1m\n.end\n"
        )
        summary = log_parser.parse_success_summary(
            FIXTURES / "ltspice_tran_rc.raw",
            FIXTURES / "ltspice_tran_rc.log",
            0.0,
            netlist=deck,
        )
        ev = [o for o in summary["observations"] if o["code"] == "extreme_value"]
        assert ev and ev[0]["evidence"]["source_name"] == "V1"


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
