"""Contract tests for the Python API raw wrapper and curated primitive facade."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import pytest

import ltspice_mcp.api as api_module
from ltspice_mcp.api import Api, RawResult
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.state import SessionState
from tests.conftest import (
    LTSPICE_TRAN_RC_VFINAL,
    SyncApi,
    make_experiment_job,
    patch_stub_bootstrap,
    stage_recorded_fixture,
)
from tests.test_decoded_raw import header

EXPECTED_ALL = [
    "Api",
    "RawResult",
    "ApiError",
    "ApiCallError",
    "ApiSessionError",
    "ApiClosedError",
    "ApiInterrupted",
    "ApiInternalError",
    "ApiValidationError",
    "prepare_ac_arrays",
    "unwrap_phase_safe",
    "log_interp",
    "log_interp_complex",
    "detect_crossings",
    "find_crossings_any_quantity",
    "gain_at_frequencies",
    "compute_filter_metrics",
    "compute_stability_metrics",
    "compute_roll_off",
    "compute_resonances",
    "compute_return_loss",
    "integrate_noise",
    "classify_filter",
    "window_and_clean",
    "analyze_edge",
    "analyze_pulse_response",
    "analyze_disturbance_response",
    "analyze_timing_between",
    "analyze_periodic",
    "analyze_thd",
    "compute_signal_stats",
    "time_weighted_quantiles",
    "compute_measurement_stats",
    "analyze_ac_structure",
    "parse_spice_value",
    "Quantity",
    "SearchDirection",
    "CrossingDirection",
    "FilterType",
    "StabilityLabel",
    "CornerKind",
    "CrossingWithQuantity",
    "GainAtPoint",
    "ReturnLossOutput",
    "FilterMetricsOutput",
    "StabilityMetricsOutput",
    "Crossover",
    "PhaseMargin",
    "GainMargin",
    "RollOffOutput",
    "ResonancesOutput",
    "ResonancePeak",
    "NoiseIntegralOutput",
    "EdgeMetricsOutput",
    "PulseResponseOutput",
    "DisturbanceResponseOutput",
    "TimingBetweenOutput",
    "PeriodicMetricsOutput",
    "SignalStatsOutput",
    "TimeWeightedQuantilesOutput",
    "ThdOutput",
    "HarmonicEntry",
    "MeasurementStatsEntry",
    "HistogramBin",
    "AcStructureResult",
    "Corner",
    "Observation",
]

METRIC_NAMES = EXPECTED_ALL[9:34]
# Not a metric: a value reader, published because SPICE literals cross the
# boundary in both directions and nothing else on the facade parses one.
VALUE_HELPER_NAMES = EXPECTED_ALL[34:35]
ALIAS_NAMES = EXPECTED_ALL[35:41]
OUTPUT_TYPE_NAMES = EXPECTED_ALL[41:]


def test_raw_result_xor_step_slicing_and_mutation_isolation(
    state_no_sim: SessionState,
    work_dir: Path,
) -> None:
    api = SyncApi(state_no_sim)
    with pytest.raises(TypeError, match="exactly one"):
        api.load_raw()
    with pytest.raises(TypeError, match="exactly one"):
        api.load_raw(raw_path="one.raw", job_id="job-one")

    raw_path = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    # The receipt hands a caller the raw path; it is the one natural positional.
    assert api.load_raw(raw_path).source == api.load_raw(raw_path=raw_path).source
    result = api.load_raw(raw_path=raw_path)
    assert isinstance(result, RawResult)
    assert result.source == raw_path.resolve()
    assert result.analysis_type == "transient"
    assert result.step_count == 3
    assert result.steps == [{"r": 1.0}, {"r": 22.0}, {"r": 680.0}]
    assert len(result.steps) == result.step_count

    first = result.trace("v(OUT)", step=0)
    last = result.trace("V(out)", step=2)
    assert first.size > 0
    assert last.size > 0
    assert not np.array_equal(first, last)
    with pytest.raises(ResultError, match=r"Signal 'V\(missing\)' not found"):
        result.trace("V(missing)")

    original_wave = first.copy()
    original_axis = result.axis(step=0)
    first[:] = -12345
    mutated_axis = result.axis(step=0)
    mutated_axis[:] = -67890
    result.steps[0]["r"] = -1

    reloaded = api.load_raw(raw_path=raw_path)
    np.testing.assert_array_equal(reloaded.trace("V(out)", step=0), original_wave)
    np.testing.assert_array_equal(reloaded.axis(step=0), original_axis)
    assert reloaded.steps[0] == {"r": 1.0}


def test_raw_result_preserves_ac_complex_trace_and_real_axis(
    state_no_sim: SessionState,
    work_dir: Path,
) -> None:
    raw_path = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    result = SyncApi(state_no_sim).load_raw(raw_path=raw_path)
    trace = result.trace("V(out)")
    axis = result.axis()

    assert result.analysis_type == "ac"
    assert result.dialect == "ltspice"
    assert np.issubdtype(trace.dtype, np.complexfloating)
    assert np.iscomplexobj(trace)
    assert np.issubdtype(axis.dtype, np.floating)
    assert not np.iscomplexobj(axis)


def test_raw_result_uses_captured_steps_without_reopening_sibling_log(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = work_dir / "fallback.raw"
    raw_path.write_bytes(b"placeholder")
    raw_path.with_suffix(".log").write_text(
        ".step gain=1\n.step gain=2\n.step gain=4\n",
        encoding="utf-8",
    )

    captured = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Transient Analysis",
                    [("time", "time"), ("V(out)", "voltage")],
                    6,
                    flags=("real", "stepped"),
                ),
                [
                    np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
                    np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
                ],
                snapshot_id="captured-steps",
                steps=[{"gain": 10}, {"gain": 20}, {"gain": 40}],
                step_offsets=[0, 2, 4],
            )
        ]
    )

    async def fake_load(source: services.AnalysisSource, _state: SessionState) -> DecodedRaw:
        assert source.raw == raw_path
        return captured

    async def forbidden_parent_parse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A decoded RAW consumer must not reopen the parent log")

    monkeypatch.setattr(services, "load_raw", fake_load)
    monkeypatch.setattr(services, "bounded_parse", forbidden_parent_parse)
    result = SyncApi(state_no_sim).load_raw(raw_path=raw_path)

    assert result.steps == [{"gain": 10}, {"gain": 20}, {"gain": 40}]
    np.testing.assert_array_equal(result.trace("V(out)", step=2), np.array([2.0, 3.0]))


def test_raw_parser_refusal_propagates_through_api(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = work_dir / "wedged-api.raw"
    raw_path.write_bytes(b"placeholder")

    async def refused_parse(source: services.AnalysisSource, _state: SessionState) -> DecodedRaw:
        assert source.raw == raw_path
        raise ResultError("RAW parser deadline exceeded")

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(services, "load_raw", refused_parse)
    api = Api()
    try:
        with pytest.raises(ResultError, match="exceeded"):
            api.load_raw(raw_path=raw_path)
    finally:
        api.close()


def test_load_raw_runs_on_the_concrete_api_private_loop(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    seen_threads: list[int] = []
    original_load = services.load_raw

    async def observed_load(source: services.AnalysisSource, state: SessionState) -> DecodedRaw:
        seen_threads.append(threading.get_ident())
        return await original_load(source, state)

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(services, "load_raw", observed_load)
    with Api() as api:
        result = api.load_raw(raw_path=raw_path)
        assert seen_threads == [api._loop_thread.ident]
        assert result.trace("V(out)").size > 0


def test_measurements_support_cases_and_relay_contained_loader_failure(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    experiment = make_experiment_job(
        state_no_sim, job_id="exp-measurements", case_id="case-selected", run_index=4, raw=raw_path
    )
    sync_api = SyncApi(state_no_sim)
    parsed = sync_api.measurements(
        job_id=experiment.job_id,
        case_id="case-selected",
    )
    assert parsed["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]

    async def refused_logs(source: services.AnalysisSource, state: SessionState):
        assert source.raw == raw_path
        assert source.trusted_job_artifact
        assert source.identity is not None
        assert source.identity["case_id"] == "case-selected"
        assert state is state_no_sim
        raise ResultError("Contained log parsing exceeded its deadline")

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(services, "load_logs", refused_logs)
    api = Api()
    try:
        with pytest.raises(ResultError, match="exceeded"):
            api.measurements(
                job_id=experiment.job_id,
                case_id="case-selected",
            )
    finally:
        api.close()


def _walk_project_types(value: Any, seen: set[Any]) -> None:
    if value in seen:
        return
    seen.add(value)
    origin = get_origin(value)
    if origin is not None:
        for argument in get_args(value):
            _walk_project_types(argument, seen)
        return
    if isinstance(value, type) and value.__module__.startswith("ltspice_mcp"):
        for nested in get_type_hints(value).values():
            _walk_project_types(nested, seen)


def test_literal_all_is_complete_and_excludes_unpublished_types() -> None:
    assert api_module.__all__ == EXPECTED_ALL
    assert not hasattr(api_module, "StatEnvelopeOutput")
    assert not hasattr(api_module, "WaveformBucket")
    assert all(callable(getattr(api_module, name)) for name in METRIC_NAMES)
    assert all(callable(getattr(api_module, name)) for name in VALUE_HELPER_NAMES)
    assert all(getattr(api_module, name) is not None for name in ALIAS_NAMES)
    assert all(callable(getattr(api_module, name)) for name in OUTPUT_TYPE_NAMES)
    assert set(EXPECTED_ALL) == {
        name for name in EXPECTED_ALL if getattr(api_module, name, None) is not None
    }


def test_every_metric_annotation_resolves_through_the_public_facade() -> None:
    facade_types = {getattr(api_module, name) for name in [*ALIAS_NAMES, *OUTPUT_TYPE_NAMES]}
    referenced: set[Any] = set()
    for name in [*METRIC_NAMES, *VALUE_HELPER_NAMES]:
        function = getattr(api_module, name)
        for annotation in get_type_hints(function).values():
            _walk_project_types(annotation, referenced)

    project_types = {
        value
        for value in referenced
        if isinstance(value, type) and value.__module__.startswith("ltspice_mcp")
    }
    assert project_types <= facade_types
    assert facade_types <= referenced


def test_archetype_build_verify_and_recorded_analysis_through_api(
    asc_state: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_stub_bootstrap(monkeypatch, asc_state)
    with Api() as api:
        built = api.edit_schematic(
            target="api_archetypes.asc",
            base="blank",
            ops=[
                {
                    "op": "add_component",
                    "reference": "D1",
                    "symbol": "diode",
                    "x": 200,
                    "y": 300,
                },
                {
                    "op": "add_component",
                    "reference": "M1",
                    "symbol": "nmos",
                    "x": 500,
                    "y": 300,
                },
                {
                    "op": "add_component",
                    "reference": "E1",
                    "symbol": "e",
                    "x": 800,
                    "y": 300,
                },
                {
                    "op": "add_component",
                    "reference": "G1",
                    "symbol": "g",
                    "x": 1100,
                    "y": 300,
                },
            ],
            return_views=["pin_legend"],
        )
        legend = {item["ref"]: item for item in built["views"]["pin_legend"]["items"]}
        assert set(legend) == {"D1", "M1", "E1", "G1"}

        verified = api.verify_circuit(
            path="api_archetypes.asc",
            checks=["symbols", "layout", "quality"],
        )
        assert set(verified["checks_run"]) == {"symbols", "layout", "quality"}

        recorded = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        analyzed = api.analyze_results(
            sources=[{"raw_path": str(recorded), "label": "recorded"}],
            recipes=[{"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}],
        )
        assert analyzed["coverage"]["runs_analyzed"] == 1
        value = analyzed["results"]["vout"]["values"][0]["value"]["value"]
        # The constant is the fixture's .MEAS result at 900u, computed by the
        # simulator from full-resolution data; the recipe interpolates the
        # stored raw points, which lands ~2ppm away on this fixture.
        assert value == pytest.approx(LTSPICE_TRAN_RC_VFINAL, abs=5e-6)
