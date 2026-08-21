"""Contract tests for the Python API raw wrapper and curated primitive facade."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints

import numpy as np
import pytest
from spicelib.raw.raw_read import RawRead

import ltspice_mcp.api as api_module
import ltspice_mcp.api._primitives as primitives_module
from ltspice_mcp.api import Api, RawResult
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import now, services
from ltspice_mcp.state import SessionState, SimulationJob
from tests.conftest import (
    LTSPICE_TRAN_RC_VFINAL,
    SyncApi,
    make_experiment_job,
    patch_stub_bootstrap,
    stage_recorded_fixture,
)

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
    "ThdOutput",
    "HarmonicEntry",
    "MeasurementStatsEntry",
    "HistogramBin",
    "AcStructureResult",
    "Corner",
    "Observation",
]

METRIC_NAMES = EXPECTED_ALL[9:33]
# Not a metric: a value reader, published because SPICE literals cross the
# boundary in both directions and nothing else on the facade parses one.
VALUE_HELPER_NAMES = EXPECTED_ALL[33:34]
ALIAS_NAMES = EXPECTED_ALL[34:40]
OUTPUT_TYPE_NAMES = EXPECTED_ALL[40:]


def _legacy_job(state: SessionState, job_id: str, raw: Path) -> SimulationJob:
    deck = state.working_dir / f"{job_id}.cir"
    deck.write_text(".tran 1m\n.end\n", encoding="utf-8")
    job = SimulationJob(
        job_id=job_id,
        netlist=deck,
        simulator="LTspice",
        status="completed",
        started_at=now(),
        completed_at=now(),
        raw_file=raw,
        log_file=raw.with_suffix(".log"),
    )
    state.jobs[job_id] = job
    return job


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


def test_raw_result_uses_bounded_sibling_log_fallback(
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

    class _RawWithoutStepParams:
        dialect = "ltspice"
        steps = None

        def get_trace_names(self) -> list[str]:
            return ["time", "V(out)"]

        def get_steps(self) -> range:
            return range(3)

        def get_raw_property(self, name: str) -> str:
            assert name == "Plotname"
            return "Transient Analysis"

        def get_wave(self, name: str, step: int = 0) -> np.ndarray:
            del name
            return np.array([step, step + 1.0])

        def get_axis(self, step: int = 0) -> np.ndarray:
            del step
            return np.array([0.0, 1.0])

    bounded_paths: list[Path] = []
    original_bounded = services.bounded_parse

    async def fake_load(_path: Path, _state: SessionState) -> RawRead:
        return cast(RawRead, _RawWithoutStepParams())

    async def observed_bounded(path: Path, thunk: Any, *, timeout_s: float) -> Any:
        bounded_paths.append(path)
        return await original_bounded(path, thunk, timeout_s=timeout_s)

    monkeypatch.setattr(services, "load_raw", fake_load)
    monkeypatch.setattr(services, "bounded_parse", observed_bounded)
    result = SyncApi(state_no_sim).load_raw(raw_path=raw_path)

    assert result.steps == [{"gain": 1.0}, {"gain": 2.0}, {"gain": 4.0}]
    np.testing.assert_array_equal(result.trace("V(out)", step=2), np.array([2.0, 3.0]))
    assert bounded_paths == [raw_path.with_suffix(".log")]


def test_experiment_and_legacy_addressing_use_disjoint_resolvers(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    experiment_raw = work_dir / "experiment.raw"
    experiment_raw.write_bytes(legacy_raw.read_bytes())
    experiment_raw.with_suffix(".log").write_bytes(legacy_raw.with_suffix(".log").read_bytes())
    legacy = _legacy_job(state_no_sim, "sim-api", legacy_raw)
    experiment = make_experiment_job(
        state_no_sim, job_id="exp-api", case_id="case-selected", run_index=4, raw=experiment_raw
    )

    legacy_calls: list[str] = []
    experiment_calls: list[str] = []
    original_legacy = services.resolve_raw_file
    original_experiment = services.experiment_run_context

    def track_legacy(job_id: str, state: SessionState, run_index: int = 0) -> Path:
        legacy_calls.append(job_id)
        return original_legacy(job_id, state, run_index)

    def track_experiment(*args: Any, **kwargs: Any):
        experiment_calls.append(args[0].job_id)
        return original_experiment(*args, **kwargs)

    monkeypatch.setattr(services, "resolve_raw_file", track_legacy)
    monkeypatch.setattr(services, "experiment_run_context", track_experiment)
    api = SyncApi(state_no_sim)

    legacy_result = api.load_raw(job_id=legacy.job_id)
    experiment_result = api.load_raw(job_id=experiment.job_id, case_id="case-selected")

    assert legacy_result.source == legacy_raw
    assert experiment_result.source == experiment_raw
    assert legacy_calls == [legacy.job_id]
    assert experiment_calls == [experiment.job_id]
    assert experiment.job_id not in legacy_calls


def test_raw_parse_deadline_propagates_through_api(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = work_dir / "wedged-api.raw"
    raw_path.write_bytes(b"placeholder")
    release = threading.Event()

    def slow_parse(_path: Path, _state: SessionState) -> RawRead:
        release.wait(5)
        return cast(RawRead, object())

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(services, "load_raw_sync", slow_parse)
    monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.05)
    api = Api()
    try:
        with pytest.raises(ResultError, match="exceeded"):
            api.load_raw(raw_path=raw_path)
    finally:
        release.set()
        services._wedged_raw_paths.pop(raw_path, None)
        api.close()


def test_load_raw_runs_on_the_concrete_api_private_loop(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    seen_threads: list[int] = []
    original_load = services.load_raw

    async def observed_load(path: Path, state: SessionState) -> RawRead:
        seen_threads.append(threading.get_ident())
        return await original_load(path, state)

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(services, "load_raw", observed_load)
    with Api() as api:
        result = api.load_raw(raw_path=raw_path)
        assert seen_threads == [api._loop_thread.ident]
        assert result.trace("V(out)").size > 0


def test_measurements_support_cases_and_enforce_a_bounded_log_parse(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    legacy = _legacy_job(state_no_sim, "sim-measurements", raw_path)
    experiment = make_experiment_job(
        state_no_sim, job_id="exp-measurements", case_id="case-selected", run_index=4, raw=raw_path
    )
    sync_api = SyncApi(state_no_sim)
    legacy_parsed = sync_api.measurements(job_id=legacy.job_id)
    parsed = sync_api.measurements(
        job_id=experiment.job_id,
        case_id="case-selected",
    )
    assert legacy_parsed["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]
    assert parsed["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]

    release = threading.Event()

    def slow_measurements(_path: Path):
        release.wait(5)
        return {}

    patch_stub_bootstrap(monkeypatch, state_no_sim)
    monkeypatch.setattr(primitives_module, "parse_measurements", slow_measurements)
    monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.05)
    api = Api()
    try:
        with pytest.raises(ResultError, match="exceeded"):
            api.measurements(
                job_id=experiment.job_id,
                case_id="case-selected",
            )
    finally:
        release.set()
        services._wedged_raw_paths.pop(raw_path.with_suffix(".log"), None)
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
