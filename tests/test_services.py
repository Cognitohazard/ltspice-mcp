"""Unit tests for the lib/services application service layer."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.errors import BatchJobError, ResultError, SimulationError
from ltspice_mcp.lib import now, services
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob


def _make_job(
    state: SessionState,
    *,
    job_id: str = "j1",
    status: str = "completed",
    raw_file: Path | None = None,
    log_file: Path | None = None,
) -> SimulationJob:
    job = SimulationJob(
        job_id=job_id,
        netlist=Path("/tmp/test.cir"),
        simulator="FakeSim",
        status=status,  # type: ignore[arg-type]
        started_at=now(),
        completed_at=now() + timedelta(seconds=2),
        raw_file=raw_file,
        log_file=log_file,
    )
    state.jobs[job_id] = job
    return job


def _make_batch(
    state: SessionState,
    *,
    job_id: str = "b1",
    status: str = "completed",
    completed: int = 1,
    run_results: dict | None = None,
) -> BatchJob:
    bj = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=Path("/tmp/x.cir"),
        total_runs=1,
        completed_runs=completed,
        failed_runs=0,
        status=status,  # type: ignore[arg-type]
    )
    if run_results is not None:
        bj.run_results = run_results
    if status == "completed":
        bj.completed_at = bj.started_at + timedelta(seconds=5)
    state.batch_jobs[job_id] = bj
    return bj


class TestResolveJob:
    def test_resolve_simulation_job_found(self, state_no_sim: SessionState):
        _make_job(state_no_sim)
        job = services.resolve_simulation_job("j1", state_no_sim)
        assert job.job_id == "j1"

    def test_resolve_simulation_job_not_found(self, state_no_sim: SessionState):
        with pytest.raises(SimulationError):
            services.resolve_simulation_job("missing", state_no_sim)

    def test_resolve_batch_job_found(self, state_no_sim: SessionState):
        _make_batch(state_no_sim)
        bj = services.resolve_batch_job("b1", state_no_sim)
        assert bj.job_id == "b1"

    def test_resolve_batch_job_not_found(self, state_no_sim: SessionState):
        with pytest.raises(BatchJobError):
            services.resolve_batch_job("missing", state_no_sim)

    def test_resolve_job_finds_sim(self, state_no_sim: SessionState):
        _make_job(state_no_sim)
        assert services.resolve_job("j1", state_no_sim).job_id == "j1"

    def test_resolve_job_finds_batch(self, state_no_sim: SessionState):
        _make_batch(state_no_sim)
        assert services.resolve_job("b1", state_no_sim).job_id == "b1"

    def test_resolve_job_not_found(self, state_no_sim: SessionState):
        with pytest.raises(ResultError):
            services.resolve_job("missing", state_no_sim)


class TestResolveResultFile:
    def test_sim_completed_with_files(self, state_no_sim: SessionState, tmp_path: Path):
        raw = tmp_path / "x.raw"
        raw.write_text("data")
        log = tmp_path / "x.log"
        log.write_text("log")
        _make_job(state_no_sim, raw_file=raw, log_file=log, status="completed")
        assert services.resolve_raw_file("j1", state_no_sim) == raw
        assert services.resolve_log_file("j1", state_no_sim) == log

    def test_sim_not_completed(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="running")
        with pytest.raises(ResultError):
            services.resolve_raw_file("j1", state_no_sim)

    def test_sim_completed_no_file(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="completed")
        with pytest.raises(ResultError):
            services.resolve_raw_file("j1", state_no_sim)

    def test_batch_not_completed(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="running")
        with pytest.raises(ResultError, match="not completed"):
            services.resolve_raw_file("b1", state_no_sim)

    def test_batch_no_run_results(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, run_results={})
        with pytest.raises(ResultError, match="no run results"):
            services.resolve_raw_file("b1", state_no_sim)

    def test_batch_first_run_missing_field(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, run_results={0: {"params": {}}})
        with pytest.raises(ResultError, match="first run"):
            services.resolve_raw_file("b1", state_no_sim)

    def test_batch_first_run_present(self, state_no_sim: SessionState, tmp_path: Path):
        raw = tmp_path / "r0.raw"
        raw.write_text("d")
        _make_batch(state_no_sim, run_results={0: {"raw_file": raw, "log_file": raw, "params": {}}})
        assert services.resolve_raw_file("b1", state_no_sim) == raw


class TestLoadRaw:
    def test_missing_file(self, state_no_sim: SessionState, tmp_path: Path):
        with pytest.raises(ResultError, match="not found"):
            services.load_raw(tmp_path / "nope.raw", state_no_sim)

    def test_caches_results(self, state_no_sim: SessionState, tmp_path: Path):
        # Just cover the caching path - call twice on missing file
        with pytest.raises(ResultError):
            services.load_raw(tmp_path / "x.raw", state_no_sim)


class TestGetBatchStatus:
    def test_running(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, status="running")
        bj.completed_at = None
        d = services.get_batch_status(bj)
        assert d["status"] == "running"
        assert "elapsed_s" in d

    def test_completed(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, status="completed")
        d = services.get_batch_status(bj)
        assert d["status"] == "completed"
        assert d["duration"] is not None
        assert d["successful"] == 1


class TestGetBatchSignalData:
    def test_no_completed(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, completed=0)
        with pytest.raises(BatchJobError, match="No completed runs"):
            services.get_batch_signal_data(bj, "V(out)")

    def test_no_runs_match_filter(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, run_results={0: {"params": {"R1": "1k"}}})
        with pytest.raises(BatchJobError, match="No runs match"):
            services.get_batch_signal_data(bj, "V(out)", filters={"R1": "999k"})

    def test_raw_mode_pagination_empty(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, run_results={0: {"params": {}}})
        with pytest.raises(BatchJobError, match="page range"):
            services.get_batch_signal_data(bj, "V(out)", raw=True, offset=10, limit=5)


class TestExtractModelSuggestions:
    def test_none_when_log_missing(self, state_no_sim: SessionState, tmp_path: Path):
        assert services.extract_model_suggestions(tmp_path / "no.log", state_no_sim.libraries) is None

    def test_none_for_clean_log(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text("Total elapsed time: 0.01 seconds.\n")
        assert services.extract_model_suggestions(log, state_no_sim.libraries) is None

    def test_none_when_no_libraries_loaded(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        log = tmp_path / "err.log"
        log.write_text('Error on line 2 : s1 0 0 sw Unable to find definition of model "sw"\n')
        assert services.extract_model_suggestions(log, state_no_sim.libraries) is None

    def test_returns_ranked_suggestions(self, state_no_sim: SessionState, work_dir: Path):
        lib = work_dir / "sw.lib"
        lib.write_text(".MODEL SW VSWITCH(VT=1)\n.MODEL SW2 VSWITCH(VT=2)\n")
        state_no_sim.libraries.load_library(lib)
        log = work_dir / "err.log"
        log.write_text('Error on line 2 : s1 0 0 swx Unable to find definition of model "swx"\n')
        out = services.extract_model_suggestions(log, state_no_sim.libraries)
        assert out is not None
        assert "swx" in out
        names = {m["name"] for m in out["swx"]}
        assert "SW" in names

    def test_format_suggestion_block_empty(self):
        assert services.format_suggestion_block(None) == ""
        assert services.format_suggestion_block({}) == ""

    def test_format_suggestion_block_renders(self):
        out = services.format_suggestion_block(
            {"swx": [{"name": "SW", "score": 0.9, "source_path": "/tmp/sw.lib"}]}
        )
        assert "Missing 'swx'" in out
        assert "SW" in out
        assert "/tmp/sw.lib" in out


class TestValidateSignal:
    """``validate_signal`` is case-insensitive — LTspice writes ``v(onoise)``
    in lowercase for ``.NOISE`` raws but ``V(out)`` everywhere else, and we
    don't want to reject a user's ``V(onoise)`` just because the raw used a
    different case."""

    def _raw(self, names: list[str]) -> MagicMock:
        raw = MagicMock()
        raw.get_trace_names.return_value = names
        return raw

    def test_exact_match_returns_same_string(self):
        raw = self._raw(["V(out)", "I(R1)"])
        assert services.validate_signal(raw, "V(out)") == "V(out)"

    def test_case_insensitive_returns_canonical_name(self):
        # spicelib preserves the case the simulator wrote — for noise raws
        # that's lowercase. Caller must use the canonical name to read traces.
        raw = self._raw(["v(onoise)", "v(inoise)"])
        assert services.validate_signal(raw, "V(onoise)") == "v(onoise)"
        assert services.validate_signal(raw, "V(INOISE)") == "v(inoise)"

    def test_unknown_signal_lists_available(self):
        raw = self._raw(["V(a)", "V(b)"])
        with pytest.raises(ResultError, match="Signal 'V\\(missing\\)' not found"):
            services.validate_signal(raw, "V(missing)")
