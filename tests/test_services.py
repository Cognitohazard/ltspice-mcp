"""Unit tests for the lib/services application service layer."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.errors import BatchJobError, JobNotFoundError, ResultError, SimulationError
from ltspice_mcp.lib import now, services
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob
from tests.conftest import FIXTURES_DIR


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
    total: int = 1,
    run_results: dict | None = None,
) -> BatchJob:
    bj = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=Path("/tmp/x.cir"),
        total_runs=total,
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
        with pytest.raises(JobNotFoundError, match="Job not found: missing"):
            services.resolve_simulation_job("missing", state_no_sim)

    def test_resolve_simulation_job_batch_id_redirects(self, state_no_sim: SessionState):
        """A batch id through the single-sim resolver is an honest redirect,
        never "not found" — the job exists."""
        _make_batch(state_no_sim)
        with pytest.raises(SimulationError) as exc:
            services.resolve_simulation_job("b1", state_no_sim)
        assert str(exc.value) == (
            "Job 'b1' is a sweep batch job — use batch_results for its per-run results."
        )

    def test_resolve_batch_job_found(self, state_no_sim: SessionState):
        _make_batch(state_no_sim)
        bj = services.resolve_batch_job("b1", state_no_sim)
        assert bj.job_id == "b1"

    def test_resolve_batch_job_not_found(self, state_no_sim: SessionState):
        with pytest.raises(BatchJobError, match="Batch job not found: missing"):
            services.resolve_batch_job("missing", state_no_sim)

    def test_resolve_batch_job_sim_id_redirects(self, state_no_sim: SessionState):
        _make_job(state_no_sim)
        with pytest.raises(BatchJobError) as exc:
            services.resolve_batch_job("j1", state_no_sim)
        assert str(exc.value) == (
            "Job 'j1' is a single simulation job — its results are "
            "available via check_job / simulation_summary."
        )

    def test_resolve_job_finds_sim(self, state_no_sim: SessionState):
        _make_job(state_no_sim)
        assert services.resolve_job("j1", state_no_sim).job_id == "j1"

    def test_resolve_job_finds_batch(self, state_no_sim: SessionState):
        _make_batch(state_no_sim)
        assert services.resolve_job("b1", state_no_sim).job_id == "b1"

    def test_resolve_job_not_found(self, state_no_sim: SessionState):
        with pytest.raises(JobNotFoundError):
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
        with pytest.raises(ResultError, match="no raw file"):
            services.resolve_raw_file("b1", state_no_sim)

    def test_batch_first_run_present(self, state_no_sim: SessionState, tmp_path: Path):
        raw = tmp_path / "r0.raw"
        raw.write_text("d")
        _make_batch(
            state_no_sim, run_results={0: {"raw_file": raw, "log_file": raw, "params": {}}}
        )
        assert services.resolve_raw_file("b1", state_no_sim) == raw


class TestLoadRaw:
    def test_missing_file(self, state_no_sim: SessionState, tmp_path: Path):
        with pytest.raises(ResultError, match="not found"):
            services.load_raw(tmp_path / "nope.raw", state_no_sim)

    def test_caches_results(self, state_no_sim: SessionState, tmp_path: Path):
        # Just cover the caching path - call twice on missing file
        with pytest.raises(ResultError):
            services.load_raw(tmp_path / "x.raw", state_no_sim)

    def test_truncated_binary_raw_raises_not_silently_short(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # A killed/interrupted sim can leave a .raw with a valid header but a
        # data section cut short. spicelib reads trace data EXACTLY and raises
        # on a short read; load_raw must surface that as ResultError, never
        # serve silently-short arrays that masquerade as a complete result.
        # Locks this safety property against a future spicelib regression to
        # a count-limited (silently-truncating) read.
        import numpy as np
        from spicelib.raw.raw_write import RawWrite, Trace

        n = 2000
        rw = RawWrite(plot_name="Transient Analysis")
        rw.add_trace(Trace("time", np.linspace(0.0, 1e-3, n), whattype="time"))
        rw.add_trace(Trace("V(out)", np.linspace(0.0, 5.0, n), whattype="voltage"))
        good = tmp_path / "good.raw"
        rw.save(good)

        # Sanity: the intact fixture parses (valid header). This proves the
        # truncated case below fails on the truncation, not a malformed header.
        raw = services.load_raw(good, state_no_sim)
        names = [t.lower() for t in raw.get_trace_names()]
        assert "time" in names and "v(out)" in names

        # Cut the data section short (header is a few hundred bytes; 60% of a
        # 2000-point file lands well inside the binary data).
        data = good.read_bytes()
        truncated = tmp_path / "truncated.raw"
        truncated.write_bytes(data[: int(len(data) * 0.6)])

        with pytest.raises(ResultError):
            services.load_raw(truncated, state_no_sim)

    def test_zero_variable_raw_is_diagnosed_as_corrupt(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # A file cut mid-header (here: 100 bytes into a real LTspice raw, in
        # the middle of the UTF-16 Title line) does NOT make spicelib raise —
        # RawRead parses it into a "valid" raw with zero variables. A real
        # SPICE raw always carries at least its axis variable, so zero
        # variables is a corruption signature; without this diagnosis,
        # consumers would report "Signal not found" against an empty
        # signal list.
        truncated = tmp_path / "trunc.raw"
        truncated.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes()[:100])

        with pytest.raises(ResultError) as exc_info:
            services.load_raw(truncated, state_no_sim)
        msg = str(exc_info.value)
        assert "zero variables" in msg
        assert "truncated or corrupt" in msg
        assert str(truncated) in msg


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
        assert (
            services.extract_model_suggestions(tmp_path / "no.log", state_no_sim.libraries) is None
        )

    def test_none_for_clean_log(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text("Total elapsed time: 0.01 seconds.\n")
        assert services.extract_model_suggestions(log, state_no_sim.libraries) is None

    def test_none_when_no_libraries_loaded(self, state_no_sim: SessionState, tmp_path: Path):
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

    def test_noise_alias_ltspice_form_to_ngspice(self):
        # Resolve the alias, don't just hint.
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.validate_signal(raw, "V(onoise)") == "onoise_spectrum"
        assert services.validate_signal(raw, "V(inoise)") == "inoise_spectrum"

    def test_noise_alias_bare_shorthand(self):
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.validate_signal(raw, "onoise") == "onoise_spectrum"
        assert services.validate_signal(raw, "inoise") == "inoise_spectrum"

    def test_noise_alias_ngspice_form_to_ltspice(self):
        raw = self._raw(["frequency", "v(onoise)", "v(inoise)"])
        assert services.validate_signal(raw, "onoise_spectrum") == "v(onoise)"

    def test_hierarchical_colon_resolves_to_dot(self):
        # LTspice V(X1:mid) <-> ngspice v(x1.mid)
        raw = self._raw(["time", "v(x1.mid)", "v(out)"])
        assert services.validate_signal(raw, "V(X1:mid)") == "v(x1.mid)"


class TestNgspicePreflightWarnings:
    """services.ngspice_preflight_warnings — shared single-run + sweep/MC check."""

    @staticmethod
    def _ngspice_cls():
        from spicelib.simulators.ngspice_simulator import NGspiceSimulator

        return NGspiceSimulator

    def test_non_ngspice_returns_empty(self, tmp_path: Path):
        net = tmp_path / "x.cir"
        net.write_text("R1 a 0 1k\n.meas tran foo FIND V(a) AT 1m\n.tran 1u 1m\n.end\n")

        class LT:  # not an NGspiceSimulator subclass
            pass

        assert services.ngspice_preflight_warnings(net, LT) == []

    def test_ngspice_meas_warned(self, tmp_path: Path):
        net = tmp_path / "m.cir"
        net.write_text("R1 a 0 1k\nV1 a 0 1\n.meas tran vfoo FIND V(a) AT 1m\n.tran 1u 1m\n.end\n")
        warns = services.ngspice_preflight_warnings(net, self._ngspice_cls())
        assert any("vfoo" in w and "batch mode" in w for w in warns)

    def test_ngspice_step_raises(self, tmp_path: Path):
        net = tmp_path / "s.cir"
        net.write_text("R1 a 0 {r}\n.step param r 1k 3k 1k\n.op\n.end\n")
        with pytest.raises(SimulationError, match=r"does not support \.step"):
            services.ngspice_preflight_warnings(net, self._ngspice_cls())


class TestScanBatchConvergence:
    """scan_batch_convergence must scan EVERY terminal batch — including an
    ``interrupted`` job recovered after restart, whose completed sub-runs are
    real results. Regression: a hardcoded ('completed','failed','cancelled')
    gate omitted 'interrupted', silently skipping the scan.
    """

    def test_interrupted_job_is_scanned(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "run0.log"
        log.write_text("Doing analysis at TEMP=27\nWarning: gmin stepping needed\n")
        bj = _make_batch(
            state_no_sim,
            status="interrupted",
            run_results={0: {"raw_file": str(log), "log_file": str(log), "params": {}}},
        )
        flagged = services.scan_batch_convergence(bj)
        assert len(flagged) == 1
        assert flagged[0]["run_index"] == 0
        assert "gmin stepping" in flagged[0]["markers"]

    def test_running_job_not_scanned(self, state_no_sim: SessionState, tmp_path: Path):
        # Still skipped while non-terminal (logs mid-write).
        log = tmp_path / "run0.log"
        log.write_text("gmin stepping\n")
        bj = _make_batch(
            state_no_sim,
            status="running",
            run_results={0: {"raw_file": str(log), "log_file": str(log), "params": {}}},
        )
        assert services.scan_batch_convergence(bj) == []


# Real ngspice log shapes for the per-run convergence walk. The clean preamble
# is verbatim ngspice-42 batch output; the failure lines are the exact formats
# ngspice prints for a non-converging bias point (a "Warning:"-prefixed gmin
# line followed by a bare source-stepping line) and for a singular matrix.
_NGSPICE_CLEAN_LOG = (
    "Note: Compatibility modes selected: ps lt ki a\n"
    "\n"
    "Circuit: * dc divider\n"
    "\n"
    'binary raw file "out.raw"\n'
    "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n"
    "\n"
    "No. of Data Columns : 4\n"
    "No. of Data Rows : 3\n"
    "\n"
    "Total elapsed time (seconds) = 0.017\n"
)
_NGSPICE_GMIN_FAIL_LINES = "Warning: gmin stepping failed\nsource stepping failed\n"
_NGSPICE_SINGULAR_LINE = "Warning: singular matrix:  check nodes out and 0\n"


def _run_entry(raw: Path, log: Path) -> dict:
    """One ``run_results`` entry, mirroring the exact shape the batch runner
    records on run completion: string paths plus an (initially empty) params
    dict — see ``runner_base.BatchRunnerBase._record_run_completion``."""
    return {"raw_file": str(raw), "log_file": str(log), "params": {}}


class TestBatchConvergenceSurfacing:
    """The per-run log walk behind ``get_batch_status`` — driven with logs
    that really exist on disk, in real ngspice formats, through the public
    service entry rather than by poking the scanner's internals."""

    def test_flagged_runs_associated_with_their_run_index(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # Three completed runs: run 0 clean, run 1 hit gmin + source stepping,
        # run 2 hit a singular matrix. The status payload must name exactly
        # runs 1 and 2, each with the markers found in ITS OWN log.
        logs = {
            0: _NGSPICE_CLEAN_LOG,
            1: _NGSPICE_CLEAN_LOG + _NGSPICE_GMIN_FAIL_LINES,
            2: _NGSPICE_CLEAN_LOG + _NGSPICE_SINGULAR_LINE,
        }
        run_results = {}
        for idx, text in logs.items():
            log = tmp_path / f"job_{idx + 1}.log"
            log.write_text(text)
            run_results[idx] = _run_entry(tmp_path / f"job_{idx + 1}.raw", log)
        bj = _make_batch(state_no_sim, completed=3, total=3, run_results=run_results)

        data = services.get_batch_status(bj)

        assert data["status"] == "completed"
        assert data["convergence_warnings"] == [
            {"run_index": 1, "markers": ["gmin stepping", "source stepping"]},
            {"run_index": 2, "markers": ["singular matrix"]},
        ]

    def test_missing_log_file_is_skipped_silently(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # A run whose log file no longer exists on disk is skipped without
        # error (read_log_text returns "" on I/O failure); the surviving
        # flagged run still surfaces with its own index.
        flagged_log = tmp_path / "job_2.log"
        flagged_log.write_text(_NGSPICE_CLEAN_LOG + _NGSPICE_GMIN_FAIL_LINES)
        run_results = {
            0: _run_entry(tmp_path / "job_1.raw", tmp_path / "job_1.log"),  # never written
            1: _run_entry(tmp_path / "job_2.raw", flagged_log),
        }
        bj = _make_batch(state_no_sim, completed=2, total=2, run_results=run_results)

        data = services.get_batch_status(bj)

        assert data["convergence_warnings"] == [
            {"run_index": 1, "markers": ["gmin stepping", "source stepping"]}
        ]

    def test_clean_job_omits_key_but_caches_completed_scan(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # All-clean logs: the status payload omits convergence_warnings
        # entirely, and the job caches the empty scan result (the cache going
        # from None to [] is what proves the walk actually ran over the logs).
        run_results = {}
        for idx in range(2):
            log = tmp_path / f"job_{idx + 1}.log"
            log.write_text(_NGSPICE_CLEAN_LOG)
            run_results[idx] = _run_entry(tmp_path / f"job_{idx + 1}.raw", log)
        bj = _make_batch(state_no_sim, completed=2, total=2, run_results=run_results)
        assert bj.convergence_warnings is None

        data = services.get_batch_status(bj)

        assert "convergence_warnings" not in data
        assert bj.convergence_warnings == []
