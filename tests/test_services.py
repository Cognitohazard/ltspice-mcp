"""Unit tests for the lib/services application service layer."""

import asyncio
import contextlib
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError
from ltspice_mcp.lib import parser_service, services
from ltspice_mcp.lib.store import parser_file_in
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    await_until,
    stage_recorded_fixture,
)
from tests.test_parser_process import _FIXTURE, _assert_gone, _started


class TestLoadRaw:
    async def test_missing_file(self, state_no_sim: SessionState, tmp_path: Path):
        with pytest.raises(ResultError, match="RAW file is unavailable"):
            await services.load_raw(
                services.source_for_raw_path(tmp_path / "nope.raw", state_no_sim), state_no_sim
            )

    async def test_caches_results(self, state_no_sim: SessionState, tmp_path: Path):
        path = stage_recorded_fixture(tmp_path, "ltspice_tran_rc")
        source = services.source_for_raw_path(path, state_no_sim)
        first = await services.load_raw(source, state_no_sim)
        assert await services.load_raw(source, state_no_sim) is first
        assert state_no_sim.results.entry_count == 1

    async def test_truncated_binary_raw_raises_not_silently_short(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # Preflight must reject a cut payload before publishing resident arrays.
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
        raw = await services.load_raw(
            services.source_for_raw_path(good, state_no_sim), state_no_sim
        )
        names = [t.lower() for t in raw.get_trace_names()]
        assert "time" in names and "v(out)" in names

        # Cut the data section short (header is a few hundred bytes; 60% of a
        # 2000-point file lands well inside the binary data).
        data = good.read_bytes()
        truncated = tmp_path / "truncated.raw"
        truncated.write_bytes(data[: int(len(data) * 0.6)])

        with pytest.raises(ResultError):
            await services.load_raw(
                services.source_for_raw_path(truncated, state_no_sim), state_no_sim
            )

    async def test_zero_variable_raw_is_diagnosed_as_corrupt(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # Dependency readers previously accepted this cut UTF-16 header as an
        # empty reader. The contained preflight must diagnose it as malformed.
        truncated = tmp_path / "trunc.raw"
        truncated.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes()[:100])

        with pytest.raises(ResultError) as exc_info:
            await services.load_raw(
                services.source_for_raw_path(truncated, state_no_sim), state_no_sim
            )
        msg = str(exc_info.value)
        assert "header" in msg.lower()
        assert str(truncated) in msg


class TestValidateSignal:
    """``resolve_signal`` is case-insensitive — LTspice writes ``v(onoise)``
    in lowercase for ``.NOISE`` raws but ``V(out)`` everywhere else, and we
    don't want to reject a user's ``V(onoise)`` just because the raw used a
    different case."""

    def _raw(self, names: list[str]) -> MagicMock:
        raw = MagicMock()
        raw.get_trace_names.return_value = names
        return raw

    def test_exact_match_returns_same_string(self):
        raw = self._raw(["V(out)", "I(R1)"])
        assert services.resolve_signal(raw, "V(out)").name == "V(out)"

    def test_case_insensitive_returns_canonical_name(self):
        # spicelib preserves the case the simulator wrote — for noise raws
        # that's lowercase. Caller must use the canonical name to read traces.
        raw = self._raw(["v(onoise)", "v(inoise)"])
        assert services.resolve_signal(raw, "V(onoise)").name == "v(onoise)"
        assert services.resolve_signal(raw, "V(INOISE)").name == "v(inoise)"

    def test_unknown_signal_lists_available(self):
        raw = self._raw(["V(a)", "V(b)"])
        with pytest.raises(ResultError, match="Signal 'V\\(missing\\)' not found"):
            services.resolve_signal(raw, "V(missing)")

    def test_noise_alias_ltspice_form_to_ngspice(self):
        # Resolve the alias, don't just hint.
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.resolve_signal(raw, "V(onoise)").name == "onoise_spectrum"
        assert services.resolve_signal(raw, "V(inoise)").name == "inoise_spectrum"

    def test_noise_alias_bare_shorthand(self):
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.resolve_signal(raw, "onoise").name == "onoise_spectrum"
        assert services.resolve_signal(raw, "inoise").name == "inoise_spectrum"

    def test_noise_alias_ngspice_form_to_ltspice(self):
        raw = self._raw(["frequency", "v(onoise)", "v(inoise)"])
        assert services.resolve_signal(raw, "onoise_spectrum").name == "v(onoise)"

    def test_dev_param_hierarchical_multi_dot(self):
        # A subckt-flattened device path: m.x1.mn.gm -> @m.x1.mn[gm], including
        # the v()/i() wrapped forms.
        raw = self._raw(["@m.x1.mn[gm]", "v(@m.x1.mn[vth])", "i(@m.x1.mn[id])"])
        assert services.resolve_signal(raw, "m.x1.mn.gm").name == "@m.x1.mn[gm]"
        assert services.resolve_signal(raw, "m.x1.mn.vth").name == "v(@m.x1.mn[vth])"
        assert services.resolve_signal(raw, "m.x1.mn.id").name == "i(@m.x1.mn[id])"

    def test_dev_param_hierarchical_without_device_letter_unique(self):
        # Dropping the leading device-type letter (x1.mn.gm) resolves when the
        # suffix is unique.
        raw = self._raw(["@m.x1.mn[gm]"])
        assert services.resolve_signal(raw, "x1.mn.gm").name == "@m.x1.mn[gm]"

    def test_dev_param_hierarchical_ambiguous_refused(self):
        # Two devices share the .mn suffix — refuse rather than guess.
        raw = self._raw(["@m.x1.mn[gm]", "@m.x2.mn[gm]"])
        with pytest.raises(ResultError, match="not found"):
            services.resolve_signal(raw, "mn.gm")

    def test_hierarchical_colon_resolves_to_dot(self):
        # LTspice V(X1:mid) <-> ngspice v(x1.mid)
        raw = self._raw(["time", "v(x1.mid)", "v(out)"])
        assert services.resolve_signal(raw, "V(X1:mid)").name == "v(x1.mid)"

    def test_device_param_shorthand_resolves_each_wrap(self):
        # 'dev.param' resolves to whichever form ngspice actually wrote:
        # bare @m1[gm], v-wrapped v(@m1[vth]), or i-wrapped i(@m1[id]).
        raw = self._raw(["@m1[gm]", "v(@m1[vth])", "i(@m1[id])"])
        assert services.resolve_signal(raw, "m1.gm").name == "@m1[gm]"
        assert services.resolve_signal(raw, "M1.VTH").name == "v(@m1[vth])"
        assert services.resolve_signal(raw, "m1.id").name == "i(@m1[id])"

    def test_device_param_not_saved_hints_save(self):
        # A dev.param that isn't in the raw points at the missing .save.
        raw = self._raw(["@m1[gm]"])
        with pytest.raises(ResultError, match=r"\.save @m1\[gds\]"):
            services.resolve_signal(raw, "m1.gds")


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


@pytest.fixture
def contained_runaway(monkeypatch):
    """Reuse the supervisor's real GIL-holding worker and detached child."""
    run = parser_service.run_parser_sync
    calls = {"runaway": True, "directories": []}

    def invoke(request, **kwargs):
        if not calls["runaway"]:
            return run(request, **kwargs)
        directory = kwargs["work_dir"]
        calls["directories"].append(directory)
        parser_file_in(directory, "parser_fixture.py").write_text(_FIXTURE, encoding="utf-8")
        return run({"mode": "runaway"}, **kwargs, _worker_module="parser_fixture")

    monkeypatch.setattr(parser_service, "run_parser_sync", invoke)
    return calls


async def _runaway_tree(calls, task: asyncio.Task):
    """The contained worker and its detached child, once the worker is at its
    runaway seam, identified while they run. Ends ``task`` if they never are."""
    try:
        directories = await await_until(
            lambda: calls["directories"], what="the parse to reach its parser call"
        )
        return await _started(directories[0])
    except BaseException:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise


class TestLoadRawParseDeadline:
    async def test_timeout_reaps_owned_tree_and_allows_fresh_parse(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
        contained_runaway,
        parser_deadline_passed,
    ):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        source = services.source_for_raw_path(path, state_no_sim)
        monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", LIVENESS_S)
        task = asyncio.create_task(services.load_raw(source, state_no_sim))
        owned = await _runaway_tree(contained_runaway, task)
        # The deadline passes now that there is a tree to reap.
        parser_deadline_passed.set()
        with pytest.raises(AnalysisDeadlineExceeded, match="exceeded"):
            await task
        _assert_gone(owned)
        assert all(not directory.exists() for directory in contained_runaway["directories"])
        assert state_no_sim.results.entry_count == 0
        contained_runaway["runaway"] = False
        parser_deadline_passed.clear()
        loaded = await services.load_raw(source, state_no_sim)
        assert loaded.get_trace_names() == ["time", "V(in)", "V(out)", "I(C1)", "I(R1)", "I(V1)"]

    async def test_cancellation_waits_for_owned_tree_and_scratch_cleanup(
        self, state_no_sim: SessionState, work_dir: Path, contained_runaway
    ):
        path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        source = services.source_for_raw_path(path, state_no_sim)
        task = asyncio.create_task(services.load_raw(source, state_no_sim))
        owned = await _runaway_tree(contained_runaway, task)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_gone(owned)
        assert all(not directory.exists() for directory in contained_runaway["directories"])
        assert state_no_sim.results.entry_count == 0
        contained_runaway["runaway"] = False
        assert (await services.load_raw(source, state_no_sim)).descriptor.analysis == "transient"

    async def test_normal_parse_unaffected_by_deadline(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        local = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        loaded = await services.load_raw(
            services.source_for_raw_path(local, state_no_sim), state_no_sim
        )
        assert loaded.get_trace_names()


async def test_log_parser_and_raw_parser_share_cancellable_admission(
    state_no_sim, work_dir, contained_runaway, monkeypatch
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    source = services.source_for_raw_path(path, state_no_sim)
    # Admission is taken in a worker thread; this says when the raw parse has
    # queued behind the log parse, which is the state the cancel must find.
    slot = state_no_sim.results.parse_slot
    entries: list[None] = []
    queued = threading.Event()
    lock = threading.Lock()

    @contextlib.contextmanager
    def watched_slot(**kwargs):
        with lock:
            entries.append(None)
            if len(entries) == 2:
                queued.set()
        with slot(**kwargs):
            yield

    monkeypatch.setattr(state_no_sim.results, "parse_slot", watched_slot)
    logs_task = asyncio.create_task(services.load_logs(source, state_no_sim))
    raw_task = None
    try:
        owned = await _runaway_tree(contained_runaway, logs_task)
        raw_task = asyncio.create_task(services.load_raw(source, state_no_sim))
        await await_until(queued.is_set, what="the raw parse to queue behind the log parse")
        raw_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await raw_task
        assert not logs_task.done()
        assert len(contained_runaway["directories"]) == 1
    finally:
        if raw_task is not None and not raw_task.done():
            raw_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await raw_task
        logs_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await logs_task
    _assert_gone(owned)
    assert all(not directory.exists() for directory in contained_runaway["directories"])
    contained_runaway["runaway"] = False
    assert (await services.load_logs(source, state_no_sim)).section("measurements")[
        "status"
    ] == "parsed"


@pytest.mark.asyncio
async def test_raw_writer_header_wins_over_session_default(
    state_no_sim: SessionState, work_dir: Path
):
    from spicelib.simulators.qspice_simulator import Qspice

    state_no_sim.default_simulator = Qspice
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")

    raw = await services.load_raw(services.source_for_raw_path(path, state_no_sim), state_no_sim)

    assert raw.dialect == "ngspice"


def test_unrecorded_job_producer_has_no_session_dialect(state_no_sim, work_dir):
    from spicelib.simulators.qspice_simulator import Qspice

    from tests.test_state import _experiment

    state_no_sim.default_simulator = Qspice
    job = _experiment(work_dir, work_dir / "deck.cir", job_id="no-producer")
    job.simulator = ""
    assert services.dialect_for_job(job, state_no_sim) is None


class TestOptionalRawSources:
    @pytest.mark.parametrize("sibling_raw", [False, True])
    def test_explicit_log_source_has_no_raw_candidate(self, state_no_sim, work_dir, sibling_raw):
        log = work_dir / "results.log"
        log.write_text("recorded log\n", encoding="ascii")
        if sibling_raw:
            log.with_suffix(".raw").write_bytes(b"unrelated sibling")
        source = services.resolve_analysis_source(state_no_sim, log_file=str(log))
        assert source.raw is None
        assert source.log == log.resolve()
        assert source.console == log.with_suffix(".exe.log").resolve()

    @pytest.mark.parametrize("mode", ["sync", "async"])
    async def test_missing_raw_refuses_before_admission_or_spawn(
        self, state_no_sim, work_dir, monkeypatch, mode
    ):
        source = services.AnalysisSource(None, work_dir / "only.log", None, None, None, False)
        admissions = []
        parse_slot = state_no_sim.results.parse_slot

        def observe_admission(**kwargs):
            admissions.append(kwargs)
            return parse_slot(**kwargs)

        monkeypatch.setattr(state_no_sim.results, "parse_slot", observe_admission)
        if mode == "sync":
            with pytest.raises(ResultError, match="source has no RAW artifact"):
                services.load_raw_sync(source, state_no_sim)
        else:
            with pytest.raises(ResultError, match="source has no RAW artifact"):
                await services.load_raw(source, state_no_sim)
        assert admissions == []
        assert not list((state_no_sim.store.root / "parsing").glob("*"))

    def test_run_source_preserves_console_without_raw(self, state_no_sim, work_dir):
        run = services.RunContext(
            raw=None,
            log=None,
            netlist=work_dir / "staged.cir",
            circuit_path=work_dir / "original.cir",
            dialect="ngspice",
            identity={"case_id": "case_0000"},
            console=work_dir / "recorded.exe.log",
        )
        source = services.source_for_run(run)
        assert source.raw is None
        assert source.console == run.console
        assert source.trusted_job_artifact

    @pytest.mark.parametrize("provenance", ["log", "raw", "token", "no_folder", "no_token"])
    def test_case_console_candidate_uses_recorded_output_inventory(
        self, state_no_sim, work_dir, provenance
    ):
        from tests.test_state import _experiment

        job = _experiment(work_dir, work_dir / "deck.cir", job_id="child_run")
        case = job.cases[0]
        job.output_folder = work_dir / "original_lineage_root"
        case.run_token = "fresh_child_attempt"
        assert job.output_folder is not None
        expected = job.output_folder / "fresh_child_attempt.exe.log"
        if provenance == "log":
            case.log_file = work_dir / "recorded_log_location" / "result.log"
            case.raw_file = work_dir / "different_raw_location" / "result.raw"
            assert case.log_file is not None
            expected = case.log_file.with_suffix(".exe.log")
        elif provenance == "raw":
            case.raw_file = work_dir / "recorded_raw_location" / "result.raw"
            assert case.raw_file is not None
            expected = case.raw_file.with_suffix(".exe.log")
        elif provenance == "no_folder":
            job.output_folder = None
            expected = None
        elif provenance == "no_token":
            case.run_token = ""
            expected = None
        run = services.experiment_run_context(job, state_no_sim, require_raw=False)
        assert run.console == expected
        assert services.source_for_run(run).console == expected
        assert case.status == "produced"
        assert job.status == "completed"

    def test_job_resolution_raw_default_and_log_opt_in(self, state_no_sim, work_dir):
        from tests.test_state import _experiment

        job = _experiment(work_dir, work_dir / "deck.cir", job_id="recorded_logs")
        job.cases[0].log_file = work_dir / "recorded.log"
        state_no_sim.job_registry.add_experiment_job(job, already_persisted=True)
        with pytest.raises(ResultError, match="did not produce a raw result"):
            services.resolve_experiment_run(job.job_id, state_no_sim)
        run = services.resolve_experiment_run(job.job_id, state_no_sim, require_raw=False)
        assert run.raw is None
        assert run.log == job.cases[0].log_file
        assert run.identity["case_id"] == job.cases[0].case_id

    @pytest.mark.parametrize(
        ("job_status", "case_status"),
        [
            ("running", "produced"),
            ("completed", "failed"),
            ("completed", "cancelled"),
        ],
    )
    def test_log_opt_in_retains_existing_status_gates(
        self, state_no_sim, work_dir, job_status, case_status
    ):
        from tests.test_state import _experiment

        job = _experiment(work_dir, work_dir / "deck.cir", job_id="status_gate")
        job.status = job_status
        job.cases[0].status = case_status
        job.cases[0].raw_file = work_dir / "existing.raw"
        with pytest.raises(ResultError):
            services.experiment_run_context(job, state_no_sim, require_raw=False)
        assert job.status == job_status
        assert job.cases[0].status == case_status
