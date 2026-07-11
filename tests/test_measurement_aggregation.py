"""measurement_stats job-id aggregation over real per-run LTspice logs.

The ``job_id`` branch of measurement_stats resolves any job: a batch job's
``run_results`` are walked (one .log per run, as recorded by the sweep/MC
runners) and the per-run .MEAS scalars aggregated; a completed single-sim
job aggregates its one log. The fixture logs were recorded from a
real 3-run LTspice parameter sweep of an RC low-pass (R1 = 1k / 2.2k / 4.7k,
C = 100n, 1 V step input) with two measurements in the deck:

    .meas tran vfinal FIND V(out) AT=0.9m     (value varies per run)
    .meas tran tcross WHEN V(out)=0.5         (level constant, time varies)

The WHEN-style measurement exercises the axis swap: its per-run scalar is the
constant trigger level 0.5, so the aggregator must switch to the folded ``at``
field (the crossing time) instead.
"""

from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob
from ltspice_mcp.tools.analysis import (
    MeasurementStatsInput,
    _aggregate_log_measurements,
    handle_measurement_stats,
)
from tests.conftest import (
    FIXTURES_DIR,
    LTSPICE_SWEEP_RUN_LOGS,
    LTSPICE_TRAN_RC_LOG,
    LTSPICE_TRAN_RC_VFINAL,
    make_batch_job,
    make_sim_job,
)

# R1 of sweep runs 0 / 1 / 2 (one LTSPICE_SWEEP_RUN_LOGS entry per run).
RUN_PARAMS = [1000.0, 2200.0, 4700.0]

# Values printed by LTspice in each per-run log (vfinal: FIND V(out) AT=0.9m).
VFINAL = [0.999876166042, 0.98323999039, 0.852486569628]
# Crossing times printed by LTspice (tcross: WHEN V(out)=0.5 -> "... AT <t>").
TCROSS_AT = [6.98285618328e-05, 0.000152998664495, 0.000326276827772]

# Stepped single-run log (3 steps in ONE log, not one log per run): vfinal is a
# FIND (value varies), tcross is a WHEN (level constant 0.5, crossing varies).
STEP_WHEN_LOG = FIXTURES_DIR / "ltspice_step_when.log"
STEP_WHEN_VFINAL = [0.63212, 0.77687, 0.95021]
STEP_WHEN_TCROSS_AT = [6.98285e-05, 1.53000e-04, 3.26000e-04]


def _make_sweep_batch(
    state: SessionState,
    *,
    job_id: str = "sweep1",
    log_files: list[Path | None] | None = None,
) -> BatchJob:
    """Completed 3-run sweep whose run_results mirror the runner's key shape.

    Each entry carries ``raw_file``/``log_file``/``params`` exactly as
    ``_record_run_completion`` stores them (string paths, 0-based int keys).
    """
    logs = LTSPICE_SWEEP_RUN_LOGS if log_files is None else log_files
    bj = make_batch_job(
        job_id,
        netlist=Path("/tmp/sweep_meas_rc.cir"),
        total_runs=len(logs),
        completed_runs=len(logs),
    )
    for i, log in enumerate(logs):
        entry: dict = {
            "raw_file": str(FIXTURES_DIR / f"unused_run{i}.raw"),
            "params": {"R1": RUN_PARAMS[i % len(RUN_PARAMS)]},
        }
        if log is not None:
            entry["log_file"] = str(log)
        bj.run_results[i] = entry
    state.batch_jobs[job_id] = bj
    return bj


async def _stats(state: SessionState, job_id: str = "sweep1") -> dict:
    result = await handle_measurement_stats(MeasurementStatsInput(job_id=job_id), state)
    assert result.structuredContent is not None
    return result.structuredContent["stats"]


def _text(result) -> str:
    """Concatenated text of every content item in a CallToolResult."""
    return "".join(getattr(c, "text", "") for c in (result.content or []))


@pytest.mark.asyncio
class TestJobAggregation:
    async def test_find_style_aggregates_on_value_axis(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        stats = await _stats(state_no_sim)

        entry = stats["vfinal"]
        assert entry["aggregated_field"] == "value"
        assert entry["total_count"] == 3
        assert entry["valid_count"] == 3
        assert entry["failure_count"] == 0
        assert entry["min"] == pytest.approx(min(VFINAL))
        assert entry["max"] == pytest.approx(max(VFINAL))
        assert entry["mean"] == pytest.approx(sum(VFINAL) / 3)
        assert entry["median"] == pytest.approx(VFINAL[1])
        # Largest R -> slowest charge -> smallest V(out) at 0.9 ms.
        assert entry["min_step_index"] == 2
        assert entry["max_step_index"] == 0

    async def test_per_run_table_maps_index_to_corner(self, state_no_sim: SessionState):
        # The per-run table lets min_step_index/max_step_index name the swept
        # corner instead of being a bare position the caller cross-references.
        _make_sweep_batch(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        sc = result.structuredContent
        assert sc is not None
        per_run = sc["per_run"]
        assert [r["run_index"] for r in per_run] == [0, 1, 2]
        assert [r["params"]["R1"] for r in per_run] == RUN_PARAMS
        entry = sc["stats"]["vfinal"]
        assert per_run[entry["min_step_index"]]["params"]["R1"] == RUN_PARAMS[2]
        assert per_run[entry["max_step_index"]]["params"]["R1"] == RUN_PARAMS[0]

    async def test_when_style_swaps_to_at_axis(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        stats = await _stats(state_no_sim)

        entry = stats["tcross"]
        # The per-run scalar is the constant trigger level (0.5 in every
        # log); the aggregator must report the crossing TIMES instead.
        assert entry["aggregated_field"] == "at"
        assert entry["valid_count"] == 3
        assert entry["min"] == pytest.approx(TCROSS_AT[0])
        assert entry["max"] == pytest.approx(TCROSS_AT[2])
        assert entry["mean"] == pytest.approx(sum(TCROSS_AT) / 3)
        assert entry["median"] == pytest.approx(TCROSS_AT[1])
        assert entry["min_step_index"] == 0
        assert entry["max_step_index"] == 2
        # Physics cross-check: crossing time tracks R*C*ln(2) (within the
        # 1 us source rise time + solver step).
        for t, r in zip(TCROSS_AT, RUN_PARAMS, strict=True):
            assert t == pytest.approx(r * 100e-9 * 0.6931, rel=0.02, abs=1.5e-6)

    async def test_run_count_in_text_output(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        assert "3 run(s)" in _text(result)

    async def test_missing_run_log_is_skipped(self, state_no_sim: SessionState):
        # Run 1's log file does not exist on disk: the aggregator skips that
        # run silently and computes stats over the remaining two.
        _make_sweep_batch(
            state_no_sim,
            log_files=[
                LTSPICE_SWEEP_RUN_LOGS[0],
                Path("/nonexistent/run1.log"),
                LTSPICE_SWEEP_RUN_LOGS[2],
            ],
        )
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        assert result.structuredContent is not None
        stats = result.structuredContent["stats"]

        entry = stats["vfinal"]
        assert entry["total_count"] == 2
        assert entry["valid_count"] == 2
        assert entry["min"] == pytest.approx(VFINAL[2])
        assert entry["max"] == pytest.approx(VFINAL[0])
        assert entry["mean"] == pytest.approx((VFINAL[0] + VFINAL[2]) / 2)
        # Level still constant across the surviving runs, times still vary:
        # the at-axis swap holds for the partial aggregate too.
        tcross = stats["tcross"]
        assert tcross["aggregated_field"] == "at"
        assert tcross["min"] == pytest.approx(TCROSS_AT[0])
        assert tcross["max"] == pytest.approx(TCROSS_AT[2])

        assert "2 run(s)" in _text(result)

    async def test_entry_without_log_file_key_is_skipped(self, state_no_sim: SessionState):
        _make_sweep_batch(
            state_no_sim,
            log_files=[LTSPICE_SWEEP_RUN_LOGS[0], None, LTSPICE_SWEEP_RUN_LOGS[2]],
        )
        stats = await _stats(state_no_sim)
        assert stats["vfinal"]["valid_count"] == 2
        assert stats["vfinal"]["mean"] == pytest.approx((VFINAL[0] + VFINAL[2]) / 2)

    async def test_all_logs_missing_errors(self, state_no_sim: SessionState):
        _make_sweep_batch(
            state_no_sim,
            log_files=[Path("/nonexistent/a.log"), Path("/nonexistent/b.log")],
        )
        with pytest.raises(ResultError, match=r"No \.MEAS results found across the runs"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)

    async def test_no_completed_runs_errors(self, state_no_sim: SessionState):
        bj = _make_sweep_batch(state_no_sim, log_files=[])
        bj.run_results = {}
        with pytest.raises(ResultError, match="no completed runs"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)


# Verbatim ngspice-42 batch-mode log (captured from `ngspice -b -r out.raw
# -o out.log` on a .tran deck with a .meas directive): batch mode with a
# rawfile evaluates NO .meas at all, and the log itself carries the reason.
_NGSPICE_BATCH_MEAS_BLOCKED_LOG = (
    "\n"
    "Note: No compatibility mode selected!\n"
    "\n"
    "\n"
    "Circuit: * rc meas\n"
    "\n"
    'binary raw file "meas.raw"\n'
    "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n"
    "\n"
    "Using SPARSE 1.3 as Direct Linear Solver\n"
    "No. of Data Columns : 4  \n"
    "\n"
    "No. of Data Rows : 526\n"
    "\n"
    "No .measure possible in batch mode (-b) with -r rawfile set!\n"
    "Remove rawfile and use .print or .plot or\n"
    "select interactive mode (optionally with .control section) instead.\n"
    "\n"
    "\n"
    "Total analysis time (seconds) = 0.001\n"
    "\n"
    "Total elapsed time (seconds) = 0.009 \n"
)


@pytest.mark.asyncio
class TestBatchNoMeasReasonRelayed:
    """When a batch yields no .MEAS at all, the error must relay WHY from the
    per-run logs (e.g. ngspice's batch-mode skip) instead of a bare 'No .MEAS
    results found' that hides a cause the log states verbatim."""

    def _batch_with_logs(self, state: SessionState, tmp_path: Path, texts: list[str]) -> None:
        logs = []
        for i, text in enumerate(texts):
            log = tmp_path / f"run{i}.log"
            log.write_text(text)
            logs.append(log)
        _make_sweep_batch(state, log_files=logs)

    async def test_ngspice_batch_mode_skip_is_relayed(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        self._batch_with_logs(state_no_sim, tmp_path, [_NGSPICE_BATCH_MEAS_BLOCKED_LOG] * 2)
        with pytest.raises(ResultError) as exc_info:
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)
        msg = str(exc_info.value)
        assert "No .MEAS results found across the runs of job 'sweep1'" in msg
        # The ngspice line is relayed, indented like the single-log branch.
        assert "\n  " in msg
        assert "No .measure possible in batch mode (-b) with -r rawfile set!" in msg
        # Identical per-run diagnostics are deduplicated, not repeated per run.
        assert msg.count("No .measure possible in batch mode") == 1

    async def test_per_run_unique_diagnostics_are_capped(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # Each run's log carries a run-unique warning line (think
        # timestamps or values), so deduplication keeps them all distinct.
        # The relay must cap the list instead of growing one line per run
        # on a large batch.
        texts = [
            f"Circuit: * mc\n\nWarning: seed {1000 + i} produced no .meas output\n"
            for i in range(12)
        ]
        self._batch_with_logs(state_no_sim, tmp_path, texts)
        with pytest.raises(ResultError) as exc_info:
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)
        msg = str(exc_info.value)
        shown = [ln for ln in msg.splitlines() if "Warning: seed" in ln]
        assert len(shown) == 8, msg
        assert "seed 1007" in msg
        assert "seed 1008" not in msg
        assert "... and 4 more distinct diagnostic lines" in msg

    async def test_readable_logs_without_diagnostics_say_so(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # Logs readable but carrying neither .MEAS results nor any diagnostic
        # line: the error must say that honestly rather than implying a cause.
        self._batch_with_logs(
            state_no_sim, tmp_path, ["Circuit: * bare\n\nNo. of Data Rows : 3\n"] * 2
        )
        with pytest.raises(ResultError, match=r"no \.MEAS results and no diagnostics"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)

    async def test_no_readable_logs_message_stays_honest(self, state_no_sim: SessionState):
        # When NO per-run log could be read there are no diagnostics to relay;
        # the error must say the logs were unreadable, not fabricate a cause.
        _make_sweep_batch(
            state_no_sim,
            log_files=[Path("/nonexistent/a.log"), Path("/nonexistent/b.log")],
        )
        with pytest.raises(ResultError, match=r"none of the per-run log files could be read"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)


def _make_single_sim(
    state: SessionState,
    *,
    job_id: str = "single1",
    status: str = "completed",
    log_file: Path | None = LTSPICE_TRAN_RC_LOG,
) -> SimulationJob:
    job = make_sim_job(job_id, status=status, log_file=log_file)
    state.jobs[job_id] = job
    return job


@pytest.mark.asyncio
class TestSingleSimJobAggregation:
    """A completed single-simulation job id aggregates its own log — the same
    physical shape as the ``log_file`` input. A plain (non-.step) run yields
    honest n=1 stats."""

    async def test_completed_single_sim_yields_n1_stats(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="single1"), state_no_sim
        )
        assert result.structuredContent is not None
        entry = result.structuredContent["stats"]["vfinal"]
        assert entry["aggregated_field"] == "value"
        assert entry["total_count"] == 1
        assert entry["valid_count"] == 1
        assert entry["failure_count"] == 0
        assert entry["min"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        assert entry["max"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        assert entry["mean"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        # The single log reads as one step, matching the log_file branch.
        assert "1 step(s)" in _text(result)

    async def test_single_run_echoes_reported_at_neutrally(self, state_no_sim: SessionState):
        # A non-stepped single run: the variation-based WHEN swap can't fire
        # (one sample, nothing to vary), so the axis stays "value" and the stat
        # is the value the simulator printed. The reported ``at`` must still
        # surface — agents read this tool as "show my .meas". But at n=1 a WHEN
        # crossing and a FIND...AT probe point are indistinguishable, so the
        # echo is a NEUTRAL fact for BOTH and must not claim either reading.
        # Fixture run0.log: tcross WHEN V(out)=0.5 AT 6.98e-5, vfinal FIND AT=0.9m.
        _make_single_sim(
            state_no_sim,
            job_id="single_when",
            log_file=FIXTURES_DIR / "ltspice_sweep_meas_run0.log",
        )
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="single_when"), state_no_sim
        )
        assert result.structuredContent is not None
        tcross = result.structuredContent["stats"]["tcross"]
        assert tcross["aggregated_field"] == "value"  # no axis swap at n=1
        assert tcross["min"] == pytest.approx(0.5)  # stat is the printed value
        assert tcross["at"] == pytest.approx(TCROSS_AT[0])  # WHEN crossing echoed
        # vfinal is a FIND...AT in the SAME log: it gets the at echo too (the
        # probe point is a true fact), but it must NOT be mislabeled a crossing
        # or have its value called "the level" — that's false for a FIND.
        vfinal = result.structuredContent["stats"]["vfinal"]
        assert vfinal["aggregated_field"] == "value"
        assert vfinal["min"] == pytest.approx(VFINAL[0])  # the found value, not a level
        assert vfinal["at"] == pytest.approx(0.0009)  # the probe point, echoed neutrally
        text = _text(result)
        assert "at=6.98286e-05" in text
        assert "crossing point" not in text  # no WHEN-specific claim for either
        assert "is the level" not in text

    async def test_running_single_sim_is_gated_on_completion(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim, status="running")
        with pytest.raises(ResultError, match=r"Job 'single1' is not completed"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="single1"), state_no_sim)

    async def test_completed_single_sim_without_log_errors(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim, log_file=None)
        with pytest.raises(ResultError, match="has no log file"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="single1"), state_no_sim)


class TestSteppedLogWhenAxis:
    """A stepped single-run log must apply the WHEN -> ``at`` axis swap exactly
    like the batch path. Before the shared swap, this path always aggregated the
    ``value`` field, so a stepped WHEN .MEAS reported the constant trigger level
    instead of the per-step crossing times. Fixture: ltspice_step_when.log."""

    def test_stepped_log_when_swaps_to_at_axis(self):
        flat_values, axis_map, steps_label, _ = _aggregate_log_measurements(STEP_WHEN_LOG)
        # Constant level (0.5) across steps + varying crossing -> aggregate ``at``.
        assert axis_map["tcross"] == "at"
        assert flat_values["tcross"] == pytest.approx(STEP_WHEN_TCROSS_AT)
        assert "3 step(s)" in steps_label

    def test_stepped_log_find_stays_on_value_axis(self):
        flat_values, axis_map, _, _ = _aggregate_log_measurements(STEP_WHEN_LOG)
        # FIND value varies per step -> stays on the value axis.
        assert axis_map["vfinal"] == "value"
        assert flat_values["vfinal"] == pytest.approx(STEP_WHEN_VFINAL)

    async def test_stepped_when_sim_job_swaps_to_at_axis(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim, job_id="stepped_when", log_file=STEP_WHEN_LOG)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="stepped_when"), state_no_sim
        )
        assert result.structuredContent is not None
        tcross = result.structuredContent["stats"]["tcross"]
        assert tcross["aggregated_field"] == "at"
        assert tcross["min"] == pytest.approx(STEP_WHEN_TCROSS_AT[0])
        assert tcross["max"] == pytest.approx(STEP_WHEN_TCROSS_AT[2])
        assert tcross["mean"] == pytest.approx(sum(STEP_WHEN_TCROSS_AT) / 3)
        # The FIND measurement in the same log stays on the value axis. Its
        # per-step ``at`` (probe point 0.9m) must NOT be echoed: the n=1 crossing
        # echo is gated on total_count==1, and this is a 3-step aggregate.
        vfinal = result.structuredContent["stats"]["vfinal"]
        assert vfinal["aggregated_field"] == "value"
        assert "at" not in vfinal


@pytest.mark.asyncio
class TestJobResolutionErrors:
    async def test_unknown_job_id(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="Job not found: nope"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="nope"), state_no_sim)

    async def test_log_file_and_job_id_mutually_exclusive(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        with pytest.raises(ResultError, match="not both"):
            await handle_measurement_stats(
                MeasurementStatsInput(job_id="sweep1", log_file=str(LTSPICE_SWEEP_RUN_LOGS[0])),
                state_no_sim,
            )


class TestMeasKindRelay:
    """The deck's .MEAS operator overrides the shape heuristic (relay beats
    inference): a FIND probe whose value is constant across runs must NOT
    swap to the probe axis, even though constant-values + varying-at is
    exactly the WHEN shape."""

    def test_meas_kinds_from_netlist(self, tmp_path: Path):
        from ltspice_mcp.tools.analysis import _meas_kinds_from_netlist

        deck = tmp_path / "deck.cir"
        deck.write_text(
            "* kinds\n"
            ".meas tran t1 WHEN V(out)=0.5\n"
            ".meas tran v2 FIND V(out) WHEN V(a)=1\n"
            ".meas op v3 FIND V(mid) AT=1\n"
            ".MEASURE AC gmax MAX mag(V(out))\n"
            ".end\n"
        )
        assert _meas_kinds_from_netlist(deck) == {
            "t1": "when",
            "v2": "value",
            "v3": "value",
            "gmax": "value",
        }
        assert _meas_kinds_from_netlist(tmp_path / "missing.cir") == {}
        assert _meas_kinds_from_netlist(None) == {}

    def test_when_constant_crossings_still_swap(self):
        """A deck-confirmed bare WHEN aggregates crossing times even when every
        run crossed at the same instant (a deterministic batch) — the constant
        trigger level is never the requested quantity."""
        from ltspice_mcp.tools.analysis import _apply_when_axis_swap

        # Trigger level (constant by construction) / crossing time (constant across runs).
        flat: dict[str, list[float | None]] = {"tcross": [1.5, 1.5, 1.5]}
        ats: dict[str, list[float | None]] = {"tcross": [2e-6, 2e-6, 2e-6]}
        axis = _apply_when_axis_swap(flat, ats, kinds={"tcross": "when"})
        assert axis["tcross"] == "at"
        assert flat["tcross"] == [2e-6, 2e-6, 2e-6]

    @pytest.mark.asyncio
    async def test_find_constant_value_does_not_swap(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        deck = tmp_path / "reg.cir"
        deck.write_text("* reg\nV1 ref 0 2.5\n.op\n.meas op vref FIND V(ref) AT=1\n.end\n")
        bj = make_batch_job("mcref", netlist=deck, total_runs=3, completed_runs=3)
        for i, at in enumerate([1, 2, 3]):
            log = tmp_path / f"run{i}.log"
            log.write_text(
                f"Circuit: * reg\n\nvref: V(ref)=2.5 at {at}\n\n"
                "Total elapsed time: 0.01 seconds.\n"
            )
            bj.run_results[i] = {"log_file": str(log), "params": {}}
        state_no_sim.batch_jobs["mcref"] = bj
        stats = await _stats(state_no_sim, "mcref")
        entry = stats["vref"]
        # Without the deck relay this is the degenerate misread: constant
        # values + varying at would swap and aggregate the probe axis 1..3.
        assert entry["aggregated_field"] == "value"
        assert entry["min"] == pytest.approx(2.5)
        assert entry["max"] == pytest.approx(2.5)


@pytest.mark.asyncio
class TestPerRunCapAndPartialWarning:
    async def test_partial_batch_carries_warning(self, state_no_sim: SessionState):
        bj = _make_sweep_batch(state_no_sim)
        bj.total_runs = 5
        bj.status = "running"
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        sc = result.structuredContent
        assert sc is not None
        assert any("3 of 5" in w and "partial" in w for w in sc["warnings"])
        assert "3 of 5" in _text(result)

    async def test_complete_batch_has_no_partial_warning(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        sc = result.structuredContent
        assert sc is not None
        assert "warnings" not in sc

    async def test_per_run_table_capped_with_explicit_truncation(self, state_no_sim: SessionState):
        # 150 runs sharing one real log: table must cap at 100 rows with the
        # full count surfaced — never a silent truncation.
        bj = make_batch_job(
            "big", netlist=Path("/tmp/big.cir"), total_runs=150, completed_runs=150
        )
        for i in range(150):
            bj.run_results[i] = {
                "log_file": str(LTSPICE_SWEEP_RUN_LOGS[0]),
                "params": {"run": i},
            }
        state_no_sim.batch_jobs["big"] = bj

        result = await handle_measurement_stats(MeasurementStatsInput(job_id="big"), state_no_sim)
        sc = result.structuredContent
        assert sc is not None
        assert len(sc["per_run"]) == 100
        assert sc["per_run_truncated"] == 150
        assert any("include_per_run=true" in w for w in sc["warnings"])

        full = await handle_measurement_stats(
            MeasurementStatsInput(job_id="big", include_per_run=True), state_no_sim
        )
        assert full.structuredContent is not None
        assert len(full.structuredContent["per_run"]) == 150
        assert "per_run_truncated" not in full.structuredContent

        omitted = await handle_measurement_stats(
            MeasurementStatsInput(job_id="big", include_per_run=False), state_no_sim
        )
        assert omitted.structuredContent is not None
        assert "per_run" not in omitted.structuredContent
