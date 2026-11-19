"""The ``time_steps`` recipe: how large the steps between a transient run's
stored points were, where the smallest was, and how many fell below a size the
caller names, with what the file does not hold said in the reply.

Each answer is read through ``analyze_results`` and through the Python API,
which share the evaluator but not the reply's rendering. The LTspice answers
are read from the recordings; the ngspice ones from live runs.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi, make_experiment_job, terminal_experiment

Reader = Callable[[SessionState, dict[str, Any]], Any]

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")


async def _through_tool(state: SessionState, request: dict[str, Any]) -> dict[str, Any]:
    reply = await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state)
    assert reply.structured_content is not None
    return reply.structured_content


async def _through_api(state: SessionState, request: dict[str, Any]) -> dict[str, Any]:
    # SyncApi runs each call on a loop of its own, which cannot nest in this one.
    return await asyncio.to_thread(SyncApi(state).analyze_results, **request)


ROUTES = pytest.mark.parametrize("read", [_through_tool, _through_api], ids=["tool", "api"])

#: The code of the observation that says the stored points are not every step
#: the solver accepted, or that whether they are was not checked.
COVERAGE_CODE = "time_steps_coverage"


def _rows(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    assert "failures" not in data, data.get("failures")
    return data["results"][key]["values"]


def _coverage(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in data.get("observations") or [] if o["code"] == COVERAGE_CODE]


@pytest.fixture
def ngspice_state(work_dir: Path) -> SessionState:
    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    available = detect_simulators(config)
    if "ngspice" not in available:
        pytest.fail("ngspice is on PATH but detect_simulators did not report it usable")
    return SessionState.create(config, available)


# ---------------------------------------------------------------------------
# One row per run of a job: the run that stepped small is plain to see
# ---------------------------------------------------------------------------

#: An RC driven by a pulse whose edges take TR. A 1 ps edge makes ngspice take
#: steps of under a picosecond at 2 us; the slower edges never do.
_RAMP_DECK = (
    "* rc driven by a pulse\n"
    ".param TR=1u\n"
    "V1 in 0 PULSE(0 1 2u {TR} {TR} 3u 10u)\n"
    "R1 in out 1k\n"
    "C1 out 0 1n\n"
    "{OPTIONS}"
    ".tran 10n 10u\n"
    ".end\n"
)


async def _ramp_job(state: SessionState, work_dir: Path, *, options: str = "") -> str:
    deck = work_dir / "ramp.cir"
    deck.write_text(_RAMP_DECK.replace("{OPTIONS}", options), encoding="utf-8")
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"ramp{len(options)}",
            "circuits": [{"path": deck.name, "id": "ramp"}],
            "variations": [{"kind": "assign", "assign": {"TR": ["1u", "100n", "1p"]}}],
            "execution": {"wait_s": 90, "simulator": "ngspice"},
        },
    )
    assert receipt["status"] == "completed", receipt
    return receipt["job_id"]


@needs_ngspice
@ROUTES
async def test_a_job_of_three_runs_gives_one_row_each_and_names_the_run_that_stepped_small(
    read: Reader, ngspice_state: SessionState, work_dir: Path
):
    job_id = await _ramp_job(ngspice_state, work_dir)
    data = await read(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps", "threshold_s": 1e-12}],
        },
    )
    rows = {row["assignments"]["TR"]: row["value"] for row in _rows(data, "ts")}
    assert set(rows) == {"1u", "100n", "1p"}
    small = rows["1p"]
    for slow in (rows["1u"], rows["100n"]):
        assert small["min_step"] < slow["min_step"] / 100
        assert slow["steps_below"] == 0
        assert "first_below_at" not in slow
    # The smallest step follows the 1 ps edge at 2 us.
    assert 2e-6 <= small["min_step_at"] <= 2.01e-6
    assert small["steps_below"] > 0
    assert 2e-6 <= small["first_below_at"] <= small["last_below_at"] <= 7e-6
    for value in rows.values():
        assert value["points"] > 2
        assert value["min_step"] <= value["median_step"] <= value["max_step"]
        # ngspice without '.options interp' stores every time point it accepted.
        assert value["stored"] == "accepted"
    assert _coverage(data) == []


@needs_ngspice
async def test_ngspice_stores_every_accepted_time_point_without_interp(
    ngspice_state: SessionState, work_dir: Path
):
    """What the recipe reports as 'accepted' for an ngspice run: '.options acct'
    makes ngspice count the time points it accepted, and the raw holds that
    many, the rejected ones not among them."""
    job_id = await _ramp_job(ngspice_state, work_dir, options=".options acct\n")
    data = await _through_tool(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    rows = _rows(data, "ts")
    accepted = await asyncio.to_thread(_accepted_counts, work_dir)
    assert len(accepted) == 3
    assert sorted(row["value"]["points"] for row in rows) == sorted(accepted)
    assert all(row["value"]["stored"] == "accepted" for row in rows)


@needs_ngspice
@ROUTES
async def test_interp_stores_the_print_step_and_says_so(
    read: Reader, ngspice_state: SessionState, work_dir: Path
):
    """Under '.options interp' ngspice stores its results on the print step,
    and a pair of stored points at one time is counted apart, never as a
    smallest step of zero."""
    job_id = await _ramp_job(ngspice_state, work_dir, options=".options interp\n")
    data = await read(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    for row in _rows(data, "ts"):
        value = row["value"]
        assert value["stored"] == "print_step"
        assert value["min_step"] > 0
        assert value["min_step"] == pytest.approx(10e-9, rel=1e-6)
        assert value["zero_steps"] >= 0
    (observation,) = _coverage(data)
    assert observation["kind"] == "coverage"
    assert "interp" in observation["detail"]


@needs_ngspice
async def test_stored_points_at_one_time_are_counted_apart(
    ngspice_state: SessionState, work_dir: Path
):
    """The count of pairs at one time is the raw's own, read independently."""
    job_id = await _ramp_job(ngspice_state, work_dir, options=".options interp\n")
    data = await _through_tool(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    expected = await asyncio.to_thread(_zero_pairs_of_each_raw, work_dir)
    assert len(expected) == 3
    assert sorted(row["value"]["zero_steps"] for row in _rows(data, "ts")) == expected


def _accepted_counts(work_dir: Path) -> list[int]:
    """The time points each ngspice log under ``work_dir`` counts as accepted."""
    counts = []
    for log in sorted(work_dir.rglob("*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        if (found := re.search(r"Accepted timepoints\s*=\s*(\d+)", text)) is not None:
            counts.append(int(found[1]))
    return counts


def _zero_pairs_of_each_raw(work_dir: Path) -> list[int]:
    return sorted(_zero_pairs(raw) for raw in sorted(work_dir.rglob("*.raw")))


def _zero_pairs(raw: Path) -> int:
    """Pairs of consecutive stored points at one time in an ngspice binary raw,
    read from its bytes."""
    data = raw.read_bytes()
    start = data.index(b"Binary:\n")
    header = data[:start].decode("ascii")
    variables = int(re.search(r"No\. Variables:\s*(\d+)", header)[1])  # type: ignore[index]
    points = int(re.search(r"No\. Points:\s*(\d+)", header)[1])  # type: ignore[index]
    values = np.frombuffer(data[start + 8 : start + 8 + points * variables * 8], "<f8")
    times = values.reshape(points, variables)[:, 0]
    return int(np.count_nonzero(np.diff(times) == 0))


# ---------------------------------------------------------------------------
# Recorded LTspice runs
# ---------------------------------------------------------------------------


def _source(build: str, case_id: str, work_dir: Path) -> dict[str, Any]:
    return {"path": str(rec.stage_run(build, case_id, work_dir / build)), "label": "dut"}


@ROUTES
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_threshold_counts_the_steps_below_it(
    read: Reader, build: str, state_no_sim: SessionState, work_dir: Path
):
    """The 1 mV charge with every point stored: the solver starts at 48.8 ps
    and doubles its step six times before it reaches 1 ns."""
    data = await read(
        state_no_sim,
        {
            "sources": [_source(build, "raw/compress_1mv_off", work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps", "threshold_s": "1n"}],
        },
    )
    (row,) = _rows(data, "ts")
    value = row["value"]
    assert value["points"] == {"ltspice17": 1078, "ltspice26": 1046}[build]
    assert value["min_step"] == pytest.approx(4.8828125e-11, rel=1e-6)
    assert value["min_step_at"] == 0.0
    assert value["max_step"] == pytest.approx(4.8828125e-6, rel=1e-6)
    assert value["median_step"] == pytest.approx(4.8828125e-6, rel=1e-6)
    assert value["zero_steps"] == 0
    assert value["steps_below"] == 6
    assert value["first_below_at"] == 0.0
    assert value["last_below_at"] < 1e-9


@ROUTES
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_window_bounds_the_points_read(
    read: Reader, build: str, state_no_sim: SessionState, work_dir: Path
):
    data = await read(
        state_no_sim,
        {
            "sources": [_source(build, "raw/compress_1mv_off", work_dir)],
            "recipes": [
                {
                    "key": "ts",
                    "metric": "time_steps",
                    "threshold_s": "1n",
                    "window": {"start": "1m", "end": "2m"},
                }
            ],
        },
    )
    (row,) = _rows(data, "ts")
    value = row["value"]
    assert value["min_step"] > 1e-9
    assert value["steps_below"] == 0
    assert 1e-3 <= value["min_step_at"] < 2e-3
    assert value["points"] < 250


#: Each step of the recorded three-resistance step response: its points and
#: its smallest step and where it starts.
STEP_TRAN = {
    "ltspice17": [
        (60, 7.184e-08, 7.232e-08),
        (62, 9.591e-08, 5.936e-07),
        (40, 8.960e-08, 9.104e-07),
    ],
    "ltspice26": [(71, 2.0e-10, 0.0)] * 3,
}


@ROUTES
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_stepped_deck_gives_one_row_per_step(
    read: Reader, build: str, state_no_sim: SessionState, work_dir: Path
):
    data = await read(
        state_no_sim,
        {
            "sources": [_source(build, "raw/step_tran", work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
            "all_steps": True,
        },
    )
    rows = _rows(data, "ts")
    assert [row["step_index"] for row in rows] == [0, 1, 2]
    for row, (points, smallest, at) in zip(rows, STEP_TRAN[build], strict=True):
        assert row["value"]["points"] == points
        assert row["value"]["min_step"] == pytest.approx(smallest, rel=1e-3)
        assert row["value"]["min_step_at"] == pytest.approx(at, rel=1e-3, abs=1e-12)


@ROUTES
async def test_a_compressed_xvii_run_carries_the_coverage_observation(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """XVII stored 13 of the 1078 points its log counts as accepted, so the
    steps are between stored points; a default reply says so at its top level."""
    data = await read(
        state_no_sim,
        {
            "sources": [_source("ltspice17", "raw/compress_1mv", work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    (row,) = _rows(data, "ts")
    assert row["value"]["points"] == 13
    assert row["value"]["stored"] == "reduced"
    (observation,) = _coverage(data)
    assert observation["kind"] == "coverage"
    assert observation["evidence"]["recipe"] == "ts"
    assert "not established" in observation["detail"]
    assert "plotwinsize=0" in observation["detail"]
    assert "rejected" not in observation["detail"]


@ROUTES
async def test_an_uncompressed_xvii_run_carries_no_coverage_observation(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """With compression off XVII stored as many points as its log counts as
    accepted (raw/compress_1mv_off: 1078 and 1078)."""
    data = await read(
        state_no_sim,
        {
            "sources": [_source("ltspice17", "raw/compress_1mv_off", work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    (row,) = _rows(data, "ts")
    assert row["value"]["stored"] == "accepted"
    assert _coverage(data) == []


@ROUTES
@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_run_that_is_not_a_transient_is_refused_with_the_fix(
    read: Reader, build: str, state_no_sim: SessionState, work_dir: Path
):
    data = await read(
        state_no_sim,
        {
            "sources": [_source(build, "raw/ac", work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    (failure,) = data["failures"]
    assert ".tran" in failure["message"]


@ROUTES
async def test_a_job_reads_each_case_from_its_own_recording(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """Three runs of one job, recorded on XVII with compression off: the run
    whose solver started at 48.8 ps is the one the rows single out."""
    job = make_experiment_job(state_no_sim, job_id="exp_steps", count=3)
    for case, case_id in zip(
        job.cases,
        ["raw/tran_uncompressed", "raw/compress_1mv_off", "raw/tran_uncompressed"],
        strict=True,
    ):
        staged = rec.stage_run("ltspice17", case_id, work_dir / case.case_id)
        assert case.raw_file is not None
        shutil.copyfile(staged, case.raw_file)
        shutil.copyfile(staged.with_suffix(".log"), case.raw_file.with_suffix(".log"))
    data = await read(
        state_no_sim,
        {
            "sources": [{"job_id": job.job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    rows = _rows(data, "ts")
    assert [row["value"]["points"] for row in rows] == [73, 1078, 73]
    assert [row["value"]["min_step"] for row in rows] == pytest.approx(
        [2.0e-10, 4.8828125e-11, 2.0e-10], rel=1e-6
    )
