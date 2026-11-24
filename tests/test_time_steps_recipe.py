"""The ``time_steps`` recipe: how large the steps between a transient run's
stored points were, where the smallest was, and how many fell below a size the
caller names, with what the file does not hold said in the reply.

Each answer is read through ``analyze_results`` and through the Python API,
which share the evaluator but not the reply's rendering. The LTspice answers
are read from the recordings; the ngspice ones from live runs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import metrics, services
from ltspice_mcp.lib.recipes import TimeStepsRecipe
from ltspice_mcp.lib.simulator import SIMULATORS, detect_simulators
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as rec
from tests.conftest import (
    SyncApi,
    make_experiment_job,
    ngspice_binary_raw,
    terminal_experiment,
)

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
    "{TRAN}\n"
    ".end\n"
)


async def _ramp_job(
    state: SessionState, work_dir: Path, *, options: str = "", tran: str = ".tran 10n 10u"
) -> str:
    deck = work_dir / "ramp.cir"
    text = _RAMP_DECK.replace("{OPTIONS}", options).replace("{TRAN}", tran)
    deck.write_text(text, encoding="utf-8")
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"ramp{len(options)}{len(tran)}",
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
            "recipes": [{"key": "ts", "metric": "time_steps", "threshold_s": "1p"}],
        },
    )
    rows = {row["assignments"]["TR"]: row["value"] for row in _rows(data, "ts")}
    assert set(rows) == {"1u", "100n", "1p"}
    small = rows["1p"]
    for slow in (rows["1u"], rows["100n"]):
        assert small["min_step"] < slow["min_step"] / 100
        assert small["steps_below"] > slow["steps_below"]
    # The smallest step follows the 1 ps edge at 2 us; the steps below 1 ps
    # fall at its edges, the rising one at 2 us and the falling one at 5 us.
    assert 2e-6 <= small["min_step_at"] <= 2.01e-6
    assert 2e-6 <= small["first_below_at"] <= small["last_below_at"] <= 5.01e-6
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
    counts = await asyncio.to_thread(_timepoint_counts, work_dir)
    assert len(counts) == 3
    assert sorted(row["value"]["points"] for row in rows) == sorted(a for a, _ in counts)
    # A run that rejected a time point stores the accepted ones only.
    for accepted, rejected in counts:
        if rejected:
            assert accepted in {row["value"]["points"] for row in rows}
    assert all(row["value"]["stored"] == "accepted" for row in rows)


@needs_ngspice
@ROUTES
async def test_a_run_saved_from_a_start_time_is_not_called_complete(
    read: Reader, ngspice_state: SessionState, work_dir: Path
):
    """A .tran start time leaves every point before it out of the file, so the
    1 ps edge at 2 us is not in what is read, and the reply says so instead of
    reporting the steps from 5 us as every accepted point."""
    job_id = await _ramp_job(ngspice_state, work_dir, tran=".tran 10n 10u 5u")
    data = await read(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    for row in _rows(data, "ts"):
        assert row["value"]["stored"] == "not_checked"
        assert row["value"]["t_start"] >= 5e-6
    observations = _coverage(data)
    assert observations
    assert all("start time" in observation["detail"] for observation in observations)


@needs_ngspice
@ROUTES
async def test_interp_stores_the_print_step_and_says_so(
    read: Reader, ngspice_state: SessionState, work_dir: Path
):
    """Under '.options interp' ngspice stores its results on the print step,
    and the row says so. ngspice-42 writes such a file with more rows than its
    header declares, the extra ones repeating a time, and the reader refuses
    it; a build that writes the file whole is read and labelled."""
    job_id = await _ramp_job(ngspice_state, work_dir, options=".options interp\n")
    data = await read(
        ngspice_state,
        {
            "sources": [{"job_id": job_id, "runs": "all"}],
            "recipes": [{"key": "ts", "metric": "time_steps"}],
        },
    )
    if "failures" in data:
        assert all("RAW" in failure["message"] for failure in data["failures"])
        return
    for row in data["results"]["ts"]["values"]:
        assert row["value"]["stored"] == "print_step"
        assert row["value"]["min_step"] > 0
    (observation,) = _coverage(data)
    assert observation["kind"] == "coverage"
    assert "interp" in observation["detail"]


@needs_ngspice
async def test_ngspice_under_interp_declares_the_print_step_and_logs_it(
    ngspice_state: SessionState, work_dir: Path
):
    """What interp_option rests on: under '.options interp' ngspice logs that
    the raw holds interpolated data and declares the print step's points,
    10u / 10n + 1 of them, in the raw's header."""
    await _ramp_job(ngspice_state, work_dir, options=".options interp\n")
    found = await asyncio.to_thread(_interp_facts, work_dir)
    assert len(found) == 3
    assert all(declared == 1001 and logged for declared, logged in found)


def _interp_facts(work_dir: Path) -> list[tuple[int, bool]]:
    """Each ngspice raw's declared point count under ``work_dir``, and whether
    its log says the data was interpolated."""
    facts = []
    for raw in sorted(work_dir.rglob("*.raw")):
        head = raw.read_bytes()[:2048].decode("ascii", errors="replace")
        declared = re.search(r"No\. Points:\s*(\d+)", head)
        log = raw.with_suffix(".log")
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if declared is not None:
            facts.append((int(declared[1]), "Interpolated raw file data" in text))
    return facts


@ROUTES
async def test_stored_points_at_one_time_are_counted_apart(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """Two stored points at one time are not a smallest step of zero: they are
    counted as zero_steps, and the steps are read between the others. The rows
    repeat a time as an ngspice file under '.options interp' does."""
    times = [0.0, 0.0, 1e-9, 2e-9, 2e-9, 2e-9, 5e-9]
    raw = work_dir / "repeated.raw"
    raw.write_bytes(
        ngspice_binary_raw([(t, 0.5) for t in times], ["time", "v(out)"], declared=len(times))
    )
    data = await read(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut", "dialect": "ngspice"}],
            "recipes": [{"key": "ts", "metric": "time_steps", "threshold_s": "1.5n"}],
        },
    )
    (row,) = _rows(data, "ts")
    value = row["value"]
    assert value["points"] == 7
    assert value["zero_steps"] == 3
    assert value["min_step"] == pytest.approx(1e-9)
    assert value["min_step_at"] == 0.0
    assert value["max_step"] == pytest.approx(3e-9)
    assert value["steps_below"] == 2
    # The file names no deck, so which points ngspice stored is not checked.
    assert value["stored"] == "not_checked"
    (observation,) = _coverage(data)
    assert "not checked" in observation["detail"]


def _timepoint_counts(work_dir: Path) -> list[tuple[int, int]]:
    """The time points each ngspice log under ``work_dir`` counts as accepted
    and as rejected."""
    counts = []
    for log in sorted(work_dir.rglob("*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        accepted = re.search(r"Accepted timepoints\s*=\s*(\d+)", text)
        rejected = re.search(r"Rejected timepoints\s*=\s*(\d+)", text)
        if accepted is not None and rejected is not None:
            counts.append((int(accepted[1]), int(rejected[1])))
    return counts


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
    assert "upper limit" in observation["detail"]
    assert "not established" not in observation["detail"]
    assert "plotwinsize=0" in observation["detail"]
    assert "rejected" not in observation["detail"]


async def _smallest(read: Reader, state: SessionState, build: str, case: str, work_dir: Path):
    data = await read(
        state,
        {
            "sources": [_source(build, case, work_dir)],
            "recipes": [{"key": "ts", "metric": "time_steps", "threshold_s": "1n"}],
        },
    )
    (row,) = _rows(data, "ts")
    return row["value"], _coverage(data)


@ROUTES
async def test_xvii_compression_leaves_the_smallest_step_far_above_the_solvers(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """An RC charge with a corner at 2 ms, on XVII. With compression off the
    file holds the 1062 points its log counts as accepted, and the solver's
    smallest step is 48.8 ps from t = 0. Compressed, the file holds 12 points
    and its smallest step is 1 ns at the corner: above the solver's smallest,
    by a factor of about twenty, with the 14 steps below 1 ns not stored."""
    off, off_notes = await _smallest(
        read, state_no_sim, "ltspice17", "raw/compress_small_step_off", work_dir
    )
    on, on_notes = await _smallest(
        read, state_no_sim, "ltspice17", "raw/compress_small_step", work_dir
    )
    assert (off["points"], off["stored"], off_notes) == (1062, "accepted", [])
    assert off["min_step"] == pytest.approx(4.8828125e-11, rel=1e-6)
    assert off["steps_below"] == 14
    assert (on["points"], on["stored"]) == (12, "reduced")
    assert on["min_step"] == pytest.approx(1e-9, rel=1e-6)
    assert on["min_step_at"] == pytest.approx(2e-3)
    assert on["steps_below"] == 1
    assert on["min_step"] >= off["min_step"]
    assert on["min_step"] / off["min_step"] == pytest.approx(20.48, rel=1e-3)
    (note,) = on_notes
    assert "upper limit" in note["detail"]
    assert "not established" not in note["detail"]


@ROUTES
async def test_ltspice_26_stores_the_same_points_with_compression_on_and_off(
    read: Reader, state_no_sim: SessionState, work_dir: Path
):
    """The same two decks on LTspice 26: both files hold the same 1062 points,
    so the smallest stored step is the same, 48.8 ps; its log prints no count
    of accepted steps, so what the points are is not checked."""
    on, on_notes = await _smallest(
        read, state_no_sim, "ltspice26", "raw/compress_small_step", work_dir
    )
    off, off_notes = await _smallest(
        read, state_no_sim, "ltspice26", "raw/compress_small_step_off", work_dir
    )
    assert on == off
    assert on["points"] == 1062
    assert on["min_step"] == pytest.approx(4.8828125e-11, rel=1e-6)
    assert on["stored"] == "not_checked"
    assert len(on_notes) == len(off_notes) == 1


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


# ---------------------------------------------------------------------------
# The guard: no smallest step without what the file leaves out
# ---------------------------------------------------------------------------


def _ngspice_shaped(work_dir: Path, name: str, start: float = 0.0) -> Path:
    raw = work_dir / f"{name}.raw"
    times = [start + t for t in (0.0, 1e-9, 3e-9, 7e-9)]
    raw.write_bytes(
        ngspice_binary_raw([(t, 0.5) for t in times], ["time", "v(out)"], declared=len(times))
    )
    return raw


#: What LTspice stored of each recording, by build: XVII under compression
#: keeps fewer points than it accepted, XVII with compression off keeps them
#: all, and LTspice 26 logs no count to compare with.
_LTSPICE_STORED = {
    ("ltspice17", "raw/compress_1mv"): "reduced",
    ("ltspice17", "raw/compress_1mv_off"): "accepted",
    ("ltspice17", "raw/step_tran"): "reduced",
    ("ltspice17", "raw/tran_window"): "reduced",
    ("ltspice26", "raw/compress_1mv"): "not_checked",
    ("ltspice26", "raw/compress_1mv_off"): "not_checked",
    ("ltspice26", "raw/step_tran"): "not_checked",
    ("ltspice26", "raw/tran_window"): "not_checked",
}

_RC = "* deck\nV1 in 0 PULSE(0 1 1n 1n 1n 2n 5n)\nR1 in out 1k\nC1 out 0 1p\n"

#: ngspice decks, each with what the server answers of its stored points:
#: an interp card read, not read, or in a file whose reading turns on the mode.
_NGSPICE_DECKS: dict[str, tuple[str, dict[str, str], str]] = {
    "plain": ("", {}, "accepted"),
    "interp": (".options interp\n", {}, "print_step"),
    "interp_in_uncalled_subckt": (
        ".subckt foo a\n.options interp\nR1 a 0 1k\n.ends\n",
        {},
        "accepted",
    ),
    "interp_in_called_subckt": (
        "X1 out foo\n.subckt foo a\n.options interp\nR1 a 0 1meg\n.ends\n",
        {},
        "print_step",
    ),
    "interp_in_include": (".include opts.inc\n", {"opts.inc": ".options interp\n"}, "print_step"),
    "interp_in_lib": (
        ".lib m.lib tt\n",
        {"m.lib": ".lib tt\n.options interp\n.endl tt\n"},
        "not_checked",
    ),
    "lib_without_interp": (
        ".lib m.lib tt\n",
        {"m.lib": ".lib tt\n.model d1 D\n.endl tt\n"},
        "accepted",
    ),
    "include_missing": (".include gone.inc\n", {}, "not_checked"),
}


def _bare(raw: Path, dialect: str) -> services.AnalysisSource:
    return services.AnalysisSource(
        raw=raw,
        log=raw.with_suffix(".log"),
        netlist=None,
        dialect=None,
        identity=None,
        trusted_job_artifact=False,
        explicit_dialect=dialect,
    )


def _sources(family: str, work_dir: Path) -> list[tuple[str, services.AnalysisSource, str]]:
    """Every kind of run of ``family`` whose stored points the server can be
    asked about, by build, compression and deck, with the answer expected."""
    found: list[tuple[str, services.AnalysisSource, str]] = []
    if family == "ltspice":
        for (build, case_id), stored in _LTSPICE_STORED.items():
            raw = rec.stage_run(build, case_id, work_dir / build)
            found.append((f"{build}:{case_id}", _bare(raw, "ltspice"), stored))
            unknown = dataclasses.replace(
                _bare(raw, "ltspice"), producer=Producer("ltspice", None, None)
            )
            found.append((f"unknown build:{case_id}", unknown, "not_checked"))
        return found
    producer = Producer(family, None, None)
    raw = _ngspice_shaped(work_dir, family)
    plain = dataclasses.replace(_bare(raw, "ngspice"), producer=producer)
    found.append((f"{family}:no deck", plain, "not_checked"))
    for name, (cards, files, stored) in _NGSPICE_DECKS.items():
        folder = work_dir / f"{family}_{name}"
        folder.mkdir()
        for file, text in files.items():
            (folder / file).write_text(text, encoding="utf-8")
        deck = folder / "deck.cir"
        deck.write_text(f"{_RC}{cards}.tran 1n 7n\n.end\n", encoding="utf-8")
        expected = stored if family == "ngspice" else "not_checked"
        found.append((f"{family}:{name}", dataclasses.replace(plain, netlist=deck), expected))
    later = _ngspice_shaped(work_dir, f"{family}_later", start=5e-9)
    deck = work_dir / f"{family}_later.cir"
    deck.write_text(f"{_RC}.tran 1n 12n 5n\n.end\n", encoding="utf-8")
    saved_late = dataclasses.replace(_bare(later, "ngspice"), producer=producer, netlist=deck)
    found.append((f"{family}:saved from a start time", saved_late, "not_checked"))
    interp_deck = work_dir / f"{family}_later_interp.cir"
    interp_deck.write_text(f"{_RC}.options interp\n.tran 1n 12n 5n\n.end\n", encoding="utf-8")
    interp_late = dataclasses.replace(saved_late, netlist=interp_deck)
    expected = "print_step" if family == "ngspice" else "not_checked"
    found.append((f"{family}:interp, saved from a start time", interp_late, expected))
    return found


@pytest.mark.parametrize("family", sorted(SIMULATORS))
async def test_no_smallest_step_is_reported_without_what_the_file_leaves_out(
    family: str, state_no_sim: SessionState, work_dir: Path
):
    """Over every simulator the server runs, a row that reports a smallest step
    says what the stored points are, as the evidence for each kind of run
    establishes it, and every answer but 'every accepted point' carries the
    coverage observation. A run whose points are known to be reduced (LTspice
    XVII under compression) is always 'reduced'."""
    recipe = TimeStepsRecipe(key="ts", metric="time_steps")
    sources = await asyncio.to_thread(_sources, family, work_dir)
    for label, source, expected in sources:
        value = await metrics.time_steps(source, recipe, 0, state_no_sim)
        assert "min_step" in value, label
        raw = await services.load_raw(source, state_no_sim)
        if await asyncio.to_thread(services.compression_drops_points, source, raw):
            assert value["stored"] == "reduced", label
        assert value["stored"] == expected, label
        if value["stored"] == "accepted":
            assert "observations" not in value, label
            continue
        (observation,) = value["observations"]
        assert observation["code"] == COVERAGE_CODE, label
        assert observation["kind"] == "coverage", label
        assert "rejected" not in observation["detail"], label
