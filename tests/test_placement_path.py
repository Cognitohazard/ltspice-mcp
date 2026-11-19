"""One checked path for every placement of a run's results beside its circuit.

``run_experiments``' ``execution.results_beside_sheet`` (as the job's runs
finish), ``jobs(action="place_beside_sheet")`` and ``plot_waveform`` with
``in_ltspice`` each put a case's results beside the circuit it ran. They ask
one question (may this run's results stand there as that circuit's own, and
what does the placement leave unchecked or untouched), so they give one
answer: ``sheet_results.place_checked``. These tests run the same case
through each.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import filelock as filelock_mod
from ltspice_mcp.lib import wsl as wsl_mod
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.lib.sheet_results import place_job_beside_sheets
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from tests._ltspice_window import a_window, as_ltspice_reads, put_windows, read_world
from tests.conftest import (
    LIVENESS_S,
    await_until,
    copied_sheet,
    job_that_ran,
    recorded_fixture_simulator,
    terminal_experiment,
)

EDIT = b"TEXT 0 0 Left 2 ;edited\n"
EARLIER = b"the results of an earlier run"


async def place(state: SessionState, job_id: str = "placed") -> dict[str, Any]:
    result = await handle_jobs(
        JobsInput.model_validate({"action": "place_beside_sheet", "job_id": job_id}), state
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(data)
    return data


async def plot(state: SessionState, job_id: str = "placed") -> dict[str, Any]:
    result = await handle_plot_waveform(
        PlotWaveformInput(job_id=job_id, signals=["V(out)"], in_ltspice=True), state
    )
    assert result.structured_content is not None
    return result.structured_content


def with_its_plot_open(sheet: Path) -> dict[str, Any]:
    """A window that has ``sheet`` open and the results beside it opened from it."""
    results = str(sheet.with_suffix(".raw"))
    return a_window(
        {str(sheet): as_ltspice_reads(sheet)},
        panes=[results],
        tied=[results],
        with_results=[str(sheet)],
    )


# ---------------------------------------------------------------------------
# plot_waveform(in_ltspice=true)
# ---------------------------------------------------------------------------


async def test_plot_waveform_does_not_give_a_changed_sheet_its_runs_results(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """The sheet was edited after the run: its results are not put beside it,
    the reply says why with the refusal's code and fix, and the run is opened
    on its own, as a run that cannot be tied to a sheet is."""
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    sheet.with_suffix(".raw").write_bytes(EARLIER)
    sheet.write_bytes(sheet.read_bytes() + EDIT)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window()])

    shown = (await plot(state_no_sim))["ltspice"]

    assert sheet.with_suffix(".raw").read_bytes() == EARLIER
    assert not sheet.with_suffix(".log").exists()
    assert shown["not_placed"]["code"] == "sheet_changed"
    assert "Run amp.asc again" in shown["not_placed"]["message"]
    assert shown["shown"] is True
    assert shown["results"] == str(raw)
    assert "sheet" not in shown and "tied" not in shown
    window = read_world(world)["windows"][0]
    assert window["shown"] == [str(raw)]
    assert "commands" not in window


async def test_plot_waveform_says_a_gone_sheet_was_not_given_its_runs_results(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    sheet.unlink()
    put_windows(state_no_sim, tmp_path / "world.json", [a_window()])

    shown = (await plot(state_no_sim))["ltspice"]

    assert shown["not_placed"]["code"] == "sheet_missing"
    assert shown["shown"] is True and shown["results"] == str(raw)
    assert not sheet.with_suffix(".raw").exists()


async def test_a_placement_refused_after_the_window_closed_a_plot_says_so(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The sheet changes between the check before the window is asked and the
    check under the locks: the window has already closed the plot of the
    sheet's results by then. Nothing is copied, the run opens on its own, and
    the reply says that plot was closed."""
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    sheet.with_suffix(".raw").write_bytes(EARLIER)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [with_its_plot_open(sheet)])
    close = OpenWindows._close_results

    def close_then_edit(self: OpenWindows, *args: Any) -> bool:
        closed = close(self, *args)
        sheet.write_bytes(sheet.read_bytes() + EDIT)
        return closed

    monkeypatch.setattr(OpenWindows, "_close_results", close_then_edit)

    shown = (await plot(state_no_sim))["ltspice"]

    assert shown["not_placed"]["code"] == "sheet_changed"
    assert sheet.with_suffix(".raw").read_bytes() == EARLIER
    assert shown["results"] == str(raw) and "sheet" not in shown
    assert "was closed to make way for these" in shown["note"]
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(sheet.with_suffix(".raw"))]
    assert window["shown"] == [str(raw)]


async def test_a_sheet_another_process_holds_opens_the_run_on_its_own(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The placement's lock wait runs out: nothing is copied, and the run opens
    on its own with the reason, as a refusal does."""
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    put_windows(state_no_sim, tmp_path / "world.json", [a_window()])
    # timing: the wait running out is the case under test; the holder never lets go
    monkeypatch.setattr(filelock_mod, "DEFAULT_TIMEOUT", 0.2)
    target = await asyncio.to_thread(Store.circuit_lock, sheet)
    with filelock_mod.file_lock(target):
        shown = (await plot(state_no_sim))["ltspice"]
    assert shown["not_placed"]["code"] == "sheet_locked"
    assert "another ltspice-mcp process" in shown["not_placed"]["message"]
    assert shown["results"] == str(raw)
    assert not sheet.with_suffix(".raw").exists()


async def test_two_plot_waveform_placements_at_one_sheet_do_not_interleave(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two calls show two runs of one sheet. The second's copy waits on the
    sheet's lock until the first's is done, and each holds the window alone."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir, job_id="first")
    job_that_ran(state_no_sim, sheet, work_dir, job_id="second")
    put_windows(state_no_sim, tmp_path / "world.json", [a_window()])
    target = await asyncio.to_thread(Store.circuit_lock, sheet)
    events: list[str] = []
    first_copying, release, first_done = threading.Event(), threading.Event(), threading.Event()
    copy, attempt = shutil.copyfileobj, filelock_mod._try_acquire

    def watched_copy(read: Any, write: Any, *args: Any) -> None:
        owner = "first" if "first" in Path(read.name).parts else "second"
        events.append(f"{owner} copies")
        if owner == "first":
            first_copying.set()
            assert release.wait(LIVENESS_S)
        copy(read, write, *args)
        events.append(f"{owner} copied")
        if owner == "second":
            # The window is the first call's until it returns.
            assert first_done.wait(LIVENESS_S)

    def watched_attempt(lock: Path, *args: Any) -> bool:
        won = attempt(lock, *args)
        if lock == target and not won:
            events.append("second waits")
        return won

    monkeypatch.setattr(shutil, "copyfileobj", watched_copy)
    monkeypatch.setattr(filelock_mod, "_try_acquire", watched_attempt)
    showing = asyncio.create_task(plot(state_no_sim, "first"))
    showing.add_done_callback(lambda _task: first_done.set())
    await await_until(first_copying.is_set, what="the first copy to start")
    racing = asyncio.create_task(plot(state_no_sim, "second"))
    await await_until(
        lambda: {"second waits", "second copies"} & set(events),
        what="the second placement to wait or copy",
    )
    release.set()
    first, second = await asyncio.gather(showing, racing)
    assert events.index("second waits") < events.index("first copied")
    last_first = len(events) - 1 - events[::-1].index("first copied")
    assert last_first < events.index("second copies")
    assert first["ltspice"]["sheet"] == second["ltspice"]["sheet"] == str(sheet)
    assert (work_dir / "runs" / "second" / "ltspice_tran_rc.raw").read_bytes() == (
        sheet.with_suffix(".raw").read_bytes()
    )


# ---------------------------------------------------------------------------
# A placement cancelled during its copy
# ---------------------------------------------------------------------------


async def test_a_cancelled_placement_holds_its_locks_until_its_copy_ends(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The copy runs in a thread a cancel cannot stop, so the locks stay
    taken until it has ended: a second placement waits for them."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir, job_id="first")
    job_that_ran(state_no_sim, sheet, work_dir, job_id="second")
    target = await asyncio.to_thread(Store.circuit_lock, sheet)
    events: list[str] = []
    first_copying, release = threading.Event(), threading.Event()
    copy, attempt = shutil.copyfileobj, filelock_mod._try_acquire

    def watched_copy(read: Any, write: Any, *args: Any) -> None:
        owner = "first" if "first" in Path(read.name).parts else "second"
        events.append(f"{owner} copies")
        if owner == "first":
            first_copying.set()
            assert release.wait(LIVENESS_S)
        copy(read, write, *args)
        events.append(f"{owner} copied")

    def watched_attempt(lock: Path, *args: Any) -> bool:
        won = attempt(lock, *args)
        if lock == target and not won:
            events.append("second waits")
        return won

    monkeypatch.setattr(shutil, "copyfileobj", watched_copy)
    monkeypatch.setattr(filelock_mod, "_try_acquire", watched_attempt)
    placing = asyncio.create_task(place(state_no_sim, "first"))
    await await_until(first_copying.is_set, what="the first copy to start")
    placing.cancel()
    racing = asyncio.create_task(place(state_no_sim, "second"))
    await await_until(
        lambda: {"second waits", "second copies"} & set(events),
        what="the second placement to wait or copy",
    )
    release.set()
    cancelled, second = await asyncio.gather(placing, racing, return_exceptions=True)
    assert isinstance(cancelled, asyncio.CancelledError)
    assert isinstance(second, dict) and second["outcome"] == "complete", second
    assert events.index("second waits") < events.index("first copied")
    last_first = len(events) - 1 - events[::-1].index("first copied")
    assert last_first < events.index("second copies")


# ---------------------------------------------------------------------------
# What a placement left beside the circuit
# ---------------------------------------------------------------------------


async def test_a_file_of_another_run_left_beside_the_sheet_is_named(
    state_no_sim: SessionState, work_dir: Path
):
    """This run wrote no operating point; the sheet's older one stays where it
    was, and the reply names it and says what is not known of it."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir)
    stale = sheet.with_name("amp.op.raw")
    stale.write_bytes(EARLIER)
    placed = await place(state_no_sim)
    assert stale.read_bytes() == EARLIER
    assert placed["left_in_place"] == [str(stale)]
    (said,) = [line for line in placed["not_checked"] if "amp.op.raw" in line]
    assert "from another run" in said and "left in place" in said
    assert "not been recorded" in said


async def test_a_placement_says_the_files_a_circuit_includes_were_not_compared(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir)
    placed = await place(state_no_sim)
    assert any(
        "were not compared with the ones the run read" in line for line in placed["not_checked"]
    )


# ---------------------------------------------------------------------------
# Every caller of the placement, one answer
# ---------------------------------------------------------------------------

_RC_DECK = "* rc\nV1 in 0 PULSE(0 1 0 1n 1n 1m 2m)\nR1 in out 1k\nC1 out 0 100n\n.tran 1m\n.end\n"


class _ExportingLTspice(LTspice):
    """An LTspice whose export of any sheet is the RC deck the recorded run
    stands for; the export, the run and the copy go through the server's code."""

    @classmethod
    def create_netlist(cls, circuit_file: Any, *_args: Any, **_kwargs: Any) -> Path:
        net = Path(circuit_file).with_suffix(".net")
        net.write_text(_RC_DECK, encoding="utf-8")
        return net


_ExportingLTspice.__name__ = "LTspice"


@pytest.fixture
def ltspice_state(config: ServerConfig, monkeypatch: pytest.MonkeyPatch) -> SessionState:
    recorded_fixture_simulator(monkeypatch)
    monkeypatch.setattr(wsl_mod, "is_wsl", lambda: False)
    return SessionState.create(config, {"ltspice": _ExportingLTspice, "ngspice": NGspiceSimulator})


#: What a caller said of one placement: ``("placed", not_checked)`` or
#: ``("refused", code, message)``, the job's id written as ``JOB``.
Said = tuple[str, ...]


def _said_by_observation(job_id: str, observations: list[dict[str, Any]]) -> Said:
    (observation,) = [
        o
        for o in observations
        if o["code"] in ("results_placed_beside_sheet", "results_not_placed")
    ]
    if observation["code"] == "results_not_placed":
        code = observation["evidence"]["reason_code"]
        # The detail is the refusal's message after the sentence naming the job folder.
        message = observation["detail"].split(" folder. ", 1)[1]
        return ("refused", code, message.replace(job_id, "JOB"))
    return ("placed", *observation["evidence"].get("not_checked", []))


async def _by_run_experiments(state: SessionState, sheet: Path, edit: bool) -> Said:
    run = ExperimentRunner.submit_netlist

    def edit_then_run(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
        sheet.write_bytes(sheet.read_bytes() + EDIT)
        return run(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        if edit:
            patch.setattr(ExperimentRunner, "submit_netlist", edit_then_run)
        receipt = await terminal_experiment(
            state,
            {
                "request_id": "beside",
                "circuits": [{"path": str(sheet), "id": "c0"}],
                "execution": {"wait_s": 30, "results_beside_sheet": True},
                "lint": "off",
            },
        )
    assert receipt["status"] == "completed", receipt
    return _said_by_observation(receipt["job_id"], receipt["observations"])


async def _as_the_job_finishes(state: SessionState, job_id: str) -> Said:
    return _said_by_observation(job_id, await place_job_beside_sheets(state.all_jobs[job_id]))


async def _by_the_jobs_action(state: SessionState, job_id: str) -> Said:
    data = await place(state, job_id)
    if "error" in data:
        error = data["error"]
        return ("refused", error["code"], error["message"].replace(job_id, "JOB"))
    return ("placed", *data.get("not_checked", []))


async def _by_plot_waveform(state: SessionState, job_id: str, world: Path) -> Said:
    put_windows(state, world, [a_window()])
    shown = (await plot(state, job_id))["ltspice"]
    if "not_placed" in shown:
        refusal = shown["not_placed"]
        return ("refused", refusal["code"], refusal["message"].replace(job_id, "JOB"))
    assert shown["sheet"].endswith("amp.asc"), shown
    return ("placed", *shown.get("not_checked", []))


CALLERS = ["run_experiments", "as_the_job_finishes", "jobs_action", "plot_waveform"]


async def _said_by(
    caller: str,
    edit: bool,
    ltspice_state: SessionState,
    work_dir: Path,
    world: Path,
) -> Said:
    sheet = copied_sheet(work_dir)
    # Another run's operating point stands beside the sheet; this run wrote none.
    sheet.with_name("amp.op.raw").write_bytes(EARLIER)
    if caller == "run_experiments":
        return await _by_run_experiments(ltspice_state, sheet, edit)
    job_id = f"ran-{caller}"
    job_that_ran(ltspice_state, sheet, work_dir, job_id=job_id)
    if edit:
        sheet.write_bytes(sheet.read_bytes() + EDIT)
    by: dict[str, Callable[[], Awaitable[Said]]] = {
        "as_the_job_finishes": lambda: _as_the_job_finishes(ltspice_state, job_id),
        "jobs_action": lambda: _by_the_jobs_action(ltspice_state, job_id),
        "plot_waveform": lambda: _by_plot_waveform(ltspice_state, job_id, world),
    }
    return await by[caller]()


@pytest.mark.parametrize("edit", [True, False], ids=["changed", "as_it_ran"])
async def test_every_caller_of_the_placement_gives_one_answer(
    ltspice_state: SessionState,
    work_dir: Path,
    tmp_path: Path,
    edit: bool,
):
    answers: dict[str, Said] = {}
    for caller in CALLERS:
        directory = work_dir / caller
        directory.mkdir()
        answers[caller] = await _said_by(
            caller, edit, ltspice_state, directory, tmp_path / f"{caller}.json"
        )
    first = answers[CALLERS[0]]
    assert all(answer == first for answer in answers.values()), answers
    if edit:
        assert first[:2] == ("refused", "sheet_changed")
    else:
        assert first[0] == "placed"
        assert any("amp.op.raw" in line for line in first[1:]), first
        assert any("were not compared with the ones the run read" in line for line in first[1:]), (
            first
        )
