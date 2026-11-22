"""A case's results beside the sheet it ran: jobs(action="place_beside_sheet"),
and run_experiments' execution.results_beside_sheet.

LTspice keeps a sheet's own results beside it under its name and opens them
from there; a job keeps its runs' in the job folder. The action copies one
case's there on request, the run option copies each sheet's as the job's runs
finish, and both refuse what the sheet's window could not take as its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import threading
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import filelock as filelock_mod
from ltspice_mcp.lib import wsl as wsl_mod
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    await_until,
    copied_sheet,
    job_that_ran,
    recorded_fixture_simulator,
    terminal_experiment,
)

pytestmark = pytest.mark.asyncio


async def place(state: SessionState, **arguments: Any) -> dict[str, Any]:
    result = await handle_jobs(
        JobsInput.model_validate(
            {"action": "place_beside_sheet", "job_id": "placed", **arguments}
        ),
        state,
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(data)
    return data


async def test_a_cases_results_are_copied_beside_its_sheet(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    first = await place(state_no_sim)
    assert first["outcome"] == "complete"
    assert first["sheet"] == str(sheet) and first["case_id"] == "case-0000"
    # Nothing was there to replace, and a false flag is left out.
    assert [(Path(f["to"]).name, f.get("replaced", False)) for f in first["files"]] == [
        ("amp.raw", False),
        ("amp.log", False),
    ]
    assert sheet.with_suffix(".raw").read_bytes() == raw.read_bytes()
    assert sheet.with_suffix(".log").read_bytes() == raw.with_suffix(".log").read_bytes()
    assert raw.is_file()  # the job keeps its own copy
    again = await place(state_no_sim)
    assert [f["replaced"] for f in again["files"]] == [True, True]


async def test_the_other_raws_a_run_wrote_go_beside_it_too(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    shutil.copyfile(raw, raw.with_name(f"{raw.stem}.op.raw"))
    state_no_sim.all_jobs["placed"].cases[0].other_raws = ["op"]
    placed = await place(state_no_sim)
    assert Path(placed["files"][-1]["to"]).name == "amp.op.raw"
    assert sheet.with_name("amp.op.raw").is_file()


async def test_a_case_with_variations_says_what_the_sheet_does_not_show(
    state_no_sim: SessionState, work_dir: Path
):
    """A job of several cases names the case, and each case's assignments
    differ from the drawn values."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir, count=3)
    refused = await place(state_no_sim)
    assert refused["error"]["code"] == "case_id_required"
    assert "case-0000, case-0001, case-0002" in refused["error"]["message"]
    placed = await place(state_no_sim, case_id="case-0002")
    assert placed["assignments"] == {"R": 2}
    assert "assignments" in placed["hint"]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("sheet", "sheet_changed"),
        ("simulator", "not_ltspice"),
        ("status", "case_not_produced"),
        ("case", "case_not_found"),
        ("results", "results_missing"),
        ("gone", "sheet_missing"),
    ],
)
async def test_what_the_sheet_could_not_take_as_its_own_is_refused(
    state_no_sim: SessionState, work_dir: Path, change: str, code: str
):
    sheet = copied_sheet(work_dir)
    raw = job_that_ran(state_no_sim, sheet, work_dir)
    job = state_no_sim.all_jobs["placed"]
    arguments: dict[str, Any] = {}
    if change == "sheet":
        sheet.write_bytes(sheet.read_bytes() + b"TEXT 0 0 Left 2 ;edited\n")
    elif change == "simulator":
        job.simulator = "NGspiceSimulator"
    elif change == "status":
        job.cases[0].status = "failed"
    elif change == "results":
        raw.unlink()
    elif change == "gone":
        sheet.unlink()
    else:
        arguments["case_id"] = "case-9999"
    refused = await place(state_no_sim, **arguments)
    assert refused["error"]["code"] == code
    assert not sheet.with_suffix(".raw").exists()


async def test_a_sheets_receipt_says_where_its_results_are(
    state_no_sim: SessionState, work_dir: Path
):
    """The receipt of a sheet's run says the results are not beside the sheet,
    and names the action that puts them there. A netlist's run says nothing,
    nor does a sheet's run on another simulator, which LTspice only exported."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir)
    job = state_no_sim.all_jobs["placed"]
    receipt = await place(state_no_sim, action="status")
    (note,) = [o for o in receipt["observations"] if o["code"] == "results_not_beside_sheet"]
    assert note["evidence"] == {"sheets": ["amp.asc"]}
    assert 'jobs(action="place_beside_sheet", job_id="placed"' in note["detail"]
    job.simulator = "NGspiceSimulator"
    receipt = await place(state_no_sim, action="status")
    assert all(o["code"] != "results_not_beside_sheet" for o in receipt.get("observations", []))
    job.simulator = "LTspice"
    job.cases[0].circuit_path = work_dir / "amp.cir"
    receipt = await place(state_no_sim, action="status")
    assert all(o["code"] != "results_not_beside_sheet" for o in receipt.get("observations", []))


# ---------------------------------------------------------------------------
# run_experiments(execution={"results_beside_sheet": true})
# ---------------------------------------------------------------------------

_RC_DECK = "* rc\nV1 in 0 PULSE(0 1 0 1n 1n 1m 2m)\nR1 in out 1k\nC1 out 0 100n\n.tran 1m\n.end\n"


class _ExportingLTspice(LTspice):
    """An LTspice whose export of any sheet is the RC deck the recorded run
    (``recorded_fixture_simulator``) stands for. Only the program is stood in
    for: the export, the run and the copy go through the server's own code."""

    @classmethod
    def create_netlist(cls, circuit_file: Any, *_args: Any, **_kwargs: Any) -> Path:
        net = Path(circuit_file).with_suffix(".net")
        net.write_text(_RC_DECK, encoding="utf-8")
        return net


# A job records its simulator's class name, which says which simulator ran; the
# server's own LTspice classes are all named for spicelib's (lib/simulator.py).
_ExportingLTspice.__name__ = "LTspice"


@pytest.fixture
def ltspice_state(config: ServerConfig, monkeypatch: pytest.MonkeyPatch) -> SessionState:
    recorded_fixture_simulator(monkeypatch)
    # An LTspice under WSL would have its runs moved to the Windows side.
    monkeypatch.setattr(wsl_mod, "is_wsl", lambda: False)
    return SessionState.create(config, {"ltspice": _ExportingLTspice, "ngspice": NGspiceSimulator})


def _run(*paths: Path, **execution: Any) -> dict[str, Any]:
    return {
        "request_id": "beside",
        "circuits": [{"path": str(path), "id": f"c{n}"} for n, path in enumerate(paths)],
        "execution": {"wait_s": 30, "results_beside_sheet": True, **execution},
        "lint": "off",
    }


def _observed(receipt: dict[str, Any], code: str) -> list[dict[str, Any]]:
    return [o for o in receipt.get("observations", []) if o["code"] == code]


async def test_a_sheets_run_leaves_its_results_beside_it(
    ltspice_state: SessionState, work_dir: Path
):
    sheet = copied_sheet(work_dir)
    receipt = await terminal_experiment(ltspice_state, _run(sheet))
    assert receipt["status"] == "completed", receipt
    recorded = FIXTURES_DIR / "ltspice_tran_rc"
    assert sheet.with_suffix(".raw").read_bytes() == recorded.with_suffix(".raw").read_bytes()
    assert sheet.with_suffix(".log").read_bytes() == recorded.with_suffix(".log").read_bytes()
    (placed,) = _observed(receipt, "results_placed_beside_sheet")
    assert placed["evidence"]["sheet"] == str(sheet)
    assert [Path(f["to"]).name for f in placed["evidence"]["files"]] == ["amp.raw", "amp.log"]
    # The receipt no longer says nothing was written beside the sheet.
    assert _observed(receipt, "results_not_beside_sheet") == []
    # The job keeps its own copy, which every read of the job takes.
    (row,) = receipt["runs"]["items"]
    kept = Path(row["raw"])
    assert kept != sheet.with_suffix(".raw") and await asyncio.to_thread(kept.is_file)


async def test_a_sheet_changed_during_the_run_is_left_alone(
    ltspice_state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    sheet = copied_sheet(work_dir)
    run = ExperimentRunner.submit_netlist

    def edit_then_run(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
        sheet.write_bytes(sheet.read_bytes() + b"TEXT 0 0 Left 2 ;edited\n")
        return run(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", edit_then_run)
    receipt = await terminal_experiment(ltspice_state, _run(sheet))
    assert receipt["status"] == "completed", receipt
    assert not sheet.with_suffix(".raw").exists()
    (note,) = _observed(receipt, "results_not_placed")
    assert note["evidence"]["reason_code"] == "sheet_changed"
    assert "Run amp.asc again" in note["detail"]
    assert _observed(receipt, "results_not_beside_sheet") == []


async def test_a_copy_that_fails_costs_only_the_copy(ltspice_state: SessionState, work_dir: Path):
    """The runs stand, the receipt says what failed, and names the action
    that copies the results once it is fixed."""
    sheet = copied_sheet(work_dir)
    sheet.with_suffix(".raw").mkdir()  # nothing can replace a folder with a file
    receipt = await terminal_experiment(ltspice_state, _run(sheet))
    assert receipt["status"] == "completed", receipt
    (note,) = _observed(receipt, "results_not_placed")
    assert note["evidence"]["reason_code"] == "copy_failed"
    assert "jobs(action='place_beside_sheet'" in note["detail"]


async def test_a_sheet_another_process_holds_is_left_alone(
    ltspice_state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A sheet whose lock another server process holds past the lock's wait
    (an edit in progress) is not copied to, and the job's runs still stand."""
    sheet = copied_sheet(work_dir)
    # timing: the wait running out is the case under test; the holder never lets go
    monkeypatch.setattr(filelock_mod, "DEFAULT_TIMEOUT", 0.2)
    held = contextlib.ExitStack()
    run = ExperimentRunner.submit_netlist

    def run_then_hold(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
        # Taken once the sheet is exported, as another process's edit would be.
        held.enter_context(filelock_mod.file_lock(Store.circuit_lock(sheet)))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", run_then_hold)
    with held:
        receipt = await terminal_experiment(ltspice_state, _run(sheet))
    assert receipt["status"] == "completed", receipt
    assert not sheet.with_suffix(".raw").exists()
    (note,) = _observed(receipt, "results_not_placed")
    assert note["evidence"]["reason_code"] == "sheet_locked"
    assert "jobs(action='place_beside_sheet'" in note["detail"]


async def test_a_job_whose_process_stops_says_it_copied_nothing(
    ltspice_state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The process running the job stops with one sheet's run finished and the
    other's still running. Nothing is copied beside either sheet, and the
    receipt says so for the sheet whose run finished, with the reason and the
    action that copies its results."""
    first, second = copied_sheet(work_dir, "first"), copied_sheet(work_dir, "second")
    run = ExperimentRunner.submit_netlist
    running: dict[str, Any] = {}

    def second_never_exits(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
        _netlist, run_filename, callback = args
        if Path(run_filename).stem.endswith("_case_1"):
            running[Path(run_filename).stem] = callback
            return object()
        return run(self, *args, **kwargs)

    async def confirmed_kill(_runner: ExperimentRunner, token: str) -> None:
        if (callback := running.pop(token, None)) is not None:
            callback(RunOutcome("", "", 0, "Stopped by test simulator"))

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", second_never_exits)
    monkeypatch.setattr(ExperimentRunner, "_kill_case", confirmed_kill)
    submitted = await handle_run_experiments(
        RunExperimentsInput.model_validate(_run(first, second, wait_s=0, max_parallel=1)),
        ltspice_state,
    )
    assert submitted.structured_content is not None, submitted.content
    job_id = submitted.structured_content["job_id"]
    await await_until(lambda: running, what="the second sheet's run to start")
    await ltspice_state.shutdown()

    receipt = await place(ltspice_state, action="status", job_id=job_id)
    assert receipt["status"] == "interrupted", receipt
    (note,) = _observed(receipt, "results_not_placed")
    assert note["evidence"] == {
        "sheet": str(first),
        "case_id": ltspice_state.all_jobs[job_id].cases[0].case_id,
        "reason_code": "server_restarted",
    }
    assert "The process running the job stopped" in note["detail"]
    assert f"jobs(action='place_beside_sheet', job_id='{job_id}'" in note["detail"]
    assert _observed(receipt, "results_not_beside_sheet") == []
    assert not first.with_suffix(".raw").exists()
    assert not second.with_suffix(".raw").exists()


@pytest.mark.parametrize(
    ("circuits", "execution", "code"),
    [
        ("sweep", {}, "several_cases_per_sheet"),
        ("twice", {}, "several_cases_per_sheet"),
        ("sheet", {"simulator": "ngspice"}, "not_ltspice"),
    ],
)
async def test_a_request_no_sheet_could_hold_is_refused_before_it_runs(
    ltspice_state: SessionState,
    work_dir: Path,
    circuits: str,
    execution: dict[str, Any],
    code: str,
):
    sheet = copied_sheet(work_dir)
    netlist = work_dir / "rc.cir"
    netlist.write_text(_RC_DECK, encoding="utf-8")
    paths = {"netlist": (netlist,), "twice": (sheet, sheet)}.get(circuits, (sheet,))
    payload = _run(*paths, **execution)
    if circuits == "sweep":
        payload["variations"] = [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}]
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(payload), ltspice_state
    )
    data = result.structured_content
    assert data is not None and result.is_error
    assert data["error"]["code"] == code
    assert data["error"]["commit_state"] == "not_started"
    if code == "several_cases_per_sheet":
        assert "jobs(action='place_beside_sheet'" in data["error"]["message"]
    assert not sheet.with_suffix(".raw").exists()
    assert ltspice_state.all_jobs == {}


async def test_a_recoverable_job_cannot_ask_for_it(work_dir: Path):
    """A resumed job's runs would finish without it, so the request says so."""
    payload = _run(work_dir / "amp.asc", recoverable=True)
    with pytest.raises(ValidationError, match="cannot be combined with recoverable"):
        RunExperimentsInput.model_validate(payload)


async def test_leaving_it_off_keeps_a_requests_identity():
    """A request made before the option existed replays as the same request."""
    plain = {"request_id": "same", "circuits": [{"path": "amp.asc"}]}
    off = {**plain, "execution": {"results_beside_sheet": False}}
    on = {**plain, "execution": {"results_beside_sheet": True}}
    stripped = [
        RunExperimentsInput.model_validate(p).strip_presentation() for p in (plain, off, on)
    ]
    assert stripped[0] == stripped[1] != stripped[2]


# ---------------------------------------------------------------------------
# A netlist's results beside it, where LTspice's batch run writes them
# ---------------------------------------------------------------------------


def _netlist(directory: Path, name: str = "rc.cir") -> Path:
    netlist = directory / name
    netlist.write_text(_RC_DECK, encoding="utf-8")
    return netlist


async def _refused_request(state: SessionState, *paths: Path) -> dict[str, Any]:
    result = await handle_run_experiments(RunExperimentsInput.model_validate(_run(*paths)), state)
    data = result.structured_content
    assert data is not None and result.is_error, data
    assert data["error"]["commit_state"] == "not_started"
    assert state.all_jobs == {}
    return data["error"]


async def test_a_netlists_run_leaves_its_results_beside_it(
    ltspice_state: SessionState, work_dir: Path
):
    """The batch runs recorded under raw/ leave each netlist's results and log
    beside it under its name, on both builds; the receipt says where they went
    and that what LTspice's window does with them was not checked."""
    netlist = _netlist(work_dir)
    receipt = await terminal_experiment(ltspice_state, _run(netlist))
    assert receipt["status"] == "completed", receipt
    recorded = FIXTURES_DIR / "ltspice_tran_rc"
    assert (work_dir / "rc.raw").read_bytes() == recorded.with_suffix(".raw").read_bytes()
    assert (work_dir / "rc.log").read_bytes() == recorded.with_suffix(".log").read_bytes()
    (placed,) = _observed(receipt, "results_placed_beside_sheet")
    assert placed["evidence"]["sheet"] == str(netlist)
    assert [Path(f["to"]).name for f in placed["evidence"]["files"]] == ["rc.raw", "rc.log"]
    (included, unchecked) = placed["evidence"]["not_checked"]
    assert "rc.cir includes or uses" in included
    assert "What LTspice's window does with results put beside a netlist" in unchecked
    assert "where LTspice writes a netlist's own results and log" in placed["detail"]
    assert "click" not in placed["detail"]
    # The standing note is a sheet's alone.
    assert _observed(receipt, "results_not_beside_sheet") == []


async def test_the_jobs_action_places_a_netlists_results(
    state_no_sim: SessionState, work_dir: Path
):
    netlist = _netlist(work_dir)
    raw = job_that_ran(state_no_sim, netlist, work_dir)
    placed = await place(state_no_sim)
    assert placed["outcome"] == "complete", placed
    assert placed["sheet"] == str(netlist)
    assert [Path(f["to"]).name for f in placed["files"]] == ["rc.raw", "rc.log"]
    assert (work_dir / "rc.raw").read_bytes() == raw.read_bytes()
    (included, unchecked) = placed["not_checked"]
    assert "rc.cir includes or uses" in included
    assert "What LTspice's window does with results put beside a netlist" in unchecked
    assert "rc.cir" in placed["hint"] and "click" not in placed["hint"]
    assert "LTspice writes a netlist's own results and log" in placed["hint"]


@pytest.mark.parametrize(
    ("change", "code"), [("changed", "sheet_changed"), ("gone", "sheet_missing")]
)
async def test_a_netlist_that_is_not_the_one_that_ran_is_refused(
    state_no_sim: SessionState, work_dir: Path, change: str, code: str
):
    """By the digest taken when the job read the netlist, its own deck."""
    netlist = _netlist(work_dir)
    job_that_ran(state_no_sim, netlist, work_dir)
    if change == "changed":
        netlist.write_text(_RC_DECK.replace("1k", "2k"), encoding="utf-8")
    else:
        netlist.unlink()
    refused = await place(state_no_sim)
    assert refused["error"]["code"] == code
    assert "sheet" not in refused["error"]["message"]
    assert not (work_dir / "rc.raw").exists()


async def test_a_circuit_not_compared_with_the_one_that_ran_says_so(
    state_no_sim: SessionState, work_dir: Path
):
    """A job that holds no digest for the circuit cannot say it is the file that
    ran; the reply says that, for a sheet as for a netlist."""
    sheet = copied_sheet(work_dir)
    job_that_ran(state_no_sim, sheet, work_dir)
    state_no_sim.all_jobs["placed"].sources = []
    placed = await place(state_no_sim)
    assert placed["outcome"] == "complete", placed
    (unchecked,) = placed["not_checked"]
    assert unchecked.startswith("amp.asc was not compared with the file job placed ran")


async def test_a_sheet_and_a_netlist_of_other_names_are_both_placed(
    ltspice_state: SessionState, work_dir: Path
):
    sheet, netlist = copied_sheet(work_dir), _netlist(work_dir)
    receipt = await terminal_experiment(ltspice_state, _run(sheet, netlist))
    assert receipt["status"] == "completed", receipt
    placed = _observed(receipt, "results_placed_beside_sheet")
    assert sorted(o["evidence"]["sheet"] for o in placed) == sorted([str(sheet), str(netlist)])
    for circuit in (sheet, netlist):
        assert circuit.with_suffix(".raw").is_file() and circuit.with_suffix(".log").is_file()
    (from_sheet,) = [o for o in placed if o["evidence"]["sheet"] == str(sheet)]
    (included,) = from_sheet["evidence"]["not_checked"]
    assert included.startswith("The files amp.asc includes or uses")


async def test_two_circuits_of_one_results_file_are_refused_naming_both(
    ltspice_state: SessionState, work_dir: Path
):
    """foo.asc and foo.net both put their results in foo.raw, which holds one run."""
    sheet = copied_sheet(work_dir, "foo")
    netlist = _netlist(work_dir, "foo.net")
    error = await _refused_request(ltspice_state, sheet, netlist)
    assert error["code"] == "several_cases_per_sheet"
    assert "foo.asc and foo.net both put their results in foo.raw" in error["message"]
    assert "jobs(action='place_beside_sheet'" in error["message"]
    assert not (work_dir / "foo.raw").exists()


async def test_one_circuit_named_by_two_spellings_is_refused(
    ltspice_state: SessionState, work_dir: Path
):
    """Two spellings that differ in case are one file where the filesystem
    folds case (Windows, macOS), so their results file is one: counted by the
    results file, with case folded, they are refused everywhere."""
    sheet = copied_sheet(work_dir, "amp")
    other = work_dir / "AMP.asc"
    if not other.exists():  # a filesystem that keeps case: make the second file
        shutil.copyfile(sheet, other)
    error = await _refused_request(ltspice_state, sheet, other)
    assert error["code"] == "several_cases_per_sheet"
    # Windows resolves both spellings to the name on disk, which reads as one
    # circuit run twice; a filesystem that keeps case names both.
    assert any(
        said in error["message"]
        for said in ("amp.asc and AMP.asc", "amp.asc runs 2 cases in this job")
    ), error["message"]
    assert "amp.raw" in error["message"]


async def test_a_netlist_whose_results_file_is_a_sheets_is_refused(
    ltspice_state: SessionState, state_no_sim: SessionState, work_dir: Path
):
    """foo.raw beside foo.net is what LTspice's window shows as foo.asc's run,
    so a netlist with a sheet of its name beside it is refused, whether or not
    the sheet is in the job: at the request and by the jobs action."""
    copied_sheet(work_dir, "foo")
    netlist = _netlist(work_dir, "foo.net")
    error = await _refused_request(ltspice_state, netlist)
    assert error["code"] == "results_belong_to_sheet"
    assert "the results file of the sheet foo.asc" in error["message"]
    assert "Run the sheet, or copy foo.net under another name" in error["message"]

    job_that_ran(state_no_sim, netlist, work_dir)
    refused = await place(state_no_sim)
    assert refused["error"]["code"] == "results_belong_to_sheet"
    assert not (work_dir / "foo.raw").exists()


async def test_a_sheet_that_appears_during_a_netlists_run_keeps_its_results_file(
    ltspice_state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The placement as the job finishes asks the same question again."""
    netlist = _netlist(work_dir, "foo.net")
    run = ExperimentRunner.submit_netlist

    def sheet_then_run(self: ExperimentRunner, *args: Any, **kwargs: Any) -> Any:
        copied_sheet(work_dir, "foo")
        return run(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", sheet_then_run)
    receipt = await terminal_experiment(ltspice_state, _run(netlist))
    assert receipt["status"] == "completed", receipt
    (note,) = _observed(receipt, "results_not_placed")
    assert note["evidence"]["reason_code"] == "results_belong_to_sheet"
    assert not (work_dir / "foo.raw").exists()


async def test_two_placements_at_one_results_file_do_not_interleave(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """foo.net and foo.cir hold different circuit locks and go to one foo.raw.
    The second placement waits on the results file's lock until the first's
    copy is done."""
    first, second = _netlist(work_dir, "foo.net"), _netlist(work_dir, "foo.cir")
    job_that_ran(state_no_sim, first, work_dir, job_id="first")
    job_that_ran(state_no_sim, second, work_dir, job_id="second")
    target = await asyncio.to_thread(Store.results_file_lock, work_dir / "foo.raw")
    events: list[str] = []
    first_copying, release = threading.Event(), threading.Event()
    copy, attempt = shutil.copyfileobj, filelock_mod._try_acquire

    def watched_copy(read: Any, write: Any, *args: Any) -> None:
        owner = "first" if "first" in read.name else "second"
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
    placing = asyncio.create_task(place(state_no_sim, job_id="first"))
    await await_until(first_copying.is_set, what="the first copy to start")
    racing = asyncio.create_task(place(state_no_sim, job_id="second"))
    await await_until(
        lambda: {"second waits", "second copies"} & set(events),
        what="the second placement to wait or copy",
    )
    release.set()
    results = await asyncio.gather(placing, racing)
    assert [r["outcome"] for r in results] == ["complete", "complete"], results
    assert events.index("second waits") < events.index("first copied")
    assert events.index("first copied") < events.index("second copies")


async def test_a_dotted_netlists_reply_says_where_its_results_go_is_unchecked(
    state_no_sim: SessionState, work_dir: Path
):
    netlist = _netlist(work_dir, "two.dots.cir")
    job_that_ran(state_no_sim, netlist, work_dir)
    placed = await place(state_no_sim)
    assert [Path(f["to"]).name for f in placed["files"]] == ["two.dots.raw", "two.dots.log"]
    assert any("more than one dot" in said for said in placed["not_checked"])
