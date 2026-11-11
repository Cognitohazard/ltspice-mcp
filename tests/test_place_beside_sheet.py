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
        ("netlist", "not_a_sheet"),
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
    elif change == "netlist":
        job.cases[0].circuit_path = work_dir / "amp.cir"
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
    assert "Run the sheet again" in note["detail"]
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
        ("netlist", {}, "not_a_sheet"),
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
