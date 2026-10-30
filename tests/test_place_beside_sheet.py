"""jobs(action="place_beside_sheet"): a case's results beside the sheet it ran.

LTspice keeps a sheet's own results beside it under its name and opens them
from there; a job keeps its runs' in the job folder. The action copies one
case's there on request and refuses what the sheet's window could not take as
its own.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from tests.conftest import FIXTURES_DIR, make_experiment_job, stage_recorded_fixture

pytestmark = pytest.mark.asyncio


def a_sheet(directory: Path, name: str = "amp") -> Path:
    sheet = directory / f"{name}.asc"
    shutil.copyfile(FIXTURES_DIR / "Draft1.asc", sheet)
    return sheet


def a_job_of(state: SessionState, sheet: Path, work_dir: Path, **job: Any) -> Path:
    """A finished job that ran ``sheet`` as it is now, its results in a folder
    of the job's own under another name. Returns the results file."""
    kept = work_dir / "runs" / "placed"
    kept.mkdir(parents=True)
    raw = stage_recorded_fixture(kept, "ltspice_tran_rc")
    made = make_experiment_job(state, job_id="placed", raw=raw, **job)
    for case in made.cases:
        case.circuit_path = sheet
    made.sources[0].path = sheet
    made.sources[0].sha256 = sha256_file(sheet)
    return raw


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
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    first = await place(state_no_sim)
    assert first["outcome"] == "complete"
    assert first["sheet"] == str(sheet) and first["case_id"] == "case-0000"
    assert [(Path(f["to"]).name, f["replaced"]) for f in first["files"]] == [
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
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
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
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir, count=3)
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
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
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
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    job = state_no_sim.all_jobs["placed"]
    receipt = await place(state_no_sim, action="status")
    (note,) = [o for o in receipt["observations"] if o["code"] == "results_not_beside_sheet"]
    assert note["evidence"] == {"sheets": ["amp.asc"]}
    assert 'jobs(action="place_beside_sheet", job_id="placed"' in note["detail"]
    job.simulator = "NGspiceSimulator"
    receipt = await place(state_no_sim, action="status")
    assert all(o["code"] != "results_not_beside_sheet" for o in receipt["observations"])
    job.simulator = "LTspice"
    job.cases[0].circuit_path = work_dir / "amp.cir"
    receipt = await place(state_no_sim, action="status")
    assert all(o["code"] != "results_not_beside_sheet" for o in receipt["observations"])
