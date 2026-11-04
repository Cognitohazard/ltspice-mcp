"""Every receipt of one job reports one call outcome, whichever action renders it.

``run_experiments``, ``jobs(status)``, ``jobs(wait)`` and ``jobs(runs)`` render
the same job. A runs page that said ``failed`` beside a status that said
``complete`` told a caller nothing came back when the runs were there to read.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_types import AnalysisStage, ExperimentStatus
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import handle_jobs
from ltspice_mcp.tools.receipts import (
    render_jobs_receipt_snapshot,
    render_receipt_snapshot,
    render_runs_envelope,
    snapshot_receipt,
)
from tests.conftest import FIXTURES_DIR
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment

_STATUSES: tuple[str, ...] = get_args(ExperimentStatus)
_MIXES = {
    "all_produced": ("produced", "produced"),
    "one_failed": ("produced", "failed"),
    "all_failed": ("failed", "failed"),
    "one_cancelled": ("produced", "cancelled"),
    "one_running": ("produced", "running"),
}


@pytest.mark.parametrize("analysis", ["not_requested", "failed"])
@pytest.mark.parametrize("mix", sorted(_MIXES))
@pytest.mark.parametrize("status", _STATUSES)
def test_every_renderer_reports_one_outcome(
    status: str, mix: str, analysis: str, state_no_sim: SessionState, work_dir: Path
):
    job = _experiment(
        work_dir,
        _circuit(work_dir),
        job_id=f"exp_outcome_{status}_{mix}",
        count=2,
        status=status,
        case_status="produced",
    )
    for case, case_status in zip(job.cases, _MIXES[mix], strict=True):
        case.status = case_status  # type: ignore[assignment]
        if case_status != "produced":
            case.raw_file = case.log_file = None
            case.failure_code = case_status
    job.completeness.recount(job.cases)
    job.analysis = AnalysisStage(status=analysis)  # type: ignore[arg-type]
    snapshot = snapshot_receipt(job, state_no_sim)

    outcomes = {
        "run_experiments": render_receipt_snapshot(snapshot)["outcome"],
        "jobs_status": render_jobs_receipt_snapshot("status", snapshot)["outcome"],
        "jobs_wait": render_jobs_receipt_snapshot("wait", snapshot, timed_out=False)["outcome"],
        "jobs_runs": render_runs_envelope(snapshot)["outcome"],
    }

    assert len(set(outcomes.values())) == 1, outcomes


def _exited_pid() -> int:
    """The pid of a process that has exited."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


@pytest.mark.asyncio
async def test_an_interrupted_job_whose_runs_were_written_reads_complete(
    state_no_sim: SessionState, work_dir: Path
):
    """A server stopped after both runs wrote their results, before it recorded
    the second: the next reader finds the results on disk, promotes the case,
    and calls the job interrupted. Its runs are all there, on every page."""
    job = _experiment(
        work_dir,
        _circuit(work_dir),
        job_id="exp_outcome_restart",
        count=2,
        status="running",
        case_status="produced",
    )
    output = work_dir / "runs"
    output.mkdir()
    job.output_folder = output
    unrecorded = job.cases[1]
    unrecorded.status = "running"
    unrecorded.raw_file = unrecorded.log_file = None
    unrecorded.run_token = f"{job.job_id}_case_1"
    shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.raw", output / f"{unrecorded.run_token}.raw")
    shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.log", output / f"{unrecorded.run_token}.log")
    job.completeness.recount(job.cases)
    job.owner_pid = _exited_pid()
    experiment_store.save_job(job)

    status = _assert_jobs_schema(await handle_jobs(_args("status", job_id=job.job_id), state_no_sim))
    runs = _assert_jobs_schema(await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim))

    assert status["status"] == runs["status"] == "interrupted"
    assert [item["status"] for item in runs["items"]] == ["produced", "produced"]
    assert status["outcome"] == runs["outcome"] == "complete"
    await state_no_sim.job_registry.drain_pending()
    record = json.loads(job.store_path.read_text(encoding="utf-8"))
    assert record["status"] == "interrupted"
