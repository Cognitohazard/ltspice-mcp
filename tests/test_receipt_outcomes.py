"""Every receipt of one job reports one call outcome, whichever action renders it.

``run_experiments``, ``jobs(status)``, ``jobs(wait)`` and ``jobs(runs)`` render
the same job. A runs page that said ``failed`` beside a status that said
``complete`` told a caller nothing came back when the runs were there to read.
"""

from __future__ import annotations

import json
import shutil
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
from tests.conftest import FIXTURES_DIR, exited_pid, make_experiment_job
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
    job.owner_pid = exited_pid()
    experiment_store.save_job(job)

    status = _assert_jobs_schema(
        await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
    )
    runs = _assert_jobs_schema(await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim))

    assert status["status"] == runs["status"] == "interrupted"
    assert [item["status"] for item in runs["items"]] == ["produced", "produced"]
    assert status["outcome"] == runs["outcome"] == "complete"
    await state_no_sim.job_registry.drain_pending()
    record = json.loads(job.store_path.read_text(encoding="utf-8"))
    assert record["status"] == "interrupted"


def test_samples_recorded_without_a_digest_are_said_to_be_uncompared(state_no_sim: SessionState):
    """A job recorded before samples were compared carries no digest of what
    each sample computed, so its receipt says the samples were not compared
    rather than leaving out the comparison as if it had found nothing."""
    job = make_experiment_job(state_no_sim, job_id="exp_old_mc_0001", count=3)
    for index, case in enumerate(job.cases):
        case.assignments = {"_random_run": index, "_random_seed": 5}
        assert case.status == "produced" and case.result_digest is None

    observations = snapshot_receipt(job, state_no_sim).observations

    (observation,) = [o for o in observations if o["code"] == "samples_not_compared"]
    assert observation["evidence"]["count"] == 3
    assert not any(o["code"] == "identical_samples" for o in observations)


@pytest.mark.parametrize(
    ("status", "third"),
    [("running", "running"), ("completed", "failed")],
    ids=["running", "failed"],
)
def test_all_samples_is_said_only_of_samples_that_all_ran(
    state_no_sim: SessionState, status: str, third: str
):
    """Two samples that computed the same numbers are reported, but not as
    "All samples" while a third is still running, or after it failed: nothing
    says what it would have computed."""
    job = make_experiment_job(state_no_sim, job_id=f"exp_partial_{third}", count=3, status=status)
    for index, case in enumerate(job.cases):
        case.assignments = {"_random_run": index}
        case.status = "produced"
        case.result_digest = "same"
    job.cases[2].status = "running" if third == "running" else "failed"
    job.cases[2].result_digest = None

    observations = snapshot_receipt(job, state_no_sim).observations

    (observation,) = [o for o in observations if o["code"] == "identical_samples"]
    assert not observation["detail"].startswith("All samples")
    assert ("so far" in observation["detail"]) is (status == "running")
    assert observation["evidence"]["groups"] == [{"circuit": "dut", "samples": 2}]


def test_a_long_list_of_alike_points_is_cut_with_its_total(state_no_sim: SessionState):
    """Twelve assign points where both samples computed one result and one
    where they differ: the observation names ten points and counts twelve,
    since a receipt carries it on every status and wait."""
    job = make_experiment_job(state_no_sim, job_id="exp_many_points", count=26)
    for index, case in enumerate(job.cases):
        point, sample = divmod(index, 2)
        case.assignments = {"vin": point, "_random_run": sample}
        case.assigned = {"vin": point}
        case.result_digest = f"point-{point}" if point < 12 else f"differs-{sample}"

    (observation,) = [
        o
        for o in snapshot_receipt(job, state_no_sim).observations
        if o["code"] == "identical_samples"
    ]

    groups = observation["evidence"]["groups"]
    assert [group["assigned"]["vin"] for group in groups] == list(range(10))
    assert observation["evidence"]["points_truncated"] == 12
    assert observation["detail"].startswith(
        "Samples of dut computed the same results at the assign points listed for them"
    )


def test_points_that_all_match_collapse_into_one_group(state_no_sim: SessionState):
    """Every point whose samples ran computed one result, though one sample
    failed: one group counts the points, and the detail says some samples did
    not run rather than "All samples"."""
    job = make_experiment_job(state_no_sim, job_id="exp_one_failed", count=6)
    for index, case in enumerate(job.cases):
        point, sample = divmod(index, 2)
        case.assignments = {"vin": point, "_random_run": sample}
        case.assigned = {"vin": point}
        case.result_digest = f"point-{point}"
    job.cases[5].status = "failed"  # point 2 keeps one produced sample: nothing to compare
    job.cases[5].result_digest = None

    (observation,) = [
        o
        for o in snapshot_receipt(job, state_no_sim).observations
        if o["code"] == "identical_samples"
    ]

    assert observation["evidence"]["groups"] == [{"circuit": "dut", "points": 2}]
    assert observation["detail"].startswith("Every sample of dut that was compared")


def test_a_case_recorded_before_it_kept_its_assign_values_is_compared_at_its_point(
    state_no_sim: SessionState,
):
    """A case recorded before ExperimentCase.assigned has its assign values
    only among its draws. Read from there, vin = 0 and vin = 1 stay apart, so
    two points whose samples each differ are not merged into one point."""
    job = make_experiment_job(state_no_sim, job_id="exp_legacy_points", count=6)
    for index, case in enumerate(job.cases):
        point, sample = divmod(index, 3)
        case.assignments = {
            "vin": point,
            "random:component:R2": 1000 + sample,
            "_random_run": sample,
            "_random_seed": 4,
        }
        case.result_digest = f"sample-{sample}"

    observations = snapshot_receipt(job, state_no_sim).observations

    assert not any(o["code"] == "identical_samples" for o in observations)


def test_each_circuit_is_described_by_what_its_own_samples_did(state_no_sim: SessionState):
    """Circuit a computed one result at both points; circuit b at one point of
    two, and its third sample's raw could not be read. The detail says each,
    so b's matching point is not read as part of "All samples of a"."""
    job = make_experiment_job(state_no_sim, job_id="exp_two_circuits", count=12)
    for index, case in enumerate(job.cases):
        circuit, rest = divmod(index, 6)
        point, sample = divmod(rest, 3)
        case.circuit = "ab"[circuit]
        case.assignments = {"v": point, "_random_run": sample}
        case.assigned = {"v": point}
        alike = circuit == 0 or point == 0
        case.result_digest = f"{case.circuit}-{point}" if alike else f"b-1-{sample}"
    job.cases[11].result_digest = None

    observations = snapshot_receipt(job, state_no_sim).observations

    (observation,) = [o for o in observations if o["code"] == "identical_samples"]
    assert observation["detail"].startswith("All samples of a computed the same results.")
    assert (
        "Samples of b computed the same results at the assign points listed"
        in (observation["detail"])
    )
    assert observation["evidence"]["groups"] == [
        {"circuit": "a", "points": 2},
        {"circuit": "b", "assigned": {"v": 0}, "samples": 3},
    ]
    (uncompared,) = [o for o in observations if o["code"] == "samples_not_compared"]
    assert uncompared["evidence"]["count"] == 1
