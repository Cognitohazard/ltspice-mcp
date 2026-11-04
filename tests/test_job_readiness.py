"""One answer to "can this job's runs be read now?" from every route that asks.

The reads by job (``services.experiment_run_context``, behind ``inspect``,
``plot_waveform`` and the Python API, and ``analyze_results``' source
resolution), ``jobs(place_beside_sheet)`` and ``jobs(wait, wait_for="runs")``
each answer it. A wait that returns while a read refuses, or a copy beside the
sheet of a case a read refuses, is three answers to one question.
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
from pathlib import Path
from typing import get_args

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import experiment_store, services
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import ExperimentJob, ExperimentStatus
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from ltspice_mcp.tools.analyze import AnalyzeSourceInput
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import handle_jobs
from tests._source_scan import PACKAGE, parsed
from tests.conftest import LIVENESS_S, await_until, recorded_fixture_simulator
from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit, _experiment

_STATUSES: tuple[str, ...] = get_args(ExperimentStatus)
_LIVE_STATUSES = ("queued", "running", "analyzing")
#: The case mixes of a two-case job: the second case still running, or both done.
_MIXES = {"sibling_running": "running", "all_terminal": "failed"}


def _expected(status: str, mix: str) -> bool:
    """The documented rule, written out: a job's runs are read once they are all
    terminal, which a status from ``analyzing`` on guarantees and a record whose
    cases are all terminal shows."""
    return status not in {"queued", "running"} or mix == "all_terminal"


def _job(work_dir: Path, status: str, mix: str) -> ExperimentJob:
    """A two-case job whose first case produced results."""
    job = _experiment(
        work_dir,
        _circuit(work_dir),
        job_id=f"exp_ready_{status}_{mix}",
        count=2,
        status=status,
        case_status="produced",
    )
    sibling = job.cases[1]
    sibling.status = _MIXES[mix]  # type: ignore[assignment]
    sibling.raw_file = sibling.log_file = None
    job.completeness.recount(job.cases)
    return job


async def _answers(job: ExperimentJob, state: SessionState) -> dict[str, bool]:
    """Whether each route accepts a read of ``job``'s produced case now."""
    case_id = job.cases[0].case_id
    answers: dict[str, bool] = {}
    refusal: ResultError | None = None
    try:
        services.experiment_run_context(job, state, case_id=case_id)
    except ResultError as exc:
        refusal = exc
    assert refusal is None or "no readable runs yet" in str(refusal), refusal
    answers["experiment_run_context"] = refusal is None
    source = AnalyzeSourceInput.model_validate(
        {"job_id": job.job_id, "label": "dut", "runs": {"case_ids": [case_id]}}
    )
    _, missing, _, _ = await analyze._resolve_sources([source], state)
    answers["analyze_results"] = all(row["code"] != "job_not_terminal" for row in missing)
    placed = _assert_jobs_schema(
        await handle_jobs(_args("place_beside_sheet", job_id=job.job_id, case_id=case_id), state)
    )
    # A case it may read goes on to be refused as a netlist's run, which a
    # sheet's would not be; one it may not read is refused before that.
    assert placed["error"]["code"] in {"job_not_terminal", "not_a_sheet"}, placed["error"]
    answers["place_beside_sheet"] = placed["error"]["code"] == "not_a_sheet"
    _, timed_out = await state.job_registry.wait(job, 0, wait_for="runs")
    answers["jobs_wait_runs"] = not timed_out
    return answers


@pytest.mark.asyncio
@pytest.mark.parametrize("mix", sorted(_MIXES))
@pytest.mark.parametrize("status", _STATUSES)
async def test_another_process_s_job_reads_one_way_on_every_route(
    status: str, mix: str, state_no_sim: SessionState, work_dir: Path
):
    """The record of a job another live process runs, read by this session."""
    job = _job(work_dir, status, mix)
    # The parent of this process: alive, and not this process.
    job.owner_pid = os.getppid()
    experiment_store.save_job(job)
    loaded = await state_no_sim.job_registry.get_or_load_async(job.job_id)
    assert loaded is not None and loaded.status == status

    answers = await _answers(loaded, state_no_sim)

    expected = _expected(status, mix)
    assert answers == dict.fromkeys(answers, expected), answers


@pytest.mark.asyncio
@pytest.mark.parametrize("mix", sorted(_MIXES))
@pytest.mark.parametrize("status", _LIVE_STATUSES)
async def test_this_process_s_job_reads_one_way_on_every_route(
    status: str, mix: str, state_no_sim: SessionState, work_dir: Path
):
    """A job this process runs, at each point the coordinator can be at: its
    runs are marked done as its last case ends, before its status changes."""
    job = _job(work_dir, status, mix)
    live = state_no_sim.job_registry.reserve(job)
    state_no_sim.add_experiment_job(job, already_persisted=True)
    if mix == "all_terminal" or status == "analyzing":
        live.mark_runs_done()

    answers = await _answers(job, state_no_sim)

    expected = _expected(status, mix)
    assert answers == dict.fromkeys(answers, expected), answers


_DECK = "* divider\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
async def test_a_recoverable_job_s_runs_read_as_soon_as_the_wait_returns(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A recoverable job's coordinator checkpoints its record before each
    status change, so a waiter on its runs wakes while the status still reads
    ``running``. The read made straight after the wait must take the runs."""
    from tests.test_resume_surface import _state

    (work_dir / "divider.cir").write_text(_DECK, encoding="utf-8")
    state = _state(work_dir)
    recorded_fixture_simulator(monkeypatch)
    recorded = ExperimentRunner.submit_netlist
    held: list[tuple] = []

    def hold(self, netlist, run_filename, callback, **kwargs):
        held.append((self, netlist, run_filename, callback, kwargs))
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", hold)
    args = RunExperimentsInput.model_validate(
        {
            "request_id": "recoverable-readiness",
            "circuits": [{"path": str(work_dir / "divider.cir")}],
            "execution": {"recoverable": True, "wait_s": 0},
            "lint": "off",
        }
    )
    result = await asyncio.wait_for(handle_run_experiments(args, state), LIVENESS_S)
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    job = state.all_jobs[result.structured_content["job_id"]]
    try:
        await await_until(lambda: len(held) == 1, what="the case to reach the simulator")
        waiter = asyncio.create_task(state.job_registry.wait(job, LIVENESS_S, wait_for="runs"))
        runner, netlist, run_filename, callback, kwargs = held[0]
        recorded(runner, netlist, run_filename, callback, **kwargs)
        _, timed_out = await waiter
        seen = job.status
        source = AnalyzeSourceInput.model_validate({"job_id": job.job_id, "label": "dut"})
        runs, missing, _, _ = await analyze._resolve_sources([source], state)

        assert not timed_out
        # The window this guards: the runs are done and the status has not moved.
        assert seen == "running"
        assert missing == []
        assert [(run.source.identity or {}).get("case_id") for run in runs] == [
            job.cases[0].case_id
        ]
    finally:
        task = state.job_registry.live[job.job_id].task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), LIVENESS_S)
        await state.job_registry.drain_pending()


# ---------------------------------------------------------------------------
# Ownership of the rule
# ---------------------------------------------------------------------------

#: The predicates readiness is made of, and the modules that own the rule.
_PREDICATES = frozenset({"runs_terminal", "has_readable_raw"})
_OWNERS = frozenset({"lib/services.py", "lib/job_lifecycle.py"})


def test_readiness_is_decided_only_by_its_owners() -> None:
    """Every other module asks ``services.runs_unready``/``case_unreadable`` or
    ``job_lifecycle.runs_done``, so a second copy of the rule cannot drift."""
    found = sorted(
        f"{path.relative_to(PACKAGE).as_posix()}:{node.lineno}"
        for path in PACKAGE.rglob("*.py")
        if path.relative_to(PACKAGE).as_posix() not in _OWNERS
        for node in ast.walk(parsed(path))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in _PREDICATES)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in _PREDICATES)
        )
    )
    assert not found, f"readiness decided outside {sorted(_OWNERS)}: {found}"
