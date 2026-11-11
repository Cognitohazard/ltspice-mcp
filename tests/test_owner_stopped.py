"""A job whose process stopped before the job finished: one record, however it stopped.

The process running a job can stop three ways before the job is done. It is
killed, and the next reader of the job's record finds it gone and reconciles
the record (``experiment_store``). The server shuts down
(``JobRegistry.cancel_running``). A script closes its ``Api``, which shuts its
session down the same way. Each way leaves the job, its cases and its attached
analysis saying the same thing, which is what ``jobs(status)`` shows a reader:
``interrupted`` while runs were left, and the completed status the record's
runs earn once only the attached analysis was. ``cancelled`` is the status of a
job someone asked to cancel, and no process stopping is that.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import experiment_store, job_registry
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.job_lifecycle import LiveJob, finished, owner_stopped_observation
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from tests.conftest import (
    LIVENESS_S,
    FakeSim,
    await_until,
    exited_pid,
    fake_simulator,
    make_experiment_job,
    needs_controlled_startup,
    patch_stub_bootstrap,
    wait_until,
)

#: The code a case records when the process running its job went away
#: (``experiment_types.OWNER_STOPPED``).
_OWNER_STOPPED = "server_restarted"

_DECK = "* divider\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"


def _payload(request_id: str, **execution: Any) -> dict[str, Any]:
    """Two cases, one at a time, with an attached analysis: whichever stage the
    process stops at, one case is launched and one is not, or every run is done
    and only the analysis is left."""
    return {
        "request_id": request_id,
        "circuits": [{"path": "first.cir"}, {"path": "second.cir"}],
        "execution": {"max_parallel": 1, **execution},
        "analyze": {"recipes": [{"key": "vout", "metric": "summary"}]},
        "lint": "off",
    }


class _Stage:
    """Where each job is when its process stops, and the simulator behind it.

    ``running``: the first case's simulator never exits on its own, so it is
    running and the second case is queued; a kill confirms the exit, as a
    killed simulator reports one. ``analyzing``: every case produces at once,
    and the attached analysis never finishes, so it is running.
    """

    def __init__(self, name: str, monkeypatch: pytest.MonkeyPatch, *, kills: bool = True):
        self.name = name
        self.analyses: list[str] = []
        self._callbacks: dict[str, Callable[[RunOutcome], None]] = {}
        if name == "analyzing":
            fake_simulator(monkeypatch)

            async def endless_analysis(args: Any, _state: SessionState) -> dict[str, Any]:
                self.analyses.extend(source.job_id for source in args.sources)
                await asyncio.Future()
                raise AssertionError("an endless analysis returned")

            monkeypatch.setattr(analyze_mod, "capture_attached_analysis", endless_analysis)
            return

        callbacks = self._callbacks

        def held(runner, _netlist: Path, run_filename: str, callback, **_kwargs):
            callbacks[Path(run_filename).stem] = callback
            return object()

        async def confirmed_kill(runner, token: str) -> None:
            if kills and (callback := callbacks.pop(token, None)) is not None:
                callback(RunOutcome("", "", 0, "Stopped by test simulator"))

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", held)
        monkeypatch.setattr(ExperimentRunner, "_kill_case", confirmed_kill)

    def reached(self, job) -> bool:
        """Whether ``job``, as its session holds it, is at this stage."""
        if self.name == "analyzing":
            return job.job_id in self.analyses and job.analysis.status == "running"
        first, second = job.cases
        return first.status == "running" and second.status == "queued"


def _session(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"fake": FakeSim})


async def _start(session: SessionState, request_id: str, stage: _Stage, **execution: Any) -> str:
    """Submit through ``run_experiments`` and wait until the job is at ``stage``."""
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(_payload(request_id, wait_s=0, **execution)),
        session,
    )
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    job = session.all_jobs[result.structured_content["job_id"]]
    await await_until(lambda: stage.reached(job), what=f"the job to reach {stage.name}")
    return job.job_id


def _crashed_copy(work_dir: Path, job_id: str) -> str:
    """The job's record as its process left it on disk when killed now: a copy
    under a job id of its own, naming a process that has exited."""
    store = Store(work_dir)
    record = json.loads(store.job_record(job_id).read_text(encoding="utf-8"))
    copy_id = f"{job_id}_crashed"
    record["job_id"] = copy_id
    record["pid"] = exited_pid()
    record.pop("owner_start", None)
    store.job_record(copy_id).write_text(json.dumps(record), encoding="utf-8")
    return copy_id


def _closed_api_job(config: ServerConfig, stage: _Stage, monkeypatch: pytest.MonkeyPatch) -> str:
    """A job a script submits with ``wait=False`` and then closes its ``Api`` on."""
    session = _session(config)
    patch_stub_bootstrap(monkeypatch, session)
    api = Api()
    try:
        receipt = api.run_experiments(wait=False, **_payload("closed-api"))
        job = session.all_jobs[receipt["job_id"]]
        wait_until(lambda: stage.reached(job), what=f"the job to reach {stage.name}")
    finally:
        api.close()
    return job.job_id


async def _as_read(reader: SessionState, job_id: str) -> dict[str, Any]:
    """What ``jobs(status)`` shows a session that runs nothing of the job."""
    result = await handle_jobs(
        JobsInput.model_validate({"action": "status", "job_id": job_id}), reader
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(data)
    assert not result.is_error, data
    analysis = data.get("analysis", {})
    return {
        "status": data["status"],
        "outcome": data["outcome"],
        "completeness": data["completeness"],
        "runs": sorted(row["status"] for row in data["runs"]["items"]),
        # Code and message: what the kill read off a killed run (its exit code,
        # its log) is evidence a crash cannot leave, and not compared.
        "failures": sorted(
            (row["code"], row["message"], row.get("count", 1)) for row in data.get("failures", [])
        ),
        "analysis": (analysis.get("status"), analysis.get("error")),
        "stopped": [
            item["detail"]
            for item in data.get("observations", [])
            if item["code"] == "server_restarted"
        ],
        "analysis_stopped": [
            item["detail"]
            for item in analysis.get("observations", [])
            if item["code"] == "server_restarted"
        ],
    }


@pytest.mark.parametrize(
    ("stage_name", "status"),
    [("running", "interrupted"), ("analyzing", "completed_with_failures")],
)
async def test_a_job_whose_process_stopped_reads_alike_however_the_process_stopped(
    stage_name: str,
    status: str,
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    for name in ("first.cir", "second.cir"):
        (work_dir / name).write_text(_DECK, encoding="utf-8")
    stage = _Stage(stage_name, monkeypatch)
    reader = SessionState.create(config, available={})
    jobs: dict[str, str] = {}

    # The server shuts down; the same job's record, as a kill now would leave
    # it, is the crash.
    server = _session(config)
    jobs["shutdown"] = await _start(server, "server-shutdown", stage)
    await server.job_registry.drain_pending()
    jobs["crash"] = _crashed_copy(work_dir, jobs["shutdown"])
    await server.shutdown()

    jobs["api_close"] = await asyncio.to_thread(_closed_api_job, config, stage, monkeypatch)

    read = {route: await _as_read(reader, job_id) for route, job_id in jobs.items()}

    assert read["crash"] == read["shutdown"] == read["api_close"], read
    verdict = read["crash"]
    assert verdict["status"] == status
    assert verdict["analysis"][0] == "failed"
    assert len(verdict["stopped"]) == len(verdict["analysis_stopped"]) == 1
    if stage_name == "running":
        assert [code for code, _message, _count in verdict["failures"]] == [_OWNER_STOPPED]
        assert verdict["runs"] == ["failed", "failed"]
    else:
        assert verdict["failures"] == []
        assert verdict["runs"] == ["produced", "produced"]
    # Every one of these words is true of a closed Api as of a server.
    words = [
        *(message for _code, message, _count in verdict["failures"]),
        verdict["analysis"][1],
        *verdict["stopped"],
    ]
    assert not [text for text in words if "server" in text.lower()], words


async def test_a_shutdown_whose_stop_outlasts_its_bound_leaves_the_same_record(
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A simulator that never confirms its exit holds the coordinator past the
    shutdown's bound, and the shutdown's own bookkeeping finishes the job. That
    record reads as the coordinator's own and a crash's do."""
    for name in ("first.cir", "second.cir"):
        (work_dir / name).write_text(_DECK, encoding="utf-8")
    reader = SessionState.create(config, available={})

    stage = _Stage("running", monkeypatch)
    stopped = _session(config)
    stopped_id = await _start(stopped, "stopped-in-bound", stage)
    await stopped.shutdown()

    wedged_stage = _Stage("running", monkeypatch, kills=False)
    wedged = _session(config)
    wedged_id = await _start(wedged, "stop-outlasts-bound", wedged_stage)
    # timing: the bound under test; the kill it bounds never confirms an exit
    monkeypatch.setattr(job_registry, "_SHUTDOWN_CANCEL_TIMEOUT_S", 0.05)
    await asyncio.wait_for(wedged.shutdown(), LIVENESS_S)

    assert await _as_read(reader, wedged_id) == await _as_read(reader, stopped_id)


@needs_controlled_startup
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize("checkpoint", ["written", "refused"])
async def test_a_shutdown_whose_bound_ends_in_the_final_checkpoint_settles_the_job_once(
    checkpoint: str,
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A recoverable job's coordinator settles a stopped job and then writes its
    final record before it changes the job's status. When the shutdown's bound
    runs out during that write, the shutdown's own bookkeeping finishes the job
    as well. The job is still settled once: one lifecycle observation, the
    status the coordinator chose, and no error added once the write returns,
    whether the disk took it or refused it."""
    from tests.test_resume_surface import _state

    for name in ("first.cir", "second.cir"):
        (work_dir / name).write_text(_DECK, encoding="utf-8")
    reader = SessionState.create(config, available={})
    stage = _Stage("running", monkeypatch)
    loop = asyncio.get_running_loop()
    held = asyncio.Event()
    released = threading.Event()
    write = experiment_store.save_job
    checkpoints: list[str] = []

    def save_job(job):
        # The first record written with a terminal status is the coordinator's
        # final checkpoint: hold it until the shutdown has finished the job.
        if job.status in TERMINAL_STATUSES and not checkpoints:
            checkpoints.append(job.status)
            loop.call_soon_threadsafe(held.set)
            released.wait(LIVENESS_S)
            if checkpoint == "refused":
                raise OSError("the disk refused the final checkpoint")
        return write(job)

    monkeypatch.setattr(experiment_store, "save_job", save_job)
    server = _state(work_dir)
    job_id = await _start(server, "checkpoint-outlasts-bound", stage, recoverable=True)
    job = server.all_jobs[job_id]
    coordinator = server.job_registry.live[job_id].task
    assert job.recovery is not None and coordinator is not None

    # The bound under test runs out the moment the coordinator holds its final
    # checkpoint, not after a time: the first stage's cancels are then asked
    # to stop, as the bound's expiry asks them, and the shutdown's own
    # bookkeeping runs next. Later stages keep their real bound.
    issue = job_registry._issue_cancels
    stages: list[int] = []

    async def first_stage_ends_at_the_checkpoint(cancels: list[Any]) -> None:
        stages.append(len(cancels))
        if len(stages) > 1:
            await issue(cancels)
            return
        tasks = [asyncio.ensure_future(cancel) for cancel in cancels]
        for task in tasks:
            task.add_done_callback(job_registry._discard_outcome)
        await asyncio.wait_for(held.wait(), LIVENESS_S)
        for task in tasks:
            task.cancel()

    monkeypatch.setattr(job_registry, "_issue_cancels", first_stage_ends_at_the_checkpoint)
    shutdown = asyncio.create_task(server.shutdown())
    await asyncio.wait_for(held.wait(), LIVENESS_S)
    # The coordinator, not the shutdown, reached the write: it settled the job
    # and has not moved it yet.
    assert job.status == "running"
    await await_until(
        lambda: job.status in TERMINAL_STATUSES, what="the shutdown to finish the job"
    )
    released.set()
    await asyncio.wait_for(shutdown, LIVENESS_S)
    await asyncio.wait({coordinator}, timeout=LIVENESS_S)

    assert checkpoints == ["interrupted"]
    assert job.status == "interrupted"
    assert job.error is None, job.error
    read = await _as_read(reader, job_id)
    assert read["status"] == "interrupted"
    assert len(read["stopped"]) == len(read["analysis_stopped"]) == 1, read
    assert [code for code, _message, _count in read["failures"]] == [_OWNER_STOPPED]


def test_a_live_job_is_settled_with_the_runs_as_the_stop_first_found_them(
    state_no_sim: SessionState,
):
    """The coordinator reads the runs as the stop reaches it; the shutdown's
    bookkeeping settles the job after its bound. A case that finishes between
    the two gives neither a second reading, and settling the job a second time
    records nothing again."""
    job = make_experiment_job(state_no_sim, job_id="exp_stopping", status="running")
    live = LiveJob(job)

    assert live.runs_done_at_owner_stop() is False  # as the stop reaches the coordinator
    job.cases[0].status = "produced"  # the run ends before the bookkeeping settles the job
    assert finished(job, "runs")
    assert live.stop_for_owner() == "interrupted"
    assert live.stop_for_owner() == "interrupted"
    assert job.observations.count(owner_stopped_observation()) == 1
