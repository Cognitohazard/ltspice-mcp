"""A finished case of a job is read while another case of the same job runs.

Whether a case can be read, and if not why, is the case's own answer
(``services.case_unreadable``): a case is read once its run has ended and its
end-of-run checks have run, whatever its siblings are doing. Every route that
reads a case by its job asks it: ``inspect(kind="results")``,
``plot_waveform``, ``analyze_results`` and the Python API's raw and log reads.
A case still running is refused with its own reason, naming what its
simulator has written so far, and costs only itself.

The job runs on the suite's stand-in simulator, with the launch of one case
held until the test releases it, so "one case still running" is a state the
test reaches by a handshake, not by timing.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import ExperimentCase, ExperimentJob
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import (
    LIVENESS_S,
    SyncApi,
    await_until,
    dispatched,
    recorded_fixture_simulator,
)


@dataclass
class _Held:
    """The launches the stand-in simulator holds, released one by one."""

    replay: Any
    launches: list[tuple] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def release(self, run_filename_part: str) -> None:
        with self.lock:
            (launch,) = [item for item in self.launches if run_filename_part in item[2]]
            self.launches.remove(launch)
        runner, netlist, run_filename, callback, kwargs = launch
        self.replay(runner, netlist, run_filename, callback, **kwargs)


def _hold_launches(monkeypatch: pytest.MonkeyPatch) -> _Held:
    """Every launch is held; ``release`` lets one run end with a recorded
    LTspice raw and log, as the real completion would."""
    recorded_fixture_simulator(monkeypatch)
    held = _Held(replay=ExperimentRunner.submit_netlist)

    def hold(self, netlist, run_filename, callback, **kwargs):
        with held.lock:
            held.launches.append((self, netlist, run_filename, callback, kwargs))
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", hold)
    return held


@dataclass
class _Running:
    state: SessionState
    job: ExperimentJob
    done: ExperimentCase
    running: ExperimentCase
    held: _Held


@pytest.fixture
async def one_case_running(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Running]:
    """A two-case job whose first case produced and whose second still runs."""
    state = state_with_sim
    deck = work_dir / "divider.cir"
    deck.write_text("* divider\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.tran 1m\n.end\n")
    held = _hold_launches(monkeypatch)
    args = RunExperimentsInput.model_validate(
        {
            "request_id": "one-case-running",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "variations": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
            "execution": {"wait_s": 0},
            "lint": "off",
        }
    )
    result = await asyncio.wait_for(handle_run_experiments(args, state), LIVENESS_S)
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    job = state.all_jobs[result.structured_content["job_id"]]
    first, second = job.cases
    try:
        await await_until(
            lambda: any(first.run_token in item[2] for item in held.launches),
            what="the first case to reach the simulator",
        )
        held.release(first.run_token)
        await await_until(lambda: first.status == "produced", what="the first case to produce")
        await await_until(
            lambda: second.status == "running", what="the second case to be running"
        )
        yield _Running(state, job, first, second, held)
    finally:
        # Let the held case end, so the job's coordinator returns.
        await await_until(
            lambda: any(second.run_token in item[2] for item in held.launches)
            or second.status != "running",
            what="the second case's launch",
        )
        if second.status == "running":
            held.release(second.run_token)
        await state.settled()


async def test_inspect_reads_the_finished_case(one_case_running: _Running):
    run = one_case_running
    data = await dispatched(
        run.state,
        "inspect",
        queries=[
            {
                "kind": "results",
                "view": "signals",
                "job_id": run.job.job_id,
                "case_id": run.done.case_id,
            }
        ],
    )
    (answer,) = data["results"]
    assert answer.get("error") is None, answer
    names = {row["name"] for row in answer["data"]["items"]}
    assert "V(out)" in names


async def test_inspect_refuses_the_running_case_with_its_own_reason(
    one_case_running: _Running,
):
    run = one_case_running
    data = await dispatched(
        run.state,
        "inspect",
        queries=[
            {
                "kind": "results",
                "view": "signals",
                "job_id": run.job.job_id,
                "case_id": run.running.case_id,
            }
        ],
    )
    (answer,) = data["results"]
    error = answer["error"]
    assert error["code"] == "run_not_finished", error
    assert run.running.case_id in error["message"]
    assert "written" in error["message"]
    assert "wait" in error["message"]


async def test_plot_waveform_draws_the_finished_case(one_case_running: _Running):
    from ltspice_mcp.tools import analysis

    run = one_case_running
    response = await analysis.handle_plot_waveform(
        analysis.PlotWaveformInput(
            job_id=run.job.job_id, case_id=run.done.case_id, signals=["V(out)"], open=False
        ),
        run.state,
    )
    assert not response.is_error, response.structured_content
    data = response.structured_content
    assert data is not None
    assert await asyncio.to_thread(Path(data["path"]).is_file)


async def test_plot_waveform_refuses_the_running_case(one_case_running: _Running):
    from ltspice_mcp.tools import analysis

    run = one_case_running
    data = await dispatched(
        run.state,
        "plot_waveform",
        job_id=run.job.job_id,
        case_id=run.running.case_id,
        signals=["V(out)"],
        open=False,
    )
    assert data["code"] == "run_not_finished", data
    assert run.running.case_id in data["error"]
    del analysis


async def test_analyze_results_reads_the_finished_case_and_lists_only_the_running_one(
    one_case_running: _Running,
):
    run = one_case_running
    data = await dispatched(
        run.state,
        "analyze_results",
        sources=[{"job_id": run.job.job_id, "label": "dut"}],
        recipes=[{"key": "v", "metric": "value", "expr": "V(out)", "at": "500u"}],
    )
    coverage = data["coverage"]
    assert coverage["runs_requested"] == 2
    assert coverage["runs_analyzed"] == 1
    assert coverage["runs_running"] == 1
    (missing,) = coverage["missing_cases"]["items"]
    assert missing["case_id"] == run.running.case_id
    assert missing["code"] == "run_not_finished"
    assert run.running.case_id in missing["detail"]
    assert data["outcome"] == "partial"
    (row,) = data["results"]["v"]["values"]
    assert row["case_id"] == run.done.case_id


async def test_the_python_api_reads_the_finished_case(one_case_running: _Running):
    run = one_case_running
    api = SyncApi(run.state)
    raw = await asyncio.to_thread(
        lambda: api.load_raw(job_id=run.job.job_id, case_id=run.done.case_id)
    )
    assert "V(out)" in {trace.name for trace in raw.descriptor.signals()}
    measured = await asyncio.to_thread(
        lambda: api.measurements(job_id=run.job.job_id, case_id=run.done.case_id)
    )
    assert isinstance(measured, dict)
    op = await asyncio.to_thread(
        lambda: api.device_op_points(job_id=run.job.job_id, case_id=run.done.case_id)
    )
    assert op is not None


@pytest.mark.parametrize("read", ["load_raw", "measurements", "device_op_points"])
async def test_the_python_api_refuses_the_running_case(one_case_running: _Running, read: str):
    from ltspice_mcp.errors import ResultError

    run = one_case_running
    api = SyncApi(run.state)
    with pytest.raises(ResultError) as refused:
        await asyncio.to_thread(
            lambda: getattr(api, read)(job_id=run.job.job_id, case_id=run.running.case_id)
        )
    assert refused.value.code == "run_not_finished"
    assert run.running.case_id in str(refused.value)


async def test_one_unreadable_case_does_not_hide_the_others(one_case_running: _Running):
    """A failure costs only its own case: the running case's refusal leaves
    the produced one read, in one call over both."""
    run = one_case_running
    data = await dispatched(
        run.state,
        "analyze_results",
        sources=[
            {
                "job_id": run.job.job_id,
                "label": "dut",
                "runs": {"case_ids": [run.running.case_id, run.done.case_id]},
            }
        ],
        recipes=[{"key": "v", "metric": "value", "expr": "V(out)", "at": "500u"}],
    )
    assert data["coverage"]["runs_analyzed"] == 1
    assert [row["case_id"] for row in data["coverage"]["missing_cases"]["items"]] == [
        run.running.case_id
    ]
    assert [row["case_id"] for row in data["results"]["v"]["values"]] == [run.done.case_id]
