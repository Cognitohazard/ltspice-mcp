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
from tests._source_scan import modules, scopes
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
        await await_until(lambda: second.status == "running", what="the second case to be running")
        yield _Running(state, job, first, second, held)
    finally:
        # Let the held case end, so the job's coordinator returns.
        await await_until(
            lambda: (
                any(second.run_token in item[2] for item in held.launches)
                or second.status != "running"
            ),
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
    names = {row["name"] for row in answer["data"]["signals"]}
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
    assert "written" in error["message"], error["message"]
    assert "wait" in error["message"]


async def test_plot_waveform_draws_the_finished_case(one_case_running: _Running):
    run = one_case_running
    data = await dispatched(
        run.state,
        "plot_waveform",
        job_id=run.job.job_id,
        case_id=run.done.case_id,
        signals=["V(out)"],
        open=False,
    )
    assert "error" not in data, data
    assert await asyncio.to_thread(Path(data["path"]).is_file)


async def test_plot_waveform_refuses_the_running_case(one_case_running: _Running):
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


async def test_a_chart_of_the_job_draws_the_finished_case_and_names_the_running_one(
    one_case_running: _Running,
):
    """A chart of several cases asks each case the one question: the case
    that produced is drawn, and the one still running is named with the code
    and the facts a read of it alone is refused with, as analyze_results
    names it under coverage.missing_cases."""
    run = one_case_running
    data = await dispatched(
        run.state,
        "plot_waveform",
        job_id=run.job.job_id,
        runs="all",
        signals=["V(out)"],
        open=False,
    )
    assert "error" not in data, data
    assert [case["case_id"] for case in data["cases"]] == [run.done.case_id]
    assert {trace["case_id"] for trace in data["traces"]} == {run.done.case_id}
    (waiting,) = data["cases_not_drawn"]
    assert waiting["case_id"] == run.running.case_id
    assert waiting["code"] == "run_not_finished"
    assert waiting["status"] == "running"
    assert "written" in waiting["detail"]

    analyzed = await dispatched(
        run.state,
        "analyze_results",
        sources=[{"job_id": run.job.job_id, "runs": "all"}],
        recipes=[{"key": "v", "metric": "signal_stats", "signal": "V(out)"}],
    )
    (missing,) = analyzed["coverage"]["missing_cases"]["items"]
    assert (missing["case_id"], missing["code"]) == (waiting["case_id"], waiting["code"])


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
    assert "V(out)" in raw.signals
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


#: The functions through which a tool or an API method reaches a job's case to
#: read it. A scope that calls one of them reads a case.
_CASE_READS = frozenset(
    {
        "experiment_run_context",
        "resolve_experiment_run",
        "readable_case",
        "experiment_case",
        "case_run_context",
        "case_refusal",
        "case_unreadable",
        "source_for_run",
    }
)
#: What answers a job's question, not a case's: whether its runs are done.
_JOB_READINESS = frozenset(
    {"runs_done", "runs_terminal", "TERMINAL_STATUSES", "NON_TERMINAL_LIVE_STATUSES"}
)
#: The one place that decides whether a case can be read, and the wording of
#: its refusal.
_OWNERS = frozenset({"case_unreadable", "case_refusal", "_not_finished"})
#: Functions that read a case's status to show it, never to decide on it, each
#: with what it shows. ``test_a_status_shown_is_never_decided_on`` holds every
#: other scope of their module to reading no status, from a case or from them.
_STATUS_SHOWN = {
    "tools/analysis.py": {
        "_case_named": "the status shown for a case a chart of several cases does not draw",
    },
}
#: Functions that ask the job's question on purpose, called from a scope that
#: also reads a case, each with why its answer is not a read's.
_JOB_QUESTIONS = {
    # A placement waits for the job's own copy of its cases beside their
    # circuits, which a job run with execution.results_beside_sheet makes
    # once its runs end; the read made in the same call does not.
    "check_case": "placement",
}


def _called(nodes: list) -> dict[str, int]:
    import ast

    names: dict[str, int] = {}
    for node in nodes:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.setdefault(func.id, node.lineno)
            elif isinstance(func, ast.Attribute):
                names.setdefault(func.attr, node.lineno)
    return names


def _reads_job_readiness(nodes: list, *, status: bool) -> list[str]:
    import ast

    found: list[str] = []
    for node in nodes:
        if status and isinstance(node, ast.Attribute) and node.attr == "status":
            found.append(f"line {node.lineno}: .status")
        elif isinstance(node, ast.Name) and node.id in _JOB_READINESS:
            found.append(f"line {node.lineno}: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _JOB_READINESS:
            found.append(f"line {node.lineno}: .{node.attr}")
    return found


def test_no_reader_of_a_case_decides_by_a_status_outside_the_one_answer():
    """Whether a case can be read is decided only in ``services.case_unreadable``.

    No tool, no API method and no function of ``services`` on the way to a
    case's results reads a status, a job's or a case's, or asks whether a
    job's runs are done, in a scope that reads a case; nor does it call a
    function of the package that asks whether a job's runs are done (the
    job-wide gate this replaced sat behind such a helper). A second answer
    there would bring back the job-wide gate, or a rule of its own.
    """
    trees = modules()
    # Every function of the package that asks the job's question itself.
    asks_job: set[str] = set()
    for tree in trees.values():
        for name, nodes in scopes(tree):
            short = name.rsplit(".", 1)[-1]
            if short not in _OWNERS and _reads_job_readiness(nodes, status=False):
                asks_job.add(short)
    offenders: list[str] = []
    for module, tree in trees.items():
        if not (module.startswith(("tools/", "api/")) or module == "lib/services.py"):
            continue
        for name, nodes in scopes(tree):
            if name.rsplit(".", 1)[-1] in _OWNERS:
                continue
            called = _called(nodes)
            reads_case = bool(called.keys() & _CASE_READS) or (
                module == "lib/services.py" and name in _CASE_READS
            )
            if not reads_case:
                continue
            offenders.extend(
                f"{module}:{name} {where}" for where in _reads_job_readiness(nodes, status=True)
            )
            offenders.extend(
                f"{module}:{name} line {line}: calls {helper}, which asks the job's question"
                for helper, line in called.items()
                if helper in asks_job and helper not in _JOB_QUESTIONS
            )
    assert offenders == []


def test_a_status_shown_is_never_decided_on():
    """A function listed in ``_STATUS_SHOWN`` reads a case's status to put it
    in a reply. No other scope of its module reads a ``.status``, nor a
    ``"status"`` key, as ``_case_named(case)["status"]`` or a row's
    ``item.get("status")`` would: whether a case is read is
    ``services.case_unreadable``'s answer alone."""
    import ast

    trees = modules()
    offenders: list[str] = []
    for module, owners in _STATUS_SHOWN.items():
        for name, nodes in scopes(trees[module]):
            if name.rsplit(".", 1)[-1] in owners:
                continue
            for node in nodes:
                if isinstance(node, ast.Attribute) and node.attr == "status":
                    offenders.append(f"{module}:{name} line {node.lineno}: .status")
                elif (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "status"
                ):
                    offenders.append(f"{module}:{name} line {node.lineno}: ['status']")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "status"
                ):
                    offenders.append(f"{module}:{name} line {node.lineno}: .get('status')")
    assert offenders == []

