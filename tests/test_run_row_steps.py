"""A run row says how its case's results are stepped.

The question: how are the results of this case stepped, how many steps and by
which parameters. It is decided once, when the case's run ends, from what the
run wrote (the raw's header and the log's ``.step`` lines, read by the raw
reader), and kept on the case's record (``ExperimentCase.steps``). Every reader
asks the record: each route that shows a run row (the ``run_experiments``
receipt, ``jobs`` status, wait and runs, and the Python API) and a chart of
several cases, whose ``cases[].n_steps`` is the same fact.

The runs are recordings of LTspice 26 and LTspice XVII
(``tests/fixtures/ltspice_recorded``), handed back by the suite's stand-in
simulator.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import experiment_store, services
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests.conftest import SyncApi, replaying_simulator

RECORDED = Path(__file__).parent / "fixtures" / "ltspice_recorded"
BUILDS = ["ltspice26", "ltspice17"]


def _recorded(build: str, name: str) -> tuple[Path, Path]:
    return RECORDED / build / "raw" / f"{name}.raw", RECORDED / build / "raw" / f"{name}.log"


async def _ran(
    state: SessionState,
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    raw: Path,
    log: Path,
    request_id: str,
    count: int = 1,
) -> dict[str, Any]:
    """A finished job of ``count`` cases, each leaving a copy of ``raw`` and
    ``log``; returns its receipt."""
    replaying_simulator(monkeypatch, raw, log)
    deck = work_dir / f"{request_id}.cir"
    deck.write_text("* rc\nV1 in 0 1\nR1 in out {r}\nC1 out 0 1n\n.param r=1k\n.tran 1m\n.end\n")
    payload: dict[str, Any] = {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 60},
        "lint": "off",
    }
    if count > 1:
        values = [f"{index + 1}k" for index in range(count)]
        payload["variations"] = [{"kind": "assign", "assign": {"r": values}}]
    result = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


async def _jobs(state: SessionState, **arguments: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(arguments), state)
    assert result.structured_content is not None, result.content
    return result.structured_content


def _rows(reply: dict[str, Any]) -> list[dict[str, Any]]:
    runs = reply.get("runs")
    if isinstance(runs, dict):
        return list(runs["items"])
    return list(reply["items"])


def _api_replies(state: SessionState, job_id: str) -> dict[str, dict[str, Any]]:
    api = SyncApi(state)
    return {
        "api.status": api.jobs(action="status", job_id=job_id),
        "api.wait": api.wait(job_id, timeout_s=30),
        "api.runs": api.jobs(action="runs", job_id=job_id),
    }


async def _every_route(
    state: SessionState, receipt: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """The run rows of the job on every route that shows them."""
    job_id = receipt["job_id"]
    replies = {
        "receipt": receipt,
        "status": await _jobs(state, action="status", job_id=job_id),
        "wait": await _jobs(state, action="wait", job_id=job_id, timeout_s=30),
        "runs": await _jobs(state, action="runs", job_id=job_id),
        **(await asyncio.to_thread(_api_replies, state, job_id)),
    }
    return {route: _rows(reply) for route, reply in replies.items()}


@pytest.mark.asyncio
class TestTheRowSaysHowItsCaseIsStepped:
    @pytest.mark.parametrize("build", BUILDS)
    async def test_a_stepped_sweep_says_how_many_steps_and_by_which_parameters(
        self, state_with_sim: SessionState, work_dir: Path, monkeypatch, build: str
    ):
        raw, log = _recorded(build, "step_tran")
        receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, f"sweep-{build}")
        for route, rows in (await _every_route(state_with_sim, receipt)).items():
            (row,) = rows
            assert row["steps"] == {"count": 3, "parameters": ["r"]}, (route, row)

    @pytest.mark.parametrize("build", BUILDS)
    async def test_a_stepped_op_says_its_steps_too(
        self, state_with_sim: SessionState, work_dir: Path, monkeypatch, build: str
    ):
        raw, log = _recorded(build, "step_op")
        receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, f"op-{build}")
        job = state_with_sim.all_jobs[receipt["job_id"]]
        (case,) = job.cases
        assert case.steps is not None and case.steps.count is not None
        assert case.steps.count > 1
        for route, rows in (await _every_route(state_with_sim, receipt)).items():
            (row,) = rows
            assert row["steps"]["count"] == case.steps.count, (route, row)
            assert row["steps"]["parameters"] == list(case.steps.parameters), (route, row)

    @pytest.mark.parametrize("build", BUILDS)
    async def test_an_unstepped_case_has_no_steps(
        self, state_with_sim: SessionState, work_dir: Path, monkeypatch, build: str
    ):
        raw, log = _recorded(build, "tran")
        receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, f"plain-{build}")
        job = state_with_sim.all_jobs[receipt["job_id"]]
        (case,) = job.cases
        assert case.steps is not None and case.steps.count == 1
        for route, rows in (await _every_route(state_with_sim, receipt)).items():
            (row,) = rows
            assert "steps" not in row, (route, row)

    async def test_a_case_whose_raw_and_log_disagree_says_so_with_the_reason(
        self, state_with_sim: SessionState, work_dir: Path, monkeypatch
    ):
        """A stepped raw beside a log with no .step line: the record says the
        number of steps was not established and why, and no row gives a number."""
        raw, _ = _recorded("ltspice26", "step_tran")
        unstepped_log = Path(__file__).parent / "fixtures" / "ltspice_tran_rc.log"
        receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, unstepped_log, "disagree")
        job = state_with_sim.all_jobs[receipt["job_id"]]
        (case,) = job.cases
        assert case.steps is not None
        assert case.steps.count is None
        assert case.steps.not_established
        for route, rows in (await _every_route(state_with_sim, receipt)).items():
            (row,) = rows
            assert set(row["steps"]) == {"not_established"}, (route, row)
            assert row["steps"]["not_established"] == case.steps.not_established


def _legacy(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` as the previous version wrote it: every case carries
    ``step_index`` and ``step_values``, and none carries ``steps``."""
    old = json.loads(json.dumps(data))
    for case in old["cases"]:
        case.pop("steps", None)
        case["step_index"] = None
        case["step_values"] = {}
    return old


@pytest.mark.asyncio
class TestARecordWrittenBefore:
    async def test_it_loads_replies_and_round_trips(
        self, state_with_sim: SessionState, work_dir: Path, monkeypatch
    ):
        raw, log = _recorded("ltspice26", "step_tran")
        receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "legacy")
        job = state_with_sim.all_jobs[receipt["job_id"]]
        data = json.loads(
            json.dumps(experiment_store.serialize_job(job), default=str, allow_nan=False)
        )
        old = _legacy(data)

        loaded = experiment_store.deserialize_job(
            old, job.store_path, liveness=experiment_store.OwnerLiveness.ALIVE
        )
        # Read back and written again, it is the record it was: a recovery
        # journal refuses a candidate that does not round-trip.
        again = json.loads(
            json.dumps(experiment_store.serialize_job(loaded), default=str, allow_nan=False)
        )
        assert again == old
        (case,) = loaded.cases
        assert case.steps is None

        # A row of it says its steps were not recorded, never that it has none.
        job.cases[0] = case
        (row,) = _rows(await _jobs(state_with_sim, action="runs", job_id=job.job_id))
        assert set(row["steps"]) == {"not_established"}
        assert "recorded" in row["steps"]["not_established"]


@pytest.mark.asyncio
async def test_a_chart_over_its_cell_limit_is_refused_before_its_files_load(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A chart of many cases over the cell limit is refused from the records
    and the request, and loads no results file to find it out."""
    import ltspice_mcp.tools.analysis as analysis
    from ltspice_mcp.errors import ResultError
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw = Path(__file__).parent / "fixtures" / "ltspice_tran_rc.raw"
    log = raw.with_suffix(".log")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "many", count=50)

    one = await handle_plot_waveform(
        PlotWaveformInput(job_id=receipt["job_id"], run_index=0, signals=["V(out)"], open=False),
        state_with_sim,
    )
    assert one.structured_content is not None
    points = one.structured_content["points_per_series"][0]
    # One case draws two rows (the axis and its series); fifty draw 51.
    monkeypatch.setattr(analysis, "_PLOT_MAX_CELLS", 10 * points)

    loads: list[Path | None] = []
    real = services.load_raw

    async def counted(source, state):
        loads.append(source.raw)
        return await real(source, state)

    monkeypatch.setattr(services, "load_raw", counted)
    with pytest.raises(ResultError, match="cells"):
        await handle_plot_waveform(
            PlotWaveformInput(
                job_id=receipt["job_id"], runs="all", signals=["V(out)"], open=False
            ),
            state_with_sim,
        )
    assert loads == []


@pytest.mark.asyncio
async def test_a_charts_n_steps_and_a_rows_steps_are_the_one_record_field(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    from ltspice_mcp.lib.experiment_types import CaseSteps
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw, log = _recorded("ltspice26", "step_tran")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "same", count=2)
    job = state_with_sim.all_jobs[receipt["job_id"]]
    # The record is the answer: what it says, both replies say.
    job.cases[1].steps = CaseSteps(count=7, parameters=("q",), values_from="log")
    chart = await handle_plot_waveform(
        PlotWaveformInput(job_id=job.job_id, runs="all", signals=["V(out)"], open=False),
        state_with_sim,
    )
    assert chart.structured_content is not None
    rows = _rows(await _jobs(state_with_sim, action="runs", job_id=job.job_id))
    for drawn, row in zip(chart.structured_content["cases"], rows, strict=True):
        assert drawn["n_steps"] == row["steps"]["count"]
    assert chart.structured_content["cases"][1]["n_steps"] == 7


def test_no_reply_builder_reads_a_step_from_a_case_record():
    """The case record holds how its results are stepped (``steps``); no
    module of the package reads a step index or step values off a case."""
    import ast

    from tests._source_scan import modules

    offenders = [
        f"{module}:{node.lineno}"
        for module, tree in modules().items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"step_index", "step_values"}
        and isinstance(node.value, ast.Name)
        and "case" in node.value.id
    ]
    assert offenders == []


#: The modules of the raw reader: the only ones that decide whether results
#: are stepped, from a raw's header and the log's step lines.
_STEPPING_DECIDERS = frozenset(
    {
        "lib/raw_header.py",
        "lib/raw_decode.py",
        "lib/decoded_raw.py",
        "lib/result_kinds.py",
        "lib/parser_protocol.py",
        "lib/raw_parser.py",
    }
)
#: Other reads of the stepped flag, each with why it is no decision on a
#: case's results.
_STEPPED_FLAG_READERS = {
    "lib/experiment_runner.py": "a running case's progress, from the raw it is writing",
}


def test_nothing_outside_the_raw_reader_decides_whether_results_are_stepped():
    import ast

    from tests._source_scan import modules

    offenders = [
        f"{module}:{node.lineno}: .{node.attr}"
        for module, tree in modules().items()
        if module not in _STEPPING_DECIDERS and module not in _STEPPED_FLAG_READERS
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"stepped", "step_status"}
    ]
    assert offenders == []


def test_the_recorded_inputs_are_recorded_on_both_builds():
    for build in BUILDS:
        for name in ("step_tran", "step_op", "tran"):
            raw, log = _recorded(build, name)
            assert raw.is_file() and log.is_file(), (build, name)


@pytest.mark.asyncio
async def test_a_plot_index_the_cases_do_not_have_is_a_fault_of_the_request(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The results files are readable; the request names a plot they do not
    hold. Each route says so with a code of its own, not result_unreadable."""
    from ltspice_mcp.errors import LTSpiceMCPError
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw = Path(__file__).parent / "fixtures" / "ltspice_tran_rc.raw"
    receipt = await _ran(
        state_with_sim, monkeypatch, work_dir, raw, raw.with_suffix(".log"), "plots", count=2
    )
    for selection in ({"run_index": 0}, {"runs": "all"}):
        with pytest.raises(LTSpiceMCPError) as refused:
            await handle_plot_waveform(
                PlotWaveformInput(
                    job_id=receipt["job_id"],
                    plot_index=5,
                    signals=["V(out)"],
                    open=False,
                    **selection,
                ),
                state_with_sim,
            )
        assert refused.value.code == "plot_not_found", selection


@pytest.mark.asyncio
async def test_loading_the_cases_of_a_chart_has_one_bound_on_its_time(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """When the bound on the time a chart spends loading ends, each case not
    loaded is named with a code saying so; none is passed off as drawn."""
    import ltspice_mcp.tools.analysis as analysis
    from ltspice_mcp.errors import ResultError
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw = Path(__file__).parent / "fixtures" / "ltspice_tran_rc.raw"
    receipt = await _ran(
        state_with_sim, monkeypatch, work_dir, raw, raw.with_suffix(".log"), "bound", count=3
    )
    monkeypatch.setattr(analysis, "_LOAD_TIME_S", 0.0)
    with pytest.raises(ResultError) as refused:
        await handle_plot_waveform(
            PlotWaveformInput(
                job_id=receipt["job_id"], runs="all", signals=["V(out)"], open=False
            ),
            state_with_sim,
        )
    assert "load_time_exhausted" in str(refused.value)
