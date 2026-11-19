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
import shutil
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
    # A Python API result answers an absent field with an empty value, so the
    # key itself is asked.
    if "runs" in reply.keys():  # noqa: SIM118 - the result's own keys, not its default
        return list(reply["runs"]["items"])
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
            assert len(rows) == 1, (route, rows)
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
            assert len(rows) == 1, (route, rows)
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
            assert len(rows) == 1, (route, rows)
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
            assert len(rows) == 1, (route, rows)
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
        assert row["steps"] == {"not_recorded": True}


@pytest.mark.asyncio
async def test_a_chart_over_its_cell_limit_is_refused_without_loading_every_file(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A chart of many cases over the cell limit is refused as the loaded
    check refuses it, and the files past the point the cases loaded prove it
    are not loaded."""
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
    # The headers say the chart would be over; it is refused once the cases
    # loaded are over by themselves (ten draw 11 rows), before the rest load.
    assert 10 <= len(loads) <= 10 + analysis._LOADS_AT_ONCE


@pytest.mark.asyncio
async def test_a_charts_n_steps_is_the_file_it_drew_and_a_record_that_differs_is_said(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The record decides the run row; once a case's file is loaded, the plot
    loaded decides its n_steps in the chart. Where the record and the loaded
    first plot disagree (the file is not the one the run wrote, or the record
    was wrong), the reply names the case with a code; a case with no record
    takes its n_steps from the file."""
    from ltspice_mcp.lib.experiment_types import CaseSteps
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw, log = _recorded("ltspice26", "step_tran")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "same", count=3)
    job = state_with_sim.all_jobs[receipt["job_id"]]
    job.cases[1].steps = CaseSteps(count=7, parameters=("q",), values_from="log")
    job.cases[2].steps = None  # recorded before steps were kept
    chart = await handle_plot_waveform(
        PlotWaveformInput(job_id=job.job_id, runs="all", signals=["V(out)"], open=False),
        state_with_sim,
    )
    data = chart.structured_content
    assert data is not None
    assert [case["n_steps"] for case in data["cases"]] == [3, 3, 3]
    assert all("steps" not in case for case in data["cases"])
    (differs,) = [
        item for item in data.get("observations", []) if item["code"] == "steps_not_as_recorded"
    ]
    assert job.cases[1].case_id in differs["detail"]
    assert differs["evidence"] == {
        "cases": [{"case_id": job.cases[1].case_id, "recorded": 7, "loaded": 3}]
    }
    assert "plot_waveform" not in json.dumps(data["cases"])


@pytest.mark.asyncio
async def test_a_raw_that_cannot_be_read_when_its_run_ends_records_no_steps(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed read says nothing of how the results are stepped: the record
    keeps none and the row says not_recorded, never a reason built from the
    parser's error."""
    raw, log = _recorded("ltspice17", "tran_killed")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "killed")
    job = state_with_sim.all_jobs[receipt["job_id"]]
    (case,) = job.cases
    assert case.raw_file is not None
    assert case.steps is None
    for route, rows in (await _every_route(state_with_sim, receipt)).items():
        (row,) = rows
        assert row["steps"] == {"not_recorded": True}, (route, row)


@pytest.mark.asyncio
async def test_a_cancel_ends_the_steps_read_at_once(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The steps read at the end of a run is held in the real parse service;
    a cancel of the job ends it at once, and the case keeps its results with
    no steps recorded."""
    import threading

    from ltspice_mcp.lib import parser_service
    from tests.conftest import LIVENESS_S, await_until

    raw, log = _recorded("ltspice26", "step_tran")
    replaying_simulator(monkeypatch, raw, log)
    entered = threading.Event()
    cancelled_parse = threading.Event()
    uncancellable: list[bool] = []
    run = parser_service.run_parser_sync

    def held(request, **kwargs):
        if '"chosen": {' in json.dumps(request):
            cancel = kwargs.get("cancel")
            if cancel is None:
                # A read no stop can reach: it is let through, and said.
                uncancellable.append(True)
                return run(request, **kwargs)
            entered.set()
            # Held until the call is cancelled, then run as the real parse is.
            if cancel.wait(LIVENESS_S):
                cancelled_parse.set()
        return run(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", held)
    deck = work_dir / "held.cir"
    deck.write_text("* rc\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1n\n.tran 1m\n.end\n")
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "held-steps",
                "circuits": [{"path": str(deck), "id": "dut"}],
                "execution": {"wait_s": 0},
                "lint": "off",
            }
        ),
        state_with_sim,
    )
    receipt = result.structured_content
    assert receipt is not None
    await await_until(entered.is_set, what="the steps read to start")
    cancelled = await asyncio.wait_for(
        _jobs(
            state_with_sim,
            action="cancel",
            job_id=receipt["job_id"],
            control_token=receipt["control_token"],
        ),
        LIVENESS_S,
    )
    assert cancelled.get("outcome") != "error", cancelled
    job = state_with_sim.all_jobs[receipt["job_id"]]
    await _jobs(state_with_sim, action="wait", job_id=job.job_id, timeout_s=LIVENESS_S)
    assert uncancellable == []
    assert cancelled_parse.is_set()
    (case,) = job.cases
    assert case.raw_file is not None
    assert case.steps is None


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
    selections: list[dict[str, Any]] = [{"run_index": 0}, {"runs": "all"}]
    for selection in selections:
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


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize(
    "analysis",
    [".tran 1u 100u", ".dc V1 0 1 0.5 V2 0 1 0.5"],
    ids=["tran", "nested_dc"],
)
async def test_an_ngspice_run_records_its_one_step(work_dir: Path, analysis: str):
    """ngspice has no .step: its raw is not marked stepped, and a nested .dc
    is one sweep of one plot. Its case records one step, and its row carries
    no steps. Measured with the ngspice on PATH."""
    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.simulator import detect_simulators

    config = ServerConfig(simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir])
    available = detect_simulators(config)
    assert "ngspice" in available
    state = SessionState.create(config, available)
    deck = work_dir / "ng.cir"
    deck.write_text(
        f"* divider\nV1 in 0 1\nV2 mid 0 0\nR1 in out 1k\nR2 out mid 1k\n{analysis}\n.end\n"
    )
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": f"ng-{analysis.split()[0][1:]}",
                "circuits": [{"path": str(deck), "id": "dut"}],
                "execution": {"wait_s": 60},
                "lint": "off",
            }
        ),
        state,
    )
    assert result.structured_content is not None
    job = state.all_jobs[result.structured_content["job_id"]]
    (case,) = job.cases
    assert case.status == "produced", case
    assert case.steps is not None and case.steps.count == 1
    (row,) = _rows(await _jobs(state, action="runs", job_id=job.job_id))
    assert "steps" not in row


@pytest.mark.asyncio
async def test_a_chart_the_loaded_cases_fit_is_drawn_whatever_the_headers_say(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The headers count every case; a case whose data cannot be read is not
    drawn, and the chart of the others fits the limit, so it is drawn and the
    case is named: the count made before loading refuses nothing that would
    be drawn."""
    import ltspice_mcp.tools.analysis as analysis
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw = Path(__file__).parent / "fixtures" / "ltspice_tran_rc.raw"
    receipt = await _ran(
        state_with_sim, monkeypatch, work_dir, raw, raw.with_suffix(".log"), "fits", count=5
    )
    job = state_with_sim.all_jobs[receipt["job_id"]]
    one = await handle_plot_waveform(
        PlotWaveformInput(job_id=job.job_id, run_index=0, signals=["V(out)"], open=False),
        state_with_sim,
    )
    assert one.structured_content is not None
    points = one.structured_content["points_per_series"][0]
    broken = job.cases[4].raw_file
    assert broken is not None
    whole = broken.read_bytes()
    broken.write_bytes(whole[: whole.index(b"B\x00i\x00n\x00a\x00r\x00y\x00:\x00") + 40])
    # Five cases draw six rows, four draw five.
    monkeypatch.setattr(analysis, "_PLOT_MAX_CELLS", 5 * points + 1)

    drawn = await handle_plot_waveform(
        PlotWaveformInput(job_id=job.job_id, runs="all", signals=["V(out)"], open=False),
        state_with_sim,
    )
    data = drawn.structured_content
    assert data is not None
    assert data["series_count"] == 4
    (unread,) = data["cases_not_drawn"]
    assert (unread["case_id"], unread["code"]) == (job.cases[4].case_id, "result_unreadable")


def _steps_differ(reply: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in reply.get("observations", []) if item["code"] == "steps_not_as_recorded"
    ]


@pytest.mark.asyncio
async def test_every_chart_of_a_jobs_case_says_a_file_that_differs_from_its_record(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """One rule for a chart of one case and of several: the file loaded
    decides n_steps, and a record that gives another number is said with the
    same code. The input: a 3-step job whose second case's results are
    rewritten after the run with a file of 1 step. A chart by path names no
    case, so it has no record to compare."""
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw, log = _recorded("ltspice26", "step_tran")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "rewritten", count=3)
    job = state_with_sim.all_jobs[receipt["job_id"]]
    case = job.cases[1]
    assert case.steps is not None and case.steps.count == 3
    assert case.raw_file is not None and case.log_file is not None
    one_raw, one_log = _recorded("ltspice26", "tran")
    shutil.copy(one_raw, case.raw_file)
    shutil.copy(one_log, case.log_file)
    expected = {"cases": [{"case_id": case.case_id, "recorded": 3, "loaded": 1}]}

    async def chart(**arguments: Any) -> dict[str, Any]:
        result = await handle_plot_waveform(
            PlotWaveformInput(signals=["V(out)"], open=False, **arguments), state_with_sim
        )
        assert result.structured_content is not None, result.content
        return result.structured_content

    for one in (
        {"job_id": job.job_id, "run_index": case.run_index},
        {"job_id": job.job_id, "case_id": case.case_id},
        {"job_id": job.job_id, "runs": [case.run_index]},
    ):
        reply = await chart(**one)
        assert reply["n_steps"] == 1, one
        (differs,) = _steps_differ(reply)
        assert differs["evidence"] == expected, one

    several = await chart(job_id=job.job_id, runs="all")
    assert [item["n_steps"] for item in several["cases"]] == [3, 1, 3]
    (differs,) = _steps_differ(several)
    assert differs["evidence"] == expected

    by_path = await chart(path=str(case.raw_file))
    assert by_path["n_steps"] == 1
    assert _steps_differ(by_path) == []


@pytest.mark.asyncio
async def test_a_chart_of_a_cases_other_raw_is_not_compared_with_its_record(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The record gives the steps of the case's own results. The input: cases
    whose results are a 3-step transient and whose other raw (fra_1) is an
    unstepped frequency response; a chart of that other raw draws 1 step a
    case and says nothing of the record, which is not about that file."""
    from ltspice_mcp.lib.raw_parser import other_raw_path
    from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform

    raw, log = _recorded("ltspice26", "step_tran")
    receipt = await _ran(state_with_sim, monkeypatch, work_dir, raw, log, "other", count=2)
    job = state_with_sim.all_jobs[receipt["job_id"]]
    fra = RECORDED / "ltspice26" / "raw" / "fra.fra_1.raw"
    for case in job.cases:
        assert case.steps is not None and case.steps.count == 3
        assert case.raw_file is not None
        shutil.copy(fra, other_raw_path(Path(case.raw_file), "fra_1"))
        # What a run's completion lists of the other raws it finds beside its
        # raw (``raw_parser.other_raws``); the stand-in writes none itself.
        case.other_raws = ["fra_1"]

    selections: list[dict[str, Any]] = [{"runs": "all"}, {"run_index": 0}]
    for arguments in selections:
        result = await handle_plot_waveform(
            PlotWaveformInput(
                job_id=job.job_id, other_raw="fra_1", signals="all", open=False, **arguments
            ),
            state_with_sim,
        )
        reply = result.structured_content
        assert reply is not None and not result.is_error, result.content
        counts = (
            [item["n_steps"] for item in reply["cases"]]
            if "cases" in reply
            else [reply["n_steps"]]
        )
        assert set(counts) == {1}, arguments
        assert _steps_differ(reply) == [], arguments
