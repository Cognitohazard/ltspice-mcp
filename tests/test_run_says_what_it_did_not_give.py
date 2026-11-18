"""A run's reply says what the run did not give, and where the rest is read.

A field left out of a reply means "checked, and nothing to report", so a reply
that covers only part of a run says which part it did not cover:

- each run row names the measurements LTspice printed as failed in that run,
  and the steps of a stepped run whose value is not established, and the
  receipt's hint names the recipe that reads the values;
- an analysis, and a receipt, say when a requested ``.meas`` failed in every
  run read.

Every case is built from recordings under ``tests/fixtures/ltspice_recorded``
through the recorded-run stand-in (``replaying_simulator``).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.store import OwnerLiveness
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import LIVENESS_S, FakeSim, SyncApi, dispatched, replaying_simulator
from tests.ltspice_recorder import INPUTS

#: Three cases of one deck: R1 is a resistor in every deck below.
_THREE_CASES = [{"kind": "assign", "assign": {"R1": ["1k", "2k", "3k"]}}]
_MEASUREMENTS = {"key": "m", "metric": "measurements"}
_ROUTE = '"metric": "measurements"'


def _recorded_run(
    build: str, case: str, raw_case: str, directory: Path
) -> tuple[Path, Path, Path]:
    """The deck of recording ``case`` in ``directory``, with the log ``build``
    wrote for it and, beside it, the raw ``build`` wrote for ``raw_case``
    (the recording of a log case keeps its log alone)."""
    name = Path(case).name
    deck = directory / f"{name}.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    log = directory / f"{name}.log"
    shutil.copyfile(rec.recorded(build, f"{case}.log"), log)
    raw = directory / f"{name}.raw"
    shutil.copyfile(rec.recorded(build, f"{raw_case}.raw"), raw)
    return deck, raw, log


def _measured(row: dict[str, Any]) -> dict[str, Any]:
    """What a run row says about the run's measurements."""
    return {
        key: row[key]
        for key in (
            "failed_measurements",
            "failed_measurement_steps",
            "measurements_not_established",
            "measurements_not_printed",
            "measurements_coverage",
        )
        if key in row
    }


def _every_run_failed(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in observations if item["code"] == "meas_failed_in_every_run"]


async def _run(
    state: SessionState,
    deck: Path,
    request_id: str,
    variations: list[dict[str, Any]] = _THREE_CASES,
    **extra: Any,
) -> dict[str, Any]:
    receipt = await dispatched(
        state,
        "run_experiments",
        request_id=request_id,
        circuits=[{"path": str(deck), "id": deck.stem}],
        variations=variations,
        execution={"wait_s": LIVENESS_S},
        lint="off",
        **extra,
    )
    assert receipt["status"] == "completed", receipt
    return receipt


async def _every_receipt(state: SessionState, receipt: dict[str, Any]) -> dict[str, dict]:
    """The receipt as each route returns it: ``run_experiments``, ``jobs``
    status and wait, and the Python API's ``jobs`` status."""
    job_id = receipt["job_id"]
    api = SyncApi(state)
    return {
        "run_experiments": receipt,
        "jobs status": await dispatched(state, "jobs", action="status", job_id=job_id),
        "jobs wait": await dispatched(
            state, "jobs", action="wait", job_id=job_id, timeout_s=LIVENESS_S
        ),
        "api jobs status": await asyncio.to_thread(
            lambda: api.jobs(action="status", job_id=job_id)
        ),
        # The receipt api.run_experiments returns is this wait's.
        "api jobs wait": await asyncio.to_thread(
            lambda: api.jobs(action="wait", job_id=job_id, timeout_s=LIVENESS_S)
        ),
    }


async def _every_row(state: SessionState, receipt: dict[str, Any]) -> dict[str, list[dict]]:
    """Each route's run rows: the receipts' and ``jobs`` runs, by MCP and by
    the Python API."""
    job_id = receipt["job_id"]
    api = SyncApi(state)
    rows = {
        route: data["runs"]["items"]
        for route, data in (await _every_receipt(state, receipt)).items()
    }
    rows["jobs runs"] = (await dispatched(state, "jobs", action="runs", job_id=job_id))["items"]
    rows["api jobs runs"] = (
        await asyncio.to_thread(lambda: api.jobs(action="runs", job_id=job_id))
    )["items"]
    return rows


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_run_row_names_the_measurements_ltspice_printed_failed(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_failed``: both builds print ``never`` and ``depends``
    FAIL'ed and take ``before`` and ``after``. Every run row names the two,
    on every route, and every receipt's hint names the recipe that reads the
    values, by MCP and by the Python API."""
    deck, raw, log = _recorded_run(build, "log/meas_failed", "raw/tran_uncompressed", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state_with_sim, deck, "meas-failed-rows")
    for route, rows in (await _every_row(state_with_sim, receipt)).items():
        assert len(rows) == 3, route
        for row in rows:
            assert _measured(row) == {"failed_measurements": ["never", "depends"]}, route
    for route, data in (await _every_receipt(state_with_sim, receipt)).items():
        if route == "jobs wait":
            # The route is a standing note: this session's first receipt sent
            # it in full, so a wait names it by code and the call that repeats
            # it; status, and the Python API, send it in full.
            assert "measurements recipe" in data["hint"], (route, data["hint"])
            assert 'jobs(action="status"' in data["hint"], (route, data["hint"])
            continue
        assert _ROUTE in data["hint"], (route, data["hint"])
        assert "api.analyze_results" in data["hint"], route


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_run_row_names_the_steps_a_stepped_measurement_did_not_give(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_when_step_fails``: the WHEN is crossed at the first step
    only. LTspice 26 prints ``failed`` in the second row, so the row names
    ``w_one`` failed at step 1. XVII prints 0 there, which names no failure:
    the row says the value at step 1 is not established."""
    deck, raw, log = _recorded_run(build, "log/meas_when_step_fails", "raw/step_tran", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state_with_sim, deck, "meas-step-rows")
    expected = (
        {"failed_measurements": ["w_one"], "failed_measurement_steps": {"w_one": [1]}}
        if rec.generation(build) == "current"
        else {"measurements_not_established": {"w_one": [1]}}
    )
    for route, rows in (await _every_row(state_with_sim, receipt)).items():
        for row in rows:
            assert _measured(row) == expected, route
    assert _ROUTE in receipt["hint"]


async def test_a_run_with_no_failed_measurement_says_nothing_of_them(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_when_threshold_tran`` on LTspice 26 prints a value for every
    requested ``.meas``, so the run rows and the hint say nothing of them."""
    deck, raw, log = _recorded_run(
        "ltspice26", "log/meas_when_threshold_tran", "raw/tran_uncompressed", work_dir
    )
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state_with_sim, deck, "meas-clean")
    for row in receipt["runs"]["items"]:
        assert _measured(row) == {}
    assert _ROUTE not in receipt["hint"]


def test_a_record_written_before_measurements_were_kept_says_they_were_not_read(
    work_dir: Path,
):
    """A run that ended before the server kept its measurements says so: it
    never reads as "none failed". One whose log could not be read says that."""
    from ltspice_mcp.lib import now
    from ltspice_mcp.lib.experiment_types import ExperimentCase
    from ltspice_mcp.tools.receipts import runs_page

    def case(**fields: Any) -> ExperimentCase:
        return ExperimentCase(
            case_id="c",
            run_index=0,
            circuit="dut",
            circuit_path=work_dir / "dut.cir",
            staged_deck=work_dir / "dut.cir",
            deck_sha256="0" * 64,
            **fields,
        )

    old = case(status="produced", submitted_at=now(), log_diagnostics={})
    unread = case(
        status="produced",
        submitted_at=now(),
        log_diagnostics={},
        measurement_facts={"coverage": "log_unread"},
    )
    clean = case(status="produced", submitted_at=now(), log_diagnostics={}, measurement_facts={})
    queued = case()
    rows = runs_page([old, unread, clean, queued])["items"]
    assert [_measured(row) for row in rows] == [
        {"measurements_coverage": "not_read"},
        {"measurements_coverage": "log_unread"},
        {},
        {},
    ]


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_measurement_facts_survive_a_reload_and_reset_on_resume(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The facts a row carries are persisted with the job, and a record
    written before they were kept loads as not read."""
    from ltspice_mcp.lib import experiment_store

    deck, raw, log = _recorded_run(build, "log/meas_failed", "raw/tran_uncompressed", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state_with_sim, deck, "meas-failed-reload")
    job = state_with_sim.all_jobs[receipt["job_id"]]
    record = json.loads(json.dumps(experiment_store.serialize_job(job), default=str))

    def load(data: dict[str, Any]) -> Any:
        return experiment_store.deserialize_job(data, job.store_path, liveness=OwnerLiveness.ALIVE)

    loaded = load(record)
    assert [case.measurement_facts for case in loaded.cases] == [
        {"failed": ["never", "depends"], "requested": ["before", "never", "depends", "after"]}
    ] * 3
    for case in record["cases"]:
        case.pop("measurement_facts")
    old = load(record)
    assert [case.measurement_facts for case in old.cases] == [None] * 3


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_measurement_that_failed_in_every_run_is_said_once(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_failed`` run as three cases: ``never`` and ``depends``
    failed in every run, and ``before`` and ``after`` did not. The receipt,
    ``jobs`` and the analysis say so in one observation each, by MCP and by
    the Python API, and do not claim every requested ``.meas`` failed."""
    deck, raw, log = _recorded_run(build, "log/meas_failed", "raw/tran_uncompressed", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(
        state_with_sim, deck, "meas-failed-everywhere", analyze={"recipes": [_MEASUREMENTS]}
    )
    found: dict[str, list[dict[str, Any]]] = {}
    for route, data in (await _every_receipt(state_with_sim, receipt)).items():
        found[f"{route} receipt"] = _every_run_failed(data.get("observations", []))
    found["attached analysis"] = _every_run_failed(
        receipt["analysis"]["result"].get("observations", [])
    )
    job = [{"job_id": receipt["job_id"], "label": "job"}]
    reply = await dispatched(
        state_with_sim, "analyze_results", sources=job, recipes=[_MEASUREMENTS]
    )
    found["analyze_results"] = _every_run_failed(reply.get("observations", []))
    api = SyncApi(state_with_sim)
    analyzed = await asyncio.to_thread(
        lambda: api.analyze_results(sources=job, recipes=[_MEASUREMENTS])
    )
    found["api analyze_results"] = _every_run_failed(analyzed.get("observations", []))
    for route, observations in found.items():
        assert len(observations) == 1, (route, observations)
        (observation,) = observations
        assert observation["evidence"]["measurements"] == ["never", "depends"], route
        assert observation["evidence"]["runs"] == 3, route
        assert observation["evidence"]["requested"] == 4, route
        assert "every requested" not in observation["detail"], route


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_every_requested_measurement_failed_in_every_run_is_said(
    build: str, state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``deck/save_omits_meas``: the deck's one ``.meas`` reads a node its
    ``.save`` leaves out. XVII prints it FAIL'ed; LTspice 26 prints an error
    and no measurement at all. Run as three cases, each row names it, as
    failed or as not printed, and the receipt and the analysis say every
    requested ``.meas`` gave no value in any run."""
    deck, raw, log = _recorded_run(build, "deck/save_omits_meas", "deck/save_omits_meas", work_dir)
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(
        state_with_sim, deck, "meas-all-failed", analyze={"recipes": [_MEASUREMENTS]}
    )
    row_field = (
        "failed_measurements" if rec.generation(build) == "xvii" else "measurements_not_printed"
    )
    for route, rows in (await _every_row(state_with_sim, receipt)).items():
        assert [_measured(row) for row in rows] == [{row_field: ["vfinal"]}] * 3, route
    assert _ROUTE in receipt["hint"]
    for observations in (
        receipt.get("observations", []),
        receipt["analysis"]["result"].get("observations", []),
    ):
        (observation,) = _every_run_failed(observations)
        assert observation["evidence"]["measurements"] == ["vfinal"]
        assert observation["evidence"]["requested"] == 1
        assert "every requested .meas" in observation["detail"]


async def test_a_measurement_that_failed_in_some_runs_only_is_not_said(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_when_step_fails`` on LTspice 26: ``w_one`` failed at one
    step of each run, not at every step, so no run failed it whole."""
    deck, raw, log = _recorded_run(
        "ltspice26", "log/meas_when_step_fails", "raw/step_tran", work_dir
    )
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state_with_sim, deck, "meas-step-not-everywhere")
    assert _every_run_failed(receipt.get("observations", [])) == []


_PATTERN_CALLS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "_re"}


def test_only_the_log_parser_reads_a_failed_measurement_line():
    """Which line says a measurement failed is decided in ``log_parser``
    (``failed_measurement_on``); the classifier asks it. No other module
    keeps a pattern of its own for the line."""
    import ast

    from tests._source_scan import modules

    holders = set()
    for path, tree in modules().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if (
                name in _PATTERN_CALLS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and "FAIL" in first.value
            ):
                holders.add(path)
    assert holders == {"lib/log_parser.py"}


def test_one_function_says_a_measurement_failed_in_every_run():
    """The observation is built in one place, which the analysis and the
    receipt both call."""
    import ast

    from tests._source_scan import modules

    builders = set()
    callers = set()
    for path, tree in modules().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "meas_failed_in_every_run":
                builders.add(path)
            if isinstance(node, ast.Name) and node.id == "measurements_failed_in_every_run":
                callers.add(path)
    assert builders == {"lib/result_observations.py"}
    assert callers == {"tools/analyze.py", "tools/receipts.py"}


def _core_section(title: str) -> str:
    from ltspice_mcp.lib import guide

    return guide.core().split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def _metrics() -> set[str]:
    from typing import get_args

    from ltspice_mcp.lib.recipes import RECIPE_MODELS

    return {get_args(model.model_fields["metric"].annotation)[0] for model in RECIPE_MODELS}


def test_every_recipe_the_core_names_is_a_recipe_and_the_instructions_name_them_too():
    """The core's list of recipes by question names recipes the server has,
    and the server's instructions point at the same ones."""
    import re

    from ltspice_mcp.server import CONSOLIDATED_INSTRUCTIONS

    section = _core_section("Ask a recipe before numpy")
    named = {
        word
        for line in section.splitlines()
        if line.startswith("- ")
        for word in re.findall(r"`([a-z_]+)`", line)
    }
    assert named >= {"tone", "transient_response", "measurements", "stability"}
    assert named <= _metrics()
    for metric in named - {"summary"}:
        assert metric in CONSOLIDATED_INSTRUCTIONS, metric


def test_the_core_example_calls_a_recipe_whose_arguments_validate():
    """The core's example calls one recipe through ``api.analyze_results``,
    beside ``load_raw``, and its arguments are ones the tool accepts."""
    import ast
    import re

    from ltspice_mcp.tools.analyze import AnalyzeResultsInput

    (block,) = re.findall(r"```python\n(.*?)```", _core_section("The API on one page"), re.S)
    calls = {
        node.func.attr: node
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "api"
    }
    assert {"analyze_results", "load_raw", "run_experiments"} <= set(calls)
    keywords = {kw.arg: kw.value for kw in calls["analyze_results"].keywords}
    recipes = ast.literal_eval(keywords["recipes"])
    AnalyzeResultsInput.model_validate({"sources": [{"job_id": "job"}], "recipes": recipes})


async def test_the_core_example_reads_its_rows_as_the_reply_gives_them(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The core example's recipe, run by the Python API over a job of three
    cases, gives one row per case with the mean its loop reads."""
    import ast
    import re

    (block,) = re.findall(r"```python\n(.*?)```", _core_section("The API on one page"), re.S)
    (call,) = [
        node
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "analyze_results"
    ]
    recipes = ast.literal_eval({kw.arg: kw.value for kw in call.keywords}["recipes"])
    # A 5 ms transient of V(out), so the example's window from 4 ms reads.
    deck = work_dir / "compress_1mv_off.cir"
    shutil.copyfile(INPUTS / "raw/compress_1mv_off.cir", deck)
    replaying_simulator(
        monkeypatch,
        rec.recorded("ltspice26", "raw/compress_1mv_off.raw"),
        rec.recorded("ltspice26", "raw/compress_1mv_off.log"),
    )
    receipt = await _run(state_with_sim, deck, "core-example")
    api = SyncApi(state_with_sim)
    stats = await asyncio.to_thread(
        lambda: api.analyze_results(sources=[{"job_id": receipt["job_id"]}], recipes=recipes)
    )
    rows = stats["results"]["vout"]["values"]
    assert len(rows) == 3
    assert all(isinstance(row["value"]["mean"], float) for row in rows)
    assert [row.get("assignments", {}) for row in rows] == [
        {"R1": value} for value in ("1k", "2k", "3k")
    ]


#: Each recording a deck with subcircuits ran as: whether its raw holds no
#: node inside an instance, by build, and the node the reply gives as one.
_SUBCIRCUIT_RUNS = {
    # No .save, the setting off: each build stores only the port currents.
    "raw/subckt_names_tran": (
        "raw/subckt_names_tran",
        {"ltspice26": "V(xa:xin:mid)", "ltspice17": "V(a:in:mid)"},
    ),
    "log/subckt_names_off": (
        "log/subckt_names_off",
        {"ltspice26": "V(xa:xin:mid)", "ltspice17": "V(a:in:mid)"},
    ),
    # The same decks with "Save subcircuit node voltages" on.
    "raw/subckt_names_tran_saved": ("raw/subckt_names_tran", {}),
    "log/subckt_names_off_saved": ("log/subckt_names_off", {}),
    # A .save in LTspice 26's spelling, which XVII does not save.
    "raw/save_subckt_names": ("raw/save_subckt_names", {"ltspice17": "V(a:in:mid)"}),
    # A .save in XVII's spelling, which LTspice 26 does not save.
    "raw/save_subckt_names_no_x": (
        "raw/save_subckt_names_no_x",
        {"ltspice26": "V(xa:xin:mid)"},
    ),
    # Subcircuits with only port nodes.
    "raw/op_subckt": ("raw/op_subckt", {}),
}


@pytest.fixture
def state_ltspice(config: ServerConfig) -> SessionState:
    """A session whose stand-in simulator is LTspice, as a deck's reader and a
    run's producer are named by the simulator's family."""
    return SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})


def _subcircuit_nodes(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in observations if item["code"] == "subcircuit_nodes_not_saved"]


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("recording", sorted(_SUBCIRCUIT_RUNS))
async def test_a_raw_with_no_node_inside_the_decks_subcircuits_says_so(
    build: str,
    recording: str,
    state_ltspice: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A run whose raw holds no node voltage inside the deck's subcircuit
    instances, where a subcircuit's body has a net of its own, says so on the
    receipt, ``jobs`` and every read by job, by MCP and by the Python API,
    with both recorded causes. A raw that holds one, or a deck whose
    subcircuits have only ports, says nothing."""
    deck_case, expected = _SUBCIRCUIT_RUNS[recording]
    example = expected.get(build)
    deck = work_dir / f"{Path(recording).name}.cir"
    shutil.copyfile(INPUTS / f"{deck_case}.cir", deck)
    replaying_simulator(
        monkeypatch,
        rec.recorded(build, f"{recording}.raw"),
        rec.recorded(build, f"{recording}.log"),
    )
    receipt = await _run(state_ltspice, deck, f"subckt-nodes-{build}", variations=[])
    job_id = receipt["job_id"]
    case_id = receipt["runs"]["items"][0]["case_id"]
    api = SyncApi(state_ltspice)
    found = {
        route: data.get("observations", [])
        for route, data in (await _every_receipt(state_ltspice, receipt)).items()
    }
    job = [{"job_id": job_id}]
    analyzed = await dispatched(
        state_ltspice, "analyze_results", sources=job, recipes=[{"key": "s", "metric": "summary"}]
    )
    found["analyze_results by job"] = analyzed.get("observations", [])
    api_analyzed = await asyncio.to_thread(
        lambda: api.analyze_results(sources=job, recipes=[{"key": "s", "metric": "summary"}])
    )
    found["api analyze_results by job"] = api_analyzed.get("observations", [])
    raw = await asyncio.to_thread(lambda: api.load_raw(job_id=job_id, case_id=case_id))
    found["api load_raw by job"] = list(raw.observations)
    for route, observations in found.items():
        said = _subcircuit_nodes(observations)
        if example is None:
            assert said == [], (route, said)
            continue
        assert len(said) == 1, (route, observations)
        (observation,) = said
        assert observation["evidence"]["example"] == example, route
        assert observation["evidence"]["circuit"] == deck.stem, route
        assert deck.stem in observation["detail"], route
        assert "Save subcircuit node voltages" in observation["detail"], route
        saves = deck_case.startswith("raw/save_")
        assert observation["evidence"].get("saves", False) is saves, route
        assert ("the deck's .save cards" in observation["detail"]) is saves, route


_NGSPICE_DECK = """* a node inside nested subcircuit instances
V1 vdd 0 3
Xa vdd outa stage
.subckt stage top out
Rd top out 10k
Xin top out inner
.ends
.subckt inner p n
Rin p mid 100k
Rlo mid n 50k
.ends
{save}.tran 1u 10u
.end
"""


@pytest.mark.parametrize("save", ["", ".save v(outa)\n"])
async def test_ngspice_says_so_where_its_raw_holds_no_inner_node_live(work_dir: Path, save: str):
    """On the ngspice on PATH: the reply says the raw holds no node inside the
    subcircuits exactly where the raw it wrote holds none, and with a
    ``.save`` names that as the cause, never LTspice's setting."""
    import shutil as _shutil

    from ltspice_mcp.lib.raw_parser import first_plot_traces
    from ltspice_mcp.lib.simulator import detect_simulators
    from tests.conftest import terminal_experiment

    if _shutil.which("ngspice") is None:
        pytest.skip("ngspice is not on PATH")
    deck = work_dir / "nested.cir"
    deck.write_text(_NGSPICE_DECK.format(save=save), encoding="utf-8", newline="\n")
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ngspice")
    state = SessionState.create(config, available=detect_simulators(config))
    reply = await terminal_experiment(
        state,
        {
            "request_id": f"ngspice-inner-{bool(save)}",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 60},
            "lint": "off",
        },
    )
    assert reply["status"] == "completed", reply
    (row,) = reply["runs"]["items"]
    traces = first_plot_traces(Path(row["raw"]))
    assert traces is not None
    inner = [name for name, _ in traces if name.lower() == "v(xa.xin.mid)"]
    said = _subcircuit_nodes(reply.get("observations", []))
    assert bool(said) is not bool(inner), (traces, said)
    if save:
        # ngspice keeps only what a .save names.
        (observation,) = said
        assert "the deck's .save cards" in observation["detail"]
        assert "Save subcircuit node voltages" not in observation["detail"]
        assert observation["evidence"]["example"] == "V(xa.xin.mid)"


def _compression(observations: list[dict[str, Any]]) -> list[str]:
    return [
        item["code"]
        for item in observations
        if item["code"] in ("compressed_transient", "coarse_compressed_waveform")
    ]


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("recording", ["tran", "tran_uncompressed"])
async def test_a_read_of_a_compressed_xvii_transient_says_so(
    build: str,
    recording: str,
    state_ltspice: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``raw/tran``: an RC step response, compression on. XVII stores 56
    points where the same deck with compression off (``raw/tran_uncompressed``)
    stores 73, not so few that the run is coarse, and every read and recipe
    that takes its samples says it was saved compressed: ``analyze_results``,
    ``inspect`` results, ``plot_waveform`` and the Python API's ``load_raw``
    and ``analyze_results``. LTspice 26's, and the run with compression off
    on either build, say nothing of it. The measurements recipe takes no
    samples."""
    deck = work_dir / f"{recording}.cir"
    shutil.copyfile(INPUTS / f"raw/{recording}.cir", deck)
    replaying_simulator(
        monkeypatch,
        rec.recorded(build, f"raw/{recording}.raw"),
        rec.recorded(build, f"raw/{recording}.log"),
    )
    receipt = await _run(state_ltspice, deck, f"compressed-{build}", variations=[])
    job_id = receipt["job_id"]
    case_id = receipt["runs"]["items"][0]["case_id"]
    job = [{"job_id": job_id}]
    stats = [{"key": "v", "metric": "signal_stats", "signal": "V(out)"}]
    api = SyncApi(state_ltspice)
    found: dict[str, list[str]] = {}
    reply = await dispatched(state_ltspice, "analyze_results", sources=job, recipes=stats)
    found["analyze_results"] = _compression(reply.get("observations", []))
    page = await dispatched(
        state_ltspice,
        "inspect",
        queries=[{"kind": "results", "view": "signals", "job_id": job_id}],
    )
    found["inspect results"] = _compression(page["results"][0]["data"].get("observations", []))
    plotted = await dispatched(
        state_ltspice,
        "plot_waveform",
        job_id=job_id,
        case_id=case_id,
        signals=["V(out)"],
        open=False,
    )
    found["plot_waveform"] = _compression(plotted.get("observations", []))
    raw = await asyncio.to_thread(lambda: api.load_raw(job_id=job_id, case_id=case_id))
    found["api load_raw"] = _compression(list(raw.observations))
    analyzed = await asyncio.to_thread(lambda: api.analyze_results(sources=job, recipes=stats))
    found["api analyze_results"] = _compression(analyzed.get("observations", []))
    compressed = recording == "tran" and rec.generation(build) == "xvii"
    assert found == dict.fromkeys(found, ["compressed_transient"] if compressed else [])
    measured = await dispatched(
        state_ltspice, "analyze_results", sources=job, recipes=[_MEASUREMENTS]
    )
    assert "compressed_transient" not in _compression(measured.get("observations", []))


def _recorded_pair(build: str, case: str, directory: Path) -> Path:
    target = directory / build
    target.mkdir(exist_ok=True)
    for suffix in (".raw", ".log"):
        shutil.copyfile(rec.recorded(build, f"raw/{case}{suffix}"), target / f"{case}{suffix}")
    return target / f"{case}.raw"


def test_the_guide_and_the_reply_state_what_the_compressed_recordings_show(
    state_no_sim: SessionState, work_dir: Path
):
    """``raw/compress_1mv`` and ``raw/compress_1mv_off``: XVII stores 13 points
    with compression on; a trapezoid over them is 0.41% below its own AVG and
    interpolation at 1 ms 1.15% below its FIND, while its .meas results are
    within 0.25% of the uncompressed run's. LTspice 26 stores 1046 points
    either way, though its compressed raw's flags lack ``nocompression`` too.
    The guide and the observations state these numbers and no others."""
    import numpy as np

    from ltspice_mcp.lib import guide
    from ltspice_mcp.lib.result_observations import _XVII_MEAS

    api = SyncApi(state_no_sim)

    def read(build: str, case: str) -> tuple[Any, dict[str, float]]:
        path = _recorded_pair(build, case, work_dir)
        raw = api.load_raw(path)
        measured = api.measurements(path.with_suffix(".log"))["measurements"]
        return raw, {name: entry["values"][0] for name, entry in measured.items()}

    raw, meas = read("ltspice17", "compress_1mv")
    _, uncompressed = read("ltspice17", "compress_1mv_off")
    t, v = np.asarray(raw.axis()), np.asarray(raw.trace("V(out)"))
    assert len(t) == 13
    trapezoid = float(np.sum((v[1:] + v[:-1]) * np.diff(t)) / 2 / (t[-1] - t[0]))
    assert round((trapezoid / meas["vavg"] - 1) * 100, 2) == -0.41
    assert round((float(np.interp(1e-3, t, v)) / meas["vat"] - 1) * 100, 2) == -1.15
    for name in ("vavg", "vat"):
        assert abs(meas[name] / uncompressed[name] - 1) < 0.0025, name
    for case in ("compress_1mv", "compress_1mv_off"):
        assert len(api.load_raw(_recorded_pair("ltspice26", case, work_dir)).axis()) == 1046
    text = " ".join(guide.read("ltspice").split())
    for stated in ("13 points", "0.41%", "1.15%", "within 0.25%"):
        assert stated in text, stated
    for stated in ("13 stored points", "1.15%", "within 0.25%"):
        assert stated in _XVII_MEAS, stated


@pytest.mark.parametrize(
    ("code", "builder", "callers"),
    [
        (
            "subcircuit_nodes_not_saved",
            "subcircuit_nodes_not_saved",
            {"lib/experiment_runner.py"},
        ),
        (
            "compressed_transient",
            "compressed_xvii_observation",
            {"lib/services.py", "tools/analyze.py"},
        ),
    ],
)
def test_each_observation_of_what_a_raw_holds_is_built_in_one_place(
    code: str, builder: str, callers: set[str]
):
    """Each observation is built by one function, which only the readers
    that answer for a run call; whether an XVII transient dropped points is
    decided by ``services.compression_drops_points`` alone."""
    import ast

    from tests._source_scan import modules

    built = set()
    called = set()
    decides = set()
    for path, tree in modules().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == code:
                built.add(path)
            if isinstance(node, ast.Name) and node.id == builder:
                called.add(path)
            if (
                isinstance(node, (ast.Name, ast.Attribute))
                and (getattr(node, "id", None) or getattr(node, "attr", None))
                == "compression_drops_points"
            ):
                decides.add(path)
    assert built == {"lib/result_observations.py"}
    assert called == callers
    assert decides == {"lib/services.py", "tools/analyze.py"}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_read_by_path_of_a_compressed_xvii_transient_says_so(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``raw/tran`` read by its path, with no job to name its producer: the
    build is the one its files name, and XVII's compressed raw says so while
    LTspice 26's says nothing."""
    raw = SyncApi(state_no_sim).load_raw(_recorded_pair(build, "tran", work_dir))
    expected = ["compressed_transient"] if rec.generation(build) == "xvii" else []
    assert _compression(list(raw.observations)) == expected


@pytest.mark.parametrize("family", ["qspice", "xyce"])
async def test_a_run_on_a_simulator_whose_measures_are_not_read_says_so(
    family: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The server reads ``.meas`` results only from LTspice and ngspice logs,
    so a QSPICE or Xyce run's rows say its measurements were not read, and
    neither name any as not given nor send the caller to a recipe that cannot
    read them. The stand-in hands back a recorded log; only the simulator's
    family decides."""
    from ltspice_mcp.lib.simulator import SIMULATORS

    stand_in = type(SIMULATORS[family].__name__, (FakeSim,), {})
    state = SessionState.create(config, available={family: stand_in})
    deck, raw, log = _recorded_run(
        "ltspice26", "log/meas_failed", "raw/tran_uncompressed", work_dir
    )
    replaying_simulator(monkeypatch, raw, log)
    receipt = await _run(state, deck, f"unread-{family}", variations=[])
    for row in receipt["runs"]["items"]:
        assert _measured(row) == {"measurements_coverage": "simulator_unsupported"}
    assert _ROUTE not in receipt["hint"]
    assert _every_run_failed(receipt.get("observations", [])) == []
