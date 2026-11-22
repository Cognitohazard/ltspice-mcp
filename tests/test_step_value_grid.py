"""Which step of a stepped run a value belongs to, on every route that answers it.

The question is the step-value grid's (``[grid.step-value]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). A run keeps its step
values in its log's ``.step`` lines (``log_parser.read_step_line``, bound to
the raw's sweeps by ``raw_decode._step_facts``) or, one point a step, in its
raw; a value names a step as ``step_values`` reads it (``step_axis_key`` for
the name, ``same_step_value`` and ``steps_with`` for the value), and the step
selector of every tool asks it (``recipes.StepSelector``).

A cell is one value of one stepped name, as the deck writes both, and the
steps (from 1) that hold it, from the deck's ``.step`` cards. Every recorded
cell, on each build, is read through every route: the log's own step rows,
the decoded raw's steps and its ``get_steps`` filter, ``steps_with`` and the
selector over the run's steps, ``analyze_results``' ``step`` with a raw
recipe and with the measurements recipe, ``plot_waveform``'s ``step``,
``run_experiments``' attached analysis as its receipt and ``jobs(status)``
return it, and the Python API's ``analyze_results`` and ``load_raw`` steps.
A route a case gives nothing to read (no raw, no ``.meas``) is said, per case.
ngspice runs no ``.step``, so the question has no ngspice answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.recipes import StepSelector
from ltspice_mcp.lib.step_values import same_step_value, step_axis_key, steps_with
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests.conftest import LIVENESS_S, SyncApi, dispatched, replaying_simulator

GRID = grid("step-value")

#: How many routes answer otherwise than a cell, per build, each declared on
#: its cell (``differs``, with the route and the reason). A count may only
#: fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 0, "ltspice17": 0}


def _recorded(build: str, case_id: str) -> bool:
    return case_id in rec.manifest(build)["cases"]


def _runs() -> list[Any]:
    runs = []
    for case_id in sorted({cell.case for cell in GRID.cells}):
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if _recorded(build, case_id):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


def _value(cell: Cell) -> str:
    return cell.extra["step_value"]


def _axis(cell: Cell) -> str:
    return cell.extra["axis"]


def _steps(cell: Cell) -> list[int]:
    """The cell's steps, from 0."""
    return [step - 1 for step in cell.extra["steps"]]


# The index, held to the decks -----------------------------------------------


def test_a_cell_names_a_value_its_deck_steps_under_the_name_it_spells():
    for cell in GRID.cells:
        deck = (rec.INPUTS / rec.CASES.case(cell.case).source).read_text(encoding="utf-8")
        steps = [line for line in deck.splitlines() if line.lower().startswith(".step")]
        assert cell.item == f"{_axis(cell)}={_value(cell)}", cell.id
        assert len(steps) == int(cell.coordinates["nesting"]), cell.id
        mine = [line for line in steps if f" {_axis(cell)} " in f" {line} ".replace("(", " (")]
        mine = mine or [line for line in steps if _axis(cell) in line]
        assert len(mine) == 1, (cell.id, steps)
        (line,) = mine
        assert _axis(cell) in line, (cell.id, line)
        if cell.coordinates["sweep"] == "list":
            assert f" {_value(cell)}" in line.split(" list ", 1)[1].join("  "), (cell.id, line)
        words = line.split()
        sweep = {"dec": "decade", "oct": "octave"}.get(words[1].lower())
        if sweep is None:
            sweep = "list" if " list " in line.lower() else "linear"
        assert cell.coordinates["sweep"] == sweep, (cell.id, line)
        stepped = (
            "temp"
            if _axis(cell).lower() == "temp"
            else "param"
            if " param " in line.lower()
            else "model"
            if "(" in _axis(cell)
            else "source"
        )
        assert cell.coordinates["stepped"] == stepped, (cell.id, line)
        case = (
            "lower"
            if _axis(cell) == _axis(cell).lower()
            else "upper"
            if _axis(cell) == _axis(cell).upper()
            else "mixed"
        )
        assert cell.coordinates["name_case"] == case, cell.id
        assert all(step >= 1 for step in cell.extra["steps"]), cell.id


def test_every_case_of_the_grid_is_waiting_or_has_no_prediction_left():
    """A recorded cell declares what the recording shows: a prediction
    (``not_established``) is settled against its recording, and the key
    removed, once the case is recorded."""
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle its steps against the "
                "recording and remove not_established"
            )


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(
            1
            for cell in GRID.cells
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        )
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route ---------------------------------------------------------------


def _step_or_refusal(reply: dict[str, Any], key: str) -> int | str:
    if reply.get("failures"):
        (message,) = {failure["message"] for failure in reply["failures"]}
        return message
    (row,) = reply["results"][key]["per_run"]["items"]
    return row["step_index"]


def _recipe(cell: Cell, plot: str, signal: str) -> dict[str, Any]:
    """A raw recipe that reads one step of the run: its signal's statistics
    over a swept step, or, where each step holds one point, the value there
    (an operating point's bias point)."""
    if cell.coordinates["kept"] == "log":
        return {"metric": "signal_stats", "signal": signal}
    if "operating point" in plot.lower():
        return {"metric": "operating_point"}
    return {"metric": "value", "expr": signal}


def _signal(names: list[str]) -> str:
    return next(name for name in names if name.lower().startswith("v("))


async def _every_route(
    build: str,
    case_id: str,
    cell: Cell,
    state: SessionState,
    staged: Path,
    monkeypatch: pytest.MonkeyPatch,
    not_run: dict[str, str],
) -> dict[str, Any]:
    """What each route reads for the cell's value under its name: the steps a
    route that lists them gives, from 0, or the one step a selector reads, or
    the message it refuses with."""
    axis, value = _axis(cell), _value(cell)
    log = staged.with_suffix(".log")
    deck = staged.with_suffix(".cir")
    routes: dict[str, Any] = {}
    has_log = rec.has(build, f"{case_id}.log")
    has_raw = rec.has(build, f"{case_id}.raw")
    if has_log and (rows := parse_step_iterations(log)):
        routes["log rows, steps_with"] = steps_with(rows, {axis: value})
    if not has_raw:
        not_run["raw routes"] = "the case keeps no raw"
        steps = parse_step_iterations(log)
    else:
        decoded = rec.decode_file(staged, staged.parent / "decoded", log=log if has_log else None)
        raw = decoded.raw
        steps = [dict(step.parameters) for step in raw.descriptor.steps]
        routes["decoded raw, steps_with"] = steps_with(steps, {axis: value})
        routes["decoded raw, get_steps"] = list(raw.get_steps(**{axis: value}))
        signal = _signal(raw.get_trace_names())
        recipe = _recipe(cell, raw.get_plot_names()[0], signal)
        reply = await dispatched(
            state,
            "analyze_results",
            sources=[{"path": str(staged), "label": "dut"}],
            recipes=[{"key": "k", **recipe}],
            step={"axis": axis, "value": value},
            include={"per_run": True},
        )
        routes["analyze_results raw recipe"] = _step_or_refusal(reply, "k")
        if cell.coordinates["kept"] == "log":
            plotted = await dispatched(
                state,
                "plot_waveform",
                path=str(staged),
                signals=[signal],
                step={"axis": axis, "value": value},
                open=False,
            )
            routes["plot_waveform"] = (
                plotted["traces"][0]["step"] if "traces" in plotted else plotted
            )
        else:
            not_run["plot_waveform"] = "a run of one point a step has no axis to chart"
        api = SyncApi(state)
        api_reply = await asyncio.to_thread(
            lambda: api.analyze_results(
                sources=[{"path": str(staged), "label": "dut"}],
                recipes=[{"key": "k", **recipe}],
                step={"axis": axis, "value": value},
                include={"per_run": True},
            )
        )
        routes["api analyze_results"] = _step_or_refusal(api_reply, "k")
        loaded = await asyncio.to_thread(lambda: api.load_raw(path=staged))
        routes["api load_raw steps"] = [
            index
            for index, values in enumerate(loaded.steps or ())
            if axis in values and same_step_value(values[axis], value)
        ]
        if has_log:
            replaying_simulator(monkeypatch, staged, log)
            receipt = await dispatched(
                state,
                "run_experiments",
                circuits=[{"path": str(deck), "id": "dut"}],
                analyze={
                    "recipes": [{"key": "k", **recipe}],
                    "step": {"axis": axis, "value": value},
                    "include": {"per_run": True},
                },
                execution={"wait_s": LIVENESS_S},
                lint="off",
            )
            assert receipt["status"] == "completed", receipt
            status = await dispatched(state, "jobs", action="status", job_id=receipt["job_id"])
            for reply, shown in ((receipt, "run_experiments receipt"), (status, "jobs status")):
                routes[shown] = _step_or_refusal(reply["analysis"]["result"], "k")
        else:
            not_run["run_experiments"] = "the case keeps no log to replay"
    try:
        routes["selector"] = StepSelector(axis=axis, value=value).pick(steps)
    except ResultError as refused:
        routes["selector"] = str(refused)
    if has_log and _measures(case_id):
        source = staged if has_raw else log
        reply = await dispatched(
            state,
            "analyze_results",
            sources=[{"path": str(source), "label": "dut"}],
            recipes=[{"key": "m", "metric": "measurements"}],
            step={"axis": axis, "value": value},
            include={"per_run": True},
        )
        routes["analyze_results measurements"] = _step_or_refusal(reply, "m")
    else:
        not_run["analyze_results measurements"] = "the deck has no .meas"
    return routes


def _measures(case_id: str) -> bool:
    """Whether the case's deck has a .meas, which the measurements recipe reads."""
    deck = rec.INPUTS / rec.CASES.case(case_id).source
    return ".meas" in deck.read_text(encoding="utf-8").lower()


def _stage(build: str, case_id: str, directory: Path) -> Path:
    staged = rec.stage_run(build, case_id, directory)
    (directory / staged.with_suffix(".cir").name).write_bytes(
        (rec.INPUTS / rec.CASES.case(case_id).source).read_bytes()
    )
    return staged


#: The routes that list every step a value names; the others read one.
_LISTS = frozenset(
    {
        "log rows, steps_with",
        "decoded raw, steps_with",
        "decoded raw, get_steps",
        "api load_raw steps",
    }
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_names_its_steps_on_every_route(
    build: str,
    case_id: str,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each cell's value, under the name the deck spells, names the steps its
    deck's ``.step`` cards give it: every step on a route that lists them, the
    first on a route that reads one. A route that answers otherwise is
    declared on its cell (``differs``)."""
    for index, cell in enumerate(GRID.cells_of(case_id)):
        directory = work_dir / f"cell{index}"
        staged = _stage(build, case_id, directory)
        not_run: dict[str, str] = {}
        routes = await _every_route(
            build, case_id, cell, state_with_sim, staged, monkeypatch, not_run
        )
        print(f"{build}/{cell.id}: routes not run: {not_run}")
        declared = {
            entry["route"]: entry["reason"]
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        }
        want = _steps(cell)
        for route, got in routes.items():
            expected: Any = want if route in _LISTS else want[0]
            if got != expected:
                assert route in declared, (cell.id, build, route, got, expected)
            else:
                assert route not in declared, (cell.id, build, route, "no longer differs")


def test_a_stepped_temperature_is_found_by_either_of_its_names():
    """``step_axis_key`` reads a raw's ``temperature`` for a deck's ``temp``,
    and the other way about: the name a cell of a stepped temperature passes
    is one of them."""
    assert step_axis_key("temp", ["temperature"]) in ("temperature", "temp")
    for cell in GRID.cells:
        if cell.coordinates["stepped"] == "temp":
            assert _axis(cell).casefold() == "temp", cell.id
