"""Which field of a printed measurement holds its result, the same on every route.

A ``.meas ... WHEN`` with no ``FIND`` searches for a crossing: LTspice prints
the level it looked for after ``=`` and the point it found after an uppercase
``AT``, so the point is the result. A ``FIND``, a ``DERIV`` or a ``FIND ...
WHEN`` prints its result after ``=`` and the point it read at after a
lowercase ``at`` (LTspice 26 and XVII, recorded as ``log/meas_forms`` and the
other logs below). In a stepped table each row prints the result alone, the
crossing of a WHEN included (``log/meas_when_threshold_tran_step``).

``log_parser.measurement_answer`` answers which field that is, from the
printed form, the build that printed it and, where the run's deck is known,
its card; every reader asks it.

The question is the printed-measurement grid's (``[grid.printed-measurement]``
in ``inputs/cases.toml``, ``docs/TESTING.md``, "Grids"): what does this log
say a measurement's result is. ``test_every_recorded_cell_of_the_grid_reads_as_its_form_means``
reads every recorded cell of the grid, on each build, through all eight
routes that read a measurement (the measurements recipe, the summary recipe,
``inspect``'s measurements view and the Python API's ``measurements()``, each
over the log's path and over a job), and requires the result its form means.
The tests after it pin the decision itself on inputs no recording holds.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_parser import (
    MeasurementAnswer,
    answered_measurements,
    measurement_answer,
    measurement_cards,
    parse_measurements,
    printed_form,
    printed_measurements,
    prints_measurement,
    read_log_text,
    result_unreadable,
)
from ltspice_mcp.lib.metrics import measurements_at_step
from ltspice_mcp.lib.simulator_build import Producer, log_banner_build
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests.conftest import FIXTURES_DIR, SyncApi, dispatched, make_experiment_job
from tests.ltspice_recorder import INPUTS

#: What a route gives for a step whose result it names as not established.
NOT_ESTABLISHED = "not established"

Answer = list[float | str | None] | str


def _stage(build: str, case_id: str, directory: Path) -> tuple[Path, Path, Path]:
    """The case's deck and log in ``directory``, with a raw beside the log: the
    one the build wrote for the case, or for ``deck/after_end`` where the
    recording keeps the log alone. Every reply here reads the measurements
    from the log."""
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(case_id).name
    deck = directory / f"{name}.cir"
    shutil.copyfile(INPUTS / rec.CASES.case(case_id).source, deck)
    log = deck.with_suffix(".log")
    shutil.copyfile(rec.recorded(build, f"{case_id}.log"), log)
    raw = deck.with_suffix(".raw")
    own = f"{case_id}.raw"
    shutil.copyfile(rec.recorded(build, own if rec.has(build, own) else "deck/after_end.raw"), raw)
    return deck, raw, log


def _numbers(field: Any, count: int) -> list[float | None]:
    if isinstance(field, list):
        return list(field)
    return [field] * count


def _entry_answer(entry: dict[str, Any]) -> Answer:
    """A measurement entry's result per step, by the field it names; the
    reason where it names none."""
    if "answer_unknown" in entry:
        return "unknown"
    values = entry["values"]
    # An entry that names no field is read as a caller reads it: its values.
    if entry.get("answer", "value") == "at":
        numbers: list[float | str | None] = list(_numbers(entry.get("at"), len(values)))
    else:
        numbers = list(values)
    for step in entry.get("not_established", {}).get("steps", ()):
        assert numbers[step] is None, (entry, step)
        numbers[step] = NOT_ESTABLISHED
    return numbers


def _recipe_answers(reply: dict[str, Any]) -> dict[str, Answer]:
    """Each measurement's result per step from an ``all_steps`` recipe reply."""
    # A log that prints no measurement gives no row.
    rows = {row["step_index"]: row["value"] for row in reply["results"]["m"].get("values", [])}
    answers: dict[str, Answer] = {}
    for step in sorted(rows):
        value = rows[step]
        for name, number in value.get("measured", {}).items():
            if name in value.get("answer_unknown", {}):
                answers[name] = "unknown"
            else:
                held = answers.setdefault(name, [])
                assert isinstance(held, list)
                if step in value.get("not_established", {}).get(name, {}).get("steps", ()):
                    assert number is None, (name, step)
                    number = NOT_ESTABLISHED
                held.append(number)
    return answers


def _inspect_answers(item: dict[str, Any]) -> dict[str, Answer]:
    assert item["ok"], item
    answers: dict[str, Answer] = {}
    for row in sorted(item["data"].get("measurements", []), key=lambda row: row["value_index"]):
        name = row["measurement"]
        if "answer_unknown" in row:
            answers[name] = "unknown"
            continue
        held = answers.setdefault(name, [])
        assert isinstance(held, list)
        number = row.get("at") if row.get("answer", "value") == "at" else row.get("value")
        if "not_established" in row:
            assert number is None, row
            number = NOT_ESTABLISHED
        held.append(number)
    return answers


def _same(left: Answer, right: Answer) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return len(left) == len(right) and all(
        a == b
        or (
            isinstance(a, float | int)
            and isinstance(b, float | int)
            and math.isclose(a, b, rel_tol=1e-12)
        )
        for a, b in zip(left, right, strict=True)
    )


_RECIPE = [{"key": "m", "metric": "measurements"}]
_EACH_STEP = {"fields": ["value", "step_index"]}


#: The field each route that names one says holds a measurement's result:
#: route -> measurement -> ``value`` or ``at``. Filled by :func:`_every_route`.
Fields = dict[str, dict[str, str | None]]


def _entry_fields(entries: dict[str, Any]) -> dict[str, str | None]:
    return {name: entry.get("answer") for name, entry in entries.items()}


def _inspect_fields(item: dict[str, Any]) -> dict[str, str | None]:
    return {row["measurement"]: row.get("answer") for row in item["data"].get("measurements", [])}


#: What each route says beside a measurement's value, by route and
#: measurement: its ``unreadable`` (what the log printed where a number
#: belongs, and why that is no number) and its ``printed_once`` (a value the
#: log printed once for a run of several steps: how many, and which step's it
#: is where that is established). Filled by :func:`_every_route`.
Said = dict[str, dict[str, dict[str, Any]]]

#: The statements :data:`Said` collects.
_STATEMENTS = ("unreadable", "printed_once")


def _say(said: dict[str, dict[str, Any]], name: str, holder: dict[str, Any]) -> None:
    for key in _STATEMENTS:
        if key in holder:
            said.setdefault(name, {})[key] = holder[key]


def _entries_said(entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    said: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        _say(said, name, entry)
    return said


def _recipe_said(reply: dict[str, Any]) -> dict[str, dict[str, Any]]:
    said: dict[str, dict[str, Any]] = {}
    for row in reply["results"]["m"].get("values", []):
        for key in _STATEMENTS:
            for name, statement in row["value"].get(key, {}).items():
                said.setdefault(name, {})[key] = statement
    return said


def _inspect_said(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    said: dict[str, dict[str, Any]] = {}
    for row in item["data"].get("measurements", []):
        _say(said, row["measurement"], row)
    return said


async def _every_route(
    build: str,
    case_id: str,
    state: SessionState,
    directory: Path,
    fields: Fields | None = None,
    said: Said | None = None,
) -> dict[str, dict[str, Answer]]:
    """Each measurement's result per step, by route: the measurements recipe,
    the summary recipe, ``inspect``'s measurements view and the Python API,
    each over the log's path and over a job. ``fields``, when given, gets the
    field each route that names one says holds the result, and ``said`` what
    each route says beside it (:data:`Said`)."""
    deck, raw, log = _stage(build, case_id, directory)
    routes: dict[str, dict[str, Answer]] = {}
    named: Fields = {} if fields is None else fields
    told: Said = {} if said is None else said
    reply = await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=_RECIPE,
        all_steps=True,
        include=_EACH_STEP,
    )
    routes["recipe, by path"] = _recipe_answers(reply)
    told["recipe, by path"] = _recipe_said(reply)
    summary = await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"fields": ["value"]},
    )
    (row,) = summary["results"]["s"]["values"]
    routes["summary, by path"] = {
        name: _entry_answer(entry) for name, entry in row["value"].get("measurements", {}).items()
    }
    named["summary, by path"] = _entry_fields(row["value"].get("measurements", {}))
    told["summary, by path"] = _entries_said(row["value"].get("measurements", {}))
    page = await dispatched(
        state, "inspect", queries=[{"kind": "results", "view": "measurements", "path": str(log)}]
    )
    routes["inspect, by path"] = _inspect_answers(page["results"][0])
    named["inspect, by path"] = _inspect_fields(page["results"][0])
    told["inspect, by path"] = _inspect_said(page["results"][0])
    api = SyncApi(state)
    read = await asyncio.to_thread(lambda: api.measurements(path=log))
    routes["api measurements, by path"] = {
        name: _entry_answer(entry) for name, entry in read["measurements"].items()
    }
    named["api measurements, by path"] = _entry_fields(read["measurements"])
    told["api measurements, by path"] = _entries_said(read["measurements"])

    job_id = "exp_" + re.sub(r"\W", "_", f"{build}_{case_id}")
    job = make_experiment_job(state, job_id=job_id, raw=raw, netlist=deck)
    # A run records its simulator's class name and the build it named.
    job.simulator = "LTspice"
    job.cases[0].simulator_version = rec.manifest(build)["reported_build"]
    reply = await dispatched(
        state,
        "analyze_results",
        sources=[{"job_id": job_id, "label": "job"}],
        recipes=_RECIPE,
        all_steps=True,
        include=_EACH_STEP,
    )
    routes["recipe, by job"] = _recipe_answers(reply)
    told["recipe, by job"] = _recipe_said(reply)
    by_job = await dispatched(
        state,
        "analyze_results",
        sources=[{"job_id": job_id, "label": "job"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"fields": ["value"]},
    )
    (row,) = by_job["results"]["s"]["values"]
    routes["summary, by job"] = {
        name: _entry_answer(entry) for name, entry in row["value"].get("measurements", {}).items()
    }
    named["summary, by job"] = _entry_fields(row["value"].get("measurements", {}))
    told["summary, by job"] = _entries_said(row["value"].get("measurements", {}))
    page = await dispatched(
        state, "inspect", queries=[{"kind": "results", "view": "measurements", "job_id": job_id}]
    )
    routes["inspect, by job"] = _inspect_answers(page["results"][0])
    named["inspect, by job"] = _inspect_fields(page["results"][0])
    told["inspect, by job"] = _inspect_said(page["results"][0])
    read = await asyncio.to_thread(lambda: api.measurements(job_id=job_id))
    routes["api measurements, by job"] = {
        name: _entry_answer(entry) for name, entry in read["measurements"].items()
    }
    named["api measurements, by job"] = _entry_fields(read["measurements"])
    told["api measurements, by job"] = _entries_said(read["measurements"])
    return routes


# The printed-measurement grid --------------------------------------------------

GRID = grid("printed-measurement")

#: A number as LTspice prints one in a ``.meas`` result.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?"

#: What a printed result's text is, read as LTspice's formats mean it: a
#: number; ``(<dB>dB,<angle>°)`` by default and ``(<magnitude>,<angle>°)``
#: under ``meascplxfmt=polar``, a complex result, real when its angle is 0 or
#: 180 degrees; ``(<re>,<im>)`` under ``meascplxfmt=cartesian``, real when its
#: imaginary part is 0; a number with its unit (``90.6295°``, ``infdB``).
_POLAR = re.compile(rf"\(({_NUMBER}|[-+]?inf)(dB)?,({_NUMBER})\W*\)", re.IGNORECASE)
_CARTESIAN = re.compile(rf"\(({_NUMBER}),({_NUMBER})\)", re.IGNORECASE)
_WITH_UNIT = re.compile(rf"({_NUMBER}|[-+]?inf)(?:°|dB)", re.IGNORECASE)


def _printed_number(text: str, complex_format: str) -> float:
    text = text.strip()
    if complex_format == "cartesian" and (match := _CARTESIAN.fullmatch(text)):
        re_part, im_part = float(match[1]), float(match[2])
        return re_part if im_part == 0 else abs(complex(re_part, im_part))
    if match := _POLAR.fullmatch(text):
        magnitude = 10 ** (float(match[1]) / 20) if match[2] else float(match[1])
        angle = float(match[3]) % 360
        if math.isclose(angle, 0, abs_tol=1e-6) or math.isclose(angle, 360, abs_tol=1e-6):
            return magnitude
        if math.isclose(angle, 180, abs_tol=1e-6):
            return -magnitude
        return magnitude
    if match := _WITH_UNIT.fullmatch(text):
        return float(match[1])
    return float(text)


@dataclass(frozen=True)
class _Printed:
    """How a log printed one measurement, read by its name: on a line (the
    text after its last ``=``, and the point after ``AT``/``at``), on a
    TRIG/TARG's line with no colon, as a stepped table (each row's first
    column), or as FAIL'ed; None where the log does not print it."""

    printed: str
    value: str | None = None
    point: str | None = None
    rows: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()


def _printed(text: str, name: str) -> _Printed | None:
    lines = text.splitlines()
    escaped = re.escape(name)
    found: list[_Printed] = []
    for index, line in enumerate(lines):
        line = line.rstrip()
        if re.fullmatch(rf'Measurement "{escaped}" FAIL\'ed', line, re.IGNORECASE):
            found.append(_Printed("failed"))
        elif re.fullmatch(rf"Measurement: {escaped}", line, re.IGNORECASE):
            rows = []
            for row in lines[index + 2 :]:
                cells = row.strip().split("\t")
                if not row.strip() or not cells[0].isdigit():
                    break
                rows.append(cells[1])
            heading = re.match(r"\s*step\t(.*)$", lines[index + 1].rstrip())
            assert heading, lines[index + 1]
            found.append(
                _Printed("table", rows=tuple(rows), columns=tuple(heading[1].split("\t")))
            )
        elif match := re.fullmatch(
            rf"{escaped}=(\S+) FROM {_NUMBER} TO {_NUMBER}", line, re.IGNORECASE
        ):
            found.append(_Printed("line_bare", value=match[1]))
        elif match := re.fullmatch(
            rf"{escaped}: .*=(?P<value>.*?)"
            rf"(?: FROM {_NUMBER} TO {_NUMBER}| (?:at|AT) (?P<point>{_NUMBER}))?",
            line,
            re.IGNORECASE,
        ):
            found.append(_Printed("line", value=match["value"], point=match["point"]))
    assert len(found) <= 1, (name, found)
    return found[0] if found else None


def _expected(cell: Cell, build: str, shown: _Printed) -> Answer:
    """The result per step the log gives ``cell``, read as its form means:
    the value after ``=`` or the point after ``AT`` on a line, the first
    column of each row of a table. A row XVII prints 0 in is not established
    (it prints 0 both for a step that failed and for a result of 0), and a
    row LTspice 26 prints ``failed`` in has no result."""
    at = cell.at(build)
    result = GRID.expected(cell, build)
    complex_format = at.get("complex_format", "")
    if shown.printed == "failed":
        return "failed"
    if shown.printed == "table":
        numbers: list[float | str | None] = []
        for row in shown.rows:
            if row == "failed":
                numbers.append(None)
            elif rec.generation(build) == "xvii" and _printed_number(row, complex_format) == 0:
                numbers.append(NOT_ESTABLISHED)
            else:
                numbers.append(_printed_number(row, complex_format))
        return numbers
    text = shown.point if result == "point" else shown.value
    assert text is not None, (cell.id, shown)
    return [_printed_number(text, complex_format)]


def _check_outcome(cell: Cell, build: str, shown: _Printed) -> None:
    """The log shows the outcome the cell declares on ``build``: a failure
    printed FAIL'ed; a result or a failure in one step printed in a table
    that has the steps the cell names; and a table of a form whose result is
    a point (a WHEN) holds that point alone, under a heading of one column."""
    at = cell.at(build)
    outcome = at["outcome"]
    steps = [step - 1 for step in cell.extra.get("steps", ())]
    assert shown.printed == at["printed"], (cell.id, build, shown)
    if outcome == "failed":
        assert shown.printed == "failed", (cell.id, build)
        return
    assert shown.printed != "failed", (cell.id, build)
    if shown.printed != "table":
        assert outcome == "value", (cell.id, build)
        return
    assert shown.rows, (cell.id, build, "a table with no rows")
    assert all(0 <= step < len(shown.rows) for step in steps), (cell.id, build, steps, shown.rows)
    if GRID.expected(cell, build) == "point":
        assert len(shown.columns) == 1, (cell.id, build, shown.columns)
    for step, row in enumerate(shown.rows):
        zero = row != "failed" and _printed_number(row, at.get("complex_format", "")) == 0
        if step not in steps:
            assert row != "failed", (cell.id, build, step, row)
            assert not zero or cell.extra.get("zero_by_form"), (cell.id, build, step, row)
        elif outcome == "step_zero":
            assert zero, (cell.id, build, step, row)
        elif rec.generation(build) == "xvii":
            # XVII prints a step whose measurement failed as 0
            # (log/meas_when_step_fails).
            assert zero, (cell.id, build, step, row)
        else:
            assert row == "failed", (cell.id, build, step, row)


def _grid_runs() -> list[Any]:
    """Every (build, case) of the grid; a case waiting for its recording
    skips with its behaviour's ``pending``."""
    runs = []
    for case_id in sorted({cell.case for cell in GRID.cells}):
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if rec.has(build, f"{case_id}.log"):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


@pytest.mark.parametrize(("build", "case_id"), _grid_runs())
async def test_every_recorded_cell_of_the_grid_reads_as_its_form_means(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """Each cell of the printed-measurement grid in this case, through all
    eight routes: the measurement is read under its printed name; its result
    is what its form means (the value after ``=``, the point after ``AT``,
    a table's first column), in the field the form names on each route that
    names one; the outcome is the cell's (a result, ``failed``, or not
    established); and a measurement a build refuses is read on no route."""
    text = read_log_text(rec.recorded(build, f"{case_id}.log"))
    cells = GRID.cells_of(case_id)
    printed = printed_measurements(text)
    expected: dict[str, Answer] = {}
    for cell in cells:
        shown = _printed(text, cell.item)
        if build in cell.refused:
            assert shown is None, (cell.id, build, cell.refused[build])
            assert cell.item.lower() not in printed, cell.id
            continue
        assert shown is not None, f"{cell.id}: {build}'s log does not print it"
        _check_outcome(cell, build, shown)
        expected[cell.item.lower()] = _expected(cell, build, shown)
        # The reader recognises the line the cell declares.
        form = printed_form(printed[cell.item.lower()])
        assert form is not None and form.startswith(
            {"line": "line", "line_bare": "line_from_to", "table": "table", "failed": "failed"}[
                cell.at(build)["printed"]
            ]
        ), (cell.id, form)

    fields: Fields = {}
    said: Said = {}
    routes = await _every_route(build, case_id, state_no_sim, work_dir, fields, said)
    for route, answers in routes.items():
        assert set(answers) == set(expected), (route, set(answers) ^ set(expected))
    for cell in cells:
        name = cell.item.lower()
        if name not in expected:
            continue
        want: Answer = expected[name]
        # A cell whose log printed a number that is not finite declares the
        # text (``not_a_number``): its value is null on every route, and each
        # says what the log printed. An undeclared one fails the comparison.
        not_a_number = cell.extra.get("not_a_number")
        if not_a_number is not None:
            assert isinstance(want, list) and any(
                not math.isfinite(v) for v in want if isinstance(v, float)
            ), (cell.id, want)
            want = [None if isinstance(v, float) and not math.isfinite(v) else v for v in want]
            for route in routes:
                statement = said[route].get(name, {}).get("unreadable")
                assert statement is not None, (cell.id, route, said[route])
                assert repr(not_a_number) in statement["reason"], (cell.id, route, statement)
        # A cell whose log printed one value for a run of several steps
        # declares the step, from 1, and how many (``printed_once``): every
        # route says both; no route says it of any other measurement.
        once = cell.extra.get("printed_once")
        for route in routes:
            statement = said[route].get(name, {}).get("printed_once")
            if once is None:
                assert statement is None, (cell.id, route, statement)
                continue
            assert statement is not None, (cell.id, route, said[route])
            assert statement.get("step") == once["step"] - 1, (cell.id, route, statement)
            assert statement["step_count"] == once["of"], (cell.id, route, statement)
        line = cell.at(build)["printed"] in ("line", "line_bare")
        field = "at" if line and GRID.expected(cell, build) == "point" else "value"
        for route, answers in routes.items():
            got = answers[name]
            assert got != "unknown", (cell.id, route)
            if want == "failed":
                assert isinstance(got, list) and got and all(v is None for v in got), (
                    cell.id,
                    route,
                    got,
                )
            else:
                assert _same(got, want), (cell.id, route, got, want)
        for route, named in fields.items():
            assert named[name] == field, (cell.id, route, named[name], field)


@pytest.mark.parametrize(
    "case_id", ["deck/pow_neg_base_cube_root", "deck/pow_neg_base_square_root"]
)
async def test_a_warning_that_quotes_a_param_is_no_measurement_on_any_route(
    case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """LTspice XVII warns of a ``.param`` it cannot resolve by quoting it,
    ``WARNING: Can't resolve .param p1=(-8)**(1/3)``, a line shaped like a
    result line. LTspice prints every measurement's name in lower case on
    both builds (``log/meas_grid_name_upper``), so a line headed
    ``WARNING`` is the message, and no route reports a measurement for a deck
    that measures nothing."""
    text = read_log_text(rec.recorded("ltspice17", f"{case_id}.log"))
    (warning,) = [line for line in text.splitlines() if line.startswith("WARNING: Can't resolve")]
    assert not prints_measurement(warning)
    assert printed_measurements(text) == {}
    routes = await _every_route("ltspice17", case_id, state_no_sim, work_dir)
    for route, answers in routes.items():
        assert answers == {}, (route, answers)


async def test_a_printed_value_that_is_not_a_number_says_what_the_log_printed_on_every_route(
    state_no_sim: SessionState, work_dir: Path
):
    """LTspice 26 prints a stepped .fra run's gain margin, where the loop's
    phase never crosses, as ``infdB`` at ``inf`` (``raw/fra_step``). A reply
    carries no infinite number, so every route gives null, and every route
    says why with what the log printed: the one reason the log decoder gives
    (``log_parser.parse_measurements``), not a count of non-finite values or
    a list of names on one route alone."""
    text = read_log_text(rec.recorded("ltspice26", "raw/fra_step.log"))
    assert "     1\tinfdB\tinf" in text.splitlines()
    said: Said = {}
    routes = await _every_route("ltspice26", "raw/fra_step", state_no_sim, work_dir, said=said)
    for route, answers in routes.items():
        assert answers["gainmargin_1"] == [None, None], (route, answers)
        statement = said[route].get("gainmargin_1", {}).get("unreadable")
        assert statement is not None, (route, said[route])
        assert "'infdB'" in statement["reason"], (route, statement)
        # The phase margin printed beside it is a number, and says nothing.
        assert "phasemargin_1" not in said[route], (route, said[route])


async def test_one_value_printed_for_a_stepped_op_names_its_step_and_the_steps_on_every_route(
    state_no_sim: SessionState, work_dir: Path
):
    """LTspice 26 prints one line for each measurement of a stepped .op, and
    no ``.step`` line (``log/meas_step_op``, three steps). The value is the
    first step's (:func:`test_the_line_a_stepped_op_prints_holds_its_first_steps_value`).
    Every route says so, says the run had three steps whose other values are
    not in the log, and names the recipe that reads them from the results."""
    said: Said = {}
    routes = await _every_route("ltspice26", "log/meas_step_op", state_no_sim, work_dir, said=said)
    for route, answers in routes.items():
        assert set(answers) == {"vout", "iin"}, (route, answers)
        for name in ("vout", "iin"):
            statement = said[route].get(name, {}).get("printed_once")
            assert statement is not None, (route, name, said[route])
            assert statement.get("step") == 0, (route, name, statement)
            assert statement["step_count"] == 3, (route, name, statement)
            assert "'value'" in statement["reason"], (route, name, statement)


async def test_one_value_printed_by_a_build_no_evidence_names_says_its_step_is_not_established(
    state_no_sim: SessionState, work_dir: Path
):
    """The same log with its banner removed names no build: a recording
    establishes the step for LTspice 26 alone, so the value keeps
    no step. The recipe's row names none either, and each route says the run
    had three steps and how to read them."""
    log = work_dir / "meas_step_op.log"
    lines = rec.recorded("ltspice26", "log/meas_step_op.log").read_bytes().splitlines(True)
    assert lines[0].startswith(b"LTspice 26")
    log.write_bytes(b"".join(lines[1:]))
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=_RECIPE,
        all_steps=True,
        include=_EACH_STEP,
    )
    (row,) = reply["results"]["m"]["values"]
    assert "step_index" not in row, row
    read = await asyncio.to_thread(lambda: SyncApi(state_no_sim).measurements(path=log))
    for name in ("vout", "iin"):
        for statement in (
            row["value"]["printed_once"][name],
            read["measurements"][name]["printed_once"],
        ):
            assert "step" not in statement, statement
            assert statement["step_count"] == 3, statement
            assert "not established" in statement["reason"], statement
            assert "'value'" in statement["reason"], statement


async def test_a_step_asked_of_a_stepped_op_log_is_refused_with_what_the_log_holds(
    state_no_sim: SessionState, work_dir: Path
):
    """A step chosen of a log that printed one value for a run of three steps
    (``log/meas_step_op``) is refused with what ``printed_once`` says: three
    steps, the printed value is the first's, and the recipe that reads the
    others. Through the tool and the Python API, by path and by job; and a
    step index the log's table does not hold, read in process, says the
    same."""
    deck, raw, log = _stage("ltspice26", "log/meas_step_op", work_dir)
    job = make_experiment_job(state_no_sim, job_id="exp_step_op", raw=raw, netlist=deck)
    job.simulator = "LTspice"
    job.cases[0].simulator_version = rec.manifest("ltspice26")["reported_build"]
    api = SyncApi(state_no_sim)
    for source in ({"path": str(log)}, {"job_id": "exp_step_op"}):
        arguments = {
            "sources": [{**source, "label": "run"}],
            "recipes": _RECIPE,
            "step": {"axis": "v", "value": 2},
        }
        tool = await dispatched(state_no_sim, "analyze_results", **arguments)
        through_api = await asyncio.to_thread(lambda a=arguments: api.analyze_results(**a))
        for route, reply in (("tool", tool), ("api", through_api)):
            (message,) = [failure["message"] for failure in reply["failures"]]
            assert "3 steps" in message, (source, route, message)
            assert "step 0's value, the first" in message, (source, route, message)
            assert "'value' recipe" in message, (source, route, message)
            assert "1 step" not in message, (source, route, message)
    table = answered_measurements(
        parse_measurements(rec.recorded("ltspice26", "log/meas_step_op.log")),
        Producer("ltspice", None, None),
    )
    with pytest.raises(ResultError) as refused:
        measurements_at_step(table, 2)
    assert "3 steps" in str(refused.value) and "1 step(s)" not in str(refused.value)
    assert measurements_at_step(table, 0)["measurements"]["vout"]["values"] == [
        table["measurements"]["vout"]["values"][0]
    ]


async def test_the_line_a_stepped_op_prints_holds_its_first_steps_value(
    state_no_sim: SessionState, work_dir: Path
):
    """The evidence for which step the line holds. ``raw/step_op`` runs the
    deck of ``log/meas_step_op`` without its ``.meas`` cards (the same
    divider, ``.step param v list 1 2 3``), and LTspice 26's results file of
    it holds V(out) and I(V1) at each step. The values the measurement log
    printed once are the first step's, read the way the reply names (the
    value recipe over every step), and no other step's."""
    printed = parse_measurements(rec.recorded("ltspice26", "log/meas_step_op.log"))
    raw = work_dir / "step_op.raw"
    shutil.copyfile(rec.recorded("ltspice26", "raw/step_op.raw"), raw)
    shutil.copyfile(rec.recorded("ltspice26", "raw/step_op.log"), raw.with_suffix(".log"))
    for name, expr in (("vout", "V(out)"), ("iin", "I(V1)")):
        reply = await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(raw), "label": "run"}],
            recipes=[{"key": "v", "metric": "value", "expr": expr}],
            all_steps=True,
            include={"fields": ["value", "step_index"]},
        )
        steps = [row["value"]["value"] for row in reply["results"]["v"]["values"]]
        assert len(steps) == 3, reply
        (value,) = printed["measurements"][name]["values"]
        assert value is not None
        matching = [
            i for i, number in enumerate(steps) if math.isclose(number, value, rel_tol=1e-6)
        ]
        assert matching == [0], (name, value, steps)


# ngspice -------------------------------------------------------------------

_NGSPICE_LINE = re.compile(r"^(\w+)\s+=\s+(\S+)", re.M)


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
def test_ngspice_prints_a_crossing_as_the_value_and_the_server_reads_no_such_line(
    state_no_sim: SessionState, work_dir: Path
):
    """ngspice prints each measurement as ``name = value``, a WHEN's crossing
    as its value, where LTspice prints the level and the crossing after
    ``AT``. The log decoder reads no such line (a run the server starts with a
    raw file prints none: ``No .measure possible in batch mode``), so the
    ngspice reading in ``log_parser`` holds no printed form, and no route
    reads a field of one."""
    import subprocess

    version = subprocess.run(
        ["ngspice", "-v"], capture_output=True, text=True, check=False, timeout=60
    ).stdout
    print("ngspice that ran:", next(line for line in version.splitlines() if "ngspice" in line))
    deck = work_dir / "forms.cir"
    deck.write_text(
        "* WHEN and FIND WHEN in ngspice\n"
        "V1 in 0 PULSE(0 1 0 1u 1u 1 2)\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        ".tran 1u 1m\n"
        ".meas tran m_when WHEN V(out)=0.5\n"
        ".meas tran m_findwhen FIND V(in) WHEN V(out)=0.5\n"
        ".end\n",
        encoding="ascii",
    )
    log = deck.with_suffix(".log")
    subprocess.run(
        ["ngspice", "-n", "-b", "-o", str(log), str(deck)],
        cwd=work_dir,
        capture_output=True,
        check=False,
        timeout=60,
    )
    printed = dict(_NGSPICE_LINE.findall(read_log_text(log)))
    assert float(printed["m_when"]) == pytest.approx(100e-6 * math.log(2), rel=0.02)
    assert float(printed["m_findwhen"]) == pytest.approx(1.0)
    assert printed_measurements(read_log_text(log)) == {}
    assert SyncApi(state_no_sim).measurements(path=log)["measurements"] == {}


# The decision itself ---------------------------------------------------------

_LT26 = Producer("ltspice", "LTspice 26.1.1 for Windows", None)
_UNKNOWN = Producer(None, None, None)


@pytest.mark.parametrize(
    ("form", "answer"),
    [
        ("line_AT", "at"),
        ("line_at", "value"),
        ("line_from_to", "value"),
        ("line", "value"),
        ("table_level", "value"),
        ("table_at", "value"),
        ("table_from_to", "value"),
        ("table", "value"),
    ],
)
def test_a_recorded_form_names_its_field_for_any_ltspice_or_an_unknown_writer(
    form: str, answer: str
):
    """The two recorded builds read every form alike, so a build that is not
    recorded and a writer that is not known read it the same: no recorded
    producer but LTspice prints these forms."""
    for producer in (_LT26, Producer("ltspice", "LTspice 24.0.9 for Windows", None), _UNKNOWN):
        assert measurement_answer(form, producer) == MeasurementAnswer(answer)  # type: ignore[arg-type]


def test_a_deriv_when_is_answered_by_its_printed_form_as_a_deriv_at_is():
    """Both builds print a DERIV ... WHEN as a DERIV ... AT: the derivative
    after ``=`` and the point after a lowercase ``at`` (``log/meas_deriv_when``),
    so its card is any other card's and its result is its value."""
    deck_cards = measurement_cards(INPUTS / "log/meas_deriv_when.cir")
    assert deck_cards["d_when"] == deck_cards["d_at"] == "value"
    for form in ("line_at", "table_at"):
        assert measurement_answer(form, _LT26, deck_cards["d_when"]).field == "value"


def test_a_card_the_printed_form_contradicts_leaves_the_answer_unknown():
    """A WHEN with no FIND is recorded only after an uppercase AT or under a
    level heading, and any other card only otherwise."""
    assert measurement_answer("line_AT", _LT26, "when").field == "at"
    assert measurement_answer("table_level", _LT26, "when").field == "value"
    for form, card in (("line_at", "when"), ("line_AT", "value"), ("table_level", "value")):
        answer = measurement_answer(form, _LT26, card)  # type: ignore[arg-type]
        assert answer.field is None and "deck's card" in str(answer.unknown)


def test_an_unrecorded_form_or_simulator_leaves_the_answer_unknown():
    for producer in (Producer("qspice", None, None), Producer("xyce", None, None)):
        answer = measurement_answer("line_AT", producer, "when")
        assert answer.field is None and "no recording" in str(answer.unknown)
    for form in (None, "something_else"):
        answer = measurement_answer(form, _LT26)
        assert answer.field is None and "no recording" in str(answer.unknown)
    answer = measurement_answer("line_AT", Producer("ngspice", None, None))
    assert answer.field is None


def test_a_failed_measurement_holds_no_number_in_either_field():
    assert measurement_answer("failed", _UNKNOWN).field == "value"


async def test_a_table_with_a_level_beside_a_point_is_a_form_no_recording_shows(
    state_no_sim: SessionState, work_dir: Path
):
    """A stepped WHEN's table printed with its level in the first column and
    the crossing beside it under ``at`` (``ltspice_step_when.log``, written by
    hand): LTspice prints the crossing alone under the level heading
    (``log/meas_when_threshold_tran_step``). The field holding the result is
    not known, so no number is given for it, and the measurement beside it
    reads."""
    log = work_dir / "step_when.log"
    shutil.copyfile(FIXTURES_DIR / "ltspice_step_when.log", log)
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=_RECIPE,
        include={"fields": ["value"]},
    )
    (row,) = reply["results"]["m"]["values"]
    value = row["value"]
    assert set(value["answer_unknown"]) == {"tcross"}
    assert "aggregated_field" not in value["stats"]["tcross"]
    assert value["stats"]["tcross"]["valid_count"] == 0
    assert value["stats"]["vfinal"]["aggregated_field"] == "value"
    assert value["stats"]["vfinal"]["valid_count"] == 3
    read = await asyncio.to_thread(lambda: SyncApi(state_no_sim).measurements(path=log))
    assert "answer" not in read["measurements"]["tcross"]
    assert read["measurements"]["vfinal"]["answer"] == "value"


def test_a_find_of_a_derivative_where_a_condition_is_met_is_a_find(work_dir: Path):
    """``FIND deriv(V(out)) WHEN ...`` is a FIND, whose result is its value,
    though its card holds both DERIV and WHEN."""
    deck = work_dir / "find_deriv.cir"
    deck.write_text(
        "* t\nV1 a 0 1\n"
        ".meas tran f_d FIND deriv(V(a)) WHEN V(a)=0.5\n"
        ".meas tran d_w DERIV V(a) WHEN V(a)=0.5\n"
        ".tran 1m\n.end\n",
        encoding="ascii",
    )
    cards = measurement_cards(deck)
    assert cards == {"f_d": "value", "d_w": "value"}
    assert measurement_answer("line_at", _LT26, cards["f_d"]).field == "value"


def test_a_deriv_when_from_a_simulator_with_no_recordings_is_unknown():
    answer = measurement_answer("line_at", Producer("qspice", None, None), "value")
    assert answer.field is None and "no recording" in str(answer.unknown)


def test_a_null_whose_field_is_unknown_has_that_one_reason():
    entry = {"values": [None], "answer_unknown": "why", "unreadable": {"line": "", "reason": "x"}}
    assert result_unreadable(entry) is None
    assert result_unreadable({"values": [None], "answer": "at", "unreadable": {}}) is None
    assert result_unreadable({"values": [None], "answer": "value", "unreadable": {"r": 1}})


# A 0 in a stepped table --------------------------------------------------------

_XVII = Producer("ltspice", rec.manifest("ltspice17")["reported_build"], None)


def _w_expr(build: str, producer: Producer, *, banner: bool = True) -> dict[str, Any]:
    """``w_expr`` of ``log/meas_when_step_forms`` as read for ``producer``:
    a WHEN that crosses at 0 s in both steps, printed ``0`` by both builds."""
    table = cast(Any, parse_measurements(rec.recorded(build, "log/meas_when_step_forms.log")))
    assert table["measurements"]["w_expr"]["values"] == [0.0, 0.0]
    if not banner:
        table.pop("printed_by", None)
    return cast(Any, answered_measurements(table, producer))["measurements"]["w_expr"]


@pytest.mark.parametrize(
    "producer",
    [
        _LT26,
        Producer("ltspice", "LTspice 24.1.9 for Windows", None),
        Producer(None, "LTspice 25.0.1 for Windows", None),
        _UNKNOWN,  # the log's own banner names LTspice 26
    ],
)
def test_a_zero_a_current_build_printed_in_a_stepped_table_is_its_result(producer: Producer):
    """LTspice 24 and later print ``failed`` for a step that fails, so a 0
    they print is the result: here a crossing at 0 s."""
    entry = _w_expr("ltspice26", producer)
    assert entry["values"] == [0.0, 0.0]
    assert "not_established" not in entry


@pytest.mark.parametrize(
    ("build", "producer", "who"),
    [
        ("ltspice17", _XVII, "LTspice XVII printed 0"),
        ("ltspice17", _UNKNOWN, "may have been written by LTspice XVII"),
        ("ltspice26", _UNKNOWN, "may have been written by LTspice XVII"),
        ("ltspice17", Producer("ltspice", None, "scad3.exe"), "LTspice IV"),
    ],
)
def test_a_zero_xvii_may_have_printed_in_a_stepped_table_is_not_established(
    build: str, producer: Producer, who: str
):
    """XVII prints a crossing at 0 s as 0, as it prints a step whose WHEN
    finds no crossing, so the two read the same. A log that names no build
    may be XVII's (the LTspice 26 log here with its banner removed)."""
    entry = _w_expr(build, producer, banner=False)
    assert entry["values"] == [None, None]
    assert entry["not_established"]["steps"] == [0, 1]
    assert who in entry["not_established"]["reason"]
    assert "crossing at 0 s" in entry["not_established"]["reason"]


def test_a_zero_xvii_printed_in_another_stepped_table_says_no_recording_shows_its_failure():
    """Only a stepped WHEN's failure is recorded on XVII; for a FIND's table
    the reason says that no recording shows how a failed step prints."""
    table = cast(Any, parse_measurements(rec.recorded("ltspice17", "log/meas_step.log")))
    entry = table["measurements"]["s_find"]
    assert entry["printed_form"] == "table_at"
    entry["values"] = [0.0, *entry["values"][1:]]
    read = cast(Any, answered_measurements(table, _XVII))["measurements"]["s_find"]
    assert read["values"][0] is None
    assert read["not_established"]["steps"] == [0]
    assert "no recording shows" in read["not_established"]["reason"]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_build_a_log_names_on_its_first_line(build: str):
    """LTspice 26 opens its log with its banner; XVII opens it with its
    ``Circuit:`` line and names no build."""
    path = rec.recorded(build, "log/meas_step.log")
    named = log_banner_build(read_log_text(path))
    table = cast(Any, parse_measurements(path))
    if rec.generation(build) == "xvii":
        assert named is None
        assert "printed_by" not in table
    else:
        assert named == rec.manifest(build)["reported_build"]
        assert table["printed_by"] == named
    assert "printed_by" not in cast(Any, answered_measurements(table, _UNKNOWN))


@pytest.mark.parametrize(
    ("case_id", "name"),
    [
        ("log/meas_grid_dc_step_value", "trig_targ_v"),
        ("log/meas_grid_ac_step_step_outcomes", "max_sf"),
    ],
)
async def test_a_failed_step_of_a_table_with_bounds_reads_on_every_route(
    case_id: str, name: str, state_no_sim: SessionState, work_dir: Path
):
    """LTspice 26 prints a step that failed in a table whose header names a
    ``FROM`` and a ``TO`` column as ``failed`` alone, with no bound beside it
    (``     2\tfailed`` under ``step\ttrig_targ_v\tFROM\tTO``). The step reads
    as failed and the one before it as its number, on every route, and the
    run's other measurements read too."""
    text = read_log_text(rec.recorded("ltspice26", f"{case_id}.log"))
    lines = text.splitlines()
    header = lines.index(f"Measurement: {name}")
    assert lines[header + 1].split("\t")[-2:] == ["FROM", "TO"]
    assert lines[header + 3].split("\t") == ["     2", "failed"]
    routes = await _every_route("ltspice26", case_id, state_no_sim, work_dir)
    for route, answers in routes.items():
        assert set(answers) == set(printed_measurements(text)), (route, answers)
        first, second = answers[name]
        assert isinstance(first, float), (route, answers[name])
        assert second is None, (route, answers[name])


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_stepped_run_whose_every_measurement_failed_reads_each_step_on_every_route(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """A stepped run whose every measurement failed at every step prints each
    as one ``Measurement "max_f" FAIL'ed`` line and no table
    (``log/meas_grid_tran_step_failed``, two ``.step`` lines). Each step's
    row is read and each measurement failed at it, on every route, rather
    than the measurements recipe refusing to pair a step with no table row."""
    case_id = "log/meas_grid_tran_step_failed"
    text = read_log_text(rec.recorded(build, f"{case_id}.log"))
    assert 'Measurement "max_f" FAIL\'ed' in text.splitlines()
    assert sum(line.startswith(".step ") for line in text.splitlines()) == 2
    routes = await _every_route(build, case_id, state_no_sim, work_dir)
    for route, answers in routes.items():
        assert answers["max_f"] == [None, None], (route, answers)
