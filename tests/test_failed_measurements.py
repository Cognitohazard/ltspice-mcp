"""A measurement is named failed where LTspice printed it failed, on every route.

LTspice prints ``Measurement "name" FAIL'ed`` for a measurement it could not
take, and LTspice 24 and later print ``failed`` in the row of a stepped table
for a step where it could not. Every reply that reports measurements names
exactly those in ``failed_measurements``: the log's measurements read through
the Python API and ``inspect``, the run summary, the measurements recipe's
row, and the analysis attached to ``run_experiments`` as its receipt and
``jobs`` return it. Each value ``inspect`` pages is marked ``failed`` where
LTspice printed it failed. A recipe row read at one step names those that
failed at that step; the summary's measurements are the whole run's at
whichever step it reads.

A measurement with no number for another reason is not among them. One whose
printed value is not a number (a WHEN's threshold written as an expression,
recorded as ``log/meas_when_threshold_ac``) is under ``unreadable``; one whose
printed value is not finite (a stepped FRA's gain margin of ``infdB``,
recorded as ``raw/fra_step``) is under a recipe row's ``not_finite``.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
from ltspice_mcp.lib.raw_parser import build_simulation_summary
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import LIVENESS_S, SyncApi, dispatched, replaying_simulator
from tests.ltspice_recorder import INPUTS

pytestmark = pytest.mark.asyncio

_FAIL_LINE = re.compile(r"""^Measurement "(\w+)" FAIL'ed""", re.M)
_WHEN_LINE = re.compile(r"^(\w+): [^\r\n=]*=([^\r\n]*?) AT ", re.M)
_TABLE = re.compile(r"^Measurement: (\S+)\r?\n(.*?)(?=\r?\n\s*\r?\n|\Z)", re.M | re.S)
_FAILED_ROW = re.compile(r"^\s*(\d+)\tfailed\s*$", re.M)


def _printed_failed(log: Path) -> list[str]:
    """The measurements the log printed FAIL'ed, in the order printed."""
    return _FAIL_LINE.findall(read_log_text(log))


def _printed_failed_anywhere(text: str) -> set[str]:
    """The measurements a log printed as failed, in lower case: FAIL'ed, or
    with ``failed`` for a row of their stepped table."""
    tables = {name for name, rows in _TABLE.findall(text) if _FAILED_ROW.search(rows)}
    return {name.lower() for name in {*_FAIL_LINE.findall(text), *tables}}


def _printed_failed_rows(text: str) -> set[tuple[str, int]]:
    """Each row of a stepped table a log printed ``failed`` in, as the
    measurement in lower case and the row's index from 0 (the table numbers
    its rows from 1)."""
    return {
        (name.lower(), int(number) - 1)
        for name, rows in _TABLE.findall(text)
        for number in _FAILED_ROW.findall(rows)
    }


def _measurement_logs() -> list[str]:
    """Every recorded log that prints a measurement, as ``build/case_id``."""
    return sorted(
        f"{build}/{path.relative_to(rec.recorded(build, '')).with_suffix('').as_posix()}"
        for build in rec.BUILDS
        for path in rec.recorded(build, "").rglob("*.log")
        if path.stat().st_size and printed_measurements(read_log_text(path))
    )


def _printed_unreadable(log: Path) -> set[str]:
    """The WHENs whose threshold the log printed as other than a number."""
    unreadable = set()
    for name, threshold in _WHEN_LINE.findall(read_log_text(log)):
        try:
            float(threshold)
        except ValueError:
            unreadable.add(name)
    return unreadable


def _failed(value: dict[str, Any]) -> list[str]:
    """A reply's ``failed_measurements``, which is left out when empty."""
    return value.get("failed_measurements", [])


def _row(reply: dict[str, Any], key: str) -> dict[str, Any]:
    (row,) = reply["results"][key]["values"]
    return row["value"]


_RECIPES = [{"key": "m", "metric": "measurements"}, {"key": "s", "metric": "summary"}]
#: A summary's default row holds its scalar facts; its lists are read by name.
_WHOLE = {"fields": ["value"]}
#: The same, with the step each row of an ``all_steps`` call reads.
_EACH_STEP = {"fields": ["value", "step_index"]}
#: ``log/meas_step`` steps r over these, one row of each table a step.
_MEAS_STEP_R = ("1k", "2k", "20k")


def _failed_by_step(reply: dict[str, Any], key: str) -> list[list[str]]:
    """Each step's ``failed_measurements`` in an ``all_steps`` reply, by step."""
    rows = {row["step_index"]: row["value"] for row in reply["results"][key]["values"]}
    assert sorted(rows) == list(range(len(rows)))
    return [_failed(rows[step]) for step in sorted(rows)]


def _inspected(item: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """An ``inspect`` measurements page's ``failed_measurements``, and at each
    step the measurements whose value there it marks ``failed``."""
    assert item["ok"], item
    rows = item["data"]["measurements"]
    assert len(rows) == item["page"]["total"]  # the whole log on one page
    steps = max(row["value_index"] for row in rows) + 1
    marked = [
        [row["measurement"] for row in rows if row["value_index"] == step and row.get("failed")]
        for step in range(steps)
    ]
    return _failed(item["data"]), marked


def _entry_failed_by_step(read: dict[str, Any], steps: int) -> list[list[str]]:
    """At each step, the measurements of the log's measurements (the Python
    API's ``measurements()``) LTspice printed failed there: one named failed
    whose entry lists that step in ``failed_steps``, or that lists none."""
    return [
        [
            name
            for name in read["failed_measurements"]
            if step in read["measurements"][name].get("failed_steps", range(steps))
        ]
        for step in range(steps)
    ]


def _stepped_run(build: str, directory: Path) -> tuple[Path, Path, Path]:
    """``log/meas_step``'s deck and log in ``directory``, with a stepped raw
    beside the log: the one ``build`` wrote for ``raw/step_tran``, the same RC
    stepped over three resistances (4k third, where meas_step steps 20k), as
    the recording keeps meas_step's log alone. Every reply reads the
    measurements and the steps from the log."""
    deck = directory / "meas_step.cir"
    shutil.copyfile(INPUTS / "log/meas_step.cir", deck)
    log = deck.with_suffix(".log")
    shutil.copyfile(rec.recorded(build, "log/meas_step.log"), log)
    raw = deck.with_suffix(".raw")
    shutil.copyfile(rec.recorded(build, "raw/step_tran.raw"), raw)
    return deck, raw, log


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_every_route_names_the_measurements_ltspice_printed_failed(
    build: str,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``log/meas_when_threshold_ac``: XVII prints ``f_param`` FAIL'ed, and
    both builds print the thresholds of ``f_expr`` and ``f_meas`` (and
    LTspice 26 that of ``f_param``) as expressions. The FAIL'ed one is named
    failed on every route, and the unreadable ones on none."""
    recorded_log = rec.recorded(build, "log/meas_when_threshold_ac.log")
    failed, unreadable = _printed_failed(recorded_log), _printed_unreadable(recorded_log)
    assert unreadable and not set(failed) & unreadable
    assert bool(failed) == (rec.generation(build) == "xvii")
    log = work_dir / "when.log"
    shutil.copyfile(recorded_log, log)
    raw = log.with_suffix(".raw")
    shutil.copyfile(rec.recorded(build, "raw/ac.raw"), raw)
    deck = work_dir / "when.cir"
    shutil.copyfile(INPUTS / "log/meas_when_threshold_ac.cir", deck)
    named: dict[str, list[str]] = {}
    rows: dict[str, dict[str, Any]] = {}

    # A log read by its path: the recipe's row and the run's summary. The
    # unreadable ones are WHENs, whose result is the crossing printed after
    # AT, so they read; only the failed one has no number.
    by_path = await dispatched(
        state_with_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=[_RECIPES[0]],
    )
    rows["recipe row, by path"] = row = _row(by_path, "m")
    assert {name for name, number in row["measured"].items() if number is None} == set(failed)
    assert "unreadable" not in row
    summary = await dispatched(
        state_with_sim,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[_RECIPES[1]],
        include=_WHOLE,
    )
    named["summary, by path"] = _failed(_row(summary, "s"))
    page = await dispatched(
        state_with_sim,
        "inspect",
        queries=[{"kind": "results", "view": "measurements", "path": str(log)}],
    )
    (item,) = page["results"]
    named["inspect, by path"] = _failed(item["data"])

    # A job: the analysis attached to its receipt, and as jobs returns it.
    replaying_simulator(monkeypatch, raw, log)
    receipt = await dispatched(
        state_with_sim,
        "run_experiments",
        request_id="failed-measurements",
        circuits=[{"path": str(deck), "id": "when"}],
        analyze={"recipes": _RECIPES, "include": _WHOLE},
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    attached = receipt["analysis"]["result"]
    rows["receipt, recipe row"] = _row(attached, "m")
    named["receipt, summary"] = _failed(_row(attached, "s"))
    status = await dispatched(state_with_sim, "jobs", action="status", job_id=receipt["job_id"])
    # jobs returns the analysis' default rows, where a summary keeps its
    # scalar facts and a measurements row its names.
    rows["jobs, recipe row"] = _row(status["analysis"]["result"], "m")

    # The Python API, over the log and over the job.
    api = SyncApi(state_with_sim)
    job = [{"job_id": receipt["job_id"], "label": "job"}]
    read = await asyncio.to_thread(lambda: api.measurements(log))
    named["api measurements, by path"] = read["failed_measurements"]
    read = await asyncio.to_thread(lambda: api.measurements(job_id=receipt["job_id"]))
    named["api measurements, by job"] = read["failed_measurements"]
    analyzed = await asyncio.to_thread(
        lambda: api.analyze_results(sources=job, recipes=_RECIPES, include=_WHOLE)
    )
    rows["api recipe row, by job"] = _row(analyzed, "m")
    named["api summary, by job"] = _failed(_row(analyzed, "s"))

    named.update({route: _failed(value) for route, value in rows.items()})
    assert named == dict.fromkeys(named, failed)
    # Each row's nulls are named by why, and none of these printed a number
    # that is not finite.
    for route, value in rows.items():
        assert "not_finite" not in value, route
        assert set(failed) <= {
            name for name, number in value["measured"].items() if number is None
        }


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_step_ltspice_printed_failed_is_named_on_that_steps_row(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``log/meas_step``: with 20k the output never reaches 0.5 V, and
    LTspice 26 prints ``failed`` in the row of that step (the third) of
    ``s_when``'s table. The log's measurements name it failed at that step,
    and so do that step's row, and no other step's, and the row over every
    step. XVII prints 0 there, which names no failure."""
    log = work_dir / "meas_step.log"
    shutil.copyfile(rec.recorded(build, "log/meas_step.log"), log)
    failed = ["s_when"] if rec.generation(build) == "current" else []
    api = SyncApi(state_no_sim)

    read = await asyncio.to_thread(lambda: api.measurements(log))
    assert read["failed_measurements"] == failed
    assert read["measurements"]["s_when"].get("failed_steps") == ([2] if failed else None)
    page = await dispatched(
        state_no_sim,
        "inspect",
        queries=[{"kind": "results", "view": "measurements", "path": str(log)}],
    )
    assert _failed(page["results"][0]["data"]) == failed
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=[{"key": "m", "metric": "measurements"}],
        all_steps=True,
        include={"per_run": True},
    )
    rows = [row["value"] for row in reply["results"]["m"]["per_run"]["items"]]
    assert [_failed(row) for row in rows] == [[], [], failed]
    assert [row["measured"]["s_when"] is None for row in rows] == [False, False, bool(failed)]
    assert all("not_finite" not in row for row in rows)
    whole = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=[{"key": "m", "metric": "measurements"}],
    )
    assert _failed(_row(whole, "m")) == failed


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_measurement_failed_at_one_step_is_named_alike_on_every_route(
    build: str,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``log/meas_step``, where LTspice 26 prints ``s_when`` failed at the
    third step alone, through every route that reports measurements:
    ``analyze_results`` over the whole run, at each step and with each step
    selected; ``inspect``; the analysis ``run_experiments`` attaches, in its
    receipt and as ``jobs`` returns it; and the Python API. Each names
    ``s_when`` failed where it reads the whole run, and at the third step
    where it reads one step. XVII prints 0 there, and every route names
    none."""
    deck, raw, log = _stepped_run(build, work_dir)
    failed = ["s_when"] if rec.generation(build) == "current" else []
    each_step = [[], [], failed]
    # What each route names failed over the whole run, and at each step.
    whole: dict[str, list[str]] = {}
    stepped: dict[str, list[list[str]]] = {}
    source = [{"path": str(raw), "label": "run"}]

    reply = await dispatched(
        state_with_sim, "analyze_results", sources=source, recipes=_RECIPES, include=_WHOLE
    )
    whole["recipe row, whole run"] = _failed(_row(reply, "m"))
    whole["summary, no step selected"] = _failed(_row(reply, "s"))
    reply = await dispatched(
        state_with_sim,
        "analyze_results",
        sources=source,
        recipes=_RECIPES,
        all_steps=True,
        include=_EACH_STEP,
    )
    stepped["recipe rows, all_steps"] = _failed_by_step(reply, "m")
    # The summary's measurements are the whole run's at any step it reads
    # (test_a_summary_names_failures_at_any_step_and_a_step_row_its_own).
    for step, names in enumerate(_failed_by_step(reply, "s")):
        whole[f"summary, all_steps, step {step}"] = names
    selected = []
    for step, value in enumerate(_MEAS_STEP_R):
        reply = await dispatched(
            state_with_sim,
            "analyze_results",
            sources=source,
            recipes=_RECIPES,
            step={"axis": "r", "value": value},
            include=_WHOLE,
        )
        selected.append(_failed(_row(reply, "m")))
        whole[f"summary, step {step} selected"] = _failed(_row(reply, "s"))
    stepped["recipe row, each step selected"] = selected
    page = await dispatched(
        state_with_sim,
        "inspect",
        queries=[{"kind": "results", "view": "measurements", "path": str(log)}],
    )
    whole["inspect, by path"], stepped["inspect rows, by path"] = _inspected(page["results"][0])

    replaying_simulator(monkeypatch, raw, log)
    receipt = await dispatched(
        state_with_sim,
        "run_experiments",
        request_id="failed-at-one-step",
        circuits=[{"path": str(deck), "id": "meas_step"}],
        analyze={"recipes": _RECIPES, "include": _EACH_STEP, "all_steps": True},
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    job_id = receipt["job_id"]
    attached = receipt["analysis"]["result"]
    stepped["receipt, recipe rows"] = _failed_by_step(attached, "m")
    for step, names in enumerate(_failed_by_step(attached, "s")):
        whole[f"receipt, summary at step {step}"] = names
    # jobs returns the analysis' default rows, where a measurements row keeps
    # its names and a summary only its scalar facts.
    status = await dispatched(state_with_sim, "jobs", action="status", job_id=job_id)
    stepped["jobs, recipe rows"] = _failed_by_step(status["analysis"]["result"], "m")
    page = await dispatched(
        state_with_sim,
        "inspect",
        queries=[{"kind": "results", "view": "measurements", "job_id": job_id}],
    )
    whole["inspect, by job"], stepped["inspect rows, by job"] = _inspected(page["results"][0])

    api = SyncApi(state_with_sim)
    for route, target in (("by path", {"path": log}), ("by job", {"job_id": job_id})):
        read = await asyncio.to_thread(lambda target=target: api.measurements(**target))
        whole[f"api measurements, {route}"] = read["failed_measurements"]
        stepped[f"api measurements, {route}"] = _entry_failed_by_step(read, len(_MEAS_STEP_R))
    job = [{"job_id": job_id, "label": "job"}]
    analyzed = await asyncio.to_thread(
        lambda: api.analyze_results(
            sources=job, recipes=_RECIPES, all_steps=True, include=_EACH_STEP
        )
    )
    stepped["api recipe rows, by job"] = _failed_by_step(analyzed, "m")
    for step, names in enumerate(_failed_by_step(analyzed, "s")):
        whole[f"api summary at step {step}"] = names
    inspected = await asyncio.to_thread(
        lambda: api.inspect(
            queries=[{"kind": "results", "view": "measurements", "job_id": job_id}]
        )
    )
    whole["api inspect, by job"], stepped["api inspect rows, by job"] = _inspected(
        inspected["results"][0]
    )

    assert whole == dict.fromkeys(whole, failed)
    assert stepped == dict.fromkeys(stepped, each_step)


async def test_a_summary_names_failures_at_any_step_and_a_step_row_its_own(
    state_no_sim: SessionState, work_dir: Path
):
    """The run summary and a measurements row answer at a selected step for
    different spans, by design. A row read at one step is that step's, so it
    names what failed there. The summary's measurements are the whole run's,
    every step's value with the steps a measurement failed at, whichever step
    selects its range and point count, so its ``failed_measurements`` names
    what failed at any step. ``log/meas_step``: LTspice 26 prints ``s_when``
    failed at the third step, and the second is selected."""
    if not rec.has("ltspice26", "log/meas_step.log"):
        pytest.skip("LTspice 26 recorded no stepped measurement tables")
    _deck, raw, _log = _stepped_run("ltspice26", work_dir)
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=_RECIPES,
        step={"axis": "r", "value": "2k"},
        include=_WHOLE,
    )
    row, summary = _row(reply, "m"), _row(reply, "s")
    assert row["measured"]["s_when"] is not None
    assert "failed_measurements" not in row
    entry = summary["measurements"]["s_when"]
    assert len(entry["values"]) == summary["step_count"] == len(_MEAS_STEP_R)
    assert entry["failed_steps"] == [2]
    assert summary["failed_measurements"] == ["s_when"]


@pytest.mark.parametrize("recorded", _measurement_logs())
async def test_inspect_marks_each_value_ltspice_printed_failed(
    recorded: str, state_no_sim: SessionState, work_dir: Path
):
    """In every recorded log that prints a measurement, ``inspect``'s
    measurements view marks a value ``failed`` exactly where the log printed
    it failed: each value of a measurement printed FAIL'ed, and each row of a
    stepped table printed ``failed``. A value it marks holds no number."""
    build, _, case_id = recorded.partition("/")
    log = work_dir / f"{Path(case_id).name}.log"
    shutil.copyfile(rec.recorded(build, f"{case_id}.log"), log)
    text = read_log_text(log)
    whole = {name.lower() for name in _FAIL_LINE.findall(text)}
    rows_printed = _printed_failed_rows(text)

    page = await dispatched(
        state_no_sim,
        "inspect",
        queries=[{"kind": "results", "view": "measurements", "path": str(log)}],
    )
    (item,) = page["results"]
    assert item["ok"], item
    rows = item["data"]["measurements"]
    assert len(rows) == item["page"]["total"]
    for row in rows:
        name = row["measurement"].lower()
        printed = name in whole or (name, row["value_index"]) in rows_printed
        assert row.get("failed", False) == printed, row
        if printed:
            assert "value" not in row, row
    marked = {(row["measurement"].lower(), row["value_index"]) for row in rows if "failed" in row}
    assert {name for name, _ in marked} == _printed_failed_anywhere(text)


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_measurement_taken_at_some_steps_was_produced(build: str, tmp_path: Path):
    """A run summary reconciles the deck's .meas against the log: one LTspice
    printed failed at every step was not produced, and one it printed failed
    at some steps (``s_when`` in LTspice 26's ``log/meas_step``) and took at
    the others was. The summary still names it failed."""
    raw = rec.decode(build, "raw/tran", tmp_path / "raw").raw
    requested = {"meas": ["s_when", "s_max"], "four": []}
    stepped = rec.decode_log(build, "log/meas_step", tmp_path / "stepped")
    summary = build_simulation_summary(raw, stepped, requested=requested)
    current = rec.generation(build) == "current"
    assert summary.get("failed_measurements", []) == (["s_when"] if current else [])
    assert not [o for o in summary["observations"] if o["code"] == "unmet_request"]

    failed = rec.decode_log(build, "log/meas_failed", tmp_path / "failed")
    summary = build_simulation_summary(raw, failed, requested={"meas": ["never"], "four": []})
    (unmet,) = [o for o in summary["observations"] if o["code"] == "unmet_request"]
    assert unmet["evidence"] == {"name": "never", "request_kind": "meas", "reason": "failed"}


async def test_a_value_printed_as_not_finite_is_not_named_failed(
    state_no_sim: SessionState, work_dir: Path
):
    """``raw/fra_step``: LTspice 26 prints each step's gain margin as
    ``infdB``, at ``inf``, a margin that is infinite and was taken. No route
    names it failed; a step's row names it ``not_finite``."""
    if not rec.has("ltspice26", "raw/fra_step.log"):
        pytest.skip("LTspice 26 recorded no stepped FRA run")
    log = work_dir / "fra_step.log"
    shutil.copyfile(rec.recorded("ltspice26", "raw/fra_step.log"), log)
    api = SyncApi(state_no_sim)

    read = await asyncio.to_thread(lambda: api.measurements(log))
    assert read["failed_measurements"] == []
    assert read["measurements"]["gainmargin_1"]["values"] == [None, None]
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=[{"key": "m", "metric": "measurements"}],
        all_steps=True,
        include={"per_run": True},
    )
    rows = reply["results"]["m"]["per_run"]["items"]
    assert len(rows) == 2
    for row in rows:
        value = row["value"]
        assert value["measured"]["gainmargin_1"] is None
        assert value["measured"]["phasemargin_1"] is not None
        assert "failed_measurements" not in value
        assert value["not_finite"] == ["gainmargin_1"]


@pytest.mark.parametrize("recorded", _measurement_logs())
async def test_each_null_of_a_row_is_named_once_by_why(
    recorded: str, state_no_sim: SessionState, work_dir: Path
):
    """In every recorded log that prints a measurement, the log's measurements
    name as failed exactly those the log printed as failed, and a recipe row,
    over every step or of one, names each of its nulls in one place: failed,
    unreadable or not finite."""
    build, _, case_id = recorded.partition("/")
    log = work_dir / f"{Path(case_id).name}.log"
    shutil.copyfile(rec.recorded(build, f"{case_id}.log"), log)
    printed = _printed_failed_anywhere(read_log_text(log))

    decoded = rec.decode_log(build, case_id, work_dir / "decoded").value("measurements")
    assert decoded is not None
    assert {name.lower() for name in decoded["failed_measurements"]} == printed
    entries = decoded["measurements"].values()
    if all(number is not None for entry in entries for number in entry["values"]):
        return  # every value has a number, so no row holds a null to name
    rows: list[dict[str, Any]] = []
    # Each step's row; a stepped .op prints no .step line to number its rows
    # by, and is read over every step at once.
    for steps in ({"all_steps": True}, {}):
        reply = await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(log), "label": "log"}],
            recipes=[{"key": "m", "metric": "measurements"}],
            include={"per_run": True},
            **steps,
        )
        items = reply["results"].get("m", {}).get("per_run", {}).get("items", [])
        rows = [row["value"] for row in items]
        if rows:
            break
    assert rows
    for value in rows:
        failed = set(_failed(value))
        assert {name.lower() for name in failed} <= printed
        if "measured" not in value:
            continue
        nulls = {name for name, number in value["measured"].items() if number is None}
        unreadable = set(value.get("unreadable", {}))
        not_finite = set(value.get("not_finite", []))
        assert failed | not_finite <= nulls
        assert nulls <= failed | unreadable | not_finite
        assert not failed & unreadable and not failed & not_finite and not unreadable & not_finite
