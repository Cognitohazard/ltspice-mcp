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
its card; every reader asks it. These tests send each recorded measurement
through every route that reads one and require one answer from all of them:
the measurements recipe over a bare log path and over a job, the summary's
``measurements``, the Python API's ``measurements()`` and ``inspect``'s
measurements view.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi, dispatched, make_experiment_job
from tests.ltspice_recorder import INPUTS

pytestmark = pytest.mark.asyncio

#: An unstepped result line whose point follows an uppercase ``AT``: the
#: point is the result.
_UPPER_AT = re.compile(
    r"^(\w+): [^\r\n]*=[^\r\n]*? AT ([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*$", re.M
)

Answer = list[float | None] | str


def _measurement_logs() -> list[tuple[str, str]]:
    """Every recorded log that prints a measurement, with a deck in the
    recording's inputs, as ``(build, case_id)``."""
    found = []
    for build in rec.BUILDS:
        root = rec.recorded(build, "")
        for path in sorted(root.rglob("*.log")):
            case_id = path.relative_to(root).with_suffix("").as_posix()
            if not (INPUTS / f"{case_id}.cir").exists():
                continue
            if path.stat().st_size and printed_measurements(read_log_text(path)):
                found.append((build, case_id))
    return found


_LOGS = _measurement_logs()


def _stage(build: str, case_id: str, directory: Path) -> tuple[Path, Path, Path]:
    """The case's deck and log in ``directory``, with a raw beside the log: the
    one the build wrote for the case, or for ``deck/after_end`` where the
    recording keeps the log alone. Every reply here reads the measurements
    from the log."""
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(case_id).name
    deck = directory / f"{name}.cir"
    shutil.copyfile(INPUTS / f"{case_id}.cir", deck)
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
        return _numbers(entry.get("at"), len(values))
    return list(values)


def _recipe_answers(reply: dict[str, Any]) -> dict[str, Answer]:
    """Each measurement's result per step from an ``all_steps`` recipe reply."""
    rows = {row["step_index"]: row["value"] for row in reply["results"]["m"]["values"]}
    answers: dict[str, Answer] = {}
    for step in sorted(rows):
        value = rows[step]
        for name, number in value["measured"].items():
            if name in value.get("answer_unknown", {}):
                answers[name] = "unknown"
            else:
                held = answers.setdefault(name, [])
                assert isinstance(held, list)
                held.append(number)
    return answers


def _inspect_answers(item: dict[str, Any]) -> dict[str, Answer]:
    assert item["ok"], item
    answers: dict[str, Answer] = {}
    for row in sorted(item["data"]["measurements"], key=lambda row: row["value_index"]):
        name = row["measurement"]
        if "answer_unknown" in row:
            answers[name] = "unknown"
            continue
        held = answers.setdefault(name, [])
        assert isinstance(held, list)
        held.append(row.get("at") if row.get("answer", "value") == "at" else row.get("value"))
    return answers


def _same(left: Answer, right: Answer) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return len(left) == len(right) and all(
        a == b or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12))
        for a, b in zip(left, right, strict=True)
    )


_RECIPE = [{"key": "m", "metric": "measurements"}]
_EACH_STEP = {"fields": ["value", "step_index"]}


async def _every_route(
    build: str, case_id: str, state: SessionState, directory: Path
) -> dict[str, dict[str, Answer]]:
    deck, raw, log = _stage(build, case_id, directory)
    routes: dict[str, dict[str, Answer]] = {}
    routes["recipe, by path"] = _recipe_answers(
        await dispatched(
            state,
            "analyze_results",
            sources=[{"path": str(log), "label": "log"}],
            recipes=_RECIPE,
            all_steps=True,
            include=_EACH_STEP,
        )
    )
    summary = await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"fields": ["value"]},
    )
    (row,) = summary["results"]["s"]["values"]
    routes["summary, by path"] = {
        name: _entry_answer(entry) for name, entry in row["value"]["measurements"].items()
    }
    page = await dispatched(
        state, "inspect", queries=[{"kind": "results", "view": "measurements", "path": str(log)}]
    )
    routes["inspect, by path"] = _inspect_answers(page["results"][0])
    api = SyncApi(state)
    read = await asyncio.to_thread(lambda: api.measurements(path=log))
    routes["api measurements, by path"] = {
        name: _entry_answer(entry) for name, entry in read["measurements"].items()
    }

    job_id = "exp_" + re.sub(r"\W", "_", f"{build}_{case_id}")
    job = make_experiment_job(state, job_id=job_id, raw=raw, netlist=deck)
    job.cases[0].simulator_version = rec.manifest(build)["reported_build"]
    routes["recipe, by job"] = _recipe_answers(
        await dispatched(
            state,
            "analyze_results",
            sources=[{"job_id": job_id, "label": "job"}],
            recipes=_RECIPE,
            all_steps=True,
            include=_EACH_STEP,
        )
    )
    by_job = await dispatched(
        state,
        "analyze_results",
        sources=[{"job_id": job_id, "label": "job"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"fields": ["value"]},
    )
    (row,) = by_job["results"]["s"]["values"]
    routes["summary, by job"] = {
        name: _entry_answer(entry) for name, entry in row["value"]["measurements"].items()
    }
    page = await dispatched(
        state, "inspect", queries=[{"kind": "results", "view": "measurements", "job_id": job_id}]
    )
    routes["inspect, by job"] = _inspect_answers(page["results"][0])
    read = await asyncio.to_thread(lambda: api.measurements(job_id=job_id))
    routes["api measurements, by job"] = {
        name: _entry_answer(entry) for name, entry in read["measurements"].items()
    }
    return routes


@pytest.mark.parametrize(("build", "case_id"), _LOGS, ids=[f"{b}/{c}" for b, c in _LOGS])
async def test_every_route_reads_a_measurement_from_the_same_field(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    routes = await _every_route(build, case_id, state_no_sim, work_dir)
    first_route, first = next(iter(routes.items()))
    for route, answers in routes.items():
        assert set(answers) == set(first), (route, first_route)
        for name, answer in answers.items():
            assert _same(answer, first[name]), (name, route, answer, first_route, first[name])
    # A crossing printed after an uppercase AT is the result on every route.
    text = read_log_text(rec.recorded(build, f"{case_id}.log"))
    for name, point in _UPPER_AT.findall(text):
        assert first[name.lower()] == [float(point)], (name, first[name.lower()])
