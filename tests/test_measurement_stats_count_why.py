"""A measurement's statistics count as failed only the steps the simulator
printed failed, and every other step with no number under its reason.

The question is "why has this step of a measurement no number". The log
decoder answers it (``log_types.no_number_reasons``, from the one list
``log_types.NO_NUMBER_REASONS``), and the statistics of the ``measurements``
recipe ask it: ``failure_count`` counts the steps LTspice printed failed
(``Measurement "name" FAIL'ed``, or ``failed`` in a row of a stepped table),
and ``no_number`` counts every other step with no number under the name the
row gives its reason under (``unreadable``, ``answer_unknown``,
``not_established``). ``total_count`` is ``valid_count`` plus
``failure_count`` plus the counts in ``no_number``.

The expected values are read from the recordings: the log text, and the
printed-measurement grid's declared outcome for each recorded cell.

Evidence by build. A printed failure: both builds (``log/meas_failed``),
and in a stepped table's row LTspice 26 alone (``log/meas_step``,
``log/meas_when_step_fails``). A step not established: XVII alone, which
prints 0 there (the same logs). A value that is not finite: LTspice 26 alone
(``raw/fra_step``, a gain margin of ``infdB``; XVII has no ``.fra``). A value
printed once for a run of several steps: LTspice 26 alone
(``log/meas_step_op``); it is a number, so it is counted as one. No
recording, on either build, gives a result printed as text where a number
belongs (every recorded threshold printed as text is a WHEN's level, whose
result is the crossing) or a result whose answer is unknown; those are
counted by the same rule, shown here with written inputs.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.api import compute_measurement_stats, no_number_reasons, result_numbers
from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
from ltspice_mcp.lib.log_types import FAILED, NO_NUMBER_REASONS
from ltspice_mcp.lib.signal_analysis import NO_REASON_GIVEN
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_tool
from tests import _ltspice_recorded as rec
from tests import _source_scan
from tests._recorded_grids import grid
from tests.conftest import FIXTURES_DIR, LIVENESS_S, SyncApi, dispatched, replaying_simulator
from tests.ltspice_recorder import INPUTS

GRID = grid("printed-measurement")

_FAIL_LINE = re.compile(r"""^Measurement "([^"]+)" FAIL'ed""", re.M)
_TABLE = re.compile(r"^Measurement: (\S+)\r?\n(.*?)(?=\r?\n\s*\r?\n|\r?\nMeasurement:|\Z)", re.M | re.S)
_ROW = re.compile(r"^\s*(\d+)\t([^\t\r\n]*)", re.M)

#: The statistics' counts of the steps with no number.
Counts = dict[str, Any]


def _counts(entry: dict[str, Any]) -> Counts:
    """A stats entry's counts: total, numbers, printed failures, and the rest
    by reason (a count of 0 is left out of a reply)."""
    return {
        "total": entry["total_count"],
        "valid": entry["valid_count"],
        "failed": entry.get("failure_count", 0),
        "no_number": entry.get("no_number", {}),
    }


def _expect(total: int, valid: int, failed: int = 0, **no_number: int) -> Counts:
    return {"total": total, "valid": valid, "failed": failed, "no_number": no_number}


def _table_cells(text: str) -> dict[str, list[str]]:
    """Each stepped table's first column, by measurement in lower case, as
    the log printed it."""
    return {
        name.lower(): [cell.strip() for _number, cell in _ROW.findall(rows)]
        for name, rows in _TABLE.findall(text)
    }


def _printed_failed_steps(text: str, name: str, total: int) -> int:
    """How many steps of ``name`` the log printed failed: every step for one
    printed FAIL'ed, else the rows of its table that print ``failed``."""
    if name.lower() in {found.lower() for found in _FAIL_LINE.findall(text)}:
        return total
    cells = _table_cells(text).get(name.lower(), [])
    return sum(cell.lower() == "failed" for cell in cells)


def _measurement_logs() -> list[tuple[str, str]]:
    """Every recorded log that prints a measurement, as (build, case_id)."""
    return sorted(
        (build, path.relative_to(rec.recorded(build, "")).with_suffix("").as_posix())
        for build in rec.BUILDS
        for path in rec.recorded(build, "").rglob("*.log")
        if path.stat().st_size and printed_measurements(read_log_text(path))
    )


def _staged_log(build: str, case_id: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{Path(case_id).name}.log"
    shutil.copyfile(rec.recorded(build, f"{case_id}.log"), log)
    return log


def _value(reply: dict[str, Any], key: str = "m") -> dict[str, Any]:
    (row,) = reply["results"][key]["values"]
    return row["value"]


_RECIPE = [{"key": "m", "metric": "measurements"}]
_FULL = {"fields": ["value"]}


async def _stats_by_path(state: SessionState, log: Path, **kw: Any) -> dict[str, Any] | None:
    """The measurements recipe's full row over a log read by its path, or
    None where the log holds no measurement the recipe reads."""
    reply = await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(log), "label": "log"}],
        recipes=_RECIPE,
        include=_FULL,
        **kw,
    )
    values = reply["results"]["m"].get("values")
    return values[0]["value"] if values else None


# -- the counts add up, over every recorded log ---------------------------------


@pytest.mark.parametrize(("build", "case_id"), _measurement_logs())
async def test_the_counts_add_up_and_count_as_failed_only_what_was_printed_failed(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """For every measurement of every recorded log on each build:
    ``total_count`` is the numbers plus the printed failures plus every other
    step with no number by its reason, and ``failure_count`` is the steps the
    log printed failed, read from its text."""
    log = _staged_log(build, case_id, work_dir)
    value = await _stats_by_path(state_no_sim, log)
    if value is None:
        pytest.skip("the recipe reads no measurement from this log")
    text = read_log_text(log)
    for name, entry in value["stats"].items():
        counts = _counts(entry)
        assert counts["total"] == counts["valid"] + counts["failed"] + sum(
            counts["no_number"].values()
        ), name
        assert set(counts["no_number"]) <= {r.field for r in NO_NUMBER_REASONS} | {
            NO_REASON_GIVEN
        }, name
        assert counts["failed"] == _printed_failed_steps(text, name, counts["total"]), name
        # A step the decoder gives no reason for would be counted under
        # NO_REASON_GIVEN: no recording has one.
        assert NO_REASON_GIVEN not in counts["no_number"], name


# -- each reason, from the recordings, through every route ----------------------


def _grid_item(case: str, item: str) -> Any:
    (cell,) = [cell for cell in GRID.cells_of(case) if cell.item == item]
    return cell


def _meas_step_expected(build: str) -> Counts:
    """``log/meas_step``'s ``s_when``: the grid declares it fails in one step
    (``step_fails``, its third). LTspice 26 prints ``failed`` in that row;
    XVII prints 0, which the log cannot tell from a crossing at 0 s, so the
    step is not established, and no failure is counted for it."""
    cell = _grid_item("log/meas_step", "s_when")
    at = cell.at(build)
    assert at["outcome"] == "step_fails"
    (step,) = cell.extra_on("steps", build)
    cells = _table_cells(read_log_text(rec.recorded(build, "log/meas_step.log")))["s_when"]
    total = len(cells)
    if rec.generation(build) == "xvii":
        assert float(cells[step - 1]) == 0.0
        return _expect(total, total - 1, not_established=1)
    assert cells[step - 1].lower() == "failed"
    return _expect(total, total - 1, failed=1)


def _every_stats_route_counts(
    rows: dict[str, dict[str, Any]], name: str, expected: Counts
) -> None:
    """Each route's row that carries statistics gives ``expected`` for
    ``name``."""
    seen = {route: _counts(row["stats"][name]) for route, row in rows.items()}
    assert seen == dict.fromkeys(seen, expected)


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_stepped_measurement_counts_each_step_by_why_on_every_route(
    build: str,
    state_with_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``log/meas_step`` through every route that gives a measurement's
    statistics: ``analyze_results`` by path (the full row, the default lean
    row, a row projected to its stats, each step's row), the analysis
    ``run_experiments`` attaches, as its receipt, ``jobs`` status and ``jobs``
    wait return it, the Python API's ``analyze_results`` and its
    ``compute_measurement_stats`` given ``no_number_reasons`` over
    ``measurements()``, and ``run_code``.
    LTspice 26 counts one failure; XVII none, and one step not
    established."""
    expected = _meas_step_expected(build)
    deck = work_dir / "meas_step.cir"
    shutil.copyfile(INPUTS / "log/meas_step.cir", deck)
    log = deck.with_suffix(".log")
    shutil.copyfile(rec.recorded(build, "log/meas_step.log"), log)
    raw = deck.with_suffix(".raw")
    shutil.copyfile(rec.recorded(build, "raw/step_tran.raw"), raw)
    rows: dict[str, dict[str, Any]] = {}
    source = [{"path": str(log), "label": "log"}]

    async def by_path(**kw: Any) -> dict[str, Any]:
        return await dispatched(
            state_with_sim, "analyze_results", sources=source, recipes=_RECIPE, **kw
        )

    rows["full row"] = _value(await by_path(include=_FULL))
    # A stepped run's spread is its answer, so the lean row keeps its stats.
    rows["default lean row"] = _value(await by_path())
    rows["projected to stats"] = _value(await by_path(include={"fields": ["value.stats"]}))
    stepped = await by_path(all_steps=True, include={"fields": ["value", "step_index"]})
    by_step = {row["step_index"]: row["value"] for row in stepped["results"]["m"]["values"]}
    step_counts = [_counts(by_step[step]["stats"]["s_when"]) for step in sorted(by_step)]
    assert sum(c["failed"] for c in step_counts) == expected["failed"]
    summed: dict[str, int] = {}
    for c in step_counts:
        for reason, count in c["no_number"].items():
            summed[reason] = summed.get(reason, 0) + count
    assert summed == expected["no_number"]

    replaying_simulator(monkeypatch, raw, log)
    receipt = await dispatched(
        state_with_sim,
        "run_experiments",
        request_id="stats-count-why",
        circuits=[{"path": str(deck), "id": "meas_step"}],
        analyze={"recipes": _RECIPE, "include": _FULL},
        execution={"wait_s": LIVENESS_S},
    )
    assert receipt["status"] == "completed", receipt
    job_id = receipt["job_id"]
    rows["receipt"] = _value(receipt["analysis"]["result"])
    status = await dispatched(state_with_sim, "jobs", action="status", job_id=job_id)
    rows["jobs status"] = _value(status["analysis"]["result"])
    waited = await dispatched(
        state_with_sim, "jobs", action="wait", job_id=job_id, timeout_s=LIVENESS_S
    )
    rows["jobs wait"] = _value(waited["analysis"]["result"])

    api = SyncApi(state_with_sim)
    analyzed = await asyncio.to_thread(
        lambda: api.analyze_results(
            sources=[{"job_id": job_id, "label": "job"}], recipes=_RECIPE, include=_FULL
        )
    )
    rows["api analyze_results"] = _value(analyzed)
    # A caller's own statistics over the log's measurements, with the reasons
    # the log decoder gives.
    read = await asyncio.to_thread(lambda: api.measurements(job_id=job_id))
    own = compute_measurement_stats(
        {name: result_numbers(entry) for name, entry in read["measurements"].items()},
        reasons={name: no_number_reasons(read, name) for name in read["measurements"]},
    )
    rows["api compute_measurement_stats"] = {"stats": own}

    code = (
        "import json\n"
        f"r = api.analyze_results(sources=[{{'path': {str(log)!r}, 'label': 'log'}}], "
        "recipes=[{'key': 'm', 'metric': 'measurements'}], include={'fields': ['value']})\n"
        "print(json.dumps(r['results']['m']['values'][0]['value']))\n"
    )
    ran = await dispatched(state_with_sim, "run_code", code=code)
    rows["run_code"] = json.loads(ran["stdout"].strip().splitlines()[-1])

    _every_stats_route_counts(rows, "s_when", expected)


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_measurement_printed_failed_is_counted_as_a_failure(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``log/meas_failed``: the grid declares ``never`` and ``depends``
    failed whole, and both builds print them FAIL'ed. Each is one failure,
    by path and through the Python API."""
    log = _staged_log(build, "log/meas_failed", work_dir)
    text = read_log_text(log)
    api = SyncApi(state_no_sim)
    read = await asyncio.to_thread(
        lambda: api.analyze_results(
            sources=[{"path": str(log), "label": "log"}], recipes=_RECIPE, include=_FULL
        )
    )
    for route, value in (
        ("by path", await _stats_by_path(state_no_sim, log)),
        ("api", _value(read)),
    ):
        assert value is not None
        for name in ("never", "depends"):
            assert _grid_item("log/meas_failed", name).at(build)["outcome"] == "failed"
            assert name in _FAIL_LINE.findall(text)
            assert _counts(value["stats"][name]) == _expect(1, 0, failed=1), (route, name)
    # The default lean row of a run with one value per name leaves its stats
    # out (``measured`` carries every number): the null names its reason.
    lean = _value(
        await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(log), "label": "log"}],
            recipes=_RECIPE,
        )
    )
    assert "stats" not in lean
    for name in ("never", "depends"):
        assert lean["measured"][name] is None
        assert name in lean["failed_measurements"]


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_value_printed_as_not_finite_is_counted_as_unreadable(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``raw/fra_step``: LTspice 26 prints a stepped ``.fra`` run's gain
    margin ``infdB`` in both steps (the grid's ``not_a_number``). They are
    counted under ``unreadable``, not as failures. XVII has no ``.fra``."""
    if not rec.has(build, "raw/fra_step.log"):
        pytest.skip(f"no recording of a stepped .fra run on {build}")
    cell = _grid_item("raw/fra_step", "GainMargin_1")
    printed = cell.extra["not_a_number"]
    log = _staged_log(build, "raw/fra_step", work_dir)
    cells = _table_cells(read_log_text(log))["gainmargin_1"]
    not_finite = sum(printed in cell_text for cell_text in cells)
    assert not_finite == len(cells) > 0
    value = await _stats_by_path(state_no_sim, log)
    assert value is not None
    assert _counts(value["stats"]["gainmargin_1"]) == _expect(
        len(cells), 0, unreadable=not_finite
    )


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_value_printed_once_for_several_steps_is_a_number_not_a_failure(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """``log/meas_step_op``: LTspice 26 prints one line for each measurement
    of a run of three steps (the grid's ``printed_once``). The line holds a
    number, so it is counted as one, and no failure is. XVII prints no
    measurement of a stepped .op."""
    cell = _grid_item("log/meas_step_op", "vout")
    if build in cell.refused:
        pytest.skip(cell.refused[build])
    log = _staged_log(build, "log/meas_step_op", work_dir)
    value = await _stats_by_path(state_no_sim, log)
    assert value is not None
    assert value["printed_once"]["vout"]["step_count"] == cell.extra["printed_once"]["of"]
    for name in ("vout", "iin"):
        assert _counts(value["stats"][name]) == _expect(1, 1), name


async def test_a_result_whose_answer_is_unknown_is_counted_under_that_reason(
    state_no_sim: SessionState, work_dir: Path
):
    """No recording gives a result whose answer is not known. A stepped WHEN
    table printed with its level beside the crossing (``ltspice_step_when.log``,
    written by hand: LTspice prints the crossing alone under the level) is a
    form no recording shows, so no step of ``tcross`` has a number, and each
    is counted under ``answer_unknown``, not as a failure."""
    log = work_dir / "step_when.log"
    shutil.copyfile(FIXTURES_DIR / "ltspice_step_when.log", log)
    value = await _stats_by_path(state_no_sim, log)
    assert value is not None
    assert "tcross" in value["answer_unknown"]
    expected = {"tcross": _expect(3, 0, answer_unknown=3), "vfinal": _expect(3, 3)}
    # The Python API's own statistics over its measurements: the printed
    # values of tcross are not taken for its results.
    read = await asyncio.to_thread(lambda: SyncApi(state_no_sim).measurements(log))
    own = compute_measurement_stats(
        {name: result_numbers(entry) for name, entry in read["measurements"].items()},
        reasons={name: no_number_reasons(read, name) for name in read["measurements"]},
    )
    for route, stats in (("by path", value["stats"]), ("api", own)):
        for name, counts in expected.items():
            assert _counts(dict(stats[name])) == counts, (route, name)


# -- the Python API's function, over a caller's own list ------------------------


def test_a_null_with_no_reason_is_not_a_failure():
    """A ``None`` in a caller's own list comes with no reason: it is counted
    under ``reason_not_given``, never as a failure; a reason given is counted
    under its name, and only the simulator's printed failure as a failure."""
    (entry,) = compute_measurement_stats({"m": [1.0, None, 3.0]}).values()
    assert _counts(dict(entry)) == _expect(3, 2, reason_not_given=1)
    reasons = {"m": [None, "failed_measurements", "not_established", "unreadable", None]}
    (entry,) = compute_measurement_stats(
        {"m": [1.0, None, None, None, None]}, reasons=reasons
    ).values()
    assert _counts(dict(entry)) == _expect(
        5, 1, failed=1, not_established=1, unreadable=1, reason_not_given=1
    )


# -- one list of reasons serves the row and the statistics ----------------------

#: The measurements row's fields that are not a reason a step has no number:
#: the numbers and their points, the statistics, the run's diagnostics, and a
#: value printed once for several steps, which is a number.
_NOT_A_REASON = {
    "diagnostics",
    "diagnostics_coverage",
    "measured",
    "at",
    "stats",
    "printed_once",
    # A waveform's value, declared in the same row value schema so the
    # presenter thins its series (``series("waveform")``).
    "x",
    "series",
    "points_returned",
    "points_total",
}


def test_one_list_of_reasons_names_the_rows_fields_and_the_statistics_counts():
    """Every reason in ``NO_NUMBER_REASONS`` is a field of the measurements
    row and, but the printed failure (``failure_count``), a count the
    statistics' schema declares; and every reason field of the row is in the
    list, so a reason the decoder adds reaches both."""
    value = analyze_tool._ATTRIBUTED_VALUE_SCHEMA["properties"]["value"]["properties"]
    fields = [reason.field for reason in NO_NUMBER_REASONS]
    assert set(value) - _NOT_A_REASON == set(fields)
    stats = value["stats"]["additionalProperties"]["properties"]
    assert "failure_count" in stats
    counted = stats["no_number"]["properties"]
    assert set(counted) == {*fields, NO_REASON_GIVEN} - {FAILED}


# -- no second place calls a null a failure --------------------------------------

_FAILURE_NAME = re.compile(r"fail", re.I)
_A_COUNT = re.compile(r"total|valid|count|len", re.I)


def _operand_names(node: ast.AST) -> set[str]:
    """The names and attribute names an expression reads."""
    return {
        sub.id if isinstance(sub, ast.Name) else sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, (ast.Name, ast.Attribute))
    }


def _tests_for_none(node: ast.AST) -> bool:
    """Whether ``node`` takes a count of nulls: it tests a value for None,
    counts the Nones of a list (``values.count(None)``), or subtracts one
    count from another (``total - valid``)."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Compare)
            and any(isinstance(op, (ast.Is, ast.IsNot)) for op in sub.ops)
            and any(
                isinstance(c, ast.Constant) and c.value is None
                for c in [sub.left, *sub.comparators]
            )
        ):
            return True
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "count"
            and any(isinstance(arg, ast.Constant) and arg.value is None for arg in sub.args)
        ):
            return True
        if (
            isinstance(sub, ast.BinOp)
            and isinstance(sub.op, ast.Sub)
            and any(_A_COUNT.search(name) for name in _operand_names(sub))
        ):
            return True
    return False


def _failure_counts_from_none(tree: ast.AST) -> list[tuple[str, int]]:
    """Each count named for failure in ``tree`` (a dict key, an assignment or
    a keyword) whose value tests for None or takes a difference."""
    found: list[tuple[str, int]] = []
    for scope, nodes in _source_scan.scopes(tree):
        for node in nodes:
            pairs: list[tuple[str, ast.AST]] = []
            if isinstance(node, ast.Dict):
                pairs = [
                    (key.value, value)
                    for key, value in zip(node.keys, node.values, strict=True)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                pairs = [
                    (target.id, node.value)
                    for target in targets
                    if isinstance(target, ast.Name) and node.value is not None
                ]
            elif isinstance(node, ast.keyword) and node.arg is not None:
                pairs = [(node.arg, node.value)]
            for name, value in pairs:
                if _FAILURE_NAME.search(name) and _tests_for_none(value):
                    found.append((scope, getattr(value, "lineno", 0)))
    return found


def test_no_code_counts_a_null_as_a_failure():
    """No code under ``src/`` derives a count named for failure from a test
    for None, a count of Nones or a difference of counts (``total -
    valid``): such a count calls a null a failure whatever the simulator
    printed. The statistics count failures by the reason the log decoder
    gives (``signal_analysis._count_no_number``). The scan finds each form
    it looks for, the one it replaced among them."""
    found = [
        (path, scope, line)
        for path, tree in _source_scan.modules().items()
        for scope, line in _failure_counts_from_none(tree)
    ]
    assert found == []
    for old in (
        'def f(values):\n    return {"failure_count": total - valid_count}\n',
        "def f(values):\n    failure_count = values.count(None)\n",
        "def f(values):\n    failed = sum(v is None for v in values)\n",
    ):
        assert _failure_counts_from_none(ast.parse(old)), old


# -- ngspice, live ----------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize("control", [False, True], ids=["dot_meas", "control_meas"])
async def test_an_ngspice_run_with_a_failed_measurement_counts_no_null_as_a_failure(
    control: bool, work_dir: Path
):
    """A real ngspice run of a deck whose WHEN is never crossed, with its
    measurements as ``.meas`` cards and as ``meas`` commands of a control
    script. Whatever an ngspice build prints for it, every statistic the
    measurements recipe gives counts as failed only a measurement its row
    names failed, and its counts add up; where the recipe reads no
    measurement it says so in its failures. On ngspice 42 neither form gives
    a row (in batch mode ngspice measures nothing, and the log decoder reads
    no ngspice measurement line), so there this checks the refusal alone; a
    build that prints measurements the decoder reads is checked in full."""
    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.simulator import detect_simulators
    from tests.conftest import terminal_experiment

    config = ServerConfig(
        simulator="ngspice", working_dir=work_dir, allowed_paths=[work_dir], log_level="DEBUG"
    )
    state = SessionState.create(config, detect_simulators(config))
    measures = "WHEN v(out)=5\n", "MAX v(out)\n"
    if control:
        cards = (
            ".control\nrun\n"
            f"meas tran never {measures[0]}meas tran vmax {measures[1]}.endc\n"
        )
    else:
        cards = f".meas tran never {measures[0]}.meas tran vmax {measures[1]}"
    (work_dir / "m.cir").write_text(
        "* never crossed\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n"
        f".tran 1u 5m\n{cards}.end\n",
        encoding="ascii",
    )
    receipt = await terminal_experiment(
        state,
        {
            "request_id": f"ng-stats-{int(control)}",
            "circuits": [{"path": "m.cir", "id": "dut"}],
            "execution": {"wait_s": LIVENESS_S, "simulator": "ngspice"},
        },
    )
    assert receipt["status"] == "completed", receipt
    reply = await dispatched(
        state,
        "analyze_results",
        sources=[{"job_id": receipt["job_id"], "runs": "all", "label": "dut"}],
        recipes=_RECIPE,
        include=_FULL,
    )
    rows = reply["results"]["m"].get("values") or []
    if not rows:
        assert any(f["code"] == "recipe_failed" for f in reply["failures"]), reply
    for row in rows:
        value = row["value"]
        for name, entry in value.get("stats", {}).items():
            counts = _counts(entry)
            assert counts["total"] == counts["valid"] + counts["failed"] + sum(
                counts["no_number"].values()
            )
            if counts["failed"]:
                assert name in value.get("failed_measurements", []), value
