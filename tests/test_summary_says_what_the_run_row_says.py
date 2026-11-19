"""What the simulator said about a run, and how severe it called it: one
answer, the run's classified conditions (``log_conditions.run_diagnostics``).

The run row's ``diagnostics`` are those conditions. The ``summary`` recipe's
``errors``, ``warnings`` and ``meas_errors``, and the severity of each line it
relays, are built from the same conditions: a line is an error or a warning
only where the simulator wrote that word on it, and a condition with no word
is in neither list (it is in ``diagnostics``). Each case runs through
``analyze_results`` and the Python API, on recordings.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    FakeSim,
    SyncApi,
    dispatched,
    replaying_simulator,
)
from tests.ltspice_recorder import INPUTS


def _observations(data: Any) -> list[dict[str, Any]]:
    """Every observation anywhere in a reply."""
    found: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("code"), str) and "detail" in data:
            found.append(data)
        for value in data.values():
            found.extend(_observations(value))
    elif isinstance(data, list):
        for value in data:
            found.extend(_observations(value))
    return found


async def _row_and_summaries(
    state: SessionState, deck: Path, raw: Path, log: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]]:
    """The run row, and the summary's whole value with the reply's
    observations, by MCP and by the Python API."""
    replaying_simulator(monkeypatch, raw, log)
    receipt = await dispatched(
        state,
        "run_experiments",
        request_id=f"said-{deck.stem}",
        circuits=[{"path": str(deck), "id": deck.stem}],
        execution={"wait_s": LIVENESS_S},
        lint="off",
    )
    (row,) = receipt["runs"]["items"]
    job = [{"job_id": receipt["job_id"]}]
    api = SyncApi(state)
    summary = {"key": "s", "metric": "summary"}
    found = {}
    for route, data in (
        (
            "tool",
            await dispatched(
                state,
                "analyze_results",
                sources=job,
                recipes=[summary],
                include={"fields": ["value"]},
            ),
        ),
        (
            "api",
            await asyncio.to_thread(
                lambda: api.analyze_results(
                    sources=job, recipes=[summary], include={"fields": ["value"]}
                )
            ),
        ),
    ):
        values = data["results"]["s"].get("values", [])
        assert len(values) == 1, (route, data)
        found[route] = (values[0]["value"], _observations(data))
    return row, found


def _ltspice(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})


def _recorded(work_dir: Path, case: str) -> Path:
    deck = work_dir / f"{Path(case).name}.cir"
    shutil.copyfile(INPUTS / f"{case}.cir", deck)
    return deck


#: The words the simulator writes on a line that the summary reads as an error,
#: and as a warning.
_ERROR_WORDS = {"fatal error", "error"}
_WARNING_WORDS = {"warning"}


def _texts(row: dict[str, Any], words: set[str]) -> set[str]:
    return {
        d["first_text"]
        for d in row.get("diagnostics", [])
        if d.get("severity", "").casefold() in words
    }


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_summary_says_what_the_run_row_says_the_simulator_said(
    build: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_bad_syntax``: each build prints an error for the ``.meas``
    it cannot evaluate, and the run produces. The summary's errors are the
    row's conditions the simulator called errors, its warnings those it
    called warnings, and each relayed observation carries the severity the
    simulator wrote or none."""
    deck = _recorded(work_dir, "log/meas_bad_syntax")
    row, found = await _row_and_summaries(
        _ltspice(config),
        deck,
        rec.recorded(build, "raw/tran_uncompressed.raw"),
        rec.recorded(build, "log/meas_bad_syntax.log"),
        monkeypatch,
    )
    written = {d["severity"] for d in row.get("diagnostics", []) if "severity" in d}
    for route, (value, observations) in found.items():
        assert set(value.get("errors", [])) == _texts(row, _ERROR_WORDS), (route, value, row)
        assert _texts(row, _WARNING_WORDS) <= set(value.get("warnings", [])), route
        for item in observations:
            if item.get("kind") == "relay" and "severity" in item:
                assert item["severity"] in written, (route, item)


async def test_a_line_the_simulator_gave_no_severity_is_in_neither_list(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """LTspice 26 writes ``…cir(7): Expected ")" here.`` for a ``.meas`` it
    cannot read, with no word for how severe (recorded: log/meas_bad_syntax).
    The summary lists it as neither an error nor a warning, gives no
    ``meas_errors`` for it and relays no severity for it; the row's
    ``diagnostics``, which the summary carries, hold it."""
    deck = _recorded(work_dir, "log/meas_bad_syntax")
    row, found = await _row_and_summaries(
        _ltspice(config),
        deck,
        rec.recorded("ltspice26", "raw/tran_uncompressed.raw"),
        rec.recorded("ltspice26", "log/meas_bad_syntax.log"),
        monkeypatch,
    )
    (condition,) = [d for d in row["diagnostics"] if "Expected" in d["first_text"]]
    assert "severity" not in condition
    for route, (value, observations) in found.items():
        assert not any("Expected" in line for line in value.get("errors", [])), route
        assert not any("Expected" in line for line in value.get("warnings", [])), route
        assert "meas_errors" not in value, route
        assert condition in value["diagnostics"], route
        assert not [
            o
            for o in observations
            if o.get("kind") == "relay" and "Expected" in o["detail"] and "severity" in o
        ], route


async def test_an_xvii_measurement_error_is_a_meas_error_with_the_simulators_severity(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """LTspice XVII writes the ``.meas`` it cannot take and ``Error: FIND can
    not be evaluated over an interval.`` (recorded: log/meas_bad_syntax). The
    summary names that ``.meas`` in ``meas_errors``, and relays it once, as
    that measurement, with the severity XVII wrote, ``Error``. The other two
    ``.meas`` were measured, so no batch abort is claimed for the broken one's
    own miss."""
    deck = _recorded(work_dir, "log/meas_bad_syntax")
    _row, found = await _row_and_summaries(
        _ltspice(config),
        deck,
        rec.recorded("ltspice17", "raw/tran_uncompressed.raw"),
        rec.recorded("ltspice17", "log/meas_bad_syntax.log"),
        monkeypatch,
    )
    for route, (value, observations) in found.items():
        (entry,) = value["meas_errors"]
        assert entry["directive"].casefold().startswith(".meas tran broken"), (route, entry)
        relayed = [o for o in observations if o["code"] == "meas_parse_error"]
        assert [o["severity"] for o in relayed] == ["Error"], (route, relayed)
        assert not [o for o in observations if o["code"] == "log_error"], route
        assert not [o for o in observations if o["code"] == "meas_batch_abort"], route


def test_a_condition_is_as_severe_as_the_strongest_word_the_simulator_wrote_on_it(
    tmp_path: Path,
):
    """LTspice XVII's log/err_timestep: the ``convergence_failed`` block starts
    ``Trouble finding operating point...`` and a later line says ``Fatal
    Error``. The condition's severity is ``Fatal Error``, so the summary lists
    it as an error; LTspice 26 writes the same block with no word, so there it
    is in neither list."""
    import numpy as np

    from ltspice_mcp.lib.log_conditions import run_diagnostics
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from tests.conftest import make_raw_mock

    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    for build, expected in (("ltspice17", "Fatal Error"), ("ltspice26", None)):
        if build not in rec.BUILDS:
            continue
        producer = _producer(build)
        logs = rec.decode_log(build, "log/err_timestep", tmp_path / build)
        (condition,) = [
            c
            for c in run_diagnostics(logs.value("diagnostics")["conditions"], producer)[
                "diagnostics"
            ]
            if c["code"] == "convergence_failed"
        ]
        assert condition.get("severity") == expected, (build, condition)
        errors = build_simulation_summary(raw, logs, producer=producer).get("errors", [])
        assert (condition["first_text"] in errors) == (expected is not None), (build, errors)


async def test_an_operating_point_found_by_stepping_is_no_error_and_says_no_demotion(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """ngspice's ``op_source_stepping``: gmin stepping fails, source stepping
    finds the bias point, and the run produces. The stepping rungs are one
    ``operating_point_fallback`` condition whose first line ngspice wrote as a
    ``Note``: the summary lists none of them as an error, and so demotes none
    to a warning."""
    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    state = SessionState.create(
        config, available={"ngspice": type(NGspiceSimulator.__name__, (FakeSim,), {})}
    )
    runs = FIXTURES_DIR / "ngspice_runs"
    deck = work_dir / "op_source_stepping.cir"
    shutil.copyfile(runs / "op_source_stepping.cir", deck)
    row, found = await _row_and_summaries(
        state, deck, runs / "op_source_stepping.raw", runs / "op_source_stepping.log", monkeypatch
    )
    assert any(d["code"] == "operating_point_fallback" for d in row["diagnostics"]), row
    for route, (value, _observations_found) in found.items():
        assert not any("stepping" in line.casefold() for line in value.get("errors", [])), route
        assert not any("surfaced as a warning" in line for line in value.get("warnings", [])), (
            route
        )


@pytest.mark.parametrize("family", ["xyce", "qspice"])
def test_a_simulator_with_no_recordings_keeps_the_errors_and_warnings_it_wrote(
    family: str, tmp_path: Path
):
    """A simulator whose log wording the server has no recordings of is still
    read for the lines it starts with ``Error:`` or ``Warning:``: the run row
    names them, and the summary lists them, as it did before it read the row."""
    import numpy as np

    from ltspice_mcp.lib.log_conditions import run_diagnostics
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from ltspice_mcp.lib.simulator_build import Producer
    from tests._ltspice_recorded import _decode_paths
    from tests.conftest import make_raw_mock

    log = tmp_path / "run.log"
    log.write_text("A run\nError: no model named qx\nWarning: a default was used\n")
    logs = _decode_paths(None, log, tmp_path / "decoded", producing=family).logs
    producer = Producer(family, None, None)
    said = run_diagnostics(logs.value("diagnostics")["conditions"], producer)
    assert said["coverage"] == "unverified"
    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    summary = build_simulation_summary(raw, logs, producer=producer)
    assert summary["errors"] == ["Error: no model named qx"]
    assert "Warning: a default was used" in summary["warnings"]


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------

#: The scopes that read the ``errors`` or ``warnings`` the log decoder's line
#: classifier (``log_parser.extract_log_diagnostics``) gives, and why each
#: may: none of them tells a reader how severe the simulator called a line.
_LINE_CLASSIFIER_READERS = {
    ("lib/runner_base.py", "collect_run_outcome"): "whether the run failed, not a severity shown",
    ("lib/log_decode.py", "decode_logs"): "whether the simulator refused the deck before it ran",
    ("lib/log_parser.py", "empty_measurements"): "the reason a refused run has no measurements",
    ("lib/services.py", "solve_failure_lines"): "solve-failure lines by phrase, on both lists",
    ("lib/metrics.py", "read_log_warnings"): "unrecognized-variable lines by phrase",
    ("lib/metrics.py", "aggregate_log_measurements"): (
        "the text of why a log has no measurements, in a refusal, with no severity"
    ),
}

#: Words in an expression that make it the line classifier's output: its
#: diagnostics, or the measurement parser's, which carries its lists.
_CLASSIFIER_OUTPUT = ("diag", "meas_data")


def test_only_the_named_readers_read_the_line_classifiers_lists():
    """No code outside ``log_conditions`` decides whether a log line is an
    error or a warning for a reply: the summary and its relays read the run's
    conditions (``log_conditions.said_as``), and the line classifier's lists
    are read only by the scopes listed, each for a question that is not a
    severity."""
    import ast

    from tests._source_scan import modules

    def holds_output(text: str) -> bool:
        return any(word in text.casefold() for word in _CLASSIFIER_OUTPUT)

    readers = set()
    for path, tree in modules().items():
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # A name bound to the classifier's output under another name
            # (``section = logs.value("diagnostics")``) is read as that output.
            aliases = {
                target.id
                for node in ast.walk(scope)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                and holds_output(ast.unparse(node.value))
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(scope):
                key = base = None
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in ("errors", "warnings", "meas_errors")
                ):
                    key, base = node.slice.value, ast.unparse(node.value)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in ("errors", "warnings", "meas_errors")
                ):
                    key, base = node.args[0].value, ast.unparse(node.func.value)
                if key is not None and base is not None:
                    root = base.split(".", 1)[0].split("[", 1)[0].split("(", 1)[0]
                    if holds_output(base) or root in aliases:
                        readers.add((path, scope.name))
    assert readers == set(_LINE_CLASSIFIER_READERS)


def _producer(build: str) -> Any:
    from ltspice_mcp.lib.simulator_build import Producer

    return Producer(
        "ltspice",
        "LTspice 26.1.1 for Windows" if build == "ltspice26" else None,
        "LTspice.exe" if build == "ltspice26" else "XVIIx64.exe",
    )


def _recorded_logs() -> list[tuple[str, str]]:
    found = []
    for build in rec.BUILDS:
        folder = rec.recorded(build, "log")
        for log in sorted(Path(folder).glob("*.log")):
            found.append((build, f"log/{log.stem}"))
    return found


@pytest.mark.parametrize(("build", "case"), _recorded_logs())
def test_every_recorded_log_gives_the_summary_the_run_rows_errors_and_warnings(
    build: str, case: str, tmp_path: Path
):
    """On every recorded log of both builds, the summary's errors are the
    run's conditions the simulator called errors, its log warnings those it
    called warnings, and no other line is either."""
    import numpy as np

    from ltspice_mcp.lib.log_conditions import run_diagnostics, said_as
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from tests.conftest import make_raw_mock

    producer = _producer(build)
    logs = rec.decode_log(build, case, tmp_path)
    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    summary = build_simulation_summary(raw, logs, producer=producer)
    section = logs.value("diagnostics")
    said = (run_diagnostics(section["conditions"], producer) if section else {}).get(
        "diagnostics", []
    )
    errors = [c["first_text"] for c in said if said_as(c) == "error"]
    warnings = [c["first_text"] for c in said if said_as(c) == "warning"]
    assert summary.get("errors", []) == errors
    assert [
        w for w in summary.get("warnings", []) if w in {c["first_text"] for c in said}
    ] == warnings
