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
import json
import re
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


#: A line on which the simulator wrote how severe it is, stated here apart from
#: the classifier: ``Fatal Error``/``Error``/``ERROR`` make it an error,
#: ``Warning``/``WARNING`` a warning.
_WORD = re.compile(r"^(Fatal Error|Error|ERROR|Warning|WARNING)\b")


def _word(line: str) -> str | None:
    """``"error"``, ``"warning"`` or None for one line, by the word on it."""
    found = _WORD.match(line.strip())
    if found is None:
        return None
    return "warning" if found[1].casefold() == "warning" else "error"


def _worded_lines(text: str) -> list[tuple[str, str]]:
    """Each line of a log that carries an error or a warning word, with it,
    the log split as the server splits it (``log_lines``, which parts a line
    LTspice XVII glued to its progress output)."""
    from ltspice_mcp.lib.log_conditions import log_lines

    return [(line.strip(), word) for line in log_lines(text) if (word := _word(line)) is not None]


def _entry_words(entry: str) -> set[str]:
    return {word for line in entry.splitlines() if (word := _word(line)) is not None}


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_summary_says_what_the_run_row_says_the_simulator_said(
    build: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """``log/meas_bad_syntax``: each build prints an error for the ``.meas``
    it cannot evaluate, and the run produces. Every line of the summary's
    errors carries an error word and every warning a warning word, each row
    condition the simulator called an error has its line among the errors,
    and each relayed observation carries the severity the simulator wrote or
    none."""
    deck = _recorded(work_dir, "log/meas_bad_syntax")
    row, found = await _row_and_summaries(
        _ltspice(config),
        deck,
        rec.recorded(build, "raw/tran_uncompressed.raw"),
        rec.recorded(build, "log/meas_bad_syntax.log"),
        monkeypatch,
    )
    written = {d["severity"] for d in row.get("diagnostics", []) if "severity" in d}
    called_errors = [
        d
        for d in row.get("diagnostics", [])
        if d.get("severity", "").casefold() in ("error", "fatal error")
    ]
    for route, (value, observations) in found.items():
        errors = value.get("errors", [])
        assert all("error" in _entry_words(e) for e in errors), (route, errors)
        assert all(
            _entry_words(w) == {"warning"} for w in value.get("warnings", []) if _entry_words(w)
        ), route
        for condition in called_errors:
            assert any(
                e in condition["first_text"] or condition["first_text"] in e for e in errors
            ), (
                route,
                condition,
                errors,
            )
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
    Error``. The condition's severity is ``Fatal Error``, and the summary lists
    as the error the line LTspice called one, which names the node; LTspice 26
    writes the same block with no word, so there it is in neither list."""
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
        if expected is None:
            assert not [e for e in errors if "operating point" in e], (build, errors)
        else:
            (said,) = [e for e in errors if expected in e]
            assert said.startswith("Fatal Error:") and 'node "a"' in said, (build, errors)
            assert condition["first_text"] not in errors, (build, errors)


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
        # A severity is listed with the line it was written on: ngspice's
        # ``Note:`` rungs are no warning, its ``Warning:`` rung is one.
        stepping = [w for w in value.get("warnings", []) if "stepping" in w.casefold()]
        assert all(w.startswith("Warning:") for w in stepping), (route, stepping)
        assert not any("surfaced as a warning" in line for line in value.get("warnings", [])), (
            route
        )


def _summary_of_log(text: str, producer: Any, tmp_path: Path) -> dict[str, Any]:
    """The summary of a run whose log is ``text``, written by ``producer``."""
    import numpy as np

    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from tests._ltspice_recorded import _decode_paths
    from tests.conftest import make_raw_mock

    log = tmp_path / "run.log"
    log.write_text(text, encoding="utf-8")
    family = producer.family or "ltspice"
    logs = _decode_paths(None, log, tmp_path / "decoded", producing=family).logs
    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    return build_simulation_summary(raw, logs, producer=producer)


def test_two_different_error_lines_of_one_condition_are_both_listed(tmp_path: Path):
    """LTspice XVII writes two different ``Error:`` lines, which fall under one
    code (``simulator_error``): both are in ``errors``, the run row counts
    them as one condition of count 3, and a repeated identical line is one
    entry."""
    from ltspice_mcp.lib.log_conditions import classify

    text = "Circuit: * two errors\n\nError: first thing\nError: second thing\nError: first thing\n"
    summary = _summary_of_log(text, _producer("ltspice17"), tmp_path)
    assert summary["errors"] == ["Error: first thing", "Error: second thing"]
    (row,) = [c for c in classify({"log": text}, "ltspice17") if c["code"] == "simulator_error"]
    assert row["count"] == 3


def test_two_broken_measurements_on_xvii_are_both_meas_errors(tmp_path: Path):
    """LTspice XVII writes each ``.meas`` it cannot take with its ``Error:``
    line (recorded: log/meas_bad_syntax); two of them are one
    ``measurement_error`` condition, and both are in ``meas_errors``."""
    from ltspice_mcp.lib.encoding import read_spice_text

    recorded = read_spice_text(rec.recorded("ltspice17", "log/meas_bad_syntax.log"))
    first = ".meas tran broken find v(out at 0.5m"
    assert first in recorded
    block = recorded[recorded.index(first) :].split("\n\n", 1)[0]
    second = block.replace("broken", "broken2")
    text = recorded.replace(block, block + "\n" + second)
    summary = _summary_of_log(text, _producer("ltspice17"), tmp_path)
    assert [e["directive"] for e in summary["meas_errors"]] == [
        first,
        first.replace("broken", "broken2"),
    ]


def test_lines_past_a_conditions_bound_are_counted(tmp_path: Path):
    """Past :data:`log_conditions.WRITTEN_KEPT` distinct worded lines of one
    condition, the summary lists the bound and says how many it left out."""
    from ltspice_mcp.lib.log_conditions import WRITTEN_KEPT

    extra = 3
    text = "".join(f"Error: thing {n}\n" for n in range(WRITTEN_KEPT + extra))
    summary = _summary_of_log(text, _producer("ltspice17"), tmp_path)
    assert len(summary["errors"]) == WRITTEN_KEPT
    assert summary["lines_left_out"] == {"simulator_error": extra}


def test_lines_that_are_neither_take_no_place_and_are_not_counted(tmp_path: Path):
    """``Note:`` lines are neither errors nor warnings: past the bound they
    are not counted as left out, and they do not push a warning past it."""
    from ltspice_mcp.lib.log_conditions import WRITTEN_KEPT
    from ltspice_mcp.lib.simulator_build import Producer

    notes = "".join(f"Note: v{n}: has no value, DC 0 assumed\n" for n in range(WRITTEN_KEPT + 5))
    text = notes + "Warning: v9: has no value, DC 0 assumed\n"
    summary = _summary_of_log(text, Producer("ngspice", None, None), tmp_path)
    assert "lines_left_out" not in summary, summary.get("lines_left_out")
    assert "Warning: v9: has no value, DC 0 assumed" in summary.get("warnings", []), summary


def test_lines_left_out_on_both_channels_are_added(tmp_path: Path):
    """A code with conditions on the log and the console counts the lines each
    left out."""
    import numpy as np

    from ltspice_mcp.lib.log_conditions import WRITTEN_KEPT
    from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
    from ltspice_mcp.lib.parser_worker import parse_request
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from ltspice_mcp.lib.simulator_build import Producer
    from tests._ltspice_recorded import RAW_LIMITS, parser_request
    from tests.conftest import make_raw_mock

    log = tmp_path / "run.log"
    log.write_text("".join(f"Error: log {n}\n" for n in range(WRITTEN_KEPT + 2)), encoding="utf-8")
    console = tmp_path / "run.console"
    console.write_text(
        "".join(f"Error: console {n}\n" for n in range(WRITTEN_KEPT + 3)), encoding="utf-8"
    )
    request = parser_request(None, log, producing="ngspice")
    request["sources"]["console"] = str(console)
    decoded = tmp_path / "decoded"
    decoded.mkdir()
    parse_request(request, decoded)
    metadata = json.loads((decoded / "result.json").read_bytes())
    logs = read_parsed_artifacts(
        metadata, decoded, limits=RAW_LIMITS, require_raw=False, request=request
    ).logs
    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    summary = build_simulation_summary(raw, logs, producer=Producer("ngspice", None, None))
    assert summary["lines_left_out"] == {"simulator_error": 5}


def _line_classifier_examples() -> list[str]:
    """One line for every pattern the line classifier
    (``log_parser.classify_line``) reads an error or a warning by, taken from
    where each is defined: a phrase list gives each phrase, and a pattern has
    the example beside its name here. A pattern the classifier reads that has
    no example fails the test that uses these."""
    import ast
    import inspect

    from ltspice_mcp.lib import log_parser

    examples = {
        "_RE_FILE_ERROR": 'deck.cir(7): Expected ")" here.',
        "_RE_FATAL": "Fatal Error: Unknown subcircuit called in: X1",
        "_RE_LINE_ERROR": "Error on line 4 : r1 a b r",
        "_RE_ERROR": "ERROR: Node a is floating.",
        "_RE_WARNING": "Warning -- a default was used",
        "_RE_COMPONENT_REFUSAL": "R1: Resistance must not be zero.",
        "_RE_UNSOLVABLE_MATRIX": "V1 and L1 form a loop: matrix is singular",
        "_RE_NGSPICE_MEAS_BLOCKED": "No .measure possible in batch mode (-b) with -r rawfile set!",
        "_RE_NGSPICE_FOUR_BLOCKED": ".fourier line ignored since rawfile was produced.",
        "_RE_NGSPICE_UNIMPLEMENTED": "unimplemented dot command '.backanno'",
        "_RE_NGSPICE_NOTE": "Note: v1 has no value, DC 0 assumed",
    }
    # Not a pattern a line is listed by: the caret under a file-line block,
    # and the notes the classifier leaves out.
    not_listing = {"_RE_CARET", "_NGSPICE_NOTE_SKIP"}
    tree = ast.parse(inspect.getsource(log_parser.classify_line))
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and re.fullmatch(r"_[A-Z][A-Z0-9_]*", node.id)
    }
    phrases = {name for name in used if name.endswith("_PHRASES")}
    unknown = used - phrases - set(examples) - not_listing
    assert not unknown, f"no example line for {sorted(unknown)}"
    lines = list(examples.values())
    for name in sorted(phrases):
        lines.extend(f"{phrase} at the end" for phrase in getattr(log_parser, name))
    # The stepping rungs, read through is_op_stepping_failure.
    lines.extend(f"{p} here" for p in log_parser._OP_STEPPING_FAILURE_PHRASES)
    # A convergence failure behind a ``Warning:`` prefix, which the classifier
    # lists as an error whatever the prefix says.
    lines.extend(
        f"Warning: {p}"
        for p in (
            *log_parser._OP_STEPPING_FAILURE_PHRASES,
            *log_parser._CONVERGENCE_FAILURE_PHRASES,
        )
    )
    return lines


@pytest.mark.parametrize("family", ["xyce", "qspice"])
async def test_a_simulator_with_no_recordings_keeps_the_lines_the_line_classifier_listed(
    family: str, config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """For a simulator whose log wording has no recordings there is no word of
    its own to prefer, so the line classifier's reading stands: each line it
    lists as an error or a warning (one per pattern it knows) is in the same
    list of the summary, through ``analyze_results`` and the Python API."""
    from ltspice_mcp.lib.log_parser import extract_log_diagnostics
    from ltspice_mcp.lib.simulator import SIMULATORS
    from tests.test_replies_name_what_they_left_out import _XYCE_TRAN_RAW, _qspice_ac_qraw

    examples = _line_classifier_examples()
    log = work_dir / f"{family}.log"
    log.write_text("\n\n".join(examples) + "\n", encoding="utf-8")
    before = extract_log_diagnostics(log)
    raw = work_dir / f"{family}.raw"
    raw.write_bytes(_qspice_ac_qraw() if family == "qspice" else _XYCE_TRAN_RAW)
    deck = _recorded(work_dir, "log/meas_failed")
    state = SessionState.create(
        config, available={family: type(SIMULATORS[family].__name__, (FakeSim,), {})}
    )
    row, found = await _row_and_summaries(state, deck, raw, log, monkeypatch)
    assert row.get("diagnostics_coverage") == "unverified", row
    # A severity is only ever the simulator's own word: a line read by the
    # classifier's patterns carries none on the row, and no relay gets one.
    words = {"Fatal Error", "Error", "ERROR", "Warning", "WARNING", "Note"}
    assert {d["severity"] for d in row["diagnostics"] if "severity" in d} <= words, row
    for route, (_value, observations) in found.items():
        relayed = {
            o["severity"] for o in observations if o.get("kind") == "relay" and "severity" in o
        }
        assert relayed <= words, (route, relayed)
    for example in examples:
        first = example.splitlines()[0]
        listed = [name for name in ("errors", "warnings") if any(first in e for e in before[name])]
        assert listed, f"the line classifier lists no {example!r}"
        for route, (value, _observations_found) in found.items():
            now = [
                name
                for name in ("errors", "warnings")
                if any(first in e for e in value.get(name, []))
            ]
            assert now == listed, (route, example, now, listed)


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
_CLASSIFIER_OUTPUT = (
    "diag",
    "meas_data",
    '"measurements")',
    "measurementsoutput",
    "logdiagnostics",
)


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
                arg.arg
                for arg in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
                if arg.annotation is not None and holds_output(ast.unparse(arg.annotation))
            } | {
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
    for log in sorted((FIXTURES_DIR / "ngspice_runs").glob("*.log")):
        found.append(("ngspice", log.stem))
    return found


@pytest.mark.parametrize(("build", "case"), _recorded_logs())
def test_every_line_the_simulator_gave_a_word_is_in_the_summary(
    build: str, case: str, tmp_path: Path
):
    """On every recorded log of both LTspice builds, and every ngspice run's
    log, each line on which the simulator wrote an error word is in the
    summary's ``errors``, each with a warning word in ``errors`` or
    ``warnings``, unless the summary counts lines it left out; and no entry of
    either list lacks a word. The words are read off the log's own lines here,
    not by the classifier."""
    import numpy as np

    from ltspice_mcp.lib.log_parser import read_log_text
    from ltspice_mcp.lib.raw_parser import build_simulation_summary
    from ltspice_mcp.lib.simulator_build import Producer
    from tests._ltspice_recorded import _decode_paths
    from tests.conftest import make_raw_mock

    if build == "ngspice":
        path = FIXTURES_DIR / "ngspice_runs" / f"{case}.log"
        producer = Producer("ngspice", None, None)
        logs = _decode_paths(None, path, tmp_path / "decoded", producing="ngspice").logs
    else:
        path = rec.recorded(build, f"{case}.log")
        producer = _producer(build)
        logs = rec.decode_log(build, case, tmp_path)
    axis = np.linspace(0.0, 1e-3, 5)
    raw = make_raw_mock(["time", "V(out)"], axis, {"time": axis, "V(out)": np.ones(5)})
    summary = build_simulation_summary(raw, logs, producer=producer)
    errors, warnings = summary.get("errors", []), summary.get("warnings", [])
    for entry in errors:
        assert "error" in _entry_words(entry), entry
    text = read_log_text(path)
    if summary.get("lines_left_out"):
        return
    for line, word in _worded_lines(text):
        lists = errors if word == "error" else errors + warnings
        assert any(line in entry for entry in lists), (line, word, errors, warnings)
