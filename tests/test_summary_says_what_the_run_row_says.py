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
    summary names that ``.meas`` in ``meas_errors``, and relays it with the
    severity XVII wrote, ``Error``."""
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
