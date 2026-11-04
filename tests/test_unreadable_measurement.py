"""A measurement whose printed value is not a number fails alone, on every door.

LTspice prints a WHEN's threshold after ``=`` as the deck writes it: a number,
a parameter, an expression or another measurement (recorded as
``log/meas_when_threshold_tran``). The server reads a number there and cannot
read the others. Such an entry fails by itself, with the line LTspice printed
and the reason, while every other measurement of the log reads; each door
reports the same entry the same way.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi

_CASE = "log/meas_when_threshold_tran"


@pytest.fixture(params=rec.BUILDS)
def log(request: pytest.FixtureRequest, work_dir: Path) -> Path:
    path = work_dir / "when_threshold.log"
    shutil.copyfile(rec.recorded(request.param, f"{_CASE}.log"), path)
    return path


def _printed(log: Path, name: str) -> str:
    """The line LTspice printed for ``name``."""
    text = log.read_text(encoding="utf-8", errors="replace")
    return next(line.strip() for line in text.splitlines() if line.lower().startswith(f"{name}:"))


def test_the_measurement_api_fails_the_entry_alone(state_no_sim: SessionState, log: Path):
    measured = SyncApi(state_no_sim).measurements(log)["measurements"]
    entry = measured["t_param"]
    assert entry["values"] == [None]
    assert entry["unreadable"]["line"] == _printed(log, "t_param")
    assert "'vth'" in entry["unreadable"]["reason"]
    assert measured["t_number"]["values"][0] is not None
    assert "unreadable" not in measured["t_number"]


def test_inspect_gives_the_failed_entry_its_line_and_reason(state_no_sim: SessionState, log: Path):
    query = {"kind": "results", "view": "measurements", "path": str(log)}
    rows = SyncApi(state_no_sim).inspect(queries=[query])["results"][0]["data"]["measurements"]
    by_name: dict[str, Any] = {row["measurement"]: row for row in rows}
    assert by_name["t_param"].get("value") is None
    assert by_name["t_param"]["unreadable"]["line"] == _printed(log, "t_param")
    assert by_name["t_number"].get("value") is not None
    assert "unreadable" not in by_name["t_number"]


def test_the_measurements_recipe_fails_the_entry_alone(state_no_sim: SessionState, log: Path):
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"path": str(log), "label": "run"}],
        recipes=[{"key": "m", "metric": "measurements"}],
    )
    assert reply.get("failures", []) == []
    (row,) = reply["results"]["m"]["values"]
    value = row["value"]
    assert value["measured"]["t_param"] is None
    assert value["measured"]["t_number"] is not None
    assert "t_param" in value["failed_measurements"]
    assert value["unreadable"]["t_param"]["line"] == _printed(log, "t_param")
    assert "t_number" not in value["unreadable"]
