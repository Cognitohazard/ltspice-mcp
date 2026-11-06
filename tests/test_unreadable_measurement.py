"""A measurement whose printed value is not a number fails alone, on every door.

LTspice prints a WHEN's threshold after ``=`` as the deck writes it: a number,
a parameter, an expression, a braced expression or another measurement
(recorded as ``log/meas_when_threshold_tran``). The server reads a number
there and cannot read the others; LTspice 26's braced form,
``t_braced: V(out)={vth} AT ...``, is a line spicelib's reader does not match
at all. Each such entry fails by itself, with the line LTspice printed, the
reason and the crossing, while every other measurement of the log reads; each
door reports the same entries the same way.
"""

from __future__ import annotations

import re
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
    # A transient raw of the same build beside it, so the run's summary reads
    # this log as its own.
    shutil.copyfile(rec.recorded(request.param, "raw/tran.raw"), path.with_suffix(".raw"))
    return path


def _printed(log: Path) -> dict[str, str]:
    """Each measurement's line as LTspice printed it, by name."""
    text = log.read_text(encoding="utf-8", errors="replace")
    return {
        found[1]: found[0].strip()
        for found in re.finditer(r"^(\w+): [^\r\n]*=[^\r\n]*", text, re.M)
        if found[1] != "Circuit"
    }


def _unreadable(log: Path) -> dict[str, str]:
    """The WHENs whose threshold LTspice printed as other than a number, by
    name, with what it printed."""
    thresholds = {
        name: line.split("=", 1)[1].rsplit(" AT ", 1)[0]
        for name, line in _printed(log).items()
        if " AT " in line
    }
    unreadable = {}
    for name, threshold in thresholds.items():
        try:
            float(threshold)
        except ValueError:
            unreadable[name] = threshold
    assert "t_param" in unreadable and "t_number" not in unreadable
    return unreadable


def _crossing(line: str) -> float:
    return float(line.rsplit(" AT ", 1)[1])


def test_the_braced_threshold_is_unreadable_where_ltspice_prints_the_braces(log: Path):
    assert ("t_braced" in _unreadable(log)) == ("{vth}" in _printed(log)["t_braced"])


def test_the_measurement_api_fails_each_entry_alone(state_no_sim: SessionState, log: Path):
    measured = SyncApi(state_no_sim).measurements(log)["measurements"]
    printed, unreadable = _printed(log), _unreadable(log)
    assert set(printed) <= set(measured)
    for name, threshold in unreadable.items():
        entry = measured[name]
        assert entry["values"] == [None], name
        assert entry["unreadable"]["line"] == printed[name]
        assert repr(threshold) in entry["unreadable"]["reason"]
        assert entry["at"] == _crossing(printed[name])
    for name in set(measured) - set(unreadable):
        assert measured[name]["values"][0] is not None, name
        assert "unreadable" not in measured[name], name


def test_inspect_gives_each_failed_entry_its_line_and_reason(
    state_no_sim: SessionState, log: Path
):
    query = {"kind": "results", "view": "measurements", "path": str(log)}
    rows = SyncApi(state_no_sim).inspect(queries=[query])["results"][0]["data"]["measurements"]
    by_name: dict[str, Any] = {row["measurement"]: row for row in rows}
    printed, unreadable = _printed(log), _unreadable(log)
    assert set(printed) <= set(by_name)
    for name in unreadable:
        assert by_name[name].get("value") is None, name
        assert by_name[name]["unreadable"]["line"] == printed[name]
        assert by_name[name]["at"] == _crossing(printed[name])
    for name in set(by_name) - set(unreadable):
        assert by_name[name].get("value") is not None, name
        assert "unreadable" not in by_name[name], name


def test_the_measurements_recipe_fails_each_entry_alone(state_no_sim: SessionState, log: Path):
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"path": str(log), "label": "run"}],
        recipes=[{"key": "m", "metric": "measurements"}],
    )
    assert reply.get("failures", []) == []
    (row,) = reply["results"]["m"]["values"]
    value = row["value"]
    printed, unreadable = _printed(log), _unreadable(log)
    assert set(printed) <= set(value["measured"])
    for name in unreadable:
        assert value["measured"][name] is None, name
        assert name in value["failed_measurements"]
        assert value["unreadable"][name]["line"] == printed[name]
        assert value["at"][name] == _crossing(printed[name])
    for name in set(value["measured"]) - set(unreadable):
        assert value["measured"][name] is not None, name
    assert set(value["unreadable"]) == set(unreadable)


def test_the_run_summary_carries_each_failed_entry(state_no_sim: SessionState, log: Path):
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"path": str(log.with_suffix(".raw")), "label": "run"}],
        recipes=[{"key": "s", "metric": "summary"}],
        include={"fields": ["value.measurements"]},
    )
    assert reply.get("failures", []) == []
    (row,) = reply["results"]["s"]["values"]
    summarized = row["value"]["measurements"]
    measured = SyncApi(state_no_sim).measurements(log)["measurements"]
    assert summarized == measured
    assert all("unreadable" in summarized[name] for name in _unreadable(log))
