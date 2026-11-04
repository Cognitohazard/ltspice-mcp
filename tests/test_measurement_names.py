"""Which measurement a caller's name refers to, on every route that takes one.

The log reader holds a ``.meas`` result under its name lower-cased, whatever
case the deck wrote it in. A caller names it as the deck does, so every reader
of a measurement name matches it in any case, and one the log does not hold is
refused with the names it does hold rather than answered with nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.result_observations import reconciliation_observations
from ltspice_mcp.lib.signal_analysis import compute_measurement_stats
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi

#: An LTspice 26 log of one transient run with one measurement the deck named
#: ``Vout_Max``.
_LOG = (
    "LTspice 26.0.2 for Windows\n"
    "Circuit: C:\\work\\rc.cir\n"
    "Start Time: Tue Jun  9 21:07:33 2026\n"
    "solver = Normal\n"
    "tnom = 27\n"
    "temp = 27\n"
    "method = trap\n"
    ".OP point found by inspection.\n"
    "Total elapsed time: 0.023 seconds.\n"
    "\n"
    "Vout_Max: MAX(V(out))=0.999955832958 FROM 0 TO 0.001\n"
)
_VALUE = 0.999955832958

SPELLINGS = ["Vout_Max", "vout_max", "VOUT_MAX"]


@pytest.fixture
def log(work_dir: Path) -> Path:
    path = work_dir / "rc.log"
    path.write_text(_LOG, encoding="utf-8")
    return path


def _analyze(state: SessionState, log: Path, **recipe: Any) -> dict[str, Any]:
    return SyncApi(state).analyze_results(
        sources=[{"path": str(log), "label": "run"}],
        recipes=[{"key": "m", "metric": "measurements", **recipe}],
    )


def test_the_log_holds_the_name_lower_cased(state_no_sim: SessionState, log: Path):
    measured = SyncApi(state_no_sim).measurements(log)
    assert list(measured["measurements"]) == ["vout_max"]


@pytest.mark.parametrize("name", SPELLINGS)
def test_names_selects_the_measurement_in_any_case(
    state_no_sim: SessionState, log: Path, name: str
):
    reply = _analyze(state_no_sim, log, names=[name])
    assert reply["failures"] == []
    (row,) = reply["results"]["m"]["values"]
    assert row["value"]["measured"] == {"vout_max": _VALUE}


@pytest.mark.parametrize("name", SPELLINGS)
def test_a_spec_on_the_field_judges_the_measurement_in_any_case(
    state_no_sim: SessionState, log: Path, name: str
):
    reply = _analyze(state_no_sim, log, field=name, spec={"max": 2}, reduce=["max"])
    assert reply["failures"] == []
    result = reply["results"]["m"]
    spec = result["spec"]
    assert (spec["field"], spec["verdict"], spec["pass_count"], spec["no_value_count"]) == (
        name,
        "pass",
        1,
        0,
    )
    assert [(row["field"], row["value"]) for row in result["reduced"]] == [(name, _VALUE)]


@pytest.mark.parametrize("name", SPELLINGS)
def test_measurement_stats_finds_the_measurement_in_any_case(name: str):
    stats = compute_measurement_stats({"vout_max": [_VALUE]}, measurement=name)
    assert list(stats) == ["vout_max"]
    assert stats["vout_max"]["mean"] == _VALUE


@pytest.mark.parametrize("name", SPELLINGS)
def test_reconciliation_counts_the_measurement_produced_in_any_case(name: str):
    summary = {"measurements": {"vout_max": {"values": [_VALUE]}}}
    assert reconciliation_observations(summary, {"meas": [name]}) == []


@pytest.mark.parametrize("names", [["vout_min"], ["Vout_Max", "vout_min"]])
def test_a_name_the_log_does_not_hold_is_refused_with_the_ones_it_does(
    state_no_sim: SessionState, log: Path, names: list[str]
):
    reply = _analyze(state_no_sim, log, names=names)
    (failure,) = reply["failures"]
    assert "No measurement named 'vout_min' in the log. It holds: vout_max" in failure["message"]


def test_a_spec_on_a_field_the_log_does_not_hold_is_refused_with_the_ones_it_does(
    state_no_sim: SessionState, log: Path
):
    reply = _analyze(state_no_sim, log, field="vout_min", spec={"max": 2})
    (failure,) = reply["failures"]
    assert "No measurement named 'vout_min' in the log. It holds: vout_max" in failure["message"]


def test_measurement_stats_refuses_a_name_it_does_not_hold():
    with pytest.raises(
        ValueError, match=r"Measurement 'vout_min' not found\. Available: vout_max"
    ):
        compute_measurement_stats({"vout_max": [_VALUE]}, measurement="vout_min")
