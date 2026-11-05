"""Every recorded log's measurement tables decode.

Each ``.log`` both builds wrote under ``tests/fixtures/ltspice_recorded`` goes
through the parser the server runs on a live result. A value printed in a
form the decoder did not read would fail the whole measurements section, and
every measurement of that run with it. A line the decoder skipped would lose
that measurement alone, so the decoded entries are checked against the
measurements the log printed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import FIXTURES


def recorded_logs() -> list[str]:
    """Every recorded log that holds anything (XVII crashes on a Solver
    setting of 2 and leaves an empty one, ``deck/solver_setting_2``)."""
    return sorted(
        path.relative_to(FIXTURES).as_posix()
        for build in rec.BUILDS
        for path in (FIXTURES / build).rglob("*.log")
        if path.stat().st_size
    )


LOGS = recorded_logs()


def test_both_builds_recorded_logs():
    assert {name.split("/")[0] for name in LOGS} == set(rec.BUILDS)
    assert "ltspice26/raw/fra_step.log" in LOGS
    # Lines that print an '=' and are not measurements: a warning, and a title.
    assert "ltspice17/deck/mos_delvto_bsim3.log" in LOGS
    assert "ltspice17/deck/title_directive.log" in LOGS


@pytest.mark.parametrize("name", LOGS)
def test_every_recorded_logs_measurements_decode(name: str, tmp_path: Path):
    """The log's entries are the measurements it printed: a result line, a
    stepped table and a FAIL'ed line each have one, LTspice 26's braced WHEN
    threshold (``log/meas_when_threshold_tran``) included, and nothing else
    has one, such as XVII's title ``Circuit: .param r=2k`` or a warning
    ``Warning: Pd = 0 is less than W.``."""
    build, _, log = name.partition("/")
    section = rec.decode_log(build, log.removesuffix(".log"), tmp_path).section("measurements")
    assert section["status"] in ("parsed", "absent"), section["error"]
    entries = (section["value"] or {}).get("measurements", {})
    printed = printed_measurements(read_log_text(FIXTURES / name))
    assert {entry.lower() for entry in entries} == set(printed)


def test_a_stepped_fra_runs_margins_read_as_the_numbers_printed(tmp_path: Path):
    """LTspice 26 prints a stepped .fra run's margins with their units:
    ``90.6295°`` for a phase margin, ``infdB`` (at ``inf``) for a gain margin
    whose phase never crosses."""
    path = rec.recorded("ltspice26", "raw/fra_step.log")
    text = path.read_text(encoding="utf-8")
    assert "     1\t90.6295°\t15900.4" in text.splitlines()
    assert "     1\tinfdB\tinf" in text.splitlines()

    section = rec.decode_log("ltspice26", "raw/fra_step", tmp_path).section("measurements")

    assert section["status"] == "parsed"
    table = section["value"]
    assert table["step_count"] == 2
    assert table["measurements"]["phasemargin_1"] == {
        "values": [90.6295, 90.1991],
        "at": [15900.4, 8007.62],
    }
    # Not a finite number, so held as null and counted.
    assert table["measurements"]["gainmargin_1"] == {"values": [None, None], "at": None}
    assert section["nonfinite_count"] == 3
