"""Every recorded log's measurement tables decode.

Each ``.log`` both builds wrote under ``tests/fixtures/ltspice_recorded`` goes
through the parser the server runs on a live result. A value printed in a
form the decoder did not read would fail the whole measurements section, and
every measurement of that run with it. A line the decoder skipped would lose
that measurement alone, so the decoded entries are checked against the
measurements the log printed. A line read as a measurement that is none would
add an entry, and would be among the printed ones too, so the entries are also
held to the names the deck behind the log measures, read from the deck alone.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from ltspice_mcp.lib import NETLIST_SUFFIXES
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text
from ltspice_mcp.lib.spice_lex import SpiceLexError, lex_deck
from ltspice_mcp.lib.spice_lex_views import MeasCard
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import FIXTURES, INPUTS


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

#: Measurements a recorded log prints that no ``.meas`` card of its deck asks
#: for: LTspice 26 prints each loop margin of a stepped ``.fra`` run as a
#: measurement table of its own (``raw/fra_step``).
UNASKED: dict[str, frozenset[str]] = {
    "raw/fra_step": frozenset({"phasemargin_1", "gainmargin_1"}),
}


def measured(build: str, case_id: str) -> set[str]:
    """The names, in lower case, of the measurements the deck behind a
    recorded log asks for: each ``.meas`` card of its input deck, or of the
    netlist the build wrote from its input sheet, and of the files copied
    beside it, and the names ``UNASKED`` lists. A card's name is the one the
    lexer gives it, which ``MeasCard`` reads too, and which it gives a card
    ``MeasCard`` refuses as malformed (``log/meas_bad_syntax``)."""
    case = rec.CASES.case(case_id)
    decks = [
        rec.export_text(build, case_id)
        if case.source.endswith(".asc")
        else read_spice_text(INPUTS / case.source)
    ]
    included = [
        read_spice_text(INPUTS / extra)
        for extra in case.extra
        if PurePosixPath(extra).suffix.lower() in NETLIST_SUFFIXES | {".inc", ".lib"}
    ]
    cards = [
        *(card for text in decks for card in lex_deck(text).cards),
        *(card for text in included for card in lex_deck(text, title=False).cards),
    ]
    names = {card.name.lower() for card in cards if card.kind == "meas" and card.name}
    return names | UNASKED.get(case_id, frozenset())


def test_a_measurement_is_named_as_meas_card_reads_it():
    """``measured`` takes each ``.meas`` card's name from the lexer, which is
    the name ``MeasCard`` reads off every card it can read."""
    for case in rec.CASES.cases:
        if PurePosixPath(case.source).suffix.lower() not in NETLIST_SUFFIXES:
            continue
        for card in lex_deck(read_spice_text(INPUTS / case.source)).cards:
            if card.kind != "meas":
                continue
            try:
                view = MeasCard.from_card(card)
            except SpiceLexError:
                continue
            assert card.name == view.name, card.body


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
    ``Warning: Pd = 0 is less than W.``. Each is a name the deck measures,
    read from the deck rather than the log: ``printed_measurements`` decides
    both sides of the first comparison, so a line it took for a measurement
    would be on both."""
    build, _, log = name.partition("/")
    case_id = log.removesuffix(".log")
    section = rec.decode_log(build, case_id, tmp_path).section("measurements")
    assert section["status"] in ("parsed", "absent"), section["error"]
    entries = {entry.lower() for entry in (section["value"] or {}).get("measurements", {})}
    printed = printed_measurements(read_log_text(FIXTURES / name))
    assert entries == set(printed)
    assert entries <= measured(build, case_id)
    assert UNASKED.get(case_id, frozenset()) <= entries


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
        "printed_form": "table_at",
    }
    # Not a finite number, so held as null and counted.
    assert table["measurements"]["gainmargin_1"] == {
        "values": [None, None],
        "at": None,
        "printed_form": "table_at",
    }
    assert section["nonfinite_count"] == 3
