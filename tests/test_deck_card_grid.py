"""Which lines of a deck are cards, and which directive a card is, on every
route that answers it.

The question is the deck-card grid's (``[grid.deck-card]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). The lexer decides it
(``lex_deck`` and ``directive_head``), and every reader of a deck's cards
asks it: the components listing of ``inspect`` and of the Python API, the
netlist reading of ``services``, the netlist graph a comparison reads, the
hierarchy, the dangling-node check, a run's source amplitudes, the targets a
variation edits, and the lexer's reading of an included file
(``tests/_deck_readings.py`` runs them).

A cell is one card whose reading sets one node's voltage in the ``.op`` raw,
or one measurement the log prints. Every recorded cell, on each build, runs
through every route: a route that reads the cell's subject otherwise than the
recording shows LTspice read it is a difference, declared on its cell
(``differs``) and counted in ``KNOWN_DIFFERENCES``. The same decks run live on
ngspice in each compatibility mode, against the server's reading for ngspice.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import spice_lex
from ltspice_mcp.lib.spice_lex import INCLUDE_HEADS, lex_deck
from ltspice_mcp.state import SessionState
from tests import _deck_readings as dr
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid

GRID = grid("deck-card")

#: How many routes answer otherwise than a recorded cell, per build, each
#: declared on its cell (``differs``, with the routes and the reason). A count
#: may only fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 11, "ltspice17": 0}


def _recorded(build: str, case_id: str) -> bool:
    return case_id in rec.manifest(build)["cases"]


def _cases() -> list[str]:
    return sorted({cell.case for cell in GRID.cells})


def _runs() -> list[Any]:
    runs = []
    for case_id in _cases():
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if _recorded(build, case_id):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


def _subject(cell: Cell) -> str:
    return cell.extra.get("subject", f"element {cell.item}")


def _volts(cell: Cell) -> dict[str, Any]:
    return cell.extra.get("volts", {})


def _lt_reading(cell: Cell, build: str, voltages: dict[str, float] | None) -> str:
    """What the recording shows LTspice read: the key of the cell's volts its
    node's voltage has, or, for a measurement, whether the log printed it."""
    if "printed" in cell.extra:
        return (
            "read"
            if cell.extra["printed"] in dr.recorded_printed(build, cell.case)
            else ("not_read")
        )
    found = dr.reading_of(_volts(cell), cell.extra["node"], voltages)
    assert found is not None, (
        f"{cell.id}: {build} gave v({cell.extra['node']}) = "
        f"{(voltages or {}).get('v(' + cell.extra['node'].lower() + ')')}, "
        "which none of the cell's readings gives"
    )
    return found


def _server_readings(cell: Cell, readings: dr.DeckReadings) -> dict[str, dr.Answer]:
    """What each route reads the cell to say, in the terms of the cell's
    readings: for a directive, read or not_read; for anything else, the
    subject's answer."""
    if cell.coordinates["directive"]:
        found: dict[str, dr.Answer] = {
            route: "read" if directive == cell.coordinates["directive"] else "not_read"
            for route, directive in readings.directives(cell.item).items()
        }
        if "subject" in cell.extra:
            when = cell.extra.get("read_when", "present")
            for route, answer in readings.answers(cell.extra["subject"]).items():
                held = "absent" if answer == "absent" else "present"
                found[f"{route} ({cell.extra['subject']})"] = (
                    "read" if held == when else "not_read"
                )
        return found
    answers = readings.answers(_subject(cell))
    if cell.extra.get("file") == "included":
        answers = {r: a for r, a in answers.items() if r in dr.FOLLOWS_INCLUDES}
    return answers


# The index, held to the decks and to the code ---------------------------------


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_names_a_card_its_deck_writes(cell: Cell):
    """A directive cell's item is the first word of a card of its deck, as
    the deck writes it (the lexer reads a card's directive from that word, so
    every card that starts with it is read alike); any other cell's subject is an element, parameter,
    model or subcircuit the deck's text names."""
    source = rec.CASES.case(cell.case).source
    from ltspice_mcp.lib.encoding import read_spice_text
    from tests.ltspice_recorder import INPUTS

    text = read_spice_text(INPUTS / source)
    for extra in rec.CASES.case(cell.case).extra:
        text += "\n" + read_spice_text(INPUTS / extra)
    if cell.coordinates["directive"]:
        starts = [line.strip().split(None, 1)[0] for line in text.splitlines() if line.strip()]
        assert cell.item in starts, (cell.id, starts)
    else:
        name = _subject(cell).split()[1]
        assert re.search(rf"(?i)(?<![\w$]){re.escape(name)}(?![\w$])", text), cell.id
    assert ("printed" in cell.extra) != ("node" in cell.extra), cell.id
    for key in _volts(cell):
        dr.as_answer(key)


def test_every_directive_the_lexer_knows_is_a_value_of_the_directive_axis():
    """The lexer's own kinds of directive, its include heads and its opaque
    region are each a directive of the grid, in a spelling the grid holds, so
    a directive it learns needs cells."""
    axis = set(GRID.axes["directive"].values)
    source = Path(spice_lex.__file__).read_text(encoding="utf-8")
    body = source[source.index("def _classify_directive") :]
    body = body[: body.index("\ndef ")]
    kinds = set(re.findall(r'"(\.[a-z]+)"', body))
    assert kinds == {".model", ".param", ".subckt", ".ends", ".end", ".meas", ".measure"}
    written = {cell.item.casefold() for cell in GRID.cells if cell.coordinates["directive"]}
    every = kinds | INCLUDE_HEADS | set(spice_lex._OPAQUE_REGIONS)
    missing = {head for head in every if head not in written and head != ".endc"}
    assert not missing, f"no cell writes {sorted(missing)}"
    for head in every - {".endc"}:
        card = lex_deck(f"* t\n{head} x\n").cards[0]
        assert dr.directive_of(card) in axis, head


def test_a_recorded_cell_has_no_prediction_left():
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle its reading against the "
                "recording, write it as reads, and remove not_established"
            )
        else:
            assert "reads" in cell.extra, f"{cell.id} declares no reading"


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(len(dr.differs_declared(cell, build)) for cell in GRID.cells)
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route -------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_reads_as_ltspice_read_it_on_every_route(
    build: str, case_id: str, tmp_path: Path, state_no_sim: SessionState, work_dir: Path
):
    voltages = dr.recorded_voltages(build, case_id, tmp_path)
    deck = dr.stage_case(case_id, work_dir)
    readings = await dr.read_every_route(state_no_sim, deck)
    found: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    for cell in GRID.cells_of(case_id):
        ltspice = _lt_reading(cell, build, voltages)
        assert ltspice == dr.declared_reading(cell, build), (
            f"{cell.id}: {build} read it as {ltspice}, not as the form means"
        )
        answers = _server_readings(cell, readings)
        assert answers, f"{cell.id}: no route answers"
        found |= {
            (cell.item, route)
            for route, answer in answers.items()
            if not dr.same_reading(answer, dr.as_answer(ltspice))
        }
        declared |= dr.differs_declared(cell, build)
    assert found == declared, (
        f"answered otherwise, not declared: {sorted(found - declared)}; "
        f"declared, not found: {sorted(declared - found)}"
    )


# ngspice, live ----------------------------------------------------------------

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: Cells where ngspice reads otherwise than the server's reading for ngspice,
#: with the reason. Not asserted to differ: another ngspice version may agree.
NGSPICE_DIFFERS: dict[str, str] = {
    "deck/after_end:R2": (
        "ngspice-42 in a compatibility mode reads the card after .end; lex_deck ends "
        "the deck at .end for every simulator"
    ),
    "deck/include_end:R2": (
        "ngspice-42 reads an included file's cards past the file's own .end; lex_deck "
        "ends an included file at its .end for every simulator"
    ),
    "deck/card_grid_directive_variants_lower:.params": (
        "ngspice reads .params as .param; the lexer reads it as an unknown directive, "
        "so no route holds its parameter"
    ),
    "deck/card_grid_directive_variants_upper:.PARAMS": (
        "ngspice reads .params as .param; the lexer reads it as an unknown directive, "
        "so no route holds its parameter"
    ),
    "deck/card_grid_directive_variants_lower:.opt": (
        "ngspice reads .opt as .options; the lexer names it .opt, and which spellings "
        "are .options is decided apart in each module that reads options"
    ),
    "deck/card_grid_directive_variants_upper:.OPT": (
        "ngspice reads .opt as .options; the lexer names it .opt, and which spellings "
        "are .options is decided apart in each module that reads options"
    ),
}

#: Cells whose reading ngspice's batch log cannot show, with the reason.
NGSPICE_UNSHOWN: dict[str, str] = {
    "deck/save_omits_meas:.save": "ngspice's batch log prints every node's operating point whatever .save names.",
    "deck/card_grid_save_lower:.save": "ngspice's batch log prints every node's operating point whatever .save names.",
    "deck/card_grid_save_upper:.SAVE": "ngspice's batch log prints every node's operating point whatever .save names.",
}


@needs_ngspice
@pytest.mark.parametrize("mode", dr.NGSPICE_MODES)
def test_the_server_reads_each_deck_for_ngspice_as_ngspice_does(mode: str, tmp_path: Path):
    """Each deck of the grid run by the ngspice on PATH in ``mode``: where its
    node voltage shows one of a cell's readings, every route of the server's
    reading for ngspice that answers gives that reading. A deck ngspice
    stops on (for another card, as often as for the cell's) and a cell it
    reads some way the grid declares no reading for are not compared."""
    wrong = []
    compared = 0
    for case_id in _cases():
        deck = dr.stage_case(case_id, tmp_path / case_id.replace("/", "_"))
        voltages, printed = dr.ngspice_run(deck, mode)
        if voltages is None:
            continue
        readings = dr.read_offline(deck, simulator="ngspice", mode=mode)
        for cell in GRID.cells_of(case_id):
            if cell.id in NGSPICE_DIFFERS or cell.id in NGSPICE_UNSHOWN:
                continue
            if "printed" in cell.extra:
                ngspice = "read" if cell.extra["printed"] in printed else None
            else:
                ngspice = dr.reading_of(_volts(cell), cell.extra["node"], voltages)
            if ngspice is None:
                continue
            compared += 1
            for route, answer in _server_readings(cell, readings).items():
                if not dr.same_reading(answer, dr.as_answer(ngspice)):
                    wrong.append((cell.id, route, answer, ngspice))
    assert compared, "ngspice showed no cell's reading"
    assert not wrong, wrong
    assert set(NGSPICE_DIFFERS) | set(NGSPICE_UNSHOWN) <= {cell.id for cell in GRID.cells}
    assert all(NGSPICE_DIFFERS.values()) and all(NGSPICE_UNSHOWN.values())
