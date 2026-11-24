"""How many nodes an element card takes, and what follows them, on every route
that answers it.

The question is the element-node grid's (``[grid.element-node]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). ``ElementSpec`` (the
shapes in ``ELEMENT_SPECS``) and ``read_instance`` decide it, and every reader
of a deck's elements asks it: the components listing of ``inspect`` and of the
Python API, the netlist graph a comparison reads, the hierarchy, and the
dangling-node check, which counts a card's nodes with a table of its own
(``spice_validator._TERMINAL_COUNTS``). ``tests/_deck_readings.py`` runs them.

A cell is one element card whose node names no other card uses, so a word of
the card that the run's results hold as a node voltage is a node LTspice gave
it, and a word that is no node is not there. Every recorded cell, on each
build, runs through every route: a route whose nodes, or whose model, differ
from LTspice's is a difference, declared on its cell (``differs``) and counted
in ``KNOWN_DIFFERENCES``. The same decks run live on ngspice in each
compatibility mode.
"""

from __future__ import annotations

import inspect as pyinspect
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import spice_lex_views
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_validator import _TERMINAL_COUNTS
from ltspice_mcp.state import SessionState
from tests import _deck_readings as dr
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests.ltspice_recorder import INPUTS

GRID = grid("element-node")

#: How many routes answer otherwise than a recorded cell, per build, each
#: declared on its cell (``differs``). A count may only fall.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 7, "ltspice17": 2}


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


def _words(cell: Cell) -> list[str]:
    """The words of the cell's card after its reference, read from the deck's
    text as a person reads it: the card is the first line that starts with
    the reference (a subcircuit's body may reuse it after)."""
    text = read_spice_text(INPUTS / rec.CASES.case(cell.case).source)
    lines = [
        line
        for line in text.splitlines()
        if line.split(None, 1)[:1] and line.split(None, 1)[0].lower() == cell.item.lower()
    ]
    return lines[0].split()[1:]


def _nodes_shown(cell: Cell, voltages: dict[str, float]) -> frozenset[str]:
    """The words of the cell's card that ``voltages`` hold as nodes."""
    return frozenset(
        word.lower()
        for word in _words(cell)
        if word.lower() not in dr.GROUND and f"v({word.lower()})" in voltages
    )


def _differences(
    cell: Cell, shown: frozenset[str], readings: dr.DeckReadings
) -> set[tuple[str, str]]:
    found = {
        (cell.item, route)
        for route, nodes in readings.nodes(cell.item).items()
        if frozenset(nodes) != shown
    }
    if "model" in cell.extra:
        found |= {
            (cell.item, f"{route} (model)")
            for route, model in readings.models(cell.item).items()
            if route != "spice_validator._instance_terminals" and model != cell.extra["model"]
        }
    return found


# The index, held to the decks and to the code ---------------------------------


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_is_a_card_of_its_deck_whose_nodes_are_words_of_it(cell: Cell):
    words = [w.lower() for w in _words(cell)]
    assert set(cell.extra["nodes"]) <= set(words), cell.id
    if "model" in cell.extra:
        assert cell.extra["model"] in words, cell.id
    letter = cell.coordinates["letter"] or next(
        (v for build in rec.BUILDS if (v := cell.at(build)["letter"])), ""
    )
    if letter:
        assert cell.item[:1].upper() == letter, cell.id


def _letters_the_view_compares() -> set[str]:
    source = pyinspect.getsource(spice_lex_views.InstanceLine.from_card)
    found = set(re.findall(r'prefix (?:==|in) \(?"([A-Z@])"', source))
    found |= {
        letter
        for group in re.findall(r"prefix in \(([^)]*)\)", source)
        for letter in re.findall(r'"([A-Z@])"', group)
    }
    return found


def test_every_letter_the_code_knows_is_a_value_of_the_letter_axis():
    """Every letter ``ElementSpec``'s table and the view's other tables know,
    and every letter the view compares a prefix with, is a letter of the grid,
    so a letter the code learns needs cells; so is every letter of the
    dangling-node check's own table."""
    known = (
        set(spice_lex_views.ELEMENT_SPECS)
        | set(spice_lex_views._EXACT_NODE_COUNT)
        | set(spice_lex_views._DEFINING_KEYS)
        | set(spice_lex_views._DEVICE_TAIL_PREFIXES)
        | _letters_the_view_compares()
        | set(_TERMINAL_COUNTS)
    )
    assert known <= set(GRID.axes["letter"].values), sorted(
        known - set(GRID.axes["letter"].values)
    )
    assert {"S", "W", "X"} <= _letters_the_view_compares()


def test_a_recorded_cell_has_no_prediction_left():
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, f"{cell.id} is recorded on {recorded}: settle it"


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(len(dr.differs_declared(cell, build)) for cell in GRID.cells)
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route -------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_card_has_the_nodes_ltspice_wired_on_every_route(
    build: str, case_id: str, tmp_path: Path, state_no_sim: SessionState, work_dir: Path
):
    voltages = dr.recorded_voltages(build, case_id, tmp_path)
    deck = dr.stage_case(case_id, work_dir)
    readings = await dr.read_every_route(state_no_sim, deck)
    found: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    for cell in GRID.cells_of(case_id):
        if voltages is None:
            assert build in cell.refused, f"{cell.id}: {build} refused the deck"
            continue
        assert build not in cell.refused, f"{cell.id}: {build} ran the deck"
        shown = _nodes_shown(cell, voltages)
        assert shown == frozenset(cell.extra["nodes"]), (
            f"{cell.id}: {build} wired {sorted(shown)}, not the nodes the form names"
        )
        assert readings.nodes(cell.item), f"{cell.id}: no route reads its nodes"
        found |= _differences(cell, shown, readings)
        declared |= dr.differs_declared(cell, build)
    assert found == declared, (
        f"answered otherwise, not declared: {sorted(found - declared)}; "
        f"declared, not found: {sorted(declared - found)}"
    )


# ngspice, live ----------------------------------------------------------------

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: Cells where ngspice wires otherwise than the server's reading for ngspice,
#: with the reason. Not asserted to differ: another ngspice version may agree.
NGSPICE_DIFFERS: dict[str, str] = {
    **{
        f"deck/element_grid_nothing_{letter.lower()}:{letter}n": (
            f"ngspice-42 ignores this {letter} card, which has nothing after its nodes; the view reads "
            "its second word as its value, and the dangling-node check counts that word as a "
            "node"
        )
        for letter in ("R", "C", "L")
    },
    "deck/element_grid_values:Kk": (
        "a K card couples two inductors and has no node, as both LTspice builds "
        "wire it; ngspice was not run on this cell since the view stopped reading "
        "the inductors' names as its nodes, so its agreement is not established"
    ),
}


@needs_ngspice
@pytest.mark.parametrize("mode", dr.NGSPICE_MODES)
def test_the_server_reads_each_card_for_ngspice_with_the_nodes_ngspice_wires(
    mode: str, tmp_path: Path
):
    """Each deck of the grid run by the ngspice on PATH in ``mode``: where it
    runs the deck, every route of the server's reading for ngspice gives each
    card the nodes ngspice's operating point holds of its words. A deck
    ngspice stops on is not compared."""
    wrong = []
    compared = 0
    for case_id in _cases():
        deck = dr.stage_case(case_id, tmp_path / case_id.replace("/", "_"))
        if "\n.ac" in deck.read_text(encoding="utf-8", errors="replace").lower():
            continue  # an AC run prints no operating point
        voltages, _ = dr.ngspice_run(deck, mode)
        if voltages is None:
            continue
        readings = dr.read_offline(deck, simulator="ngspice", mode=mode)
        for cell in GRID.cells_of(case_id):
            if cell.id in NGSPICE_DIFFERS:
                continue
            compared += 1
            shown = _nodes_shown(cell, voltages)
            wrong += [(cell.id, route) for _, route in _differences(cell, shown, readings)]
    assert compared, "ngspice ran no deck of the grid"
    assert not wrong, wrong
    assert set(NGSPICE_DIFFERS) <= {cell.id for cell in GRID.cells}
    assert all(NGSPICE_DIFFERS.values())


def test_a_card_is_read_from_the_deck_the_lexer_reads():
    """The words the index reads a card from are those of a card ``lex_deck``
    reads, so the index and the routes read one card."""
    for case_id in _cases():
        cards = lex_deck(read_spice_text(INPUTS / rec.CASES.case(case_id).source)).cards
        refs = {card.name.lower() for card in cards if card.kind == "instance" and card.name}
        for cell in GRID.cells_of(case_id):
            assert cell.item.lower() in refs, cell.id
