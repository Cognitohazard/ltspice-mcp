"""What a ``.param``, ``.model`` or ``.subckt`` card declares, on every route
that answers it.

The question is the declaration grid's (``[grid.declaration]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). The views of the three
cards decide it (``ParamCard``, ``ModelCard``, ``SubcktCard``) with the reader
of every assignment of a ``.param`` card (``assigned_params``), and the
readers of a deck's declarations ask it: the lexer's reading of the deck and
of a deck read for an edit, the targets a variation edits, the hierarchy,
which evaluates each element's value from them, and a run's source
amplitudes, which read ``.param`` constants with a pattern of their own
(``tests/_deck_readings.py`` runs them).

A cell is one name a card declares, whose value sets one node's voltage in the
``.op`` raw. Every recorded cell, on each build, runs through every route: a
route that reads the name otherwise than the recording shows LTspice read it
is a difference, declared on its cell (``differs``) and counted in
``KNOWN_DIFFERENCES``. The same decks run live on ngspice in each
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
from ltspice_mcp.state import SessionState
from tests import _deck_readings as dr
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid

GRID = grid("declaration")

#: How many routes answer otherwise than a recorded cell, per build, each
#: declared on its cell (``differs``). A count may only fall.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 1, "ltspice17": 1}


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
    return cell.extra.get("subject", f"param {cell.item}")


def _volts(cell: Cell) -> dict[str, Any]:
    return cell.extra["volts"]


def _server_readings(cell: Cell, readings: dr.DeckReadings) -> dict[str, dr.Answer]:
    """Each route's reading of the declared name, and of the element whose
    value it is, where the cell names one."""
    found = dict(readings.answers(_subject(cell)))
    if via := cell.extra.get("via"):
        found |= {
            f"{route} ({via})": answer
            for route, answer in readings.answers(f"element {via}").items()
        }
    return found


# The index, held to the decks and to the code ---------------------------------


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_names_what_its_deck_declares(cell: Cell):
    from ltspice_mcp.lib.encoding import read_spice_text
    from tests.ltspice_recorder import INPUTS

    text = read_spice_text(INPUTS / rec.CASES.case(cell.case).source).lower()
    kind, name, *rest = _subject(cell).split()
    assert kind in ("param", "model", "level", "default"), cell.id
    assert name == cell.item.lower() or kind in ("model", "default"), cell.id
    assert re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text), cell.id
    if rest:
        assert re.search(rf"(?<!\w){re.escape(rest[0])}\s*=", text), cell.id
    if via := cell.extra.get("via"):
        assert re.search(rf"(?m)^{re.escape(via.lower())}\s.*\{{{name}\}}", text), cell.id
    for key in _volts(cell):
        dr.as_answer(key)


def test_every_form_the_card_readers_accept_is_a_cell_or_outside_the_grid():
    """The forms the three views and ``assigned_params`` tell apart are each
    a value of an axis: a ``.param`` card's first and later assignments and a
    value in braces or quotes, spaces round its sign (the tokenizer merges
    them), a ``.model`` card's parameters in parentheses or bare and its level
    as a number or not, and a ``.subckt`` card's defaults after ``params:`` or
    after its first assignment. The forms they read alike are listed with the
    reason."""
    axes = {name: set(axis.values) for name, axis in GRID.axes.items()}
    assert axes["param_form"] >= {"one", "several_first", "several_later", "braces", "quotes"}
    assert {"spaced_both", "spaced_before", "spaced_after"} <= axes["param_form"]
    assert axes["model_params"] == {"parens", "bare"}
    assert axes["model_level"] == {"number", "computed"}
    assert axes["subckt_params"] == {"params_marker", "first_assignment"}
    model = pyinspect.getsource(spice_lex_views.ModelCard.from_card)
    assert "TokenKind.PARENED" in model and "TokenKind.KEY_VALUE" in model
    assert "parse_spice_value" in model, "a level is a number or not"
    subckt = pyinspect.getsource(spice_lex_views.SubcktCard.from_card)
    assert '"params"' in subckt and "implicit PARAMS:" in subckt
    param = pyinspect.getsource(spice_lex_views.ParamCard.from_card)
    assert "tokens[1]" in param, "ParamCard reads the first assignment"
    assert set(OUTSIDE_THE_GRID) and all(OUTSIDE_THE_GRID.values())


#: Forms the card readers accept that are not cells, with the reason.
OUTSIDE_THE_GRID: dict[str, str] = {
    "a .param value in quotes inside braces": (
        "the tokenizer reads braces and quotes as one value each way, the braces and "
        "quotes cells hold each"
    ),
    "a .model card's type in quotes": (
        "ModelCard strips quotes from its type; no deck names a type in quotes, and the "
        "type is not what the card declares of its parameters"
    ),
    "a .subckt default in braces": (
        "a default's value is an expression like any .param value, which the "
        "expression-value grid covers"
    ),
    "a .model card's level given twice": (
        "ModelCard takes the first; which copy a simulator keeps is the question of two "
        "definitions of one name, a later grid's"
    ),
}


def test_a_recorded_cell_has_no_prediction_left():
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, f"{cell.id} is recorded on {recorded}: settle it"
        else:
            assert "reads" in cell.extra, f"{cell.id} declares no reading"


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(len(dr.differs_declared(cell, build)) for cell in GRID.cells)
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route -------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_declaration_reads_as_ltspice_read_it_on_every_route(
    build: str, case_id: str, tmp_path: Path, state_no_sim: SessionState, work_dir: Path
):
    voltages = dr.recorded_voltages(build, case_id, tmp_path)
    deck = dr.stage_case(case_id, work_dir)
    readings = await dr.read_every_route(state_no_sim, deck)
    found: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    for cell in GRID.cells_of(case_id):
        ltspice = dr.reading_of(_volts(cell), cell.extra["node"], voltages)
        assert ltspice is not None, f"{cell.id}: {build} gave none of the cell's readings"
        assert ltspice == dr.declared_reading(cell, build), (cell.id, build, ltspice)
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
NGSPICE_DIFFERS: dict[str, str] = {}


@needs_ngspice
@pytest.mark.parametrize("mode", dr.NGSPICE_MODES)
def test_the_server_reads_each_declaration_for_ngspice_as_ngspice_does(mode: str, tmp_path: Path):
    """Each deck of the grid run by the ngspice on PATH in ``mode``: where its
    node voltage shows one of a cell's readings, every route of the server's
    reading for ngspice that answers gives that reading. A deck ngspice stops
    on is not compared, nor a route a cell declares to differ from LTspice:
    that reading is the server's for every simulator."""
    wrong = []
    compared = 0
    for case_id in _cases():
        deck = dr.stage_case(case_id, tmp_path / case_id.replace("/", "_"))
        voltages, _ = dr.ngspice_run(deck, mode)
        if voltages is None:
            continue
        readings = dr.read_offline(deck, simulator="ngspice", mode=mode)
        for cell in GRID.cells_of(case_id):
            if cell.id in NGSPICE_DIFFERS:
                continue
            ngspice = dr.reading_of(_volts(cell), cell.extra["node"], voltages)
            if ngspice is None:
                continue
            compared += 1
            expected = {route for _, route in dr.differs_declared(cell, "ltspice26")}
            expected |= {route for _, route in dr.differs_declared(cell, "ltspice17")}
            for route, answer in _server_readings(cell, readings).items():
                if route not in expected and not dr.same_reading(answer, dr.as_answer(ngspice)):
                    wrong.append((cell.id, route, answer, ngspice))
    assert compared, "ngspice showed no cell's reading"
    assert not wrong, wrong
    assert set(NGSPICE_DIFFERS) <= {cell.id for cell in GRID.cells}
    assert all(NGSPICE_DIFFERS.values())
