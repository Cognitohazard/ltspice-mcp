"""The grids of recorded LTspice behaviour: each an index over the cases of
``inputs/cases.toml``, checked here against the cases and their decks.

A grid's own test (for the printed-measurement grid,
``test_measurement_answer.py``) reads each recorded cell through every route
that answers the grid's question. This module checks the index: every cell
names a case and an item that exist with the coordinates it declares, every
coordinate the grid requires has a cell, the functions the grid names exist
and are named by a behaviour, and the count of coordinates not yet recorded
only falls.
"""

from __future__ import annotations

import functools

import pytest

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    TokenKind,
    directive_head,
    lex_deck,
)
from ltspice_mcp.lib.spice_lex_views import MeasCard
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grids, named_code_exists
from tests.ltspice_recorder import INPUTS

GRIDS = grids()

#: How many coordinates of each grid have no recorded cell, per build. A
#: count may only fall: a recording that fills a coordinate lowers it here in
#: the same change. A new coordinate with no recording raises it, with a line
#: below saying why.
#:
#: printed-measurement: the decks of behaviour log-meas-grid fill every
#: coordinate but the .fra margins on XVII (raw/fra_step is recorded on
#: LTspice 26 only), and wait for a recording run on Windows.
UNRECORDED: dict[str, dict[str, int]] = {
    "printed-measurement": {"ltspice26": 468, "ltspice17": 468},
}


@pytest.mark.parametrize("key", list(GRIDS))
class TestGridIndex:
    def test_the_grid_states_its_question_and_its_functions(self, key: str):
        grid = GRIDS[key]
        assert grid.question and grid.functions
        assert set(grid.expect) <= set(grid.axes)
        for group in grid.groups:
            assert group and set(group) <= set(grid.axes), group

    def test_every_function_the_grid_names_exists_and_a_behaviour_names_it(self, key: str):
        """A grid's functions are those a recording is evidence for, so each is
        named by the ``model`` of a behaviour, which ``test_ltspice_recorder``
        checks in turn."""
        modelled = {name for b in rec.CASES.behaviours.values() for name in b.model}
        for name in GRIDS[key].functions:
            assert ":" in name, f"{name}: a grid names functions, as module:name"
            assert named_code_exists(name), f"{name} is not in the repository"
            assert name in modelled, f"no behaviour's model names {name}"

    def test_every_cell_names_a_case_and_coordinates_on_the_axes(self, key: str):
        grid = GRIDS[key]
        cases = {case.case_id for case in rec.CASES.cases}
        seen: set[tuple[str, str]] = set()
        for cell in grid.cells:
            assert cell.case in cases, f"{cell.id}: no such case"
            assert (cell.case, cell.item) not in seen, f"{cell.id} is indexed twice"
            seen.add((cell.case, cell.item))
            assert set(cell.refused) | set(cell.on) <= set(rec.BUILDS), cell.id
            assert all(cell.refused.values()), f"{cell.id}: a refusal says why"
            for build in rec.BUILDS:
                at = cell.at(build)
                assert set(at) <= set(grid.axes), (cell.id, at)
                for axis, value in at.items():
                    allowed = (*grid.axes[axis].values, grid.axes[axis].default)
                    assert value in allowed, f"{cell.id}: {axis}={value!r}"
                grid.expected(cell, build)

    def test_every_coordinate_has_a_cell_and_none_is_excluded(self, key: str):
        """Every coordinate the grid requires has a cell, recorded or waiting
        to be, and no cell sits where an exclusion says none can."""
        grid = GRIDS[key]
        held = {c for cell in grid.cells for b in rec.BUILDS for c in grid.covered(cell, b)}
        missing = [c for c in grid.required() if c not in held]
        assert not missing, f"coordinates with no cell: {missing[:20]} ({len(missing)})"
        for cell in grid.cells:
            for build in rec.BUILDS:
                at = cell.at(build)
                for rule in grid.exclusions:
                    assert not rule.covers(at), (cell.id, build, rule.reason)

    def test_the_count_of_coordinates_not_recorded_only_falls(self, key: str):
        pinned = UNRECORDED[key]
        assert set(pinned) == set(rec.BUILDS)
        for build in rec.BUILDS:
            missing = GRIDS[key].unrecorded(build, set(rec.manifest(build)["cases"]))
            assert len(missing) <= pinned[build], (
                f"{len(missing)} coordinates of {key} have no recording on {build}, more "
                f"than the {pinned[build]} pinned: record them, or raise the pin with the "
                f"reason ({missing[:10]})"
            )
            assert len(missing) == pinned[build], (
                f"{len(missing)} coordinates of {key} have no recording on {build}: lower "
                f"the pin from {pinned[build]}"
            )

    def test_a_cell_waiting_for_its_recording_is_on_a_pending_behaviour(self, key: str):
        """A cell whose case no build has recorded belongs to a behaviour
        whose ``pending`` says how to record it."""
        recorded = {case for build in rec.BUILDS for case in rec.manifest(build)["cases"]}
        for cell in GRIDS[key].cells:
            if cell.case not in recorded:
                behaviour = rec.CASES.case(cell.case).behaviour
                assert rec.CASES.behaviours[behaviour].pending, cell.id


# The printed-measurement grid, against its decks ------------------------------

MEASUREMENTS = GRIDS["printed-measurement"]

#: The words a sweepless analysis is written with, and the others.
_ANALYSES = {
    ".tran": "tran",
    ".ac": "ac",
    ".dc": "dc",
    ".noise": "noise",
    ".op": "op",
    ".tf": "tf",
}

#: The aggregate forms, by the word that starts their card.
_AGGREGATES = frozenset({"max", "min", "pp", "avg", "rms", "integ"})

#: ``.meas`` cards of the decks of a ``log-meas`` behaviour that are not
#: cells of the grid, with the reason. None today.
OUTSIDE_THE_GRID: dict[tuple[str, str], str] = {}

#: ``MeasCard`` knows no ``tf`` analysis word (``spice_lex_views.MeasAnalysis``),
#: so it reads ``.meas tf name ...`` as a measurement named ``tf``. Whether
#: LTspice takes ``tf`` there waits for the recordings of the
#: ``log/meas_grid_tf_*`` decks; until then the index reads the name after it.
MEAS_CARD_READS_TF_AS_THE_NAME = True


@functools.cache
def _deck_cards(source: str) -> tuple[SpiceCard, ...]:
    return tuple(lex_deck(read_spice_text(INPUTS / source)).cards)


def _meas_cards(source: str) -> dict[str, tuple[str, list[str]]]:
    """Each ``.meas`` card of a deck by its name in lower case: the name as
    written and the bare words of its body, in lower case, after the name."""
    found: dict[str, tuple[str, list[str]]] = {}
    for card in _deck_cards(source):
        if card.kind != "meas":
            continue
        try:
            meas = MeasCard.from_card(card)
        except SpiceLexError:
            # A card that does not parse (log/meas_bad_syntax): its name is the
            # lexer's, and its words are read from its text.
            words = card.body.split()
            name = words[2] if words[1].lower() in _ANALYSES.values() else words[1]
            found[name.lower()] = (name, [w.lower() for w in words[words.index(name) + 1 :]])
            continue
        words = [t.text.casefold() for t in meas.body_tokens if t.kind == TokenKind.BARE]
        name = meas.name
        if meas.analysis is None and name.lower() == "tf":
            assert MEAS_CARD_READS_TF_AS_THE_NAME
            name, words = meas.body_tokens[0].text, words[1:]
        found[name.lower()] = (name, words)
    return found


def _form(words: list[str]) -> str:
    first = words[0]
    if first in _AGGREGATES or first in ("when", "param"):
        return first
    if first == "find":
        return "find_when" if "when" in words else "find_at" if "at" in words else "find"
    if first == "deriv":
        return "deriv_when" if "when" in words else "deriv_at"
    if first == "trig":
        return "trig_targ"
    raise AssertionError(f"no form starts with {first!r}")


def _analysis_and_stepping(source: str) -> tuple[str, str]:
    heads = [directive_head(card) for card in _deck_cards(source) if card.kind == "directive"]
    analyses = {_ANALYSES[head] for head in heads if head in _ANALYSES}
    assert len(analyses) <= 1, (source, analyses)
    return (analyses.pop() if analyses else ""), ("yes" if ".step" in heads else "no")


def _source(cell: Cell) -> str:
    return rec.CASES.case(cell.case).source


@pytest.mark.parametrize("cell", MEASUREMENTS.cells, ids=lambda cell: cell.id)
def test_a_measurement_cell_is_a_card_of_its_deck_as_it_declares(cell: Cell):
    """The item is a ``.meas`` card of the case's deck, read through the
    lexer, whose words give the form the cell declares; the deck runs the
    analysis and the stepping the cell declares. A ``.fra`` margin has no card,
    and its deck runs ``.fra``."""
    source = _source(cell)
    analysis, stepped = _analysis_and_stepping(source)
    cards = _meas_cards(source)
    at = cell.coordinates
    assert at["stepped"] == stepped, (cell.id, stepped)
    if at["card"] == "fra":
        assert cell.item.lower() not in cards
        heads = {directive_head(c) for c in _deck_cards(source) if c.kind == "directive"}
        assert ".fra" in heads, cell.id
        return
    assert cell.item.lower() in cards, f"{cell.id}: the deck {source} has no such .meas card"
    name, words = cards[cell.item.lower()]
    assert name == cell.item, f"{cell.id}: the deck spells it {name!r}"
    assert at["analysis"] == analysis, (cell.id, analysis)
    assert at["form"] == _form(words), (cell.id, words)
    if at["name"] == "upper":
        assert name == name.upper() != name.lower(), cell.id
    elif at["name"] == "punctuation":
        assert any(char in name for char in ".-"), cell.id


def test_every_card_of_a_measurement_deck_is_a_cell():
    """Every ``.meas`` card of a deck of a ``log-meas`` behaviour is a cell of
    the grid for each case of that deck, or is listed outside it with the
    reason."""
    indexed = {(cell.case, cell.item.lower()) for cell in MEASUREMENTS.cells}
    missing = []
    for key in rec.CASES.behaviours:
        if not key.startswith("log-meas"):
            continue
        for case in rec.CASES.of(key):
            for name in _meas_cards(case.source):
                if (case.case_id, name) not in indexed and (
                    case.case_id,
                    name,
                ) not in OUTSIDE_THE_GRID:
                    missing.append(f"{case.case_id}:{name}")
    assert not missing, missing
    assert all(OUTSIDE_THE_GRID.values())


def test_every_recorded_log_that_prints_a_measurement_is_indexed():
    """A recorded log of any behaviour that prints a measurement has each of
    them in the grid, so the grid's test reads every printed measurement the
    recordings hold."""
    from ltspice_mcp.lib.log_parser import printed_measurements, read_log_text

    indexed = {(cell.case, cell.item.lower()) for cell in MEASUREMENTS.cells}
    missing = []
    for build in rec.BUILDS:
        for case_id, entry in rec.manifest(build)["cases"].items():
            log = f"{case_id}.log"
            if log not in entry["outputs"]:
                continue
            path = rec.recorded(build, log)
            if not path.stat().st_size:
                continue
            for name in printed_measurements(read_log_text(path)):
                if (case_id, name) not in indexed:
                    missing.append(f"{build}/{case_id}:{name}")
    assert not missing, missing


def test_meas_card_reads_the_tf_analysis_word_as_the_name():
    """Pinned: ``MeasCard`` reads ``.meas tf name FIND ...`` as a measurement
    named ``tf`` with no analysis. When it learns ``tf``, this fails, and
    ``MEAS_CARD_READS_TF_AS_THE_NAME`` and the reading above come out."""
    card = next(c for c in _deck_cards("log/meas_grid_tf_find.cir") if c.kind == "meas")
    meas = MeasCard.from_card(card)
    assert (meas.analysis, meas.name) == (None, "tf")
