"""The views that answer a question about one card for every reader: a
directive's arguments, a ``.param`` card's names, a measurement's kind, and
which parenthesised parts a card can be split into."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.errors import UnreadableCardError
from ltspice_mcp.lib.metrics import meas_kinds_from_netlist, noise_input_source_unit
from ltspice_mcp.lib.spice_lex import SpiceCard, lex_deck, unreadable_cards
from ltspice_mcp.lib.spice_lex_views import (
    MeasCard,
    ModelCard,
    assigned_params,
    directive_args,
)


def _card(line: str) -> SpiceCard:
    (card,) = [c for c in lex_deck(f"* t\n{line}\n.end\n").cards if c.kind not in ("end",)]
    return card


@pytest.mark.parametrize(
    ("line", "args"),
    [
        (".noise V(out) V1 dec 10 1 1Meg", ["V(out)", "V1", "dec", "10", "1", "1Meg"]),
        # A probe is one argument whatever spaces its parentheses hold.
        (".four 1k V(out, ref) V(x)", ["1k", "V(out, ref)", "V(x)"]),
        (".tran 0 1m 0 1u ; a comment", ["0", "1m", "0", "1u"]),
        (".step param R list 1 {2*a} 3", ["param", "R", "list", "1", "{2*a}", "3"]),
    ],
)
def test_a_directives_arguments_are_its_words_as_written(line: str, args: list[str]):
    assert directive_args(_card(line)) == args


def test_every_assignment_of_a_param_card_is_read():
    names = [name for name, _token in assigned_params(_card(".param a=1 B = {a*2} c='x'"))]
    assert names == ["a", "B", "c"]


def test_an_argument_reader_refuses_a_card_it_cannot_split():
    with pytest.raises(UnreadableCardError, match=r"line 2, '\.tran 0 \{1m'"):
        directive_args(_card(".tran 0 {1m"))


@pytest.mark.parametrize(
    "line",
    [".model D D(is={1e-14)", "B1 a 0 V=f(}x)", ".meas tran m FIND V(a) AT={1"],
)
def test_a_brace_left_open_inside_parentheses_leaves_the_card_unreadable(line: str):
    """A parenthesised group's parts are split again by the views, so the
    check every rule shares (``unreadable_cards``) reads inside it too."""
    card = _card(line)
    assert [c for c, _error in unreadable_cards([card])] == [card]


def test_a_braced_value_keeps_its_own_parentheses():
    """A brace's contents are one value, an expression: its parentheses are
    its own, as before."""
    assert unreadable_cards([_card("R1 p 0 {(1+2}")]) == []


def test_a_models_numeric_parameters_leave_expressions_out():
    view = ModelCard.from_card(_card(".model n NMOS(level=1 vto=0.7 kp={k*2} u0=(1)) "))
    assert view.numeric_params() == {"LEVEL": 1.0, "VTO": 0.7}


# --------------------------------------------------------------------------
# A measurement's name and kind are its card's, on every deck
# --------------------------------------------------------------------------


def _deck(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "d.cir"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_measurements_name_and_kind_are_read_from_its_card(tmp_path: Path):
    """The recipe's own pattern missed an indented card and keyed ``t.rise``
    as ``t``; the card's name is the lexer's, and the kind is the view's."""
    path = _deck(
        tmp_path,
        "* t\nV1 a 0 1\n"
        "  .meas tran t.rise WHEN V(a)=0.5\n"
        ".meas tran m_find FIND V(a) WHEN V(a)=0.5\n"
        ".measure m_max MAX V(a)\n"
        ".tran 1m\n.end\n",
    )
    assert meas_kinds_from_netlist(path) == {"t.rise": "when", "m_find": "value", "m_max": "value"}
    cards = [MeasCard.from_card(c) for c in lex_deck(path.read_text()).cards if c.kind == "meas"]
    assert [(m.name, m.bare_when) for m in cards] == [
        ("t.rise", True),
        ("m_find", False),
        ("m_max", False),
    ]


def test_the_noise_source_is_the_second_argument_however_the_output_is_written(tmp_path: Path):
    path = _deck(tmp_path, "* t\nI1 0 a 1\nR1 a 0 1k\n.noise V(a, 0) I1 dec 10 1 1k\n.end\n")
    assert noise_input_source_unit(path) == "A"
