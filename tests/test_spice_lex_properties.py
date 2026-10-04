"""Hypothesis property tests for ``lib/spice_lex.py``.

Two load-bearing invariants:

1. **Round-trip**: ``emit(lex(text).cards) == text`` for any
   well-formed netlist. Verified by generating netlists from a small
   grammar and checking byte-identical round-trip.

2. **Parse-mutate-parse**: after a typed-view setter changes a value,
   re-lexing the emitted text produces the same value. Catches drift
   in canonical-form rerender paths.

The grammars produce valid SPICE-flavour netlists: cards split over ``+``
continuation lines, ``\\n`` or ``\\r\\n`` line endings, braced expressions,
quoted names, ``;``/``$`` inline comments and ``.SUBCKT`` blocks. Random
ASCII text isn't valid SPICE and would mostly fail (correctly) for reasons
that have nothing to do with the parser's invariants.
"""

from __future__ import annotations

import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ltspice_mcp.lib.spice_lex import emit, lex
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    ModelCard,
    ParamCard,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

# SPICE identifier — letter followed by letter/digit/underscore. Avoid
# colliding with keywords.
_ident_first = st.sampled_from(string.ascii_letters)
_ident_rest = st.text(alphabet=string.ascii_letters + string.digits + "_", min_size=0, max_size=8)
_ident = st.builds(lambda h, t: h + t, _ident_first, _ident_rest)

# Distinct identifier for refs that must start with an element prefix.
_resistor_ref = st.builds(lambda t: "R" + t, _ident_rest)
_capacitor_ref = st.builds(lambda t: "C" + t, _ident_rest)
_mosfet_ref = st.builds(lambda t: "M" + t, _ident_rest)
_subckt_ref = st.builds(lambda t: "X" + t, _ident_rest)

# Numeric value with optional SPICE suffix.
_suffix = st.sampled_from(["", "k", "Meg", "u", "n", "p", "f", "m", ""])
_number = st.one_of(
    st.integers(min_value=1, max_value=999).map(str),
    st.builds(
        lambda i, d: f"{i}.{d}",
        st.integers(min_value=0, max_value=99),
        st.integers(min_value=0, max_value=99),
    ),
)
_value = st.builds(lambda n, s: f"{n}{s}", _number, _suffix)

# A braced expression in a value slot, e.g. ``{2*rload}``.
_braced = st.builds(
    lambda n, op, name: "{" + f"{n}{op}{name}" + "}",
    _number,
    st.sampled_from(["*", "+", "-", "/"]),
    _ident,
)
_value_or_expr = st.one_of(_value, _braced)

# Node names — short identifiers or "0".
_node = st.one_of(_ident, st.just("0"), st.just("vdd"), st.just("vss"))

# Whitespace separator between tokens — at least one space.
_ws = st.sampled_from([" ", "  ", "   ", "\t", " \t"])

# Comment text: restricted ASCII, no newlines.
_comment_text = st.text(alphabet=string.ascii_letters + string.digits + " _-+/", max_size=30)

# An optional trailing ``;`` or ``$`` comment on a card's last line.
_inline_comment = st.one_of(
    st.just(""),
    st.builds(lambda mark, s: f" {mark} {s}", st.sampled_from([";", "$"]), _comment_text),
)


# ---------------------------------------------------------------------------
# Card strategies: each card is a list of raw lines without line endings
# ---------------------------------------------------------------------------


@st.composite
def _laid_out(draw: st.DrawFn, tokens: list[str]) -> list[str]:
    """Lay ``tokens`` out on one line, or split them over ``+`` continuations,
    and end the card with an optional inline comment."""
    breaks = sorted(
        draw(st.sets(st.integers(min_value=1, max_value=len(tokens) - 1), max_size=2))
        if len(tokens) > 1
        else set()
    )
    lines: list[str] = []
    for start, end in zip([0, *breaks], [*breaks, len(tokens)], strict=True):
        text = "".join(tok + draw(_ws) for tok in tokens[start:end]).rstrip(" \t")
        lines.append(text if not lines else f"+{draw(_ws)}{text}")
    lines[-1] += draw(_inline_comment)
    return lines


def _two_terminal(ref: st.SearchStrategy[str]) -> st.SearchStrategy[list[str]]:
    return st.tuples(ref, _node, _node, _value_or_expr).flatmap(lambda t: _laid_out(list(t)))


def _mosfet() -> st.SearchStrategy[list[str]]:
    model = st.one_of(_ident, _ident.map(lambda n: f'"{n}"'))
    return st.tuples(
        _mosfet_ref, _node, _node, _node, _node, model, _value_or_expr, _value
    ).flatmap(lambda t: _laid_out([*t[:6], f"W={t[6]}", f"L={t[7]}"]))


def _model() -> st.SearchStrategy[list[str]]:
    return st.tuples(_ident, _value, _value).flatmap(
        lambda t: _laid_out([".MODEL", t[0], "NMOS", f"(VTO={t[1]}", f"KP={t[2]})"])
    )


def _param() -> st.SearchStrategy[list[str]]:
    return st.tuples(_ident, _value_or_expr).map(lambda t: [f".PARAM {t[0]}={t[1]}"])


def _include() -> st.SearchStrategy[list[str]]:
    return _ident.map(lambda n: [f'.include "models/{n}.lib"'])


def _comment() -> st.SearchStrategy[list[str]]:
    return _comment_text.map(lambda s: [f"* {s}"])


def _blank() -> st.SearchStrategy[list[str]]:
    return st.just([""])


def _leaf_card() -> st.SearchStrategy[list[str]]:
    return st.one_of(
        _two_terminal(_resistor_ref),
        _two_terminal(_capacitor_ref),
        _mosfet(),
        _model(),
        _param(),
        _include(),
        _comment(),
        _blank(),
    )


@st.composite
def _subckt(draw: st.DrawFn) -> list[str]:
    name = draw(_ident)
    ports = draw(st.lists(_ident, min_size=1, max_size=3))
    body = [line for card in draw(st.lists(_leaf_card(), max_size=3)) for line in card]
    return [f".SUBCKT {name} {' '.join(ports)}", *body, f".ENDS {name}"]


def _instance_of_subckt() -> st.SearchStrategy[list[str]]:
    return st.tuples(_subckt_ref, _node, _node, _ident).flatmap(lambda t: _laid_out(list(t)))


@st.composite
def _netlist(draw: st.DrawFn) -> str:
    """1 to 10 cards and subcircuit blocks, with one line ending throughout."""
    cards = draw(
        st.lists(
            st.one_of(_leaf_card(), _instance_of_subckt(), _subckt()), min_size=1, max_size=10
        )
    )
    eol = draw(st.sampled_from(["\n", "\r\n"]))
    return "".join(line + eol for card in cards for line in card)


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


@given(_netlist())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_round_trip_byte_faithful(text: str) -> None:
    """``emit(lex(text).cards) == text`` for every generated netlist."""
    result = lex(text)
    # Every generated block is closed and every '+' follows a card.
    assert result.warnings == []
    assert emit(result.cards) == text


# ---------------------------------------------------------------------------
# Parse-mutate-parse property
# ---------------------------------------------------------------------------


@given(name=_ident, old_v=_value_or_expr, new_v=_value_or_expr)
@settings(max_examples=100)
def test_param_set_value_round_trip(name: str, old_v: str, new_v: str) -> None:
    """After ``set_value`` the new value survives a re-lex."""
    text = f".PARAM {name}={old_v}\n"
    cards = lex(text).cards
    view = ParamCard.from_card(cards[0])
    view.set_value(new_v)
    out = emit(cards)
    re_view = ParamCard.from_card(lex(out).cards[0])
    assert re_view.value == new_v
    assert re_view.name == name


@given(name=_ident, old_vto=_value, new_vto=_value, kp=_value)
@settings(max_examples=100)
def test_model_set_param_round_trip(name: str, old_vto: str, new_vto: str, kp: str) -> None:
    """``set_param`` on a model card survives a re-lex."""
    text = f".MODEL {name} NMOS(VTO={old_vto} KP={kp})\n"
    cards = lex(text).cards
    view = ModelCard.from_card(cards[0])
    view.set_param("VTO", new_vto)
    out = emit(cards)
    re_view = ModelCard.from_card(lex(out).cards[0])
    assert re_view.params.get("VTO") == new_vto
    assert re_view.params.get("KP") == kp  # unrelated param preserved


@given(
    lines=st.tuples(_resistor_ref, _node, _node, _value_or_expr).flatmap(
        lambda t: _laid_out(list(t)).map(lambda lines: (t, lines))
    ),
    new_v=_value_or_expr,
    eol=st.sampled_from(["\n", "\r\n"]),
)
@settings(max_examples=100)
def test_resistor_set_value_round_trip(
    lines: tuple[tuple[str, str, str, str], list[str]], new_v: str, eol: str
) -> None:
    """``InstanceLine.set_value`` round-trips for a resistor, whatever its
    continuation layout, line ending or trailing comment."""
    (ref, n1, n2, _old), raw = lines
    text = "".join(line + eol for line in raw)
    cards = lex(text).cards
    view = InstanceLine.from_card(cards[0])
    view.set_value(new_v)
    out = emit(cards)
    assert out.endswith(eol)
    re_view = InstanceLine.from_card(lex(out).cards[0])
    assert re_view.ref == ref
    assert re_view.value == new_v
    assert re_view.nodes == [n1, n2]
