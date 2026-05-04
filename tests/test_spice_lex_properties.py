"""Hypothesis property tests for ``lib/spice_lex.py``.

Two load-bearing invariants:

1. **Round-trip**: ``emit(lex(text).cards) == text`` for any
   well-formed netlist. Verified by generating netlists from a small
   grammar and checking byte-identical round-trip.

2. **Parse-mutate-parse**: after a typed-view setter changes a value,
   re-lexing the emitted text produces the same value. Catches drift
   in canonical-form rerender paths.

The grammars produce valid SPICE-flavour netlists. Random ASCII text
isn't valid SPICE and would mostly fail (correctly) for reasons that
have nothing to do with the parser's invariants.
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
_ident_rest = st.text(
    alphabet=string.ascii_letters + string.digits + "_", min_size=0, max_size=8
)
_ident = st.builds(lambda h, t: h + t, _ident_first, _ident_rest)

# Distinct identifier for refs that must start with an element prefix.
_resistor_ref = st.builds(lambda t: "R" + t, _ident_rest)
_capacitor_ref = st.builds(lambda t: "C" + t, _ident_rest)
_mosfet_ref = st.builds(lambda t: "M" + t, _ident_rest)

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

# Node names — short identifiers or "0".
_node = st.one_of(_ident, st.just("0"), st.just("vdd"), st.just("vss"))

# Whitespace separator between tokens — at least one space.
_ws = st.sampled_from([" ", "  ", "   ", "\t", " \t"])


# ---------------------------------------------------------------------------
# Card-line strategies
# ---------------------------------------------------------------------------


def _resistor_line() -> st.SearchStrategy[str]:
    return st.builds(
        lambda r, n1, n2, v, ws1, ws2, ws3: f"{r}{ws1}{n1}{ws2}{n2}{ws3}{v}\n",
        _resistor_ref, _node, _node, _value, _ws, _ws, _ws,
    )


def _capacitor_line() -> st.SearchStrategy[str]:
    return st.builds(
        lambda r, n1, n2, v, ws1, ws2, ws3: f"{r}{ws1}{n1}{ws2}{n2}{ws3}{v}\n",
        _capacitor_ref, _node, _node, _value, _ws, _ws, _ws,
    )


def _mosfet_line() -> st.SearchStrategy[str]:
    return st.builds(
        lambda r, d, g, s, b, m, w, ll: f"{r} {d} {g} {s} {b} {m} W={w} L={ll}\n",
        _mosfet_ref, _node, _node, _node, _node, _ident, _value, _value,
    )


def _model_line() -> st.SearchStrategy[str]:
    return st.builds(
        lambda n, vto, kp: f".MODEL {n} NMOS(VTO={vto} KP={kp})\n",
        _ident, _value, _value,
    )


def _param_line() -> st.SearchStrategy[str]:
    return st.builds(
        lambda n, v: f".PARAM {n}={v}\n",
        _ident, _value,
    )


def _comment_line() -> st.SearchStrategy[str]:
    """Star-comment lines. Body is restricted ASCII to avoid encoding
    issues; no embedded newlines."""
    safe = st.text(
        alphabet=string.ascii_letters + string.digits + " _-+/",
        min_size=0, max_size=40,
    )
    return st.builds(lambda s: f"* {s}\n", safe)


def _blank_line() -> st.SearchStrategy[str]:
    return st.just("\n")


def _any_card_line() -> st.SearchStrategy[str]:
    return st.one_of(
        _resistor_line(),
        _capacitor_line(),
        _mosfet_line(),
        _model_line(),
        _param_line(),
        _comment_line(),
        _blank_line(),
    )


def _netlist() -> st.SearchStrategy[str]:
    """Build a netlist from 1 to 10 card lines."""
    return st.lists(_any_card_line(), min_size=1, max_size=10).map("".join)


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


@given(_netlist())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_round_trip_byte_faithful(text: str) -> None:
    """``emit(lex(text).cards) == text`` for every generated netlist."""
    result = lex(text)
    # Warnings are OK (e.g. unmatched .ENDS); round-trip still required.
    assert emit(result.cards) == text


# ---------------------------------------------------------------------------
# Parse-mutate-parse property
# ---------------------------------------------------------------------------


@given(name=_ident, old_v=_value, new_v=_value)
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
def test_model_set_param_round_trip(
    name: str, old_vto: str, new_vto: str, kp: str
) -> None:
    """``set_param`` on a model card survives a re-lex."""
    text = f".MODEL {name} NMOS(VTO={old_vto} KP={kp})\n"
    cards = lex(text).cards
    view = ModelCard.from_card(cards[0])
    view.set_param("VTO", new_vto)
    out = emit(cards)
    re_view = ModelCard.from_card(lex(out).cards[0])
    assert re_view.params.get("VTO") == new_vto
    assert re_view.params.get("KP") == kp  # unrelated param preserved


@given(ref=_resistor_ref, n1=_node, n2=_node, old_v=_value, new_v=_value)
@settings(max_examples=100)
def test_resistor_set_value_round_trip(
    ref: str, n1: str, n2: str, old_v: str, new_v: str
) -> None:
    """``InstanceLine.set_value`` round-trips for a resistor."""
    text = f"{ref} {n1} {n2} {old_v}\n"
    cards = lex(text).cards
    view = InstanceLine.from_card(cards[0])
    view.set_value(new_v)
    out = emit(cards)
    re_view = InstanceLine.from_card(lex(out).cards[0])
    assert re_view.value == new_v
    assert re_view.nodes == [n1, n2]
