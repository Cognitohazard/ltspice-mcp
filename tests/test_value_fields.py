"""A part's value field is one number only when it is one token.

A source's field holds its DC value and its AC specification, as in
``V1 a 0 5V AC 1``, and each token is its own value. So no path reads such a
field by its leading number: not the value reader every path shares, and not
the netlist compare behind ``verify_circuit``. A random rule draws around the
source's value alone, ``5V``, and keeps its ``AC 1``, which is the source's own
parameter (``read_source_text``, the source-parameters recordings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.netlist_graph import compare_graphs
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    RandomVariation,
    expand_variations,
    materialize_variants,
)

#: A source, its value field, and the same field with a different AC magnitude.
FIELDS = [("V1", "5V AC 1", "5V AC 2"), ("I1", "1m AC 2", "1m AC 3")]

#: A source and two spellings of one field: each token holds the same value.
SAME_FIELDS = [
    ("V1", "5V AC 1", "5.0V AC 1"),
    ("V1", "5V AC 1", "5V AC 1.0"),
    ("V1", "5V AC 1", "5v ac 1"),
    ("I1", "1m AC 2", "1mA AC 2"),
    ("I1", "1m AC 2", "1000u AC 2"),
]


def _deck(ref: str, field: str) -> str:
    return f"* sources\n{ref} a 0 {field}\nR1 a 0 1k\n.op\n.end\n"


@pytest.mark.parametrize("field", [field for _, field, _ in FIELDS])
def test_a_field_of_several_tokens_is_not_one_value(field: str):
    with pytest.raises(ValueError, match="one token"):
        parse_spice_value(field)


@pytest.mark.parametrize(("ref", "field", "other"), FIELDS)
def test_sources_that_differ_after_the_first_token_are_not_equivalent(
    ref: str, field: str, other: str
):
    assert compare_graphs(_deck(ref, field), _deck(ref, field)).equivalent
    assert not compare_graphs(_deck(ref, field), _deck(ref, other)).equivalent


@pytest.mark.parametrize(("ref", "field", "other"), SAME_FIELDS)
def test_sources_whose_tokens_hold_the_same_values_are_equivalent(
    ref: str, field: str, other: str
):
    """Each token is compared as a value, so a field is not told apart from
    the same field written another way."""
    assert compare_graphs(_deck(ref, field), _deck(ref, other)).equivalent


@pytest.mark.parametrize(("ref", "field"), [(ref, field) for ref, field, _ in FIELDS])
def test_a_random_rule_draws_a_sources_value_and_keeps_its_ac_magnitude(
    tmp_path: Path, ref: str, field: str
):
    """The draw replaces the source's value, never a later token of the field
    (``5V AC 5.1``), and the AC magnitude after it stays as written."""
    path = tmp_path / "dut.cir"
    text = _deck(ref, field)
    path.write_text(text, encoding="utf-8")
    circuit = CircuitDeck("dut", path, text)
    variation = RandomVariation.model_validate(
        {
            "kind": "random",
            "runs": 1,
            "seed": 3,
            "rules": [{"rule": "component", "target": ref, "tolerance": 0.1}],
        }
    )

    (case,) = materialize_variants(
        circuit, expand_variations([circuit], [variation]), tmp_path / "out"
    )

    (card,) = [line for line in case.text.splitlines() if line.split()[:1] == [ref]]
    value, own = field.split(None, 1)
    words = card.split()
    assert " ".join(words[4:]) == own
    assert words[3] != value
    assert parse_spice_value(words[3]) == pytest.approx(parse_spice_value(value), rel=0.5)
