"""A part's value field is one number only when it is one token.

A source's field holds its DC value and its AC specification, as in
``V1 a 0 5V AC 1``, and each token is its own value. So no path reads such a
field by its leading number: not the value reader every path shares, not the
netlist compare behind ``verify_circuit``, and not the nominal a random rule
draws around.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.netlist_graph import compare_graphs
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    RandomVariation,
    VariationError,
    expand_variations,
    materialize_variants,
)

#: A source, its value field, and the same field with a different AC magnitude.
FIELDS = [("V1", "5V AC 1", "5V AC 2"), ("I1", "1m AC 2", "1m AC 3")]


def _deck(ref: str, field: str) -> str:
    return f"* sources\n{ref} a 0 {field}\nR1 a 0 1k\n.op\n.end\n"


@pytest.mark.parametrize(("ref", "field", "other"), FIELDS)
def test_a_field_of_several_tokens_is_not_one_value(ref: str, field: str, other: str):
    with pytest.raises(ValueError, match="one token"):
        parse_spice_value(field)


@pytest.mark.parametrize(("ref", "field", "other"), FIELDS)
def test_sources_that_differ_after_the_first_token_are_not_equivalent(
    ref: str, field: str, other: str
):
    assert compare_graphs(_deck(ref, field), _deck(ref, field)).equivalent
    assert not compare_graphs(_deck(ref, field), _deck(ref, other)).equivalent


@pytest.mark.parametrize(("ref", "field", "other"), FIELDS)
def test_a_random_rule_has_no_nominal_on_a_field_of_several_tokens(
    tmp_path: Path, ref: str, field: str, other: str
):
    """Drawing around the leading number would write the draw over a later
    token of the field (``5V AC 5.1``), so the rule is refused instead."""
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

    with pytest.raises(VariationError, match="no plain numeric nominal") as excinfo:
        materialize_variants(circuit, expand_variations([circuit], [variation]), tmp_path / "out")

    assert excinfo.value.code == "random_nominal_unavailable"
