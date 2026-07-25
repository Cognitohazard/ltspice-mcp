"""Strict variation grammar, expansion ordering, and deck materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    MismatchRule,
    RandomVariation,
    Variation,
    VariationError,
    expand_variations,
    materialize_variants,
)
from ltspice_mcp.tools.advanced import MonteCarloMismatchRule


def _deck(path: Path, circuit_id: str = "dut") -> CircuitDeck:
    text = (
        "R1 in out 1k\n"
        "C1 out 0 1n\n"
        "V1 in 0 1\n"
        ".param gain=2\n"
        ".model NM NMOS(VTO=0.7 KP=100u)\n"
        "M1 out in 0 0 NM W=10u L=1u\n"
        ".op\n"
        ".end\n"
    )
    path.write_text(text)
    return CircuitDeck(circuit_id, path, text)


def _random(*, seed: int = 7, runs: int = 2, applies_to=None) -> RandomVariation:
    return RandomVariation.model_validate(
        {
            "kind": "random",
            "runs": runs,
            "seed": seed,
            "applies_to": applies_to,
            "rules": [
                {
                    "rule": "component",
                    "target": "R1",
                    "tolerance": 0.1,
                    "distribution": "normal",
                }
            ],
        }
    )


class TestAssignExpansion:
    def test_grid_is_cartesian_and_stable(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variation = AssignVariation(
            kind="assign",
            assign={"R1": ["1k", "2k"], "C1": ["1n", "2n"]},
        )

        cases = expand_variations([circuit], [variation])

        assert [case.assignments for case in cases] == [
            {"R1": "1k", "C1": "1n"},
            {"R1": "1k", "C1": "2n"},
            {"R1": "2k", "C1": "1n"},
            {"R1": "2k", "C1": "2n"},
        ]

    def test_zip_is_lockstep(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variation = AssignVariation(
            kind="assign",
            combine="zip",
            assign={"R1": ["1k", "2k"], "C1": ["1n", "2n"]},
        )

        cases = expand_variations([circuit], [variation])

        assert [case.assignments for case in cases] == [
            {"R1": "1k", "C1": "1n"},
            {"R1": "2k", "C1": "2n"},
        ]

    def test_zip_rejects_unequal_lengths(self):
        with pytest.raises(ValidationError, match="equal lengths"):
            AssignVariation(
                kind="assign",
                combine="zip",
                assign={"R1": ["1k"], "C1": ["1n", "2n"]},
            )

    def test_si_suffix_values_validate_and_materialize(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variation = TypeAdapter(Variation).validate_python(
            {"kind": "assign", "assign": {"R1": ["2.2k"]}}
        )
        expanded = expand_variations([circuit], [variation])

        variants = materialize_variants(circuit, expanded, tmp_path / "staged")

        assert "R1 in out 2.2k" in variants[0].text
        assert variants[0].sha256 == hashlib.sha256(variants[0].text.encode("utf-8")).hexdigest()

    def test_b_source_assignment_preserves_source_kind_and_nodes(self, tmp_path: Path):
        path = tmp_path / "behavioral.cir"
        text = "V1 in 0 1\nB1 out 0 V=V(in)\nR1 out 0 1k\n.op\n.end\n"
        circuit = CircuitDeck("behavioral", path, text)
        variation = AssignVariation(
            kind="assign",
            assign={"B1": ["{V(in)*2}"]},
        )

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "staged",
        )

        assert "B1 out 0 V={V(in)*2}" in variants[0].text

    def test_pulse_source_assignment_preserves_complete_source_card(self, tmp_path: Path):
        path = tmp_path / "pulse.cir"
        text = "V1 in 0 PULSE(0 1 1n 1n 1n 5n 10n)\nR1 in 0 1k\n.tran 1n 20n\n.end\n"
        circuit = CircuitDeck("pulse", path, text)
        pulse = "PULSE(0 5 2n 1n 1n 4n 10n)"
        variation = AssignVariation(kind="assign", assign={"V1": [pulse]})

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "staged",
        )

        assert f"V1 in 0 {pulse}" in variants[0].text

    def test_param_wins_over_same_named_component(self, tmp_path: Path):
        path = tmp_path / "collision.cir"
        text = "R1 in out 1k\n.param R1=7\n.op\n.end\n"
        circuit = CircuitDeck("collision", path, text)
        variation = AssignVariation(kind="assign", assign={"R1": [9]})

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "staged",
        )

        assert ".param R1=9" in variants[0].text
        assert "R1 in out 1k" in variants[0].text

    def test_nonfirst_param_on_shared_card_is_resolved_and_rewritten(self, tmp_path: Path):
        path = tmp_path / "multi-param.cir"
        text = "R1 in out 1k\n.param first=1 target=2 last=3\n.op\n.end\n"
        circuit = CircuitDeck("multi", path, text)
        variation = AssignVariation(kind="assign", assign={"target": ["4.7k"]})

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "staged",
        )

        assert ".param first=1 target=4.7k last=3" in variants[0].text

    def test_model_swap_accepts_model_names(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variation = AssignVariation(
            kind="assign",
            assign={"M*@model": ["SLOW", "FAST"]},
        )

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "staged",
        )

        assert "M1 out in 0 0 SLOW" in variants[0].text
        assert "M1 out in 0 0 FAST" in variants[1].text

    def test_ambiguous_target_is_an_error(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variation = AssignVariation(kind="assign", assign={"missing": [1]})

        with pytest.raises(VariationError, match="neither a declared"):
            expand_variations([circuit], [variation])

    def test_stable_order_across_assign_entries(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        variations: list[Variation] = [
            AssignVariation(kind="assign", assign={"R1": [1, 2]}),
            AssignVariation(kind="assign", assign={"C1": [3, 4]}),
        ]

        cases = expand_variations([circuit], variations)

        assert [tuple(case.assignments.values()) for case in cases] == [
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
        ]


class TestCrossCircuitValidation:
    def test_applies_to_routes_only_to_named_circuit(self, tmp_path: Path):
        first = _deck(tmp_path / "a.cir", "a")
        second = _deck(tmp_path / "b.cir", "b")
        variation = AssignVariation(
            kind="assign",
            applies_to=["a"],
            assign={"R1": [1, 2]},
        )

        cases = expand_variations([first, second], [variation])

        assert [(case.circuit_id, case.case_index) for case in cases] == [
            ("a", 0),
            ("a", 1),
            ("b", 0),
        ]

    def test_duplicate_circuit_ids_are_rejected(self, tmp_path: Path):
        first = _deck(tmp_path / "a.cir", "same")
        second = _deck(tmp_path / "b.cir", "SAME")

        with pytest.raises(VariationError, match="duplicated"):
            expand_variations([first, second], [])

    def test_missing_applies_to_id_is_rejected(self, tmp_path: Path):
        circuit = _deck(tmp_path / "a.cir", "a")
        variation = AssignVariation(
            kind="assign",
            applies_to=["missing"],
            assign={"R1": [1]},
        )

        with pytest.raises(VariationError, match="unknown circuit"):
            expand_variations([circuit], [variation])


class TestRandomExpansion:
    def test_only_one_random_entry_is_allowed(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")

        with pytest.raises(VariationError, match="At most one random"):
            expand_variations([circuit], [_random(), _random(seed=8)])

    def test_seed_is_repeatable_with_distinct_case_streams(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        expanded = expand_variations([circuit], [_random(seed=42, runs=3)])

        first = materialize_variants(circuit, expanded, tmp_path / "one")
        second = materialize_variants(circuit, expanded, tmp_path / "two")

        assert [case.assignments for case in first] == [case.assignments for case in second]
        draws = [case.assignments["random:component:R1"] for case in first]
        assert len(set(draws)) == 3

    def test_case_cap_applies_after_cross_circuit_expansion(self, tmp_path: Path):
        first = _deck(tmp_path / "a.cir", "a")
        second = _deck(tmp_path / "b.cir", "b")
        variation = AssignVariation(kind="assign", assign={"R1": [1, 2, 3]})

        with pytest.raises(VariationError, match="configured maximum"):
            expand_variations([first, second], [variation], max_cases=5)

    def test_variation_id_reach_is_asymmetric_and_documented(self, tmp_path: Path):
        """A random id is recorded on every case it produces; an assign id
        reaches no case at all (several assign entries combine into one case),
        so each field description has to say which one it is."""
        circuit = _deck(tmp_path / "dut.cir")
        assign = AssignVariation(kind="assign", id="corner_sweep", assign={"R1": [1, 2]})
        random_variation = _random(runs=1).model_copy(update={"id": "mc"})

        expanded = expand_variations([circuit], [assign, random_variation])
        cases = materialize_variants(circuit, expanded, tmp_path / "out")

        assert {case.assignments["_random_id"] for case in cases} == {"mc"}
        assert not any(
            "corner_sweep" in (key, value)
            for case in cases
            for key, value in case.assignments.items()
        )

        assign_doc = AssignVariation.model_fields["id"].description or ""
        random_doc = RandomVariation.model_fields["id"].description or ""
        assert "_random_id" in random_doc
        assert "NOT" in assign_doc and "assignments" in assign_doc

    def test_mismatch_model_fields_match_shipped_tool_model(self):
        assert set(MismatchRule.model_fields) - {"rule"} == set(
            MonteCarloMismatchRule.model_fields
        )
        assert MismatchRule(rule="mismatch").model_dump(exclude={"rule"}) == (
            MonteCarloMismatchRule().model_dump()
        )
