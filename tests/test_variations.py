"""Strict variation grammar, expansion ordering, and deck materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    DeckFile,
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


_CORE = ".subckt core in out\nR1 in out 1k\n.ends\n"


def _closure(
    tmp_path: Path,
    *,
    includes: dict[str, str],
    root_body: str,
    reference: str = ".include",
) -> CircuitDeck:
    """A deck whose root body is given verbatim, over staged include files."""
    staged = tmp_path / "staged"
    staged.mkdir(exist_ok=True)
    files = []
    references = ""
    for name, text in includes.items():
        path = staged / name
        path.write_text(text)
        files.append(DeckFile(path, text))
        references += f'{reference} "{path}"\n'
    root_text = f"{references}{root_body}.op\n.end\n"
    root = staged / "dut.cir"
    root.write_text(root_text)
    return CircuitDeck("dut", root, root_text, tuple(files))


def _factored(
    tmp_path: Path,
    *,
    includes: dict[str, str],
    root_body: str = "",
) -> CircuitDeck:
    """A deck whose components live in staged include files beside it."""
    return _closure(
        tmp_path,
        includes=includes,
        root_body=f"V1 in 0 1\nX1 in out core\n{root_body}",
    )


def _random_rule(rule: dict[str, object], *, runs: int = 1, seed: int = 3) -> RandomVariation:
    return RandomVariation.model_validate(
        {"kind": "random", "runs": runs, "seed": seed, "rules": [rule]}
    )


def _case_copy(tmp_path: Path, name: str, index: int = 0) -> str:
    return (tmp_path / "staged" / f"case-{index:04d}__{name}").read_text()


class TestIncludeClosureTargets:
    def test_included_component_edits_a_per_case_copy(self, tmp_path: Path):
        circuit = _factored(tmp_path, includes={"core.inc": _CORE})
        variation = AssignVariation(kind="assign", assign={"R1": ["2k", "3k"]})

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        copies = [tmp_path / "staged" / f"case-{index:04d}__core.inc" for index in (0, 1)]
        assert "R1 in out 2k" in copies[0].read_text()
        assert "R1 in out 3k" in copies[1].read_text()
        assert str(copies[0]) in variants[0].text
        assert str(copies[1]) in variants[1].text
        assert (tmp_path / "staged" / "core.inc").read_text() == _CORE

    def test_root_declaration_wins_over_an_include(self, tmp_path: Path):
        circuit = _factored(
            tmp_path,
            includes={"core.inc": _CORE},
            root_body="R1 a b 5k\n",
        )
        variation = AssignVariation(kind="assign", assign={"R1": ["9k"]})

        variants = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert "R1 a b 9k" in variants[0].text
        assert not (tmp_path / "staged" / "case-0000__core.inc").exists()

    def test_two_includes_declaring_one_target_name_both_files(self, tmp_path: Path):
        circuit = _factored(
            tmp_path,
            includes={
                "left.inc": ".subckt left in out\nR1 in out 1k\n.ends\n",
                "right.inc": ".subckt right in out\nR1 in out 2k\n.ends\n",
            },
        )
        variation = AssignVariation(kind="assign", assign={"R1": ["3k"]})

        with pytest.raises(VariationError, match=r"left\.inc, right\.inc") as excinfo:
            expand_variations([circuit], [variation])

        assert excinfo.value.code == "ambiguous_target"

    def test_missing_target_reports_what_was_searched(self, tmp_path: Path):
        circuit = _factored(tmp_path, includes={"core.inc": _CORE})
        variation = AssignVariation(kind="assign", assign={"nowhere": [1]})

        with pytest.raises(VariationError, match="its 1 included file"):
            expand_variations([circuit], [variation])

    def test_random_component_rule_perturbs_the_include(self, tmp_path: Path):
        circuit = _factored(tmp_path, includes={"core.inc": _CORE})
        variation = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 2,
                "seed": 5,
                "rules": [{"rule": "component", "target": "R1", "tolerance": 0.1}],
            }
        )

        materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        values = [
            (tmp_path / "staged" / f"case-{index:04d}__core.inc")
            .read_text()
            .split("R1 in out ")[1]
            .split()[0]
            for index in (0, 1)
        ]
        assert len(set(values)) == 2
        assert all(float(value) != 1000.0 for value in values)


class TestPatternTargetsSpanTheClosure:
    """A glob or a prefix means every match, wherever the closure holds it.

    Stopping at the root deck is how a Monte Carlo comes to perturb half the
    devices it was asked to and still report a clean sweep.
    """

    def test_component_glob_perturbs_matches_in_every_file(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"core.inc": "R3 a b 3k\nR4 b 0 4k\n"},
            root_body="R1 in out 1k\nR2 out 0 2k\n",
        )
        variation = _random_rule({"rule": "component", "target": "R*", "tolerance": 0.1})

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert {key for key in cases[0].assignments if key.startswith("random:component:")} == {
            "random:component:R1",
            "random:component:R2",
            "random:component:R3",
            "random:component:R4",
        }
        include = _case_copy(tmp_path, "core.inc")
        assert "R3 a b 3k" not in include and "R4 b 0 4k" not in include

    def test_mismatch_prefix_reaches_devices_in_an_include(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"core.inc": "M3 d g 0 0 NM W=4u L=1u\n"},
            root_body=".model NM NMOS(VTO=0.7 KP=100u)\nM1 d g 0 0 NM W=10u L=1u\n",
        )
        variation = _random_rule(
            {"rule": "mismatch", "prefix": "M", "AVT": 3e-3, "AK": 0.02},
        )

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert {key for key in cases[0].assignments if key.startswith("random:mismatch:")} == {
            "random:mismatch:M1.VTO",
            "random:mismatch:M1.KP",
            "random:mismatch:M3.VTO",
            "random:mismatch:M3.KP",
        }
        assert "NM__M3" in _case_copy(tmp_path, "core.inc")

    def test_model_swap_glob_rewrites_every_matching_file(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"core.inc": "M2 d g 0 0 NM W=4u L=1u\n"},
            root_body="M1 d g 0 0 NM W=10u L=1u\n",
        )
        variation = AssignVariation(kind="assign", assign={"M*@model": ["FAST"]})

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert "M1 d g 0 0 FAST" in cases[0].text
        assert "M2 d g 0 0 FAST" in _case_copy(tmp_path, "core.inc")

    def test_exact_name_in_two_includes_is_still_refused(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"a.inc": ".param VDD=1\nR1 a b 1k\n", "b.inc": ".param VDD=2\nR2 b 0 2k\n"},
            root_body="V1 in 0 1\n",
        )
        variation = AssignVariation(kind="assign", assign={"VDD": [3]})

        with pytest.raises(VariationError, match="names one declaration") as excinfo:
            expand_variations([circuit], [variation])

        assert excinfo.value.code == "ambiguous_target"
        assert "declare it in the deck" in str(excinfo.value)

    def test_a_glob_over_the_same_two_includes_perturbs_both(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"a.inc": ".param VDD=1\nR1 a b 1k\n", "b.inc": ".param VDD=2\nR2 b 0 2k\n"},
            root_body="V1 in 0 1\n",
        )
        variation = _random_rule({"rule": "component", "target": "R*", "tolerance": 0.1})

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert {key for key in cases[0].assignments if key.startswith("random:component:")} == {
            "random:component:R1",
            "random:component:R2",
        }


class TestOneFileDeclaringOneNameTwice:
    """Ambiguity inside a file is the same data loss as ambiguity across two.

    Includes are exactly where several ``.subckt`` bodies and several library
    corner sections live, so the flattened first-match that used to resolve
    these silently edited one declaration and reported the whole set applied.
    """

    def test_two_subckts_declaring_one_component_are_ambiguous(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={
                "core.inc": (
                    ".subckt left in out\nR1 in out 1k\n.ends\n"
                    ".subckt right in out\nR1 in out 2k\n.ends\n"
                )
            },
            root_body="V1 in 0 1\n",
        )
        variation = AssignVariation(kind="assign", assign={"R1": ["3k"]})

        with pytest.raises(VariationError, match=r"\.subckt left, \.subckt right") as excinfo:
            expand_variations([circuit], [variation])

        assert excinfo.value.code == "ambiguous_target"

    def test_duplicates_sharing_a_site_are_not_told_to_move_there(self, tmp_path: Path):
        """Two declarations in one place cannot be resolved by moving one there."""
        circuit = _closure(
            tmp_path,
            includes={},
            root_body=".param gain=2\n.param gain=3\nR1 a b 1k\n",
        )
        variation = AssignVariation(kind="assign", assign={"gain": [5]})

        with pytest.raises(VariationError, match="all at top level") as excinfo:
            expand_variations([circuit], [variation])

        assert "move the one you mean" not in str(excinfo.value)
        assert excinfo.value.code == "ambiguous_target"

    def test_two_library_sections_declaring_one_model_are_ambiguous(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={
                "mos.lib": (
                    ".lib tt\n.model nch NMOS(VTO=0.4 KP=100u)\n.endl\n"
                    ".lib ff\n.model nch NMOS(VTO=0.3 KP=120u)\n.endl\n"
                )
            },
            root_body="M1 d g 0 0 nch W=10u L=1u\n",
            reference=".lib",
        )
        variation = _random_rule(
            {"rule": "model", "target": "nch", "param": "VTO", "tolerance": 0.2},
            runs=2,
        )

        with pytest.raises(VariationError, match="section 'ff'") as excinfo:
            expand_variations([circuit], [variation])

        assert excinfo.value.code == "ambiguous_target"
        assert "section 'tt'" in str(excinfo.value)

    def test_two_files_declaring_one_mismatch_model_are_ambiguous(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={"core.inc": ".model NM NMOS(VTO=0.5 KP=80u)\n"},
            root_body=".model NM NMOS(VTO=0.7 KP=100u)\nM1 d g 0 0 NM W=10u L=1u\n",
        )
        variation = _random_rule({"rule": "mismatch", "prefix": "M", "AVT": 3e-3})

        with pytest.raises(VariationError, match="mismatch model 'NM' has 2 declarations"):
            materialize_variants(
                circuit,
                expand_variations([circuit], [variation]),
                tmp_path / "out",
            )

    def test_a_top_level_declaration_wins_over_a_subckt_copy(self, tmp_path: Path):
        circuit = _closure(
            tmp_path,
            includes={},
            root_body=".subckt buf in out\nR1 in out 9k\n.ends\nR1 a b 1k\n",
        )
        variation = AssignVariation(kind="assign", assign={"R1": ["3k"]})

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert "R1 a b 3k" in cases[0].text
        assert "R1 in out 9k" in cases[0].text

    def test_a_top_level_lib_include_opens_no_section(self, tmp_path: Path):
        """In a deck, a bare ``.lib`` names a file; only a library has sections.

        Reading it as a section opener pushes a section nothing closes, and
        every declaration below it then reads as buried — so a target that
        resolves against the deck's own top level is refused as ambiguous.
        """
        staged = tmp_path / "staged"
        staged.mkdir()
        models = staged / "models"
        model_text = ".model NM NMOS(VTO=0.7)\n"
        models.write_text(model_text)
        root_text = ".lib models\n.subckt buf in out\nR1 in out 9k\n.ends\nR1 a b 1k\n.op\n.end\n"
        root = staged / "dut.cir"
        root.write_text(root_text)
        circuit = CircuitDeck("dut", root, root_text, (DeckFile(models, model_text),))
        variation = AssignVariation(kind="assign", assign={"R1": ["3k"]})

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert "R1 a b 3k" in cases[0].text
        assert "R1 in out 9k" in cases[0].text

    def test_a_glob_matching_one_reference_twice_is_refused(self, tmp_path: Path):
        """One reference, two declarations, two draws — and one of them reported.

        Each match draws from a stream keyed by the reference, so the second
        declaration is perturbed with a different number than the first and the
        receipt keeps whichever was written last.
        """
        circuit = _closure(
            tmp_path,
            includes={
                "core.inc": (
                    ".subckt left in out\nR1 in out 1k\n.ends\n"
                    ".subckt right in out\nR1 in out 2k\n.ends\n"
                )
            },
            root_body="V1 in 0 1\n",
        )
        variation = _random_rule({"rule": "component", "target": "R*", "tolerance": 0.1})

        with pytest.raises(VariationError, match=r"matches 'R1' at 2 declarations") as excinfo:
            materialize_variants(
                circuit,
                expand_variations([circuit], [variation]),
                tmp_path / "out",
            )

        assert excinfo.value.code == "ambiguous_target"
        assert ".subckt left" in str(excinfo.value)

    def test_a_glob_matching_one_reference_in_two_files_is_refused(self, tmp_path: Path):
        """The fan-out across files widened the same defect; it refuses there too."""
        circuit = _closure(
            tmp_path,
            includes={
                "a.inc": ".subckt left in out\nR1 in out 1k\n.ends\n",
                "b.inc": ".subckt right in out\nR1 in out 2k\n.ends\n",
            },
            root_body="V1 in 0 1\n",
        )
        variation = _random_rule({"rule": "component", "target": "R*", "tolerance": 0.1})

        with pytest.raises(VariationError, match=r"matches 'R1' at 2 declarations") as excinfo:
            materialize_variants(
                circuit,
                expand_variations([circuit], [variation]),
                tmp_path / "out",
            )

        assert excinfo.value.code == "ambiguous_target"
        assert "a.inc" in str(excinfo.value) and "b.inc" in str(excinfo.value)


class TestRulesReadWhatEarlierRulesWrote:
    """Each rule runs against the deck the rule before it produced."""

    def test_overlapping_mismatch_prefixes_see_the_injected_variant(self, tmp_path: Path):
        """A mismatch rule repoints its instances at variant models it injects.

        A second rule matching the same devices then reads those variant names,
        so a model lookup that consults only the pre-expansion index cannot see
        them and the instance reads as referencing a model that does not exist.
        """
        circuit = _closure(
            tmp_path,
            includes={},
            root_body=".model NM NMOS(VTO=0.7 KP=100u)\nM1 d g 0 0 NM W=10u L=1u\n",
        )
        variation = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 3,
                "rules": [
                    {"rule": "mismatch", "prefix": "M", "AVT": 3e-3},
                    {"rule": "mismatch", "prefix": "M1", "AVT": 2e-3},
                ],
            }
        )

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        # The second rule perturbed the card the first one injected, so both
        # variants are on the deck and the instance names the newer of them.
        assert ".MODEL NM__M1 " in cases[0].text
        assert ".MODEL NM__M1__M1 " in cases[0].text
        assert "M1 d g 0 0 NM__M1__M1" in cases[0].text

    def test_an_unparsable_twin_declaration_reports_the_reference(self, tmp_path: Path):
        """A declaration the lexer cannot read is still a declaration.

        It is skipped when the target index is built, so the repeated-reference
        guard never sees it; the writer then meets it anyway and must name the
        reference rather than surface a raw tokenizer fault.
        """
        circuit = _closure(
            tmp_path,
            includes={
                "core.inc": (
                    ".subckt left in out\nR1 in out 1k\n.ends\n"
                    '.subckt right in out\nR1 in out "1k\n.ends\n'
                )
            },
            root_body="V1 in 0 1\n",
        )
        variation = _random_rule({"rule": "component", "target": "R*", "tolerance": 0.1})

        with pytest.raises(VariationError, match="R1") as excinfo:
            materialize_variants(
                circuit,
                expand_variations([circuit], [variation]),
                tmp_path / "out",
            )

        assert excinfo.value.code == "ambiguous_target"
        assert "could not be read" in str(excinfo.value)


class TestSeedStability:
    def test_single_file_draws_are_bit_identical(self, tmp_path: Path):
        """Widening a pattern's reach must not move one recorded number.

        Every draw is keyed by the name of what it perturbs, never by a
        position in a match list, so reaching further can only add draws. These
        literals were recorded before that change; a diff here means a shipped
        Monte Carlo silently reports different values for the same seed.
        """
        circuit = _deck(tmp_path / "dut.cir")
        variation = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 2,
                "seed": 1234,
                "rules": [
                    {"rule": "component", "target": "R1", "tolerance": 0.1},
                    {"rule": "param", "target": "gain", "tolerance": 0.2},
                    {"rule": "model", "target": "NM", "param": "VTO", "tolerance": 0.15},
                    {"rule": "mismatch", "prefix": "M", "AVT": 3e-3, "AK": 0.02},
                ],
            }
        )

        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [variation]),
            tmp_path / "out",
        )

        assert [
            {key: value for key, value in case.assignments.items() if key.startswith("random:")}
            for case in cases
        ] == [
            {
                "random:component:R1": 958.0617667513966,
                "random:mismatch:M1.KP": 0.00010148208338562579,
                "random:mismatch:M1.VTO": 0.757990883147177,
                "random:model:NM.VTO": 0.7579506983921265,
                "random:param:gain": 1.9558313266252403,
            },
            {
                "random:component:R1": 976.05993006425,
                "random:mismatch:M1.KP": 0.00010097882808618597,
                "random:mismatch:M1.VTO": 0.6759447769918987,
                "random:model:NM.VTO": 0.6753772591338416,
                "random:param:gain": 1.9275251793846904,
            },
        ]
