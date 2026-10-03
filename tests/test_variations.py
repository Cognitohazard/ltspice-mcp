"""Strict variation grammar, expansion ordering, and deck materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.instance_targeting import canonical_target
from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    DeckFile,
    MismatchRule,
    RandomVariation,
    Variation,
    VariationError,
    derive_circuit_ids,
    expand_variations,
    materialize_variants,
    normalize_circuit_decks,
)
from tests._mismatch_fixtures import MINI_FET, instance_params


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

    def test_stem_ids_are_sanitised_and_disambiguated(self):
        ids = derive_circuit_ids(
            ["a/amp.cir", "b/amp.cir", "my amp.cir", "_x.cir", "AMP-2.cir", "plain.cir"],
            [None, None, None, None, None, None],
        )
        assert [name for name, _ in ids] == ["amp", "amp-2", "my_amp", "x", "AMP-2-2", "plain"]
        assert ids[0][1] is None and ids[5][1] is None
        assert all(note is not None for _, note in ids[1:5])

    def test_a_caller_id_is_kept_and_reserved_before_stems(self):
        ids = derive_circuit_ids(["amp.cir", "other.cir"], [None, "amp"])
        assert ids == [
            ("amp-2", ids[0][1]),
            ("amp", None),
        ]
        assert "'amp' is another circuit's id" in str(ids[0][1])

    def test_a_stem_with_nothing_valid_left_gets_a_generic_id(self):
        ((name, note),) = derive_circuit_ids(["___.cir"], [None])
        assert name == "circuit"
        assert note is not None

    def test_unknown_applies_to_names_the_ids_in_the_call(self, tmp_path: Path):
        circuit = _deck(tmp_path / "a.cir", "amp_v2")
        variation = AssignVariation(kind="assign", applies_to=["amp.v2"], assign={"R1": [1]})

        with pytest.raises(VariationError, match="'amp_v2'"):
            expand_variations([circuit], [variation])

    def test_missing_applies_to_id_is_rejected(self, tmp_path: Path):
        circuit = _deck(tmp_path / "a.cir", "a")
        variation = AssignVariation(
            kind="assign",
            applies_to=["missing"],
            assign={"R1": [1]},
        )

        with pytest.raises(VariationError, match="unknown circuit"):
            expand_variations([circuit], [variation])


class TestPdkNativeExpansion:
    def test_native_samples_follow_assignments_and_keep_logical_indices(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")
        native = TypeAdapter(Variation).validate_python(
            {
                "kind": "pdk_native",
                "id": "foundry",
                "runs": 2,
                "seed": 17,
                "profile": "sky130-e6f9c887-ngspice-v1",
                "mode": "combined",
                "sample_start": 8,
            }
        )
        assign = AssignVariation(kind="assign", assign={"R1": ["1k", "2k"]})
        cases = expand_variations([circuit], [assign, native])
        assert [(case.case_index, case.assignments, case.native_index) for case in cases] == [
            (0, {"R1": "1k"}, 8),
            (1, {"R1": "1k"}, 9),
            (2, {"R1": "2k"}, 8),
            (3, {"R1": "2k"}, 9),
        ]
        assert all(case.native == native and case.random is None for case in cases)
        assert [case.case_id for case in cases] == [f"dut-case-{i:04d}" for i in range(4)]
        with pytest.raises(VariationError, match="configured maximum"):
            expand_variations([circuit], [assign, native], max_cases=3)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("id", ""),
            ("id", "bad id"),
            ("runs", 0),
            ("seed", -1),
            ("seed", 2**63),
            ("seed", True),
            ("sample_start", -1),
            ("sample_start", 1.5),
            ("mode", "unknown"),
        ],
    )
    def test_native_contract_rejects_invalid_fields(self, field, value):
        data = {
            "kind": "pdk_native",
            "id": "foundry",
            "runs": 1,
            "seed": 17,
            "profile": "sky130-e6f9c887-ngspice-v1",
            "mode": "nominal",
        }
        data[field] = value
        with pytest.raises(ValidationError):
            TypeAdapter(Variation).validate_python(data)

    def test_native_and_caller_random_are_exclusive_per_circuit(self, tmp_path: Path):
        first = _deck(tmp_path / "a.cir", "a")
        second = _deck(tmp_path / "b.cir", "b")

        def native(applies_to):
            return TypeAdapter(Variation).validate_python(
                {
                    "kind": "pdk_native",
                    "id": "foundry",
                    "runs": 1,
                    "seed": 17,
                    "profile": "sky130-e6f9c887-ngspice-v1",
                    "mode": "nominal",
                    "applies_to": applies_to,
                }
            )

        assert (
            len(expand_variations([first, second], [native(["a"]), _random(applies_to=["b"])]))
            == 3
        )
        with pytest.raises(VariationError, match=r"stochastic|native|random"):
            expand_variations([first], [native(["a"]), _random(applies_to=["a"])])
        with pytest.raises(VariationError, match="native"):
            expand_variations([first], [native(["a"]), native(["a"])])


class TestRandomExpansion:
    def test_only_one_random_entry_per_circuit_is_allowed(self, tmp_path: Path):
        circuit = _deck(tmp_path / "dut.cir")

        with pytest.raises(VariationError, match="at most one") as exc_info:
            expand_variations([circuit], [_random(), _random(seed=8)])
        assert exc_info.value.code == "multiple_random_variations"

    def test_random_entries_on_different_circuits_each_run(self, tmp_path: Path):
        # The rule was one random entry per call, so two designs could not be
        # Monte-Carlo'd in one job although neither deck saw two families.
        first = _deck(tmp_path / "a.cir", "a")
        second = _deck(tmp_path / "b.cir", "b")

        cases = expand_variations(
            [first, second],
            [_random(seed=3, runs=2, applies_to=["a"]), _random(seed=8, runs=3, applies_to=["b"])],
        )

        assert [(case.circuit_id, case.random_index) for case in cases] == [
            ("a", 0),
            ("a", 1),
            ("b", 0),
            ("b", 1),
            ("b", 2),
        ]
        seeds = {case.circuit_id: case.random.seed for case in cases if case.random is not None}
        assert seeds == {"a": 3, "b": 8}

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
        assert "does not reach case assignments" in assign_doc

    def test_mismatch_min_wl_um2_must_be_positive(self):
        # min_wl_um2 is the √(W·L) denominator floor; <= 0 would divide by zero
        # (or take sqrt of a negative) in the sampler. Reject it at the boundary.
        for bad in (0.0, -1e-3):
            with pytest.raises(ValidationError):
                MismatchRule(rule="mismatch", min_wl_um2=bad)
        assert MismatchRule(rule="mismatch", min_wl_um2=1e-2).min_wl_um2 == 1e-2


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


@pytest.mark.parametrize("captured_hash", [False, True])
@pytest.mark.parametrize("drift_before_materialization", [False, True])
def test_final_file_digests_freeze_unedited_captured_bytes(
    tmp_path, captured_hash, drift_before_materialization
):
    include = tmp_path / "body.inc"
    original = "* coût\r\nR1 a 0 1k\r\n"
    data = original.encode("cp1252" if captured_hash else "utf-8")
    include.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    kwargs = {"sha256": digest} if captured_hash else {}
    circuit = CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        '* bench\n.include "body.inc"\n.end\n',
        (DeckFile(include, original, **kwargs),),
    )
    changed = b"* changed\nR1 a 0 2k\n"
    if drift_before_materialization:
        include.write_bytes(changed)
    (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path)
    if not drift_before_materialization:
        include.write_bytes(changed)
    assert dict(case.file_digests) == {
        case.path: hashlib.sha256(case.path.read_bytes()).hexdigest(),
        include: digest,
    }
    assert dict(case.file_digests)[include] != hashlib.sha256(include.read_bytes()).hexdigest()
    assert case.sha256 == dict(case.file_digests)[case.path]


def test_final_file_digests_cover_rewritten_include_bytes(tmp_path):
    leaf = tmp_path / "leaf.inc"
    leaf_text = "* coût\r\nR1 a 0 1k\r\n"
    leaf.write_bytes(leaf_text.encode("cp1252"))
    branch = tmp_path / "branch.inc"
    branch_text = '.include "leaf.inc"\r\n'
    branch.write_bytes(branch_text.encode())
    root_text = '* bench\n.include "branch.inc"\n.end\n'
    circuit = CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        root_text,
        tuple(
            DeckFile(path, text, hashlib.sha256(path.read_bytes()).hexdigest())
            for path, text in ((branch, branch_text), (leaf, leaf_text))
        ),
    )
    variation = AssignVariation(kind="assign", assign={"R1": ["2k"]})
    (case,) = materialize_variants(circuit, expand_variations([circuit], [variation]), tmp_path)
    branch_copy = tmp_path / "case-0000__branch.inc"
    leaf_copy = tmp_path / "case-0000__leaf.inc"
    assert str(leaf_copy.name) in branch_copy.read_text()
    assert "R1 a 0 2k" in leaf_copy.read_text()
    assert dict(case.file_digests) == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (case.path, branch_copy, leaf_copy)
    }
    assert leaf.read_bytes() == leaf_text.encode("cp1252")
    assert branch.read_bytes() == branch_text.encode()


# A deck LTspice exported with micro-sign suffixes, plus a micro sign in a
# comment and inside a quoted file name, where it is not a value.
_MICRO_DECK = (
    "* rc µ bench\n"
    'V1 in 0 PWL file="steps 10µs.txt"\n'
    "R1 in out 1k\n"
    "C1 out 0 23µ\n"
    ".param tau=4.7µ\n"
    ".tran 0 {10*tau}\n"
    ".end\n"
)


class TestMicroSuffixInCaseDecks:
    """A case deck is always written as UTF-8, so a micro sign in it reaches
    the simulator as bytes C2 B5 whatever the source used. LTspice XVII
    decodes a deck as cp1252, reads 'Âµ', and runs 23µ as 23. The case deck
    must spell the suffix 'u', which no reader or encoding can change."""

    @pytest.mark.parametrize("codec", ["cp1252", "utf-8"])
    def test_zero_variation_case_spells_micro_as_u(self, tmp_path: Path, codec: str):
        source = tmp_path / "bench.cir"
        source.write_bytes(_MICRO_DECK.encode(codec))
        circuit = CircuitDeck("bench", source, read_spice_text(source))

        (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path / "out")

        data = case.path.read_bytes()
        assert b"C1 out 0 23u\n" in data
        assert b".param tau=4.7u\n" in data
        # Only value positions change: the comment and the quoted file name
        # keep their micro sign.
        assert "* rc µ bench\n".encode() in data
        assert 'file="steps 10µs.txt"'.encode() in data
        assert case.sha256 == hashlib.sha256(data).hexdigest()
        assert case.text == data.decode("utf-8")

    def test_greek_mu_folds_like_the_micro_sign(self, tmp_path: Path):
        source = tmp_path / "bench.cir"
        source.write_bytes("* t\nC1 out 0 23μ\n.op\n.end\n".encode())
        circuit = CircuitDeck("bench", source, read_spice_text(source))

        (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path / "out")

        assert case.path.read_bytes() == b"* t\nC1 out 0 23u\n.op\n.end\n"

    def test_assigned_value_with_micro_sign_is_written_as_u(self, tmp_path: Path):
        source = tmp_path / "bench.cir"
        source.write_bytes(b"* t\nC1 out 0 1n\n.op\n.end\n")
        circuit = CircuitDeck("bench", source, read_spice_text(source))
        variation = AssignVariation(kind="assign", assign={"C1": ["10µ"]})

        (case,) = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )

        assert b"C1 out 0 10u\n" in case.path.read_bytes()

    def test_edited_include_copy_spells_micro_as_u(self, tmp_path: Path):
        include = tmp_path / "core.inc"
        include_text = "R1 in out 1k\nC1 out 0 1n\n"
        include.write_bytes(include_text.encode())
        circuit = CircuitDeck(
            "bench",
            tmp_path / "bench.cir",
            '* bench\n.include "core.inc"\n.end\n',
            (DeckFile(include, include_text),),
        )
        variation = AssignVariation(kind="assign", assign={"C1": ["4.7µ"]})

        (case,) = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path
        )

        copy = tmp_path / "case-0000__core.inc"
        assert b"C1 out 0 4.7u\n" in copy.read_bytes()
        assert dict(case.file_digests)[copy] == hashlib.sha256(copy.read_bytes()).hexdigest()

    def test_mis_decoded_micro_is_left_for_the_linter(self, tmp_path: Path):
        """'Âµ' is not a micro sign at the suffix position: whether it was one
        is a guess, and a guess must not rewrite a value. The linter blocks it."""
        source = tmp_path / "bench.cir"
        source.write_bytes("* t\nC1 out 0 23Âµ\n.op\n.end\n".encode())
        circuit = CircuitDeck("bench", source, read_spice_text(source))

        (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path / "out")

        assert "C1 out 0 23Âµ\n".encode() in case.path.read_bytes()


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


class TestStructuredWriteConflicts:
    @pytest.mark.parametrize("reference", ["R1", "r1"])
    def test_component_draw_cannot_be_overwritten_by_structured_edit(
        self, tmp_path: Path, reference: str
    ):
        text = f"* bench\n{reference} a 0 1k\n.end\n"
        circuit = CircuitDeck(
            "bench",
            tmp_path / "bench.cir",
            text,
            semantic_profile=SemanticProfile("ngspice", "hsa"),
        )
        variations = TypeAdapter(list[Variation]).validate_python(
            [
                {
                    "kind": "assign",
                    "instances": [{"instance": ["R1"], "attribute": "value", "values": ["2k"]}],
                },
                {
                    "kind": "random",
                    "runs": 1,
                    "seed": 4,
                    "rules": [{"rule": "component", "target": "R1", "tolerance": 0.1}],
                },
            ]
        )

        with pytest.raises(VariationError) as exc:
            materialize_variants(
                circuit, expand_variations([circuit], variations), tmp_path / "out"
            )
        assert exc.value.code == "overlapping_assignment"
        assert not list((tmp_path / "out").glob("case-*"))

    @pytest.mark.parametrize("with_disjoint_edit", [False, True])
    def test_disjoint_structured_edit_preserves_ordinary_then_random(
        self, tmp_path: Path, with_disjoint_edit: bool
    ):
        text = "* bench\n.param p=1k\nR1 a 0 {p}\nR2 b 0 2k\n.end\n"
        circuit = CircuitDeck(
            "bench",
            tmp_path / "bench.cir",
            text,
            semantic_profile=SemanticProfile("ngspice", "hsa"),
        )
        assign: dict[str, object] = {"kind": "assign", "assign": {"p": ["2k"]}}
        if with_disjoint_edit:
            assign["instances"] = [{"instance": ["R2"], "attribute": "value", "values": ["3k"]}]
        variations = TypeAdapter(list[Variation]).validate_python(
            [
                assign,
                {
                    "kind": "random",
                    "runs": 1,
                    "seed": 4,
                    "rules": [{"rule": "param", "target": "p", "tolerance": 0.1}],
                },
            ]
        )

        case = materialize_variants(
            circuit, expand_variations([circuit], variations), tmp_path / "out"
        )[0]
        draw = case.assignments["random:param:p"]
        param_line = next(
            line for line in case.text.splitlines() if line.lower().startswith(".param p=")
        )
        assert float(param_line.split("=", 1)[1]) == pytest.approx(draw)
        assert draw == pytest.approx(1896.62575638)
        assert case.assignments["p"] == "2k"
        assert ("R2 b 0 3k" in case.text) == with_disjoint_edit

    def test_ordered_random_rules_still_compose_with_disjoint_structured_edit(
        self, tmp_path: Path
    ):
        text = "* bench\nR1 a 0 1k\nR2 b 0 2k\n.end\n"
        circuit = CircuitDeck(
            "bench",
            tmp_path / "bench.cir",
            text,
            semantic_profile=SemanticProfile("ngspice", "hsa"),
        )
        variations = TypeAdapter(list[Variation]).validate_python(
            [
                {
                    "kind": "assign",
                    "instances": [{"instance": ["R2"], "attribute": "value", "values": ["3k"]}],
                },
                {
                    "kind": "random",
                    "runs": 1,
                    "seed": 4,
                    "rules": [
                        {"rule": "component", "target": "R1", "tolerance": 0.1},
                        {"rule": "component", "target": "R1", "tolerance": 0.1},
                    ],
                },
            ]
        )

        case = materialize_variants(
            circuit, expand_variations([circuit], variations), tmp_path / "out"
        )[0]
        draw = case.assignments["random:component:R1"]
        component_line = next(line for line in case.text.splitlines() if line.startswith("R1 "))
        assert float(component_line.split()[-1]) == pytest.approx(draw)
        assert draw != pytest.approx(1000)
        assert case.assignments[canonical_target(("R2",), "value")] == "3k"

        rules = variations[1]
        assert isinstance(rules, RandomVariation)
        first_rule = rules.model_copy(update={"rules": rules.rules[:1]})
        first_case = materialize_variants(
            circuit,
            expand_variations([circuit], [variations[0], first_rule]),
            tmp_path / "first",
        )[0]
        assert first_case.assignments["random:component:R1"] != pytest.approx(draw)


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


PDK_DECK = (
    "* pdk-shaped deck\n" + MINI_FET + "XM1 out1 gate 0 0 minifet W=1 L=0.15\n"
    "XM2 out2 gate 0 0 minifet W=1 L=0.15\n"
    "V1 out1 0 1\n"
    ".op\n"
    ".end\n"
)


def _pdk_deck(path: Path, text: str = PDK_DECK, circuit_id: str = "pdk") -> CircuitDeck:
    path.write_text(text)
    return CircuitDeck(circuit_id, path, text)


class TestInstanceParameterTargets:
    """``X1:delvto`` — a per-instance mismatch value on an X-wrapped device."""

    def _assign(self, **assign) -> AssignVariation:
        return AssignVariation.model_validate({"kind": "assign", "assign": assign})

    def test_a_value_lands_on_the_named_instance(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        variation = self._assign(**{"XM1:delvto": [-0.02], "XM2:delvto": [0.03]})
        cases = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )
        text = cases[0].text
        assert instance_params(text, "XM1")["mc_delvto__m0"] == "-0.02"
        assert instance_params(text, "XM2")["mc_delvto__m0"] == "0.03"
        assert instance_params(text, "XM1")["__model__"] == "minifet__mcpatch"
        # Two instances of one device type share a single patched copy.
        assert text.count(".subckt minifet__mcpatch") == 1
        assert cases[0].assignments["XM1:delvto"] == -0.02

    def test_both_parameters_reach_one_instance(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        variation = self._assign(**{"XM1:delvto": [-0.02], "XM1:mulu0": [1.05]})
        cases = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )
        params = instance_params(cases[0].text, "XM1")
        assert params["mc_delvto__m0"] == "-0.02"
        assert params["mc_mulu0__m0"] == "1.05"

    def test_a_sweep_over_one_instance_expands(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        variation = self._assign(**{"XM1:delvto": [-0.02, 0.0, 0.02]})
        cases = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )
        assert [instance_params(c.text, "XM1")["mc_delvto__m0"] for c in cases] == [
            "-0.02",
            "0",
            "0.02",
        ]

    def test_only_the_two_mismatch_parameters_are_addressable(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        with pytest.raises(VariationError) as exc:
            expand_variations([circuit], [self._assign(**{"XM1:vth0": [0.7]})])
        assert exc.value.code == "invalid_instance_param_target"

    def test_a_multi_device_body_requires_naming_the_device(self, tmp_path: Path):
        pair = (
            ".subckt pairfet d1 d2 g s b\n"
            ".param w = 1 l = 0.15\n"
            "ma d1 g s b pair_model w = {w} l = {l}\n"
            "mb d2 g s b pair_model w = {w} l = {l}\n"
            ".model pair_model nmos level = 54 vth0 = 0.7\n"
            ".ends pairfet\n"
        )
        text = "* pair\n" + pair + "XP0 da db g 0 0 pairfet W=1 L=0.15\nV1 da 0 1\n.op\n.end\n"
        circuit = _pdk_deck(tmp_path / "pair.cir", text, "pair")
        cases = expand_variations([circuit], [self._assign(**{"XP0:delvto": [-0.02]})])
        with pytest.raises(VariationError) as exc:
            materialize_variants(circuit, cases, tmp_path / "out")
        assert exc.value.code == "ambiguous_inner_device"

        named = expand_variations([circuit], [self._assign(**{"XP0.mb:delvto": [-0.02]})])
        out = materialize_variants(circuit, named, tmp_path / "named")
        assert instance_params(out[0].text, "XP0")["mc_delvto__mb"] == "-0.02"

    def test_a_target_that_also_names_a_component_is_ambiguous(self, tmp_path: Path):
        text = PDK_DECK.replace("V1 out1 0 1", "V1 out1 0 1\n.param XM1:delvto=0")
        circuit = _pdk_deck(tmp_path / "clash.cir", text, "clash")
        with pytest.raises(VariationError) as exc:
            expand_variations([circuit], [self._assign(**{"XM1:delvto": [-0.02]})])
        assert exc.value.code == "ambiguous_target"

    def test_an_unknown_instance_is_named(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        cases = expand_variations([circuit], [self._assign(**{"XZ9:delvto": [-0.02]})])
        with pytest.raises(VariationError) as exc:
            materialize_variants(circuit, cases, tmp_path / "out")
        assert exc.value.code == "instance_not_found"


class TestMismatchThroughSubcircuits:
    """A mismatch rule aimed at X-wrapped devices."""

    def _variation(self, prefix: str = "X") -> RandomVariation:
        return RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 2,
                "seed": 11,
                "rules": [{"rule": "mismatch", "prefix": prefix, "AVT": 5e-3, "AK": 0.02}],
            }
        )

    def test_each_instance_draws_its_own_shift(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        cases = materialize_variants(
            circuit, expand_variations([circuit], [self._variation()]), tmp_path / "out"
        )
        first = instance_params(cases[0].text, "XM1")
        second = instance_params(cases[0].text, "XM2")
        assert first["mc_delvto__m0"] != second["mc_delvto__m0"]
        assert first["__model__"] == "minifet__mcpatch"
        # Two runs of the same rule are different draws.
        assert instance_params(cases[1].text, "XM1")["mc_delvto__m0"] != first["mc_delvto__m0"]

    def test_the_receipt_records_the_draws_and_reconciles(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        cases = materialize_variants(
            circuit, expand_variations([circuit], [self._variation()]), tmp_path / "out"
        )
        assignments = cases[0].assignments
        assert set(assignments) >= {
            "random:mismatch:XM1.m0.delvto",
            "random:mismatch:XM1.m0.mulu0",
            "random:mismatch:XM2.m0.delvto",
            "random:mismatch:XM2.m0.mulu0",
            "random:mismatch:requested",
            "random:mismatch:matched",
            "random:mismatch:applied",
            "random:mismatch:skipped",
        }
        assert assignments["random:mismatch:requested"] == 2.0
        assert assignments["random:mismatch:applied"] == 2.0

    def test_a_body_without_a_device_is_a_named_skip(self, tmp_path: Path):
        text = PDK_DECK.replace(
            "V1 out1 0 1",
            ".subckt divider a b\nR1 a b 1k\n.ends divider\nXD1 out1 0 divider\nV1 out1 0 1",
        )
        circuit = _pdk_deck(tmp_path / "skip.cir", text, "skip")
        cases = materialize_variants(
            circuit, expand_variations([circuit], [self._variation()]), tmp_path / "out"
        )
        assert cases[0].assignments["random:mismatch:XD1:no_mos_at_depth"] == 1.0
        assert cases[0].assignments["random:mismatch:skipped"] == 1.0

    def test_a_prefix_matching_nothing_says_both_levels_were_tried(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        with pytest.raises(VariationError) as exc:
            expand_variations([circuit], [self._variation(prefix="Q")])
        assert "one subcircuit level down" in str(exc.value)

    def test_a_flat_deck_still_uses_the_model_card_mechanism(self, tmp_path: Path):
        circuit = _deck(tmp_path / "flat.cir")
        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [self._variation(prefix="M")]),
            tmp_path / "out",
        )
        assert "NM__M1" in cases[0].text
        assert "mcpatch" not in cases[0].text


class TestMismatchAcrossIncludedFiles:
    """The device library and the instances that use it live in includes."""

    def _circuit(self, tmp_path: Path) -> CircuitDeck:
        library = tmp_path / "lib.spice"
        library.write_text(MINI_FET)
        dut = tmp_path / "dut.spice"
        dut_text = "XM1 out1 gate 0 0 minifet W=1 L=0.15\n"
        dut.write_text(dut_text)
        root = tmp_path / "top.cir"
        root_text = "* top\n.include lib.spice\n.include dut.spice\nV1 out1 0 1\n.op\n.end\n"
        root.write_text(root_text)
        return CircuitDeck(
            "hier",
            root,
            root_text,
            includes=(DeckFile(library, MINI_FET), DeckFile(dut, dut_text)),
        )

    def test_the_instance_is_rewritten_in_its_own_file_and_the_chain_follows(self, tmp_path: Path):
        circuit = self._circuit(tmp_path)
        variation = AssignVariation.model_validate(
            {"kind": "assign", "assign": {"XM1:delvto": [-0.02]}}
        )
        cases = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )
        root = cases[0].text
        # The patched device subcircuit goes to the deck the simulator is handed.
        assert ".subckt minifet__mcpatch" in root
        # The edited include is written as a private per-case copy and the root
        # deck's include chain points at it.
        copy = tmp_path / "case-0000__dut.spice"
        assert copy.exists()
        assert "case-0000__dut.spice" in root
        assert instance_params(copy.read_text(), "XM1")["mc_delvto__m0"] == "-0.02"
        # The shared staged library is left exactly as it was.
        assert (tmp_path / "lib.spice").read_text() == MINI_FET


class TestOverlappingMismatchRules:
    """Two rules cannot both claim one X-wrapped device."""

    def _variation(self, *prefixes: str) -> RandomVariation:
        return RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 3,
                "rules": [
                    {"rule": "mismatch", "prefix": prefix, "AVT": 3e-3} for prefix in prefixes
                ],
            }
        )

    def test_two_prefixes_claiming_one_device_are_refused(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        cases = expand_variations([circuit], [self._variation("X", "XM")])
        with pytest.raises(VariationError) as exc:
            materialize_variants(circuit, cases, tmp_path / "out")
        assert exc.value.code == "overlapping_mismatch_rules"

    def test_disjoint_prefixes_each_perturb_their_own_devices(self, tmp_path: Path):
        text = PDK_DECK.replace("XM1 out1", "XA1 out1").replace("XM2 out2", "XB1 out2")
        circuit = _pdk_deck(tmp_path / "split.cir", text, "split")
        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [self._variation("XA", "XB")]),
            tmp_path / "out",
        )
        first = instance_params(cases[0].text, "XA1")
        second = instance_params(cases[0].text, "XB1")
        assert first["__model__"] == "minifet__mcpatch"
        assert second["__model__"] == "minifet__mcpatch"
        assert first["mc_delvto__m0"] != second["mc_delvto__m0"]

    def test_a_flat_deck_still_composes_overlapping_rules(self, tmp_path: Path):
        # The top-level mechanism layers variant model cards, and each rule
        # reads what the one before it wrote; only the subcircuit path, where
        # values share one instance line, has no way to compose.
        circuit = _deck(tmp_path / "flat.cir")
        cases = materialize_variants(
            circuit,
            expand_variations([circuit], [self._variation("M", "M1")]),
            tmp_path / "out",
        )
        assert "NM__M1__M1" in cases[0].text


class TestAssignedValueAndRuleOnOneDevice:
    """An exact value and a random rule cannot both drive one device."""

    def test_the_collision_is_refused_rather_than_applied_twice(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        assign = AssignVariation.model_validate(
            {"kind": "assign", "assign": {"XM1:delvto": [-0.02]}}
        )
        random = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 5,
                "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 3e-3}],
            }
        )
        cases = expand_variations([circuit], [assign, random])
        with pytest.raises(VariationError) as exc:
            materialize_variants(circuit, cases, tmp_path / "out")
        # The device already carries a shift, so a second one would silently
        # stack on top of it instead of replacing it.
        assert exc.value.code == "preexisting_mismatch_param"

    def test_a_rule_on_a_different_device_still_runs(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "pdk.cir")
        assign = AssignVariation.model_validate(
            {"kind": "assign", "assign": {"XM1:delvto": [-0.02]}}
        )
        random = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 5,
                "rules": [{"rule": "mismatch", "prefix": "XM2", "AVT": 3e-3}],
            }
        )
        cases = materialize_variants(
            circuit, expand_variations([circuit], [assign, random]), tmp_path / "out"
        )
        text = cases[0].text
        assert instance_params(text, "XM1")["mc_delvto__m0"] == "-0.02"
        assert "mc_delvto__m0" in instance_params(text, "XM2")
        assert instance_params(text, "XM2")["mc_delvto__m0"] != "-0.02"


class TestMixedFlatAndSubcircuitDecks:
    """Plain transistors and X-wrapped ones in one deck, one rule for each."""

    DECK = (
        "* mixed deck\n" + MINI_FET + ".model NM NMOS(LEVEL=1 VTO=0.7 KP=100u)\n"
        "M9 out9 gate 0 0 NM W=10u L=1u\n"
        "XM1 out1 gate 0 0 minifet W=1 L=0.15\n"
        "V1 out1 0 1\n"
        ".op\n"
        ".end\n"
    )

    def _variation(self) -> RandomVariation:
        return RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 4,
                "rules": [
                    {"rule": "mismatch", "prefix": "M", "AVT": 3e-3},
                    {"rule": "mismatch", "prefix": "X", "AVT": 5e-3},
                ],
            }
        )

    def test_both_rules_reach_their_own_devices(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "mixed.cir", self.DECK, "mixed")
        cases = materialize_variants(
            circuit, expand_variations([circuit], [self._variation()]), tmp_path / "out"
        )
        text = cases[0].text
        # The flat device gets its own variant model card, the wrapped one gets
        # a value on its instance line; neither mechanism displaces the other.
        assert "NM__M9" in text
        assert instance_params(text, "M9")["__model__"] == "NM__M9"
        assert instance_params(text, "XM1")["__model__"] == "minifet__mcpatch"
        assert "mc_delvto__m0" in instance_params(text, "XM1")
        assert {
            "random:mismatch:M9.VTO",
            "random:mismatch:XM1.m0.delvto",
        } <= set(cases[0].assignments)

    def test_a_rule_reaching_neither_level_is_still_refused(self, tmp_path: Path):
        circuit = _pdk_deck(tmp_path / "mixed.cir", self.DECK, "mixed")
        variation = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 1,
                "seed": 4,
                "rules": [
                    {"rule": "mismatch", "prefix": "M", "AVT": 3e-3},
                    {"rule": "mismatch", "prefix": "Q", "AVT": 5e-3},
                ],
            }
        )
        with pytest.raises(VariationError) as exc:
            expand_variations([circuit], [variation])
        assert exc.value.code == "ambiguous_target"


class TestCaseBundleIsWrittenWhole:
    """A case's edited includes and its root deck are one bundle.

    Each case writes private copies rather than editing anything a sibling
    case reads, and the root deck — the only file a simulator is handed — is
    written after the includes it points at. A caller that finds the deck
    therefore finds every file it names.
    """

    def _circuit(self, tmp_path: Path) -> CircuitDeck:
        library = tmp_path / "lib.spice"
        library.write_text(MINI_FET)
        dut = tmp_path / "dut.spice"
        dut_text = "XM1 out1 gate 0 0 minifet W=1 L=0.15\nXM2 out2 gate 0 0 minifet W=1 L=0.15\n"
        dut.write_text(dut_text)
        root = tmp_path / "top.cir"
        root_text = "* top\n.include lib.spice\n.include dut.spice\nV1 out1 0 1\n.op\n.end\n"
        root.write_text(root_text)
        return CircuitDeck(
            "bundle",
            root,
            root_text,
            includes=(DeckFile(library, MINI_FET), DeckFile(dut, dut_text)),
        )

    def test_the_root_deck_is_written_after_the_includes_it_names(self, tmp_path: Path):
        circuit = self._circuit(tmp_path)
        written: list[str] = []
        import ltspice_mcp.lib.variations as variations

        real = variations.atomic_write_text

        def record(path, text, **kwargs):
            written.append(Path(path).name)
            return real(path, text, **kwargs)

        variations.atomic_write_text = record
        try:
            variation = RandomVariation.model_validate(
                {
                    "kind": "random",
                    "runs": 2,
                    "seed": 6,
                    "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 5e-3}],
                }
            )
            cases = materialize_variants(
                circuit, expand_variations([circuit], [variation]), tmp_path / "out"
            )
        finally:
            variations.atomic_write_text = real

        for case in cases:
            include_copy = f"case-{case.case_index:04d}__dut.spice"
            deck = f"case-{case.case_index:04d}.cir"
            assert written.index(include_copy) < written.index(deck)
            assert include_copy in case.text

    def test_each_run_gets_its_own_include_content(self, tmp_path: Path):
        circuit = self._circuit(tmp_path)
        variation = RandomVariation.model_validate(
            {
                "kind": "random",
                "runs": 2,
                "seed": 6,
                "rules": [{"rule": "mismatch", "prefix": "X", "AVT": 5e-3}],
            }
        )
        cases = materialize_variants(
            circuit, expand_variations([circuit], [variation]), tmp_path / "out"
        )
        copies = [
            (tmp_path / f"case-{case.case_index:04d}__dut.spice").read_text() for case in cases
        ]
        # Per-run draws now differ inside an INCLUDED file, not only in the root
        # deck, so the copies must not be the same file twice.
        assert copies[0] != copies[1]
        assert len(copies) == 2
        # And the shared staged original is untouched by either run.
        assert (tmp_path / "dut.spice").read_text() == circuit.includes[1].text


class TestCircuitIdRejection:
    def test_a_derived_id_is_made_valid_and_says_where_it_came_from(self):
        """A file named ``_truth_op.asc`` used to be refused because its stem
        is not a valid id, although the caller never wrote one. The stem is
        now made valid, and the note names the file and how to override it."""
        ((name, note),) = derive_circuit_ids(["_truth_op.asc"], [None])

        assert name == "truth_op"
        assert note is not None
        assert "_truth_op.asc" in note
        assert "'truth_op'" in note
        assert "'id'" in note

    def test_an_explicit_id_is_not_blamed_on_the_filename(self, tmp_path: Path):
        deck = CircuitDeck("_chosen", tmp_path / "amp.asc", "")

        with pytest.raises(VariationError) as excinfo:
            normalize_circuit_decks([deck])

        message = str(excinfo.value)
        assert "derived" not in message
        # The id it refused, and the rule it refused it by.
        assert "_chosen" in message
        assert "letter or digit" in message
