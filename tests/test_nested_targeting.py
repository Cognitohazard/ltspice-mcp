"""Occurrence-exact variation and the real-PDK hierarchy prerequisites."""

from pathlib import Path
from typing import Literal

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy


def inspect_text(tmp_path: Path, text: str, simulator: Literal["ngspice", "ltspice"] = "ngspice"):
    path = tmp_path / "bench.cir"
    path.write_text("* bench\n" + text + "\n.end\n", encoding="utf-8")
    profile = SemanticProfile(simulator, "hsa" if simulator == "ngspice" else None)
    return load_hierarchy(str(path), [tmp_path], profile)


def test_scoped_model_bins_keep_family_and_local_identity(tmp_path):
    hierarchy = inspect_text(
        tmp_path,
        ".model n NMOS(level=1)\n"
        ".subckt leaf d g\nM0 d g 0 0 n w=1u l=.15u\n"
        ".model n.0 NMOS(level=54 vth0={unresolved_device_formula(l,w)})\n"
        ".model n.1 NMOS(level=54)\n.ends leaf\nXA a g leaf\nXB b g leaf",
    )
    mos = [row for row in hierarchy.instances if row.element == "M"]
    assert len(mos) == 2
    assert mos[0].model_source is None
    assert mos[0].model_reason is not None and "binned" in mos[0].model_reason
    assert [m.name for m in mos[0].model_family] == ["n.0", "n.1"]
    assert all(m.source.definition == "leaf" for m in mos[0].model_family)
    assert all(m.level == 54 for m in mos[0].model_family)


def test_inactive_staged_files_do_not_consume_hierarchy_byte_limit(tmp_path, monkeypatch):
    import ltspice_mcp.lib.hierarchy as hierarchy_module
    from ltspice_mcp.lib.instance_targeting import captured_hierarchy
    from ltspice_mcp.lib.subckt_mismatch import ClosureFile

    monkeypatch.setattr(hierarchy_module, "MAX_BYTES", 100)
    files = [
        ClosureFile(0, tmp_path / "bench.cir", "* bench\nR1 a 0 1k\n.end\n"),
        ClosureFile(1, tmp_path / "inactive.lib", "*" + "x" * 200 + "\n"),
    ]
    resolved = captured_hierarchy(files, SemanticProfile("ngspice", "hsa"))
    assert [row.reference for row in resolved.instances] == ["R1"]
    assert [file.path for file in resolved.inputs] == [files[0].path]


def test_spaced_pdk_parameter_expression_does_not_block_structure(tmp_path):
    hierarchy = inspect_text(
        tmp_path,
        ".param slope = 1.0 + MC_PR_SWITCH*AGAUSS(0,0.125,1)\nR1 a 0 1k",
    )
    assert hierarchy.instances[0].reference == "R1"
    slope = dict(hierarchy.instances[0].environment)["slope"]
    assert slope.value is None
    assert slope.expression is not None and "AGAUSS" in slope.expression


def test_probed_ngspice_sequential_declarations(tmp_path):
    hierarchy = inspect_text(
        tmp_path,
        ".param mc_mm_switch=0\n.param dependent={mc_mm_switch+1}\n.param mc_mm_switch=1\n"
        ".subckt leaf p q=2\n.param q=3\n.param q=4\nR1 p 0 {q*1000}\n.ends\n"
        "Rglobal a 0 {dependent*1000}\nXA b leaf\nXB c leaf q=5",
    )
    assert [r.value.value for r in hierarchy.instances if r.element == "R"] == [2000, 4000, 5000]


@pytest.mark.parametrize(
    ("simulator", "body"),
    [
        ("ltspice", ".param a=1\n.param a=2"),
        ("ngspice", ".param a=1 a=2"),
    ],
)
def test_unproven_or_within_card_duplicates_refused(tmp_path, simulator, body):
    with pytest.raises(NetlistError, match="duplicate"):
        inspect_text(tmp_path, body, simulator)


def test_structured_grid_clones_selected_ancestry_and_preserves_peer(tmp_path):
    from ltspice_mcp.lib.variations import (
        AssignVariation,
        CircuitDeck,
        expand_variations,
        materialize_variants,
    )

    text = "* bench\n.subckt leaf p\nR1 p 0 1k\n.ends leaf\n.subckt block p\nXleaf p leaf\n.ends block\nXA a block\nXB b block\n.end\n"
    circuit = CircuitDeck(
        "bench", tmp_path / "bench.cir", text, semantic_profile=SemanticProfile("ngspice", "hsa")
    )
    variation = AssignVariation.model_validate(
        {
            "kind": "assign",
            "instances": [
                {"instance": ["xa", "xleaf", "r1"], "attribute": "value", "values": ["2k", "3k"]}
            ],
        }
    )
    cases = materialize_variants(circuit, expand_variations([circuit], [variation]), tmp_path)
    assert len(cases) == 2
    for case, expected in zip(cases, [2000, 3000], strict=True):
        assert circuit.semantic_profile is not None
        hierarchy = load_hierarchy(str(case.path), [tmp_path], circuit.semantic_profile)
        rows = {r.instance: r for r in hierarchy.instances}
        assert rows[("XA", "Xleaf", "R1")].value.value == expected
        assert rows[("XB", "Xleaf", "R1")].value.value == 1000
        assert case.text.count(".subckt ") == 4
        assert ".subckt leaf p\nR1 p 0 1k\n.ends leaf" in case.text
        assert "XB b block\n" in case.text


@pytest.mark.parametrize(
    ("attribute", "parameter", "value"),
    [
        ("model", None, "n w=2u"),
        ("model", None, "n\n.end"),
        ("parameter", "w", "1u l=9u"),
        ("parameter", "w", "{1} l=9u"),
        ("parameter", "w", "1}"),
        ("parameter", "w", "'1"),
        ("parameter", "w", "1; ignored"),
        ("parameter", "w", "1$ignored"),
        ("parameter", "w", float("inf")),
        ("parameter", "w", float("nan")),
        ("value", "w", "1"),
        ("parameter", None, "1"),
        ("parameter", "w=x", "1"),
        ("model", None, "n\r.end"),
        ("parameter", "w", "1)"),
        ("parameter", "w", "{1}junk"),
    ],
)
def test_structured_payload_refuses_slot_escape(attribute, parameter, value):
    from ltspice_mcp.lib.variations import InstanceAssignment

    with pytest.raises(ValueError, match=r"."):
        InstanceAssignment(
            instance=["XA", "M0"], attribute=attribute, parameter=parameter, values=[value]
        )


def target_circuit(tmp_path, *, fragment=False):
    from ltspice_mcp.lib.deck_staging import sha256_file
    from ltspice_mcp.lib.variations import CircuitDeck, DeckFile

    body = "R1 p 0 1k\nM0 p g 0 0 n w={w} l={l}\n.model n NMOS(level=54)\n"
    includes = ()
    if fragment:
        path = tmp_path / "body.inc"
        path.write_text(body, encoding="utf-8")
        includes = (DeckFile(path, body, sha256_file(path)),)
        body = '.include "body.inc"\n'
    text = (
        "* bench\n.param w=1u l=.15u\n.subckt leaf p g\n"
        + body
        + ".ends leaf\n.subckt block p g\nXleaf p g leaf\n.ends block\nXA a g block\nXB b g block\n.end\n"
    )
    return CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        text,
        includes,
        semantic_profile=SemanticProfile("ngspice", "hsa"),
    )


def materialize(circuit, tmp_path, entries):
    from pydantic import TypeAdapter

    from ltspice_mcp.lib.variations import Variation, expand_variations, materialize_variants

    variations = TypeAdapter(list[Variation]).validate_python(entries)
    return materialize_variants(circuit, expand_variations([circuit], variations), tmp_path)


@pytest.mark.parametrize("reference", ["relative", "absolute", "nested"])
def test_included_fragment_isolated_and_multi_edit_clone_reused(tmp_path, reference):
    import hashlib
    from dataclasses import replace

    circuit = target_circuit(tmp_path, fragment=True)
    fragment = tmp_path / "body.inc"
    if reference == "nested":
        nested = tmp_path / "parts" / "body.inc"
        nested.parent.mkdir()
        fragment.rename(nested)
        fragment = nested
        circuit = replace(circuit, includes=(replace(circuit.includes[0], path=fragment),))
    spelling = (
        fragment.as_posix()
        if reference == "absolute"
        else fragment.relative_to(tmp_path).as_posix()
    )
    original_reference = f'.include "{spelling}"'
    circuit = replace(
        circuit, text=circuit.text.replace('.include "body.inc"', original_reference)
    )
    original = fragment.read_bytes()
    cases = materialize(
        circuit,
        tmp_path,
        [
            {
                "kind": "assign",
                "instances": [
                    {"instance": ["XA", "Xleaf", "R1"], "attribute": "value", "values": ["2k"]},
                    {
                        "instance": ["XA", "Xleaf", "M0"],
                        "attribute": "parameter",
                        "parameter": "w",
                        "values": ["2u"],
                    },
                ],
            }
        ],
    )
    assert circuit.semantic_profile is not None
    hierarchy = load_hierarchy(str(cases[0].path), [tmp_path], circuit.semantic_profile)
    rows = {r.instance: r for r in hierarchy.instances}
    assert rows[("XA", "Xleaf", "R1")].value.value == 2000
    assert rows[("XB", "Xleaf", "R1")].value.value == 1000
    assert dict(rows[("XA", "Xleaf", "M0")].geometry)["w"].value == 2e-6
    assert dict(rows[("XB", "Xleaf", "M0")].geometry)["w"].value == 1e-6
    assert cases[0].text.count(".subckt ") == 4
    assert fragment.read_bytes() == original
    assert original_reference in cases[0].text
    assert dict(cases[0].file_digests) == {
        file.path: hashlib.sha256(file.path.read_bytes()).hexdigest() for file in hierarchy.inputs
    }


@pytest.mark.parametrize(
    ("ordinary", "instances"),
    [
        (
            {"M*@model": ["n"]},
            [{"instance": ["XA", "Xleaf", "M0"], "attribute": "model", "values": ["n"]}],
        ),
        (
            {},
            [
                {"instance": ["XA", "Xleaf", "M0"], "attribute": "value", "values": ["n w=2u"]},
                {
                    "instance": ["XA", "Xleaf", "M0"],
                    "attribute": "parameter",
                    "parameter": "W",
                    "values": ["3u"],
                },
            ],
        ),
        (
            {"XA@model": ["block"]},
            [{"instance": ["XA", "Xleaf", "R1"], "attribute": "value", "values": ["2k"]}],
        ),
        (
            {},
            [
                {"instance": ["XA"], "attribute": "parameter", "parameter": "w", "values": ["2u"]},
                {
                    "instance": ["XA", "Xleaf", "M0"],
                    "attribute": "parameter",
                    "parameter": "w",
                    "values": ["3u"],
                },
            ],
        ),
    ],
)
def test_actual_write_overlap_and_ancestor_conflicts(tmp_path, ordinary, instances):
    from ltspice_mcp.lib.variations import VariationError

    circuit = target_circuit(tmp_path)
    with pytest.raises(VariationError, match=r"overlap|ancestor"):
        materialize(
            circuit, tmp_path, [{"kind": "assign", "assign": ordinary, "instances": instances}]
        )
    assert not list(tmp_path.glob("case-*"))


def test_final_geometry_precedes_exact_nested_mismatch(tmp_path):
    circuit = target_circuit(tmp_path)
    cases = materialize(
        circuit,
        tmp_path,
        [
            {
                "kind": "assign",
                "assign": {"l": [".2u"]},
                "instances": [
                    {
                        "instance": ["XA", "Xleaf", "M0"],
                        "attribute": "parameter",
                        "parameter": "l",
                        "values": [".3u"],
                    }
                ],
            },
            {
                "kind": "random",
                "runs": 1,
                "seed": 17,
                "rules": [
                    {"rule": "mismatch", "instance": ["XA", "Xleaf", "M0"], "AVT": 0.003},
                    {"rule": "param", "target": "w", "tolerance": 0.1},
                ],
            },
        ],
    )
    assert circuit.semantic_profile is not None
    hierarchy = load_hierarchy(str(cases[0].path), [tmp_path], circuit.semantic_profile)
    rows = {r.instance: r for r in hierarchy.instances}
    left = rows[("XA", "Xleaf", "M0")]
    right = rows[("XB", "Xleaf", "M0")]
    assert "delvto" in dict(left.parameters)
    assert "delvto" not in dict(right.parameters)
    assert dict(left.geometry)["l"].value == 3e-7
    assert cases[0].text.count(".subckt ") == 4
    from ltspice_mcp.lib.instance_targeting import canonical_target
    from ltspice_mcp.lib.montecarlo import (
        InstanceGeometry,
        MCSampler,
        MismatchRule,
        sample_instance_mismatch,
    )

    sampler = MCSampler(17).derive("bench:case0:run1")
    width = dict(left.geometry)["w"].value
    assert width is not None
    expected = sample_instance_mismatch(
        sampler,
        InstanceGeometry(canonical_target(left.instance, "mismatch"), "n", width, 3e-7),
        MismatchRule(prefix="M", avt=0.003),
    )
    assert dict(left.parameters)["delvto"].value == pytest.approx(expected["dvth"], rel=2e-5)


def test_source_lineage_covers_final_local_model_bins_and_included_cards(tmp_path):
    circuit = target_circuit(tmp_path, fragment=True)
    (case,) = materialize(
        circuit,
        tmp_path,
        [
            {
                "kind": "assign",
                "instances": [
                    {
                        "instance": ["XA", "Xleaf", "M0"],
                        "attribute": "parameter",
                        "parameter": "delvto",
                        "values": [0.02],
                    },
                ],
            }
        ],
    )
    assert circuit.semantic_profile is not None
    hierarchy = load_hierarchy(str(case.path), [tmp_path], circuit.semantic_profile)
    lineage = {item.case_source: item.staged_source for item in case.source_lineage}
    for row in hierarchy.instances:
        assert row.source in lineage
        for model in row.model_family:
            assert model.source in lineage
            assert lineage[model.source].definition == "leaf"
            assert Path(lineage[model.source].path).name == "body.inc"
            assert lineage[model.source].line == 3
    left = next(row for row in hierarchy.instances if row.instance == ("XA", "Xleaf", "M0"))
    assert lineage[left.source].line == 2
    assert lineage[left.source].path == str(tmp_path / "body.inc")


def test_zero_edit_lineage_captures_active_library_and_include_cards(tmp_path):
    from ltspice_mcp.lib.variations import (
        CircuitDeck,
        DeckFile,
        expand_variations,
        materialize_variants,
    )

    library = tmp_path / "models.lib"
    library.write_text(".lib tt\n.model n NMOS(level=54)\n.endl tt\n", encoding="utf-8")
    text = '* bench\n.lib "models.lib" tt\nR1 a 0 1k\n.end\n'
    circuit = CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        text,
        (DeckFile(library, library.read_text(encoding="utf-8")),),
        semantic_profile=SemanticProfile("ngspice", "hsa"),
        record_source_lineage=True,
    )
    (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path / "out")
    lineage = {item.case_source: item.staged_source for item in case.source_lineage}
    assert any(
        source.line == 2 and source.path == str(circuit.path) for source in lineage.values()
    )
    assert any(source.line == 2 and source.path == str(library) for source in lineage.values())
    assert all(Path(case_source.path).exists() for case_source in lineage)


@pytest.mark.parametrize("edit", ["none", "ordinary", "targeted"])
def test_repeated_active_include_keeps_contextual_lineage(tmp_path, edit):
    from ltspice_mcp.lib.hierarchy import Source
    from ltspice_mcp.lib.variations import CircuitDeck, DeckFile

    body = tmp_path / "body.inc"
    body_text = "R1 p 0 1k\n"
    body.write_text(body_text, encoding="utf-8")
    root = tmp_path / "bench.cir"
    text = (
        '* bench\nRtop a 0\n+ 10k\n.subckt first p\n.include "body.inc"\n.ends first\n'
        '.subckt second p\n.include "body.inc"\n.ends second\n'
        "XA a first\nXB b second\n.end\n"
    )
    root.write_text(text, encoding="utf-8")
    profile = SemanticProfile("ngspice", "hsa")
    circuit = CircuitDeck(
        "bench",
        root,
        text,
        (DeckFile(body, body_text),),
        semantic_profile=profile,
        record_source_lineage=True,
    )
    entries = []
    if edit == "ordinary":
        entries = [{"kind": "assign", "assign": {"Rtop": ["20k"]}}]
    elif edit == "targeted":
        entries = [
            {
                "kind": "assign",
                "instances": [
                    {"instance": ["XA", "R1"], "attribute": "value", "values": ["2k"]},
                ],
            }
        ]
    (case,) = materialize(circuit, tmp_path, entries)
    hierarchy = load_hierarchy(str(case.path), [tmp_path], profile)
    lineage = {item.case_source: item.staged_source for item in case.source_lineage}
    resistors = [row for row in hierarchy.instances if row.reference == "R1"]
    assert [row.instance for row in resistors] == [("XA", "R1"), ("XB", "R1")]
    assert all(row.source in lineage for row in resistors)
    assert [lineage[row.source] for row in resistors] == [Source(str(body), 1, None)] * 2
    assert len({row.source.definition for row in resistors}) == 2
    top = next(row for row in hierarchy.instances if row.reference == "Rtop")
    assert top.value.value == (20000 if edit == "ordinary" else 10000)
    assert lineage[top.source] == Source(str(root), 2, None)
    assert [row.value.value for row in resistors] == [2000 if edit == "targeted" else 1000, 1000]
    assert [item.case_source for item in case.source_lineage] == [
        occurrence.source for occurrence in hierarchy.occurrences
    ]
    assert root.read_text(encoding="utf-8") == text
    assert body.read_text(encoding="utf-8") == body_text


@pytest.mark.parametrize("edit", [False, True])
def test_lineage_keeps_distinct_library_sections(tmp_path, edit):
    from ltspice_mcp.lib.variations import CircuitDeck, DeckFile

    library = tmp_path / "body.lib"
    library_text = ".lib tt\nR1 p 0 1k\n.endl tt\n.lib ff\nR1 p 0 2k\n.endl ff\n"
    library.write_text(library_text, encoding="utf-8")
    text = (
        '* bench\n.subckt first p\n.lib "body.lib" tt\n.ends first\n'
        '.subckt second p\n.lib "body.lib" ff\n.ends second\n'
        "XA a first\nXB b second\n.end\n"
    )
    profile = SemanticProfile("ngspice", "hsa")
    circuit = CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        text,
        (DeckFile(library, library_text),),
        semantic_profile=profile,
        record_source_lineage=True,
    )
    entries = (
        [
            {
                "kind": "assign",
                "instances": [
                    {"instance": ["XA", "R1"], "attribute": "value", "values": ["3k"]},
                ],
            }
        ]
        if edit
        else []
    )
    (case,) = materialize(circuit, tmp_path, entries)
    hierarchy = load_hierarchy(str(case.path), [tmp_path], profile)
    lineage = {item.case_source: item.staged_source for item in case.source_lineage}
    resistors = [row for row in hierarchy.instances if row.reference == "R1"]
    assert [(lineage[row.source].line, lineage[row.source].section) for row in resistors] == [
        (2, "tt"),
        (5, "ff"),
    ]
    assert [lineage[row.source].definition for row in resistors] == ["first", "second"]
    assert all(lineage[row.source].path == str(library) for row in resistors)


def test_native_zero_edit_cases_reuse_lineage_and_closure_scans(tmp_path, monkeypatch):
    import ltspice_mcp.lib.variations as variations_module
    from ltspice_mcp.lib.variations import (
        CircuitDeck,
        PdkNativeVariation,
        expand_variations,
        materialize_variants,
    )

    text = "* bench\nR1 a 0 1k\n.end\n"
    circuit = CircuitDeck(
        "bench",
        tmp_path / "bench.cir",
        text,
        semantic_profile=SemanticProfile("ngspice", "hsa"),
        record_source_lineage=True,
    )
    native = PdkNativeVariation(
        kind="pdk_native", id="foundry", runs=3, seed=17, profile="pinned", mode="nominal"
    )
    expanded = expand_variations([circuit], [native])
    calls = {"lineage": 0, "closure": 0, "referrers": 0}
    for label, name in (
        ("lineage", "source_lineage"),
        ("closure", "_case_closure"),
        ("referrers", "_include_referrers"),
    ):
        original = getattr(variations_module, name)

        def counted(*args, _name=label, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(variations_module, name, counted)
    cases = materialize_variants(circuit, expanded, tmp_path / "out")
    assert calls == {"lineage": 1, "closure": 0, "referrers": 0}
    assert [case.text for case in cases] == [text] * 3
    assert [case.assignments for case in cases] == [{}, {}, {}]
    assert all(
        any(item.case_source.path == str(case.path) for item in case.source_lineage)
        for case in cases
    )


def test_profile_does_not_force_control_deck_through_target_editor(tmp_path):
    from ltspice_mcp.lib.variations import CircuitDeck, expand_variations, materialize_variants

    text = "* driver\n.control\nop\n.endc\n.end\n"
    circuit = CircuitDeck(
        "driver", tmp_path / "driver.cir", text, semantic_profile=SemanticProfile("ngspice", "hsa")
    )
    (case,) = materialize_variants(circuit, expand_variations([circuit], []), tmp_path / "out")
    assert case.text == text
    assert case.source_lineage == ()


@pytest.fixture
def real_pdk_cases(tmp_path):
    import hashlib
    import os

    from ltspice_mcp.lib.deck_staging import stage_deck
    from ltspice_mcp.lib.variations import CircuitDeck, DeckFile

    root_value = os.environ.get("LTSPICE_MCP_TEST_PDK_ROOT")
    if not root_value:
        pytest.skip("set LTSPICE_MCP_TEST_PDK_ROOT to the actual pinned sky130A tree")
    root = Path(root_value)
    spice = root / "libs.ref/sky130_fd_pr/spice"
    model = "sky130_fd_pr__nfet_01v8"
    source = tmp_path / "real.cir"
    source.write_text(
        "* two real foundry devices\n.option scale=1u\n.param mc_mm_switch=0 mc_pr_switch=0\n"
        f'.include "{root / "libs.tech/ngspice/parameters/lod.spice"}"\n'
        f'.include "{spice / (model + "__mismatch.corner.spice")}"\n'
        f'.include "{spice / (model + "__tt.corner.spice")}"\n'
        ".param mc_mm_switch=0\n.param mc_pr_switch=0\n"
        f".subckt block p d g\nR1 p 0 1k\nXN d g 0 0 {model} w=1 l=.15\n.ends block\n"
        "XA pa da g block\nXB pb db g block\nVA pa 0 1\nVB pb 0 1\nVDA da 0 .9\nVDB db 0 .9\nVG g 0 .9\n.op\n.end\n",
        encoding="utf-8",
    )
    staged = stage_deck(source, tmp_path / "staged", [tmp_path, root], origin=source)
    circuit = CircuitDeck(
        "real",
        staged.staged_deck,
        staged.text,
        tuple(DeckFile(file.staged_path, file.text, file.sha256) for file in staged.includes),
        semantic_profile=SemanticProfile("ngspice", "hsa"),
    )
    assert circuit.semantic_profile is not None
    original_hierarchy = load_hierarchy(str(source), [tmp_path, root], circuit.semantic_profile)
    original_hashes = {
        captured.path: hashlib.sha256(captured.content).hexdigest()
        for captured in original_hierarchy.inputs
    }
    mos = [r for r in original_hierarchy.instances if r.element == "M"]
    assert len(mos) == 2
    assert all(len(row.model_family) == 180 for row in mos)
    leaf = ["XA", "XN", "msky130_fd_pr__nfet_01v8"]
    cases = materialize(
        circuit,
        staged.staged_deck.parent,
        [
            {
                "kind": "assign",
                "combine": "zip",
                "instances": [
                    {"instance": ["XA", "R1"], "attribute": "value", "values": ["1k", "2k", "1k"]},
                    {
                        "instance": leaf,
                        "attribute": "parameter",
                        "parameter": "w",
                        "values": [1, 1.1, 1],
                    },
                    {
                        "instance": leaf,
                        "attribute": "parameter",
                        "parameter": "delvto",
                        "values": [0, 0, 0.02],
                    },
                ],
            }
        ],
    )
    for path, digest in original_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return circuit, cases, original_hashes


def test_real_pdk_electrical_nested_width_threshold_and_peer(real_pdk_cases, tmp_path):
    import hashlib
    import json
    import re
    import shutil
    import subprocess

    circuit, cases, hashes = real_pdk_cases
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    facts = []
    signals = [
        f"@m.{side}.xn.msky130_fd_pr__nfet_01v8[{parameter}]"
        for side in ("xa", "xb")
        for parameter in ("id", "vth")
    ]
    signals += ["@r.xa.r1[i]", "@r.xb.r1[i]"]

    def probe(deck: Path, label: str):
        driver = tmp_path / f"driver{label}.cir"
        driver.write_text(
            f"* driver\n.control\nset ngbehavior=hsa\nsource {deck.name}\nop\nprint "
            + " ".join(signals)
            + "\nquit\n.endc\n.end\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [executable, "-b", str(driver)],
            cwd=deck.parent,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (tmp_path / f"electrical{label}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Compatibility modes selected: hs a" in result.stdout
        values = {
            key.lower(): float(value)
            for key, value in re.findall(r"(@\S+)\s*=\s*([-+0-9.eE]+)", result.stdout)
        }
        assert set(signals) <= values.keys(), result.stdout + result.stderr
        return values

    for index, case in enumerate(cases):
        assert circuit.semantic_profile is not None
        hierarchy = load_hierarchy(str(case.path), [tmp_path], circuit.semantic_profile)
        mos = [row for row in hierarchy.instances if row.element == "M"]
        assert [dict(row.geometry)["w"].value for row in mos] == [
            1.1e-6 if index == 1 else 1e-6,
            1e-6,
        ]
        lineage = {item.case_source: item.staged_source for item in case.source_lineage}
        assert all(model.source in lineage for row in mos for model in row.model_family)
        public_row = mos[0].row()
        assert len(json.dumps(public_row).encode("utf-8")) < 100_000
        assert all("raw" not in model for model in public_row["model"]["family"])
        facts.append(probe(case.path, str(index)))
    assert facts[0] == pytest.approx(probe(circuit.path, "direct"), rel=1e-6)
    assert facts[1][signals[0]] > 1.05 * facts[0][signals[0]]
    assert facts[2][signals[1]] - facts[0][signals[1]] == pytest.approx(0.02, abs=2e-6)
    assert facts[2][signals[0]] < facts[0][signals[0]]
    assert [row[signals[4]] for row in facts] == pytest.approx([0.001, 0.0005, 0.001])
    assert all(
        row[signals[2]] == facts[0][signals[2]]
        and row[signals[3]] == facts[0][signals[3]]
        and row[signals[5]] == 0.001
        for row in facts
    )
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items()
    )


async def test_real_pdk_public_run_and_exact_operating_point(
    real_pdk_cases, tmp_path, monkeypatch
):
    import shutil

    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.simulator import detect_simulators
    from ltspice_mcp.state import SessionState
    from tests.conftest import terminal_experiment
    from tests.test_ngspice_e2e import _analyze

    if shutil.which("ngspice") is None:
        pytest.skip("ngspice is not on PATH")
    circuit, cases, _ = real_pdk_cases
    case = cases[2]
    path = case.path.with_name("public.cir")
    path.write_text(
        case.path.read_text(encoding="utf-8").replace(
            ".end\n",
            ".save @m.xa.xn.msky130_fd_pr__nfet_01v8[id] @m.xa.xn.msky130_fd_pr__nfet_01v8[vth]\n.save @m.xb.xn.msky130_fd_pr__nfet_01v8[id] @m.xb.xn.msky130_fd_pr__nfet_01v8[vth]\n.end\n",
        ),
        encoding="utf-8",
    )
    assert circuit.semantic_profile is not None
    hierarchy = load_hierarchy(str(path), [tmp_path], circuit.semantic_profile)
    device = next(
        row.device
        for row in hierarchy.instances
        if row.instance == ("XA", "XN", "msky130_fd_pr__nfet_01v8")
    )
    config = ServerConfig(
        working_dir=tmp_path, allowed_paths=[tmp_path], simulator="ngspice", ngbehavior="hsa"
    )
    available = detect_simulators(config)
    monkeypatch.setattr(NGspiceSimulator, "_compatibility_mode", "hsa")
    state = SessionState.create(config, available)
    try:
        receipt = await terminal_experiment(
            state,
            {
                "request_id": "targeting-real-pdk",
                "circuits": [{"path": str(path)}],
                "execution": {"simulator": "ngspice", "wait_s": 90},
            },
        )
        assert receipt["status"] == "completed", receipt
        result = await _analyze(
            state,
            receipt["job_id"],
            [{"key": "selected", "metric": "operating_point", "device": device}],
        )
        assert "selected" in result["results"], result
        points = result["results"]["selected"]["values"][0]["value"]["device_op_points"]
        assert points and all("xa.xn" in key.casefold() for key in points), points
        assert any(key.endswith("[vth])") and value > 0.1 for key, value in points.items())
        from ltspice_mcp.lib.services import resolve_job

        job = resolve_job(receipt["job_id"], state)
        assert job.cases[0].log_file is not None
        assert "Compatibility modes selected: hs a" in job.cases[0].log_file.read_text(
            encoding="utf-8"
        )
    finally:
        await state.shutdown()


def test_active_occurrences_keep_library_order_and_lexical_scope(tmp_path):
    library = tmp_path / "parts.lib"
    library.write_text(
        ".lib tt\n.param switch=0\n.model n NMOS(level=54)\n.endl tt\n", encoding="utf-8"
    )
    hierarchy = inspect_text(
        tmp_path,
        '.subckt leaf p\n.lib "parts.lib" tt\n.param switch=1\nM0 p p 0 0 n w=1u l=1u\n.ends\nXA a leaf',
    )
    cards = hierarchy.occurrences
    call = next(i for i, o in enumerate(cards) if o.card.body.startswith(".lib"))
    assert cards[call + 1].card.body == ".param switch=0"
    assert cards[call + 3].card.body == ".param switch=1"
    assert all(cards[i].source.definition == "leaf" for i in range(call, call + 4))
    (edge,) = hierarchy.include_bindings
    (binding,) = hierarchy.library_bindings
    assert edge.source == binding.source == cards[call].source
    assert edge.target == binding.target == library
    assert edge.section == binding.section == "tt"


def test_exact_scoped_capability_refuses_global_name_only_match(tmp_path):
    circuit = target_circuit(tmp_path)
    from dataclasses import replace

    circuit = replace(
        circuit,
        text=circuit.text.replace(".model n NMOS(level=54)", ".model n NMOS(level=1)").replace(
            ".param w=1u", ".model n NMOS(level=54)\n.param w=1u"
        ),
    )
    with pytest.raises(ValueError, match="exact model family"):
        materialize(
            circuit,
            tmp_path,
            [
                {
                    "kind": "random",
                    "runs": 1,
                    "rules": [
                        {"rule": "mismatch", "instance": ["XA", "Xleaf", "M0"], "AVT": 0.003}
                    ],
                }
            ],
        )
    assert not list(tmp_path.glob("case-*"))


def test_legacy_alias_and_structured_mismatch_field_overlap(tmp_path):
    circuit = target_circuit(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        materialize(
            circuit,
            tmp_path,
            [
                {
                    "kind": "assign",
                    "assign": {"XA.Xleaf.M0:delvto": [0.01]},
                    "instances": [
                        {
                            "instance": ["xa", "xleaf", "m0"],
                            "attribute": "parameter",
                            "parameter": "DELVTO",
                            "values": [0.02],
                        }
                    ],
                }
            ],
        )


def test_cross_family_multi_field_write_overlap(tmp_path):
    circuit = target_circuit(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        materialize(
            circuit,
            tmp_path,
            [
                {
                    "kind": "assign",
                    "instances": [
                        {
                            "instance": ["XA", "Xleaf", "M0"],
                            "attribute": "value",
                            "values": ["n w=2u"],
                        }
                    ],
                },
                {
                    "kind": "assign",
                    "instances": [
                        {
                            "instance": ["xa", "xleaf", "m0"],
                            "attribute": "parameter",
                            "parameter": "w",
                            "values": ["3u"],
                        }
                    ],
                },
            ],
        )


@pytest.mark.parametrize("simulator", ["ltspice", "ngspice"])
def test_profiled_flat_mismatch_retains_model_clone_engine(tmp_path, simulator):
    from ltspice_mcp.lib.variations import CircuitDeck

    circuit = CircuitDeck(
        "bench",
        tmp_path / "flat.cir",
        "* flat\n.param w=1u\n.model n NMOS(level=1 vto=.5 kp=100u)\nM1 a g 0 0 n w={w} l=1u\nM2 b g 0 0 n w={w} l=1u\n.end\n",
        semantic_profile=SemanticProfile(simulator, "hsa" if simulator == "ngspice" else None),
    )
    (case,) = materialize(
        circuit,
        tmp_path,
        [
            {"kind": "assign", "assign": {"w": ["2u"]}},
            {
                "kind": "random",
                "runs": 1,
                "seed": 13,
                "rules": [{"rule": "mismatch", "prefix": "M1", "AVT": 0.003}],
            },
        ],
    )
    assert circuit.semantic_profile is not None
    hierarchy = load_hierarchy(str(case.path), [tmp_path], circuit.semantic_profile)
    left, right = hierarchy.instances
    assert left.model_name != right.model_name == "n"
    assert dict(left.geometry)["w"].value == 2e-6
    assert left.model_source in {item.case_source for item in case.source_lineage}
