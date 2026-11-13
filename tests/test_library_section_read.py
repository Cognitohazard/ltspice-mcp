"""Which cards of a sectioned library are read: ``.lib file tt`` reads the
cards of section ``tt`` and no other (``deck_staging.read_in_section``).

Hierarchy discovery read only the selected section, while the analyses a run
expects (``runner_base.deck_requirements``) and the models the linter finds
read every section of the library, so a model or a ``.save`` that only the
unselected corner declares counted as present.
"""

from __future__ import annotations

from pathlib import Path

from ltspice_mcp.lib.deck_staging import in_other_section
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.runner_base import _deck_has_sectioned_lib, deck_requirements

_LIBRARY = (
    "* two corners\n"
    ".lib tt\n.model ntt NMOS(level=1 vto=0.7)\n.endl tt\n"
    ".lib ff\n.model nff NMOS(level=1 vto=0.6)\n.save v(d)\n.ac dec 1 1 10\n.endl ff\n"
)


def _deck(tmp_path: Path, device_model: str) -> Path:
    (tmp_path / "corners.lib").write_text(_LIBRARY, encoding="utf-8")
    deck = tmp_path / "d.cir"
    deck.write_text(
        "* sectioned library\n.lib corners.lib tt\n"
        f"M1 d g 0 0 {device_model} W=1u L=1u\nV1 d 0 1\nV2 g 0 1\n.op\n.end\n",
        encoding="utf-8",
    )
    return deck


def _model_missing(deck: Path) -> list[str]:
    findings = lint_deck(
        deck.read_text(encoding="utf-8"), deck, "ngspice", "ngspice", ngbehavior="hsa"
    )
    return [f["subject"] for f in findings if f["rule_id"] == "model-missing"]


def test_a_model_only_the_unselected_section_declares_is_missing_everywhere(tmp_path: Path):
    deck = _deck(tmp_path, "nff")
    assert _model_missing(deck) == ["M1"]
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", "hsa"))
    (fet,) = [row for row in hierarchy.instances if row.reference.upper() == "M1"]
    assert fet.model_name == "nff"
    assert fet.model_family == ()


def test_a_model_the_selected_section_declares_is_found(tmp_path: Path):
    assert _model_missing(_deck(tmp_path, "ntt")) == []


def test_the_unselected_sections_save_and_analysis_are_not_expected(tmp_path: Path):
    deck = _deck(tmp_path, "ntt")
    requirements = deck_requirements(deck)
    assert requirements.analyses == (".op",)
    assert not requirements.has_save
    assert _deck_has_sectioned_lib(deck)


def test_the_section_rule():
    assert in_other_section("ff", "tt")
    assert not in_other_section("TT", "tt")
    assert not in_other_section(None, "tt")
    assert not in_other_section("tt", None)


_OUTSIDE = (
    "* a card outside every section\n.model nout NMOS(level=1 vto=0.5)\n"
    ".lib tt\n.model ntt NMOS(level=1 vto=0.7)\n.endl tt\n"
)


def test_hierarchy_alone_leaves_out_the_cards_outside_every_section(tmp_path: Path):
    """No recording decides whether ``.lib file tt`` reads a card outside
    every section of the file. Hierarchy discovery leaves it out; the linter
    reads it, as before (``deck_staging.in_other_section``)."""
    (tmp_path / "corners.lib").write_text(_OUTSIDE, encoding="utf-8")
    deck = tmp_path / "d.cir"
    deck.write_text(
        "* t\n.lib corners.lib tt\nM1 d g 0 0 nout W=1u L=1u\nV1 d 0 1\nV2 g 0 1\n.op\n.end\n",
        encoding="utf-8",
    )
    assert _model_missing(deck) == []
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", "hsa"))
    (fet,) = [row for row in hierarchy.instances if row.reference.upper() == "M1"]
    assert fet.model_family == ()


def test_a_model_at_the_end_of_the_deepest_include_staging_takes_is_found(tmp_path: Path):
    """Staging takes an include chain eight files deep; the linter read the
    last file's models before following its references, and must still."""
    for level in range(1, 8):
        (tmp_path / f"f{level}.inc").write_text(f".include f{level + 1}.inc\n", encoding="utf-8")
    (tmp_path / "f8.inc").write_text(".model nx NMOS(level=1)\n", encoding="utf-8")
    deck = tmp_path / "d.cir"
    deck.write_text(
        "* t\n.include f1.inc\nM1 d g 0 0 nx W=1u L=1u\nV1 d 0 1\nV2 g 0 1\n.op\n.end\n",
        encoding="utf-8",
    )
    assert _model_missing(deck) == []
