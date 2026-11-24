"""Which parts of two decks are the same part.

LTspice exports a part named ``Load`` drawn with a resistor symbol as
``R§Load`` (recorded: ``deck/section_sign_utf8``), and an ngspice copy of that
deck names it ``RLoad``. So ``R§Load`` and ``RLoad`` in two decks are one
part, but in one deck they are two. A part pairs by its exact reference first
(letter case aside); only a part with no exact partner pairs by its name
without the section sign, and a fold that would pair more than one part on a
side pairs none of them and says so. Every change is reported under the
reference as the deck writes it.

Each case goes through every path that answers the question: the graph
comparison for both readers, ``verify_circuit``'s equivalence and
``structural_diff`` modes (the handler and the Python API), the export's
``diff_vs_prior`` (sidecar and managed) and ``edit_schematic``'s reference
stage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.netlist_graph import compare_graphs
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools import verify as vc
from tests._asc_ops import DIVIDER_OPS
from tests.conftest import LTSPICE_READER, SyncApi
from tests.test_edit_schematic import _build_blank
from tests.test_structural_diff_scopes import _exporter_writing
from tests.test_verify_circuit import _RES_ASC, _run, _with_ltspice, _write

_NGSPICE_READER = SemanticProfile.of(Producer("ngspice", None, None), "kiltpsa")
READERS = [
    pytest.param(LTSPICE_READER, id="ltspice"),
    pytest.param(_NGSPICE_READER, id="ngspice"),
]


def _top(*cards: str) -> str:
    return "* parts\nI1 0 a 1\n" + "".join(f"{card}\n" for card in cards) + ".op\n.end\n"


def _block(*cards: str) -> str:
    body = "".join(f"{card}\n" for card in cards)
    return f"* parts\nI1 0 a 1\nX1 a BLK\n.subckt BLK p\n{body}.ends BLK\n.op\n.end\n"


@dataclass(frozen=True)
class Case:
    """Two decks, and what each comparison says about their parts."""

    baseline: str
    compared: str
    #: ``structural_diff``: the references added, removed and changed.
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    #: Equivalence: each value difference, ``(ref, before, after)``, and the
    #: leaves only one side has.
    values: list[tuple[str, str, str]] = field(default_factory=list)
    leaves_added: list[str] = field(default_factory=list)
    leaves_removed: list[str] = field(default_factory=list)
    #: The parts a fold would pair more than one of, as written per side.
    ambiguous: list[tuple[list[str], list[str]]] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return not (
            self.added or self.removed or self.changed or self.values or self.ambiguous
        )


def _cases() -> Iterator[Any]:
    both = _top("RLoad a 0 1k", "R§Load a 0 2k")
    yield pytest.param(
        Case(
            both,
            _top("RLoad a 0 7k", "R§Load a 0 2k"),
            changed=["RLoad"],
            values=[("RLoad", "1k", "7k")],
        ),
        id="the_part_without_the_sign_changes_value",
    )
    yield pytest.param(
        Case(
            both,
            _top("RLoad a 0 1k", "R§Load a 0 5k"),
            changed=["R§Load"],
            values=[("R§Load", "2k", "5k")],
        ),
        id="the_part_with_the_sign_is_named_as_written",
    )
    yield pytest.param(
        Case(both, _top("R§Load a 0 2k"), removed=["RLoad"], leaves_removed=["RLoad"]),
        id="the_part_without_the_sign_is_removed",
    )
    yield pytest.param(
        Case(_top("R§Load a 0 2k"), both, added=["RLoad"], leaves_added=["RLoad"]),
        id="a_part_without_the_sign_is_added_beside_the_export_s",
    )
    yield pytest.param(
        Case(_top("R§Load a 0 2k"), _top("RLoad a 0 2k")),
        id="an_export_and_its_ngspice_copy_are_one_circuit",
    )
    yield pytest.param(
        Case(
            _block("RLoad p 0 1k", "R§Load p 0 2k"),
            _block("RLoad p 0 7k", "R§Load p 0 2k"),
            changed=["RLoad"],
            values=[("X1.RLoad", "1k", "7k")],
        ),
        id="in_a_subckt_the_part_without_the_sign_changes_value",
    )
    yield pytest.param(
        Case(
            _block("RLoad p 0 1k", "R§Load p 0 2k"),
            _block("R§Load p 0 2k"),
            removed=["RLoad (in .subckt BLK)"],
            leaves_removed=["X1.RLoad"],
        ),
        id="in_a_subckt_the_part_without_the_sign_is_removed",
    )
    yield pytest.param(
        Case(
            _top("R§Load a 0 1k", "RL§oad a 0 1k"),
            _top("RLoad a 0 1k"),
            ambiguous=[(["R§Load", "RL§oad"], ["RLoad"])],
        ),
        id="a_fold_two_parts_on_one_side_share_pairs_neither",
    )


CASES = list(_cases())


def _refs(rows: list[Any]) -> list[str]:
    return [row if isinstance(row, str) else row["reference"] for row in rows]


def _assert_delta(delta: dict[str, Any], case: Case) -> None:
    assert delta.get("components_added", []) == case.added
    assert delta.get("components_removed", []) == case.removed
    assert _refs(delta.get("components_changed", [])) == case.changed
    assert delta.get("components_renamed", []) == []
    assert [
        (row["before"], row["after"]) for row in delta.get("components_ambiguous", [])
    ] == case.ambiguous


def _assert_equivalence(comparison: dict[str, Any], case: Case) -> None:
    rows = [
        (row["ref"], row["reference_value"], row["candidate_value"])
        for row in comparison.get("value_mismatches", [])
    ]
    assert rows == case.values
    assert [row["ref"] for row in comparison.get("added", [])] == case.leaves_added
    assert [row["ref"] for row in comparison.get("removed", [])] == case.leaves_removed
    assert [
        (row["reference_refs"], row["candidate_refs"])
        for row in comparison.get("ambiguous", [])
    ] == case.ambiguous
    assert comparison.get("renamed", []) == []
    assert comparison["equivalent"] is case.equivalent


# ---------------------------------------------------------------------------
# The graph comparison, for each reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("case", CASES)
def test_graph_comparison(case: Case, reader: SemanticProfile):
    comparison = compare_graphs(case.baseline, case.compared, reader=reader)
    _assert_equivalence(comparison.as_dict() | {"equivalent": comparison.equivalent}, case)


# ---------------------------------------------------------------------------
# verify_circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
async def test_equivalence_mode(state_no_sim: SessionState, work_dir: Path, case: Case):
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    comparison = (await _run(state_no_sim, path=str(deck), reference=str(ref)))["comparison"]
    _assert_equivalence(comparison, case)


@pytest.mark.parametrize("case", CASES)
async def test_structural_diff_mode(state_no_sim: SessionState, work_dir: Path, case: Case):
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    _assert_delta(data["comparison"], case)
    assert data["comparison"]["equivalent"] is case.equivalent


@pytest.mark.parametrize("case", CASES)
def test_python_api_structural_diff(state_no_sim: SessionState, work_dir: Path, case: Case):
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    data = SyncApi(state_no_sim).verify_circuit(
        path=str(deck), compare={"reference": str(ref), "mode": "structural_diff"}
    )
    _assert_delta(data["comparison"], case)


# ---------------------------------------------------------------------------
# The export's diff_vs_prior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
async def test_diff_vs_prior_sidecar(config, work_dir, asc_symbols, monkeypatch, case: Case):
    monkeypatch.setattr(vc, "_create_netlist", _exporter_writing([case.compared]))
    state = _with_ltspice(config)
    asc = _write(work_dir, "s.asc", _RES_ASC)
    _write(work_dir, "s.net", case.baseline)
    data = await _run(state, path=str(asc), checks=["export"], export_to="sidecar")
    _assert_delta(data["export"]["diff_vs_prior"], case)


@pytest.mark.parametrize("case", CASES)
async def test_diff_vs_prior_managed(config, work_dir, asc_symbols, monkeypatch, case: Case):
    monkeypatch.setattr(vc, "_create_netlist", _exporter_writing([case.baseline, case.compared]))
    state = _with_ltspice(config)
    asc = _write(work_dir, "m.asc", _RES_ASC)
    await _run(state, path=str(asc), checks=["export"])
    data = await _run(state, path=str(asc), checks=["export"])
    _assert_delta(data["export"]["diff_vs_prior"], case)


# ---------------------------------------------------------------------------
# edit_schematic's reference stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
async def test_edit_schematic_reference_stage(asc_state, monkeypatch, case: Case):
    async def export(_copy, _state):
        return case.compared

    monkeypatch.setattr(se, "_export_asc_to_netlist", export)
    data = await _build_blank(
        asc_state,
        "parts",
        DIVIDER_OPS,
        dry_run=True,
        compare={"reference": case.baseline, "mode": "structural_diff"},
    )
    verification = data["verification"]
    _assert_delta(verification["comparison"], case)
    assert verification["equivalent"] is case.equivalent
