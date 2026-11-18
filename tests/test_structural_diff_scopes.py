"""Which cards of two decks are the same element or directive, and what
differs between the two decks.

An element is its scope plus its reference: the top level, or the
``.subckt`` it sits in. A hierarchical sheet's export has one ``.subckt`` per
block, after the top level, and blocks reuse references (recorded:
``blocks/nested_sheet_beside_both``, where the top level and block ``outer``
each hold an ``X§X1``). Each case below goes through every path that answers
the question: ``verify_circuit``'s ``structural_diff`` mode (the handler and
the Python API), the export's ``diff_vs_prior`` (sidecar and managed), and
``edit_schematic``'s reference stage; the equivalence mode answers the same
inputs through the graph and is pinned beside them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools import verify as vc
from tests import _ltspice_recorded as rec
from tests._asc_ops import DIVIDER_OPS
from tests.conftest import SyncApi
from tests.test_edit_schematic import _build_blank
from tests.test_verify_circuit import _RES_ASC, _run, _with_ltspice, _write

# ---------------------------------------------------------------------------
# The decks
# ---------------------------------------------------------------------------

_TOP = "V1 in 0 5\nVPRE pre 0 {top}\nX1 in out BLK\nR9 out 0 1k\n"
_BLK = ".subckt BLK a b\nVPRE p 0 {inner}\nR1 a b 1k\nR2 p b 1k\n.ends BLK\n"


def _deck(top: str = "5", inner: str = "1", *, block_first: bool = False) -> str:
    """A top-level ``VPRE`` and one inside ``BLK``, the block after the top
    level as an export writes it, or before it."""
    parts = [_TOP.format(top=top), _BLK.format(inner=inner)]
    if block_first:
        parts.reverse()
    return "* scopes\n" + "".join(parts) + ".end\n"


def _recorded(build: str) -> str:
    return read_spice_text(rec.recorded(build, "blocks/nested_sheet_beside_both.net"))


def _blocks_first(text: str) -> str:
    """The recorded export with its block definitions moved before its top
    level: the same circuit."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("* block symbol"))
    end = next(i for i, line in enumerate(lines) if line.startswith(".op"))
    top_from = 2 if lines[1].startswith("* Generated") else 1
    return "".join(lines[:top_from] + lines[start:end] + lines[top_from:start] + lines[end:])


@dataclass(frozen=True)
class Case:
    """Two decks and what the structural delta between them is."""

    baseline: str
    compared: str
    changed: list[dict[str, str]]
    #: What equivalence reports as value differences, ``(ref, before, after)``.
    values: list[tuple[str, str, str]]


def _cases() -> Iterator[Any]:
    yield pytest.param(
        Case(
            _deck(),
            _deck(top="6"),
            [{"reference": "VPRE", "before": "5", "after": "6"}],
            [("VPRE", "5", "6")],
        ),
        id="top_level_change_beside_a_block_of_the_same_reference",
    )
    yield pytest.param(
        Case(
            _deck(),
            _deck(inner="2"),
            [{"reference": "VPRE", "subckt": "BLK", "before": "1", "after": "2"}],
            [("X1.VPRE", "1", "2")],
        ),
        id="block_change_is_said_to_be_the_block_s",
    )
    yield pytest.param(
        Case(_deck(), _deck(block_first=True), [], []),
        id="block_moved_before_the_top_level_is_no_change",
    )
    for build in rec.BUILDS:
        if not rec.has(build, "blocks/nested_sheet_beside_both.net"):
            continue
        text = _recorded(build)
        x1 = "X§X1" if "X§X1" in text else "XX1"
        yield pytest.param(
            Case(text, _blocks_first(text), [], []),
            id=f"{build}_export_with_its_blocks_moved_first_is_no_change",
        )
        swapped = text.replace(f"{x1} N001 out outer", f"{x1} N001 out blk")
        assert swapped != text
        yield pytest.param(
            Case(
                text,
                swapped,
                [{"reference": x1, "before": "outer", "after": "blk"}],
                # Equivalence compares the circuits each runs: the top level
                # calls another block, so its leaves differ, not a value.
                None,  # type: ignore[arg-type]
            ),
            id=f"{build}_export_top_level_instance_beside_a_block_s_of_one_name",
        )


CASES = list(_cases())


def _assert_delta(delta: dict[str, Any], case: Case) -> None:
    assert delta.get("components_changed", []) == case.changed
    assert delta.get("components_added", []) == []
    assert delta.get("components_removed", []) == []
    assert delta.get("directives_added", []) == []
    assert delta.get("directives_removed", []) == []


# ---------------------------------------------------------------------------
# verify_circuit, structural_diff mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
async def test_structural_diff_mode(state_no_sim: SessionState, work_dir: Path, case: Case):
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    data = await _run(
        state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
    )
    comparison = data["comparison"]
    _assert_delta(comparison, case)
    assert comparison["equivalent"] is (not case.changed)


@pytest.mark.parametrize("case", CASES)
def test_python_api_structural_diff(state_no_sim: SessionState, work_dir: Path, case: Case):
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    data = SyncApi(state_no_sim).verify_circuit(
        path=str(deck), compare={"reference": str(ref), "mode": "structural_diff"}
    )
    _assert_delta(data["comparison"], case)


@pytest.mark.parametrize("case", CASES)
async def test_equivalence_mode_on_the_same_decks(
    state_no_sim: SessionState, work_dir: Path, case: Case
):
    """The graph keeps each definition's elements apart: it already answers."""
    ref = _write(work_dir, "ref.cir", case.baseline)
    deck = _write(work_dir, "cand.cir", case.compared)
    comparison = (await _run(state_no_sim, path=str(deck), reference=str(ref)))["comparison"]
    if case.values is None:
        assert comparison["equivalent"] is False
        return
    rows = [
        (row["ref"], row["reference_value"], row["candidate_value"])
        for row in comparison.get("value_mismatches", [])
    ]
    assert rows == case.values
    assert comparison["equivalent"] is (not case.values)


# ---------------------------------------------------------------------------
# The export's diff_vs_prior
# ---------------------------------------------------------------------------


def _exporter_writing(texts: list[str]):
    """An exporter that writes ``texts`` in turn as each export's netlist."""
    pending = iter(texts)

    def exporter(_cls, asc_path, timeout=0):
        net = Path(asc_path).with_suffix(".net")
        net.write_bytes(next(pending).encode("utf-8"))
        return net

    return exporter


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
        "scoped",
        DIVIDER_OPS,
        dry_run=True,
        compare={"reference": case.baseline, "mode": "structural_diff"},
    )
    verification = data["verification"]
    _assert_delta(verification["comparison"], case)
    assert verification["equivalent"] is (not case.changed)


# ---------------------------------------------------------------------------
# Directives, by scope
# ---------------------------------------------------------------------------

_AMP_BODY = ".subckt AMP a b\n{body}R1 a b {{g*1k}}\n.ends AMP\n"


def _amp(body: str = ".param g=2\n", top: str = "") -> str:
    return (
        "* amp\nV1 a 0 1\nX1 a b AMP\nR9 b 0 1k\n" + top + _AMP_BODY.format(body=body) + ".end\n"
    )


async def test_a_param_moved_from_a_body_to_the_top_level_is_a_difference(state_no_sim, work_dir):
    ref = _write(work_dir, "ref.cir", _amp())
    deck = _write(work_dir, "cand.cir", _amp("", top=".param g=2\n"))
    comparison = (
        await _run(
            state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["equivalent"] is False
    assert comparison["directives_added"] == [".param g=2"]
    assert comparison["directives_removed"] == [".param g=2 (in .subckt AMP)"]


async def test_a_directive_in_a_body_is_said_to_be_the_body_s(state_no_sim, work_dir):
    ref = _write(work_dir, "ref.cir", _amp())
    deck = _write(work_dir, "cand.cir", _amp(".param g=3\n"))
    comparison = (
        await _run(
            state_no_sim, path=str(deck), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["directives_added"] == [".param g=3 (in .subckt AMP)"]
    assert comparison["directives_removed"] == [".param g=2 (in .subckt AMP)"]


async def test_an_element_in_a_nested_definition_names_the_definitions_around_it(
    state_no_sim, work_dir
):
    def deck(value: str) -> str:
        return (
            "* nest\nV1 a 0 1\nX1 a 0 OUTER\n"
            f".subckt OUTER p q\nX1 p q INNER\n.subckt INNER m n\nR1 m n {value}\n"
            ".ends INNER\nR1 p q 1k\n.ends OUTER\n.end\n"
        )

    ref = _write(work_dir, "ref.cir", deck("1k"))
    cand = _write(work_dir, "cand.cir", deck("2k"))
    comparison = (
        await _run(
            state_no_sim, path=str(cand), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["components_changed"] == [
        {"reference": "R1", "subckt": "OUTER/INNER", "before": "1k", "after": "2k"}
    ]


async def test_an_element_only_one_side_has_is_named_with_its_scope(state_no_sim, work_dir):
    ref = _write(work_dir, "ref.cir", _deck())
    cand = _write(work_dir, "cand.cir", _deck().replace("R2 p b 1k\n", ""))
    comparison = (
        await _run(
            state_no_sim, path=str(cand), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["components_removed"] == ["R2 (in .subckt BLK)"]


# ---------------------------------------------------------------------------
# A name defined more than once
# ---------------------------------------------------------------------------


async def test_a_name_defined_twice_in_one_scope_is_said_not_compared(state_no_sim, work_dir):
    twice = _deck().replace(".end\n", _BLK.format(inner="1") + ".end\n")
    ref = _write(work_dir, "ref.cir", twice)
    cand = _write(work_dir, "cand.cir", twice.replace("VPRE p 0 1", "VPRE p 0 7", 1))
    comparison = (
        await _run(
            state_no_sim, path=str(cand), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["equivalent"] is None
    assert not comparison.get("components_changed")
    assert any(".subckt BLK" in line for line in comparison["not_compared"])


async def test_definitions_of_one_name_in_two_scopes_are_kept_apart(state_no_sim, work_dir):
    """A ``BLK`` local to ``OUTER`` is not the top-level ``BLK``."""

    def deck(local: str) -> str:
        return (
            "* t\nX1 a 0 OUTER\nX2 a 0 BLK\n"
            f".subckt OUTER p q\nX1 p q BLK\n.subckt BLK m n\nR1 m n {local}\n.ends BLK\n"
            ".ends OUTER\n.subckt BLK m n\nR1 m n 1k\n.ends BLK\n.end\n"
        )

    ref = _write(work_dir, "ref.cir", deck("5k"))
    cand = _write(work_dir, "cand.cir", deck("6k"))
    comparison = (
        await _run(
            state_no_sim, path=str(cand), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["components_changed"] == [
        {"reference": "R1", "subckt": "OUTER/BLK", "before": "5k", "after": "6k"}
    ]


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


async def test_structural_diff_mode_says_it_did_not_compare_connections(state_no_sim, work_dir):
    ref = _write(work_dir, "ref.cir", "* d\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\n.end\n")
    cand = _write(work_dir, "cand.cir", "* d\nV1 in 0 5\nR1 in out 1k\nR2 in 0 2k\n.end\n")
    comparison = (
        await _run(
            state_no_sim, path=str(cand), reference=str(ref), compare_mode="structural_diff"
        )
    )["comparison"]
    assert comparison["equivalent"] is True
    assert any(line.startswith("connections") for line in comparison["not_compared"])


def _gates(gate: str, last: str, *, numbered: tuple[str, str] = ("N001", "N002")) -> str:
    """Seven NMOS gates; six on ``gate``, the seventh on ``last``, with two
    nets the exporter numbered."""
    lines = ["* gates", "V1 vdd 0 5", "V2 en 0 5", "V3 dE 0 1", "V4 dEf 0 2"]
    for n in range(1, 7):
        lines.append(f"M{n} o{n} {gate} 0 0 NMOS")
        lines.append(f"R{n} vdd o{n} 10k")
    lines.append(f"M7 o7 {last} 0 0 NMOS")
    lines.append("R7 vdd o7 10k")
    a, b = numbered
    lines += [f"R8 vdd {a} 1k", f"R9 {a} 0 2k", f"R10 o1 {b} 1k", f"C1 {b} 0 1n"]
    lines += [".model NMOS NMOS", ".end"]
    return "\n".join(lines) + "\n"


_MOVED = [
    *({"reference": f"M{n}", "pin": 2, "before": "en", "after": "0"} for n in range(1, 7)),
    {"reference": "M7", "pin": 2, "before": "de", "after": "def"},
]


@pytest.mark.parametrize("mode", ["sidecar", "managed"])
async def test_diff_vs_prior_reports_each_pin_moved_to_another_net(
    config, work_dir, asc_symbols, monkeypatch, mode: str
):
    before, after = _gates("en", "dE"), _gates("0", "dEf", numbered=("N002", "N001"))
    texts = [after] if mode == "sidecar" else [before, after]
    monkeypatch.setattr(vc, "_create_netlist", _exporter_writing(texts))
    state = _with_ltspice(config)
    asc = _write(work_dir, "g.asc", _RES_ASC)
    if mode == "sidecar":
        _write(work_dir, "g.net", before)
    else:
        await _run(state, path=str(asc), checks=["export"])
    data = await _run(state, path=str(asc), checks=["export"], export_to=mode)
    diff = data["export"]["diff_vs_prior"]
    assert diff["connections_changed"] == _MOVED
    assert not diff.get("components_changed")
    assert not any(line.startswith("connections") for line in diff.get("not_compared", []))


async def test_diff_vs_prior_reads_renumbered_nets_as_the_same(
    config, work_dir, asc_symbols, monkeypatch
):
    """The exporter numbers unlabelled nets anew on each export."""
    monkeypatch.setattr(
        vc, "_create_netlist", _exporter_writing([_gates("en", "dE", numbered=("N002", "N001"))])
    )
    state = _with_ltspice(config)
    asc = _write(work_dir, "g.asc", _RES_ASC)
    _write(work_dir, "g.net", _gates("en", "dE"))
    data = await _run(state, path=str(asc), checks=["export"], export_to="sidecar")
    diff = data["export"]["diff_vs_prior"]
    assert "connections_changed" not in diff
    assert not any(line.startswith("connections") for line in diff.get("not_compared", []))
