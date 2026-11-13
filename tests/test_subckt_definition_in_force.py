"""Which definition of a subcircuit is in force when two share a name: the
first, in card order (``spice_lex.subckt_definitions``). ngspice warns
"redefinition of .subckt X, ignored" and keeps the first; LTspice is not
recorded.

The validator read a body by scope, which both definitions share, and kept
the ports of the later header: in force were ``a b``, it took ``x y``, and
flagged the first body's ports as left open and the second body's node as
dangling.
"""

from __future__ import annotations

from pathlib import Path

from ltspice_mcp.lib.netlist_graph import parse_netlist_graph
from ltspice_mcp.lib.spice_lex import lex_deck, shadowed_cards, subckt_definitions
from ltspice_mcp.lib.spice_validator import (
    validate_netlist_bias_topology,
    validate_netlist_dangling_nodes,
)
from tests.conftest import dispatch

_DECK = (
    "* two definitions of cell\n"
    ".subckt cell a b\nR1 a b 1k\n.ends cell\n"
    ".subckt CELL x y\nR1 x q 1k\nC1 q 0 1p\n.ends CELL\n"
    "X1 n1 0 cell\nV1 n1 0 1\n.op\n.end\n"
)


def test_the_first_definition_is_in_force():
    cards = lex_deck(_DECK).cards
    ((key, index),) = subckt_definitions(cards).items()
    assert key == ((), "cell")
    assert cards[index].line_start == 2
    assert sorted(cards[i].line_start for i in shadowed_cards(cards)) == [5, 6, 7, 8]


def test_the_validator_reads_the_definition_in_force():
    cards = lex_deck(_DECK).cards
    assert validate_netlist_dangling_nodes(cards) == []
    assert validate_netlist_bias_topology(cards) == []


async def test_verify_circuit_reads_the_definition_in_force(state_no_sim, work_dir: Path):
    path = work_dir / "d.cir"
    path.write_text(_DECK, encoding="utf-8")
    data = (await dispatch(state_no_sim, "verify_circuit", {"path": str(path)})).structured_content
    assert data is not None
    flagged = [
        f for f in data.get("findings", []) if f["rule_id"] in ("dangling_node", "floating_net")
    ]
    assert flagged == [], data


def test_the_netlist_graph_uses_the_same_definition(work_dir: Path):
    path = work_dir / "d.cir"
    path.write_text(_DECK, encoding="utf-8")
    graph = parse_netlist_graph(path)
    (duplicate,) = graph.duplicate_subckts
    assert duplicate.name.casefold() == "cell"
    assert list(graph.subckts["cell"].ports) == ["a", "b"]
