"""Which definition of a subcircuit is in force when two share a name.

ngspice keeps the first, in card order, and warns "redefinition of .subckt X,
ignored" (``netlist_graph.DuplicateSubckt``): ``spice_lex.subckt_definitions``
answers that, and the netlist graph and the inspect port table read it. What
LTspice keeps is not recorded, and the validator, which reads a deck as
LTspice does, still reads both definitions' bodies as one scope with the
later header's ports. The difference is pinned below until a recording
decides it.
"""

from __future__ import annotations

from pathlib import Path

from ltspice_mcp.lib.netlist_graph import parse_netlist_graph
from ltspice_mcp.lib.spice_lex import lex_deck, shadowed_cards, subckt_definitions
from ltspice_mcp.lib.spice_validator import validate_netlist_dangling_nodes
from ltspice_mcp.tools.inspect_tools import _subcircuit_ports

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


def test_a_definition_nested_in_one_out_of_force_is_out_of_force_too():
    cards = lex_deck(
        "* t\n.subckt a p\nR1 p 0 1\n.ends a\n"
        ".subckt a p\n.subckt inner q\nR2 q 0 1\n.ends inner\n.ends a\n.end\n"
    ).cards
    assert [cards[i].line_start for i in subckt_definitions(cards).values()] == [2]


def test_the_graph_and_the_inspect_ports_read_the_first(work_dir: Path):
    path = work_dir / "d.cir"
    path.write_text(_DECK, encoding="utf-8")
    graph = parse_netlist_graph(path)
    assert list(graph.subckts["cell"].ports) == ["a", "b"]
    assert _subcircuit_ports(lex_deck(_DECK).cards) == {"cell": ("a", "b")}


def test_the_validator_still_reads_both_definitions_until_ltspice_is_recorded():
    """Kept as it was: the later header's ports, both bodies in one scope, so
    the later header's port ``y`` reads as left open, where ngspice reads the
    first definition and finds nothing. Listed for the next
    recording run; when LTspice's choice is recorded, the validator asks
    ``subckt_definitions`` (or a simulator-keyed form of it) and this test
    changes."""
    issues = validate_netlist_dangling_nodes(lex_deck(_DECK).cards)
    assert any("'y'" in str(issue["message"]) for issue in issues), issues
