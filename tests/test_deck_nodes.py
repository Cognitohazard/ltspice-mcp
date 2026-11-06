"""Which net a node name in a deck stands for, read the same by every reader of
a deck's nodes: hierarchy discovery, the connectivity comparison and the
floating- and dangling-node checks (``lib/deck_nodes.py``).

A port named ``gnd`` is pinned against the LTspice recordings in
``tests/test_recorded_ltspice_decks.py``. No LTspice recording has a
``.global`` card or a node named ``gnd!``, so the runs here are ngspice's,
where ngspice is on PATH.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.lib.deck_nodes import node_kind
from ltspice_mcp.lib.netlist_graph import compare_graphs, flatten_graph, parse_netlist_graph
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_validator import is_ground_node, validate_netlist_bias_topology
from tests.conftest import LIVENESS_S
from tests.test_inspect_tools import _run

#: A body's ``vdd`` is the global ``vdd``.
GLOBAL = (
    "* a global rail\n.global vdd\n.subckt load out\nR1 vdd out 1k\n.ends\n"
    "V1 vdd 0 5\nX1 a load\nR2 a 0 1k\n.op\n.end\n"
)
#: The same circuit with the rail passed as a port.
PORTED = (
    "* a rail passed as a port\n.subckt load out vdd\nR1 vdd out 1k\n.ends\n"
    "V1 vdd 0 5\nX1 a vdd load\nR2 a 0 1k\n.op\n.end\n"
)
#: No ``.global``: the body's ``vdd`` is a node of its own, wired to nothing.
UNTIED = (
    "* a rail the body does not reach\n.subckt load out\nR1 vdd out 1k\n.ends\n"
    "V1 vdd 0 5\nX1 a load\nR2 a 0 1k\n.op\n.end\n"
)
#: A port named like a global: the body's ``vdd`` is still the global.
GLOBAL_PORT = (
    "* a port named like a global\n.global vdd\n.subckt leg top vdd\nR1 top vdd 1k\n.ends\n"
    "V1 vdd 0 5\nI3 0 c 1m\nX2 c mid leg\nR3 mid 0 1k\n.op\n.end\n"
)
#: ``gnd!`` is a node of its own: the ampere crosses both resistors.
GND_BANG = "* is gnd! ground\nI1 0 a 1\nR1 a gnd! 1k\nR2 gnd! 0 1k\n.op\n.end\n"


def ngspice_voltages(tmp_path: Path, deck: str, *nodes: str) -> dict[str, float]:
    """The operating point ngspice gives ``nodes`` of ``deck``."""
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not on PATH")
    control = ".control\nop\nprint " + " ".join(f"v({n})" for n in nodes) + "\n.endc\n.end\n"
    path = tmp_path / "deck.cir"
    path.write_text(deck.replace(".end\n", control), encoding="utf-8")
    done = subprocess.run(
        [executable, "-b", str(path)], capture_output=True, text=True, timeout=LIVENESS_S
    )
    found = dict(re.findall(r"^v\((\S+)\) = (\S+)$", done.stdout, re.MULTILINE))
    return {node: float(found[node]) for node in nodes}


async def hierarchy_nodes(state, work_dir: Path, deck: str) -> dict[tuple[str, ...], list]:
    """Each instance's nodes as hierarchy discovery reads them for ngspice,
    as (scope, name) pairs."""
    path = work_dir / "deck.cir"
    path.write_text(deck, encoding="utf-8")
    (result,) = await _run(
        state,
        [{"kind": "hierarchy", "path": str(path), "simulator": "ngspice", "ngbehavior": "hsa"}],
    )
    assert result["ok"], result
    return {
        tuple(row["instance"]): [(n.get("scope", []), n["name"]) for n in row["nodes"]]
        for row in result["data"]["instances"]
    }


def flat_nodes(deck: str) -> dict[str, tuple[str, ...]]:
    return {c.ref.lower(): c.nodes for c in flatten_graph(parse_netlist_graph(deck)).components}


@pytest.mark.parametrize(
    ("name", "ports", "global_names", "simulator", "kind"),
    [
        ("0", (), (), "ltspice", "ground"),
        ("GND", (), (), "ltspice", "ground"),
        ("gnd", (), (), "ngspice", "ground"),
        ("gnd", ("top", "GND"), (), "ltspice", "port"),
        ("Gnd", ("top", "gnd"), (), "ngspice", "ground"),
        ("0", ("0",), (), "ltspice", "ground"),
        ("VDD", ("vdd",), ("vdd",), "ltspice", "global"),
        ("vdd", ("Vdd",), (), "ltspice", "port"),
        ("vdd", (), ("VDD",), "ngspice", "global"),
        ("gnd!", (), (), "ltspice", "local"),
        ("mid", ("top",), ("vdd",), "ltspice", "local"),
    ],
)
def test_what_a_node_name_stands_for(
    name: str, ports: tuple[str, ...], global_names: tuple[str, ...], simulator: str, kind: str
):
    assert node_kind(name, ports=ports, global_names=global_names, simulator=simulator) == kind


def test_ngspice_wires_a_body_name_listed_by_global_to_the_global_net(tmp_path: Path):
    assert ngspice_voltages(tmp_path, GLOBAL, "a")["a"] == pytest.approx(2.5)
    assert ngspice_voltages(tmp_path, PORTED, "a")["a"] == pytest.approx(2.5)
    assert ngspice_voltages(tmp_path, UNTIED, "a")["a"] == pytest.approx(0.0)
    # Through the body's 1k to the global 5 V rail, not to the port's node.
    voltages = ngspice_voltages(tmp_path, GLOBAL_PORT, "c", "mid")
    assert voltages == {"c": pytest.approx(6.0), "mid": pytest.approx(0.0)}


async def test_every_reader_wires_a_body_name_listed_by_global_to_the_global_net(
    state_no_sim, work_dir: Path
):
    """The comparison flattens the body's resistor onto the global rail, as
    hierarchy discovery reads it, so the deck matches the one that passes the
    rail as a port and not the one whose body never reaches it; a port named
    like the global is the global too."""
    assert flat_nodes(GLOBAL)["x1.r1"] == ("vdd", "a")
    assert compare_graphs(GLOBAL, PORTED).structurally_equivalent
    assert not compare_graphs(GLOBAL, UNTIED).structurally_equivalent
    rows = await hierarchy_nodes(state_no_sim, work_dir, GLOBAL)
    assert rows[("X1", "R1")][0] == ([], "vdd")

    assert flat_nodes(GLOBAL_PORT)["x2.r1"] == ("c", "vdd")
    rows = await hierarchy_nodes(state_no_sim, work_dir, GLOBAL_PORT)
    assert rows[("X2", "R1")][1] == ([], "vdd")
    # The rail is driven, so nothing floats.
    assert validate_netlist_bias_topology(lex_deck(GLOBAL_PORT).cards) == []


def test_ngspice_reads_gnd_bang_as_a_node_of_its_own(tmp_path: Path):
    assert ngspice_voltages(tmp_path, GND_BANG, "a")["a"] == pytest.approx(2000.0)


async def test_every_reader_reads_gnd_bang_as_a_node_of_its_own(state_no_sim, work_dir: Path):
    assert flat_nodes(GND_BANG)["r1"] == ("a", "gnd!")
    to_ground = GND_BANG.replace("R1 a gnd! 1k\nR2 gnd! 0 1k", "R1 a 0 1k")
    assert not compare_graphs(GND_BANG, to_ground).structurally_equivalent
    assert not is_ground_node("gnd!")
    rows = await hierarchy_nodes(state_no_sim, work_dir, GND_BANG)
    assert rows[("R1",)][1] == ([], "gnd!")
