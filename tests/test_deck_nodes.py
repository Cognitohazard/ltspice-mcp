"""Which net a node name in a deck stands for, read the same by every reader of
a deck's nodes: hierarchy discovery, the connectivity comparison and the
floating- and dangling-node checks (``lib/deck_nodes.py``).

A port named ``gnd``, a port named like a ``.global`` net and a node named
``gnd!`` are pinned against the LTspice recordings in
``tests/test_recorded_ltspice_decks.py``. The runs here are ngspice's, where
ngspice is on PATH, and pin where hierarchy discovery reads a deck for ngspice
otherwise than the LTspice readers do.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.lib.deck_nodes import is_ground, node_kind, settled_node_kind
from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.netlist_graph import compare_graphs, flatten_graph, parse_netlist_graph
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_validator import validate_netlist_bias_topology
from ltspice_mcp.lib.trace_names import read_name
from tests.conftest import LIVENESS_S, LTSPICE_READER
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
#: A port named like a global: the body's ``vdd`` is the port to LTspice
#: (``deck/global_port``) and the global to ngspice 42.
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
        [executable, "-b", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=LIVENESS_S,
    )
    found = dict(re.findall(r"^v\((\S+)\) = (\S+)$", done.stdout, re.MULTILINE))
    return {node: float(found[node]) for node in nodes}


async def hierarchy_nodes(
    state, work_dir: Path, deck: str, simulator: str = "ngspice"
) -> dict[tuple[str, ...], list]:
    """Each instance's nodes as hierarchy discovery reads them for
    ``simulator`` (ngspice in its hsa mode), as (scope, name) pairs."""
    path = work_dir / "deck.cir"
    path.write_text(deck, encoding="utf-8")
    query = {"kind": "hierarchy", "path": str(path), "simulator": simulator}
    if simulator == "ngspice":
        query["ngbehavior"] = "hsa"
    (result,) = await _run(state, [query])
    assert result["ok"], result
    return {
        tuple(row["instance"]): [(n.get("scope", []), n["name"]) for n in row["nodes"]]
        for row in result["data"]["instances"]
    }


#: A deck read by ngspice in its hsa mode, where its reading of every name
#: here is established.
NGSPICE_HSA = SemanticProfile("ngspice", "hsa")


def flat_nodes(deck: str, reader: SemanticProfile = LTSPICE_READER) -> dict[str, tuple[str, ...]]:
    graph = parse_netlist_graph(deck, reader=reader)
    return {c.ref.lower(): c.nodes for c in flatten_graph(graph).components}


@pytest.mark.parametrize(
    ("name", "ports", "global_names", "simulator", "mode", "kind"),
    [
        ("0", (), (), "ltspice", None, "ground"),
        ("GND", (), (), "ltspice", None, "ground"),
        ("gnd", (), (), "ngspice", "hsa", "ground"),
        ("gnd", (), (), "ngspice", "", "ground"),
        ("gnd", (), (), "ngspice", "kiltpsa", "unestablished"),
        ("gnd", ("top", "GND"), (), "ltspice", None, "port"),
        ("Gnd", ("top", "gnd"), (), "ngspice", "hsa", "ground"),
        ("Gnd", ("top", "gnd"), (), "ngspice", "kiltpsa", "unestablished"),
        ("0", ("0",), (), "ltspice", None, "ground"),
        ("0", (), (), "ngspice", "kiltpsa", "ground"),
        ("VDD", ("vdd",), ("vdd",), "ltspice", None, "port"),
        ("VDD", ("vdd",), ("vdd",), "ngspice", "hsa", "global"),
        ("vdd", ("Vdd",), (), "ltspice", None, "port"),
        ("vdd", (), ("VDD",), "ngspice", "kiltpsa", "global"),
        ("gnd!", (), (), "ltspice", None, "local"),
        ("gnd!", (), (), "ngspice", "kiltpsa", "local"),
        ("mid", ("top",), ("vdd",), "ltspice", None, "local"),
    ],
)
def test_what_a_node_name_stands_for(
    name: str,
    ports: tuple[str, ...],
    global_names: tuple[str, ...],
    simulator: str,
    mode: str | None,
    kind: str,
):
    found = node_kind(
        name, ports=ports, global_names=global_names, simulator=simulator, ngbehavior=mode
    )
    assert found == kind


def test_an_ngspice_reading_names_its_mode():
    with pytest.raises(ValueError, match="ngbehavior"):
        node_kind("gnd", simulator="ngspice")


def test_where_gnds_reading_is_unestablished_the_readers_that_cannot_refuse_read_it_as_ltspice():
    """In kiltpsa mode a ``gnd`` is not established for every ngspice version
    (hierarchy discovery refuses it); the comparison and the netlist checks
    keep the answer they gave before they were told the reader, LTspice's,
    until a recording decides."""
    for ports, kind in (((), "ground"), (("top", "gnd"), "port")):
        assert settled_node_kind(
            "gnd", ports=ports, simulator="ngspice", ngbehavior="kiltpsa"
        ) == (kind)
        assert settled_node_kind("gnd", ports=ports, simulator="ltspice") == kind


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
    rail as a port and not the one whose body never reaches it."""
    for reader in (LTSPICE_READER, NGSPICE_HSA):
        assert flat_nodes(GLOBAL, reader)["x1.r1"] == ("vdd", "a")
        assert compare_graphs(GLOBAL, PORTED, reader=reader).structurally_equivalent
        assert not compare_graphs(GLOBAL, UNTIED, reader=reader).structurally_equivalent
    for simulator in ("ngspice", "ltspice"):
        rows = await hierarchy_nodes(state_no_sim, work_dir, GLOBAL, simulator)
        assert rows[("X1", "R1")][0] == ([], "vdd"), simulator


async def test_every_reader_reads_a_port_named_like_a_global_as_the_decks_reader_does(
    state_no_sim, work_dir: Path
):
    """Inside ``.subckt leg top vdd`` with ``.global vdd``, the body's ``vdd``
    is the port to both LTspice builds (``deck/global_port``) and the global
    to ngspice 42. The comparison and hierarchy discovery, told the reader,
    read each one's net."""
    assert flat_nodes(GLOBAL_PORT)["x2.r1"] == ("c", "mid")
    assert flat_nodes(GLOBAL_PORT, NGSPICE_HSA)["x2.r1"] == ("c", "vdd")
    rows = await hierarchy_nodes(state_no_sim, work_dir, GLOBAL_PORT, "ltspice")
    assert rows[("X2", "R1")][1] == ([], "mid")
    rows = await hierarchy_nodes(state_no_sim, work_dir, GLOBAL_PORT, "ngspice")
    assert rows[("X2", "R1")][1] == ([], "vdd")
    # Through the port and R3, or the rail, c and mid reach ground: nothing floats.
    for reader in (LTSPICE_READER, NGSPICE_HSA):
        assert validate_netlist_bias_topology(lex_deck(GLOBAL_PORT).cards, reader) == []


def test_ngspice_reads_gnd_bang_as_a_node_of_its_own(tmp_path: Path):
    assert ngspice_voltages(tmp_path, GND_BANG, "a")["a"] == pytest.approx(2000.0)


async def test_every_reader_reads_gnd_bang_as_a_node_of_its_own(state_no_sim, work_dir: Path):
    assert flat_nodes(GND_BANG)["r1"] == ("a", "gnd!")
    to_ground = GND_BANG.replace("R1 a gnd! 1k\nR2 gnd! 0 1k", "R1 a 0 1k")
    for reader in (LTSPICE_READER, NGSPICE_HSA):
        assert not compare_graphs(GND_BANG, to_ground, reader=reader).structurally_equivalent
    assert not is_ground("gnd!", simulator="ltspice")
    assert not is_ground("gnd!", simulator="ngspice", ngbehavior="kiltpsa")
    rows = await hierarchy_nodes(state_no_sim, work_dir, GND_BANG)
    assert rows[("R1",)][1] == ([], "gnd!")
    # A result's node pair reads the same names as ground: V(a,gnd!) needs
    # gnd!'s own voltage, where it read as V(a).
    for xvii in (False, True):
        assert read_name(["V(a)"], "V(a,gnd!)", xvii=xvii).refused == "missing"
        pair = read_name(["V(a)", "V(gnd!)"], "V(a,gnd!)", xvii=xvii)
        assert (pair.plus, pair.minus) == ("V(a)", "V(gnd!)")
        grounded = read_name(["V(a)"], "V(a,GND)", xvii=xvii)
        assert (grounded.refused, grounded.plus, grounded.minus) == (None, "V(a)", None)


async def test_every_reader_reads_a_port_named_gnd_as_the_decks_reader_does(
    state_no_sim, work_dir: Path
):
    """Inside ``.subckt ported top gnd`` the body's ``gnd`` is the port to
    LTspice and ground to ngspice 42 in its hsa mode. The comparison and
    hierarchy discovery, told the reader, read each one's net; in kiltpsa
    mode the comparison keeps LTspice's reading, which hierarchy refuses."""
    deck = (
        "* a port named gnd\n.subckt ported top gnd\nR1 top gnd 1k\n.ends\n"
        "I3 0 c 1\nX2 c mid ported\nR3 mid 0 1k\n.op\n.end\n"
    )
    assert flat_nodes(deck)["x2.r1"] == ("c", "mid")
    assert flat_nodes(deck, NGSPICE_HSA)["x2.r1"] == ("c", "0")
    kiltpsa = SemanticProfile("ngspice", "kiltpsa")
    assert flat_nodes(deck, kiltpsa)["x2.r1"] == ("c", "mid")
    rows = await hierarchy_nodes(state_no_sim, work_dir, deck)
    assert rows[("X2", "R1")][1] == ([], "0")
