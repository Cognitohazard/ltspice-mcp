"""Which definition of a subcircuit defined twice is in force, per reader.

The question every reader here answers: which definition of a subcircuit is
in force for the simulator that reads this deck? ``spice_lex.definition_rule``
answers it from the reader's ``simulator_build.Producer``, and every reader
of a subcircuit's header or body asks it:

- LTspice 26 refuses two top-level definitions in one deck, and XVII runs the
  last (recorded: ``deck/subckt_defined_twice``,
  ``test_recorded_ltspice_decks.test_a_subcircuit_defined_twice``).
- ngspice keeps the first, in every compatibility mode, at the top level,
  inside a subcircuit and across an include (measured live below).
- Any other reader (another LTspice build, one whose build is not known yet,
  QSPICE, Xyce) has nothing recorded: no definition is read, and the reply
  says so, never another simulator's answer.

Each test runs the same two-definition deck through one path that answers
the question, for every reader.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.netlist_graph import compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.simulator import NGSPICE_NO_USER_INIT
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.spice_lex import definition_rule, lex_deck, subckt_definitions
from ltspice_mcp.lib.spice_validator import (
    validate_netlist_bias_topology,
    validate_netlist_dangling_nodes,
)
from ltspice_mcp.lib.subckt_mismatch import (
    ClosureFile,
    MismatchPlanError,
    MismatchValues,
    build_plan,
    render,
)
from ltspice_mcp.tools.inspect_tools import _subcircuit_ports
from ltspice_mcp.tools.verify import _LexedDeck, _syntax_findings
from tests._mismatch_fixtures import clone_body
from tests.conftest import LIVENESS_S
from tests.ltspice_recorder import INPUTS

#: The recorded input: ``amp`` defined with 1k, then with 2k, at the top level.
DECK_PATH = INPUTS / "deck" / "subckt_defined_twice.cir"
DECK = DECK_PATH.read_text(encoding="utf-8")

#: Who reads the deck, and the body each keeps in force: the resistance in
#: ohms, or None where no definition is in force (refused or not recorded).
READERS: dict[str, tuple[Producer, float | None]] = {
    "ltspice26": (Producer("ltspice", "LTspice 26.1.1 for Windows", None), None),
    "xvii": (Producer("ltspice", "Linear Technology Corporation LTspice XVII", None), 2000.0),
    "ngspice": (Producer("ngspice", "ngspice-42", None), 1000.0),
    "ltspice25": (Producer("ltspice", "LTspice 25.1.0 for Windows", None), None),
    "ltspice-build-unknown": (Producer("ltspice", None, None), None),
}

#: The linter's finding for each reader that keeps no definition in force.
LINT_RULE = {
    "ltspice26": ("subckt-defined-twice", "error"),
    "ltspice25": ("subckt-defined-twice-unrecorded", "warning"),
    "ltspice-build-unknown": ("subckt-defined-twice-unrecorded", "warning"),
}


def _profile(reader: str) -> SemanticProfile:
    producer, _ = READERS[reader]
    return SemanticProfile.of(producer, "kiltpsa" if producer.family == "ngspice" else None)


def _body_value(reader: str) -> float | None:
    return READERS[reader][1]


@pytest.fixture(params=list(READERS))
def reader(request: pytest.FixtureRequest) -> str:
    return request.param


def test_the_answerer(reader: str):
    cards = lex_deck(DECK).cards
    found = subckt_definitions(cards, definition_rule(READERS[reader][0]))
    if _body_value(reader) is None:
        assert found.in_force == {}
        answer = found.unsettled[((), "amp")]
        assert [cards[i].line_start for i in answer.members] == [2, 5]
        assert sorted(cards[i].line_start for i in found.unsettled_cards) == [2, 3, 4, 5, 6, 7]
        assert found.shadowed == frozenset()
        assert answer.reason is not None
        assert "Rename or delete all but one definition of amp" in answer.reason
    else:
        (index,) = found.in_force.values()
        body = cards[index + 1].body
        assert parse_spice_value(body.split()[-1]) == _body_value(reader)
        assert len(found.shadowed) == 3


def test_the_netlist_graph_and_its_comparison(reader: str, tmp_path: Path):
    """The graph reads the body in force; where none is, the instance is a
    black box and the duplicate record says why. The comparison against a
    deck holding only the 2k body agrees exactly where the reader runs it."""
    graph = parse_netlist_graph(DECK, reader=_profile(reader))
    (record,) = graph.duplicate_subckts
    if _body_value(reader) is None:
        assert "amp" not in graph.subckts
        assert record.used is None and record.reason is not None
        assert len(record.ignored) == 2
    else:
        (body,) = graph.subckts["amp"].components
        assert body.value is not None
        assert parse_spice_value(body.value) == _body_value(reader)
        assert record.used is not None and record.reason is None
    only_2k = DECK.replace(".subckt amp a b\nR1 a b 1k\n.ends\n", "", 1)
    result = compare_graphs(only_2k, DECK, reader=_profile(reader))
    assert result.equivalent is (_body_value(reader) == 2000.0)
    assert [d.reason is None for d in result.duplicate_subckts] == [
        _body_value(reader) is not None
    ]


def test_the_linter(reader: str):
    findings = lint_deck(
        DECK, DECK_PATH, READERS[reader][0].family, None, reader=READERS[reader][0]
    )
    found = [(f["rule_id"], f["severity"]) for f in findings if "subckt" in f["rule_id"]]
    assert found == ([LINT_RULE[reader]] if reader in LINT_RULE else [])
    for finding in findings:
        if finding["rule_id"] in ("subckt-defined-twice", "subckt-defined-twice-unrecorded"):
            assert finding["evidence"]["lines"] == [2, 5]
            assert "Rename or delete" in finding["evidence"]["reason"]


def test_verify_circuit_syntax_says_what_the_linter_says(reader: str):
    deck = _LexedDeck(lex_deck(DECK).cards, [])
    findings = _syntax_findings(
        DECK,
        DECK_PATH,
        deck,
        "utf-8",
        cp1252_reader=None,
        family=READERS[reader][0].family or "ltspice",
        definitions=_profile(reader).subckt_rule,
    )
    found = [(f["rule_id"], f["severity"]) for f in findings if "subckt" in f["rule_id"]]
    expected = LINT_RULE.get(reader)
    assert found == ([(expected[0].replace("-", "_"), expected[1])] if expected else [])


#: Two definitions whose headers differ: the first wires ``a`` to ``b``; the
#: second leaves port ``y`` open and its own node ``q`` on a capacitor alone.
PORTS_DECK = (
    "* two definitions of cell\n"
    ".subckt cell a b\nR1 a b 1k\n.ends cell\n"
    ".subckt CELL x y\nR1 x q 1k\nC1 q 0 1p\n.ends CELL\n"
    "X1 n1 0 cell\nV1 n1 0 1\n.op\n.end\n"
)


def test_the_validator_reads_the_body_in_force(reader: str):
    """ngspice reads the first, which leaves nothing open; XVII the second,
    whose port ``y`` is wired to nothing in its body; a reader that keeps
    neither has no body read, so nothing inside one is flagged."""
    cards = lex_deck(PORTS_DECK).cards
    dangling = validate_netlist_dangling_nodes(cards, _profile(reader))
    open_y = any("'y'" in str(issue["message"]) for issue in dangling)
    assert open_y is (reader == "xvii")
    assert all("'a'" not in str(issue["message"]) for issue in dangling)
    floating = validate_netlist_bias_topology(cards, _profile(reader))
    assert floating == [] or reader == "xvii", floating


def test_inspect_reads_the_ports_in_force(reader: str):
    expected = {"ngspice": ("a", "b"), "xvii": ("x", "y")}.get(reader)
    ports = _subcircuit_ports(lex_deck(PORTS_DECK).cards, _profile(reader))
    assert ports == ({"cell": expected} if expected else {})


def test_hierarchy_discovery(reader: str, tmp_path: Path):
    """Nested targeting reads the body in force, and refuses the deck, with
    the reason and the fix, where the reader keeps none."""
    path = tmp_path / "d.cir"
    path.write_text(DECK, encoding="utf-8")
    if _body_value(reader) is None:
        with pytest.raises(NetlistError, match="Rename or delete all but one definition of amp"):
            load_hierarchy(str(path), None, _profile(reader))
        return
    hierarchy = load_hierarchy(str(path), None, _profile(reader))
    (r1,) = [
        row
        for row in hierarchy.instances
        if tuple(s.casefold() for s in row.instance) == ("x1", "r1")
    ]
    assert r1.value.value == _body_value(reader)


#: A device subcircuit defined twice, its MOS named differently in each body.
_FET = (
    ".subckt fet d g s b\nm{n} d g s b nch w=1 l=0.15\n"
    ".model nch nmos level=54 vth0=0.7 u0=0.06\n.ends fet\n"
)
MISMATCH_DECK = (
    "* two device definitions\n"
    + _FET.format(n="0")
    + _FET.format(n="1")
    + "XN0 d g 0 0 fet W=1 L=0.15\nV1 d 0 0.9\nVg g 0 0.9\n.op\n.end\n"
)


@pytest.mark.parametrize(
    ("producer", "inner"),
    [
        (Producer("ngspice", None, None), "m0"),
        (Producer("ltspice", "Linear Technology Corporation LTspice XVII", None), "m1"),
        (Producer("ltspice", "LTspice 26.1.1 for Windows", None), None),
        (Producer("qspice", None, None), None),
        (Producer("xyce", None, None), None),
        (Producer(None, None, None), None),
    ],
)
def test_the_mismatch_plan(producer: Producer, inner: str | None):
    """The mismatch draw patches the definition in force, and refuses a
    subcircuit whose reader keeps none, saying why."""
    files = [ClosureFile(index=0, path=Path("/deck/deck.cir"), text=MISMATCH_DECK)]
    rule = definition_rule(producer)
    if inner is None:
        with pytest.raises(MismatchPlanError, match="Rename or delete all but one") as exc:
            build_plan(files, prefix="XN", subckt_rule=rule)
        assert exc.value.code == "ambiguous_subckt"
        return
    plan = build_plan(files, prefix="XN", subckt_rule=rule)
    (target,) = plan.targets
    assert target.inner_ref.casefold() == inner
    # Each run copies the definition the plan chose out of the deck again.
    out = render(files, plan, {target.ref: MismatchValues(delvto=-0.01)})
    clone = clone_body(out[0], plan.clones[0].name)
    assert f"\n{inner} " in clone.casefold()


@pytest.mark.parametrize("family", ["qspice", "xyce"])
def test_a_reader_with_no_profile_gets_its_own_gap(family: str):
    """QSPICE and Xyce are read through LTspice's profile elsewhere
    (``services.session_reader``), but which definition is in force is their
    own answer, which nothing records."""
    producer = Producer(family, None, None)
    profile = SemanticProfile.of(Producer("ltspice", None, None), None, definitions_of=producer)
    assert profile.subckt_rule == definition_rule(producer)
    (record,) = parse_netlist_graph(DECK, reader=profile).duplicate_subckts
    assert record.reason is not None and record.reason.startswith(
        f"Which definition of subcircuit amp {'QSPICE' if family == 'qspice' else 'Xyce'} reads"
    )
    findings = lint_deck(DECK, DECK_PATH, family, None, reader=producer)
    assert [f["rule_id"] for f in findings if "subckt" in f["rule_id"]] == [
        "subckt-defined-twice-unrecorded"
    ]


# ---------------------------------------------------------------------------
# Places nothing records for LTspice: a definition repeated across an include
# or within an included file, two inside one subcircuit, and three of one
# name. ngspice is measured for each.
# ---------------------------------------------------------------------------

INCLUDED = ".subckt amp a b\nR1 a b 2k\n.ends\n"
#: The deck's own 1k definition, then the include's 2k one.
ACROSS_INCLUDE = "* inc\n.subckt amp a b\nR1 a b 1k\n.ends\n.include inc.lib\nI1 0 in 1\nX1 in 0 amp\n.op\n.end\n"
#: Both definitions in one included file.
WITHIN_INCLUDE = "* inc\n.include two.lib\nI1 0 in 1\nX1 in 0 amp\n.op\n.end\n"
TWO_LIB = ".subckt amp a b\nR1 a b 1k\n.ends\n.subckt amp a b\nR1 a b 2k\n.ends\n"
#: Three definitions at the top level of the deck.
THREE = DECK.replace(".ends\nI1", ".ends\n.subckt amp a b\nR1 a b 3k\n.ends\nI1", 1)
#: The include's 2k definition read first, then the deck's own 1k one.
INCLUDE_FIRST = "* inc\n.include inc.lib\n.subckt amp a b\nR1 a b 1k\n.ends\nI1 0 in 1\nX1 in 0 amp\n.op\n.end\n"
#: Two definitions of ``amp`` inside ``outer``.
NESTED = (
    "* nested\n.subckt outer p\n.subckt amp a b\nR1 a b 1k\n.ends\n"
    ".subckt amp a b\nR1 a b 2k\n.ends\nX1 p 0 amp\n.ends\nI1 0 in 1\nX0 in outer\n.op\n.end\n"
)


@pytest.mark.parametrize("reader", ["xvii", "ltspice26"])
def test_an_ltspice_definition_across_an_include_is_not_established(reader: str, tmp_path: Path):
    (tmp_path / "inc.lib").write_text(INCLUDED, encoding="utf-8")
    path = tmp_path / "d.cir"
    path.write_text(ACROSS_INCLUDE, encoding="utf-8")
    (record,) = parse_netlist_graph(path, reader=_profile(reader)).duplicate_subckts
    assert record.reason is not None
    assert "in a file the deck includes is not recorded" in record.reason
    with pytest.raises(NetlistError, match="is not recorded"):
        load_hierarchy(str(path), None, _profile(reader))


@pytest.mark.parametrize("reader", ["xvii", "ltspice26"])
def test_two_ltspice_definitions_in_one_included_file_are_not_established(
    reader: str, tmp_path: Path
):
    """The recording has both definitions in the deck itself; two in a
    library the deck includes are another place."""
    (tmp_path / "two.lib").write_text(TWO_LIB, encoding="utf-8")
    path = tmp_path / "d.cir"
    path.write_text(WITHIN_INCLUDE, encoding="utf-8")
    (record,) = parse_netlist_graph(path, reader=_profile(reader)).duplicate_subckts
    assert record.reason is not None
    assert "in a file the deck includes is not recorded" in record.reason
    with pytest.raises(NetlistError, match="is not recorded"):
        load_hierarchy(str(path), None, _profile(reader))


@pytest.mark.parametrize("reader", ["xvii", "ltspice26"])
def test_three_ltspice_definitions_are_not_established(reader: str):
    """Two definitions are recorded; what either build does with three is not."""
    cards = lex_deck(THREE).cards
    (answer,) = subckt_definitions(cards, _profile(reader).subckt_rule).unsettled.values()
    assert answer.choice == "unestablished"
    assert answer.reason is not None and "defined 3 times" in answer.reason
    findings = lint_deck(THREE, Path("t.cir"), "ltspice", None, reader=READERS[reader][0])
    assert [f["rule_id"] for f in findings if "subckt" in f["rule_id"]] == [
        "subckt-defined-twice-unrecorded"
    ]


#: ``fet`` defined at the top level and again inside ``wrap``, where it is
#: local; the top-level call reaches the top-level one.
APART = (
    "* apart\n.subckt fet a b\nR1 a b 2k\n.ends\n"
    ".subckt wrap p\n.subckt fet a b\nR1 a b 1k\n.ends\nX1 p 0 fet\n.ends\n"
    "I1 0 in 1\nX0 in 0 fet\n.op\n.end\n"
)


@pytest.mark.parametrize("reader", list(READERS))
def test_one_name_in_two_scopes_is_two_definitions(reader: str):
    """A definition inside a subcircuit is local to it: the answerer keeps
    both, and the graph, which keys definitions by name, reads neither and
    says why, rather than flatten the top-level call to the local body."""
    cards = lex_deck(APART).cards
    found = subckt_definitions(cards, _profile(reader).subckt_rule)
    assert set(found.in_force) == {((), "fet"), ((), "wrap"), (("wrap",), "fet")}
    assert found.unsettled == {}
    graph = parse_netlist_graph(APART, reader=_profile(reader))
    assert "fet" not in graph.subckts
    (record,) = graph.duplicate_subckts
    assert record.reason is not None and "more than one scope" in record.reason


@pytest.mark.parametrize("reader", ["ngspice", "xvii"])
def test_the_mismatch_plan_refuses_what_it_cannot_order(reader: str):
    """The plan knows each file's cards but not where each file is read, so
    of definitions in two files it patches neither; nor of one name in two
    scopes."""
    include_first = [
        ClosureFile(
            index=0,
            path=Path("/deck/deck.cir"),
            text=(
                "* m\n.include lib.lib\n" + _FET.format(n="ROOT") + "XN0 d g 0 0 fet W=1 L=0.15\n"
                "V1 d 0 0.9\nVg g 0 0.9\n.op\n.end\n"
            ),
        ),
        ClosureFile(index=1, path=Path("/deck/lib.lib"), text=_FET.format(n="LIB")),
    ]
    with pytest.raises(MismatchPlanError) as exc:
        build_plan(include_first, prefix="XN", subckt_rule=_profile(reader).subckt_rule)
    assert exc.value.code == "ambiguous_subckt"
    apart = MISMATCH_DECK.replace(
        _FET.format(n="1"), ".subckt wrap p\n" + _FET.format(n="1") + ".ends wrap\n", 1
    )
    files = [ClosureFile(index=0, path=Path("/deck/deck.cir"), text=apart)]
    with pytest.raises(MismatchPlanError, match="more than one scope"):
        build_plan(files, prefix="XN", subckt_rule=_profile(reader).subckt_rule)


@pytest.mark.parametrize("reader", ["xvii", "ltspice26"])
def test_an_ltspice_definition_inside_a_subcircuit_is_not_established(reader: str):
    cards = lex_deck(NESTED).cards
    found = subckt_definitions(cards, _profile(reader).subckt_rule)
    assert list(found.unsettled) == [(("outer",), "amp")]
    (answer,) = found.unsettled.values()
    assert answer.reason is not None
    assert "inside one subcircuit is not recorded" in answer.reason
    findings = lint_deck(NESTED, Path("n.cir"), "ltspice", None, reader=READERS[reader][0])
    assert [f["rule_id"] for f in findings if "subckt" in f["rule_id"]] == [
        "subckt-defined-twice-unrecorded"
    ]


# ---------------------------------------------------------------------------
# ngspice, measured
# ---------------------------------------------------------------------------

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: ngspice's default mode (no -D), the one spicelib passes, and the HSPICE one.
MODES = ("", "kiltpsa", "hsa", "ltpsa", "spice3")


def _ngspice_v_in(deck: str, mode: str, tmp_path: Path) -> tuple[float, str]:
    """``v(in)`` as ngspice computes it for ``deck`` in ``mode``, and its output."""
    path = tmp_path / "d.cir"
    control = deck.replace(".op\n.end\n", ".control\nop\nprint v(in)\n.endc\n.end\n")
    path.write_text(control, encoding="utf-8")
    command = ["ngspice", NGSPICE_NO_USER_INIT, "-b"]
    if mode:
        command += ["-D", f"ngbehavior={mode}"]
    done = subprocess.run(
        [*command, path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=LIVENESS_S,
        check=False,
    )
    output = done.stdout + done.stderr
    found = re.search(r"v\(in\)\s*=\s*(\S+)", output)
    assert found is not None, output
    return float(found[1]), output


def _server_value(deck_path: Path, mode: str) -> float:
    """The resistance of the body ngspice's reader keeps, as the graph reads it."""
    profile = SemanticProfile.of(Producer("ngspice", None, None), mode or "kiltpsa")
    (body,) = parse_netlist_graph(deck_path, reader=profile).subckts["amp"].components
    assert body.value is not None
    return parse_spice_value(body.value)


@needs_ngspice
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case", ["deck", "include", "include-first", "within-include", "three", "nested"]
)
def test_ngspice_runs_the_body_the_server_reads(mode: str, case: str, tmp_path: Path):
    """ngspice warns "redefinition of .subckt amp, ignored" and runs the first
    body it reads in every mode: at the top level, across an include in
    either order, within an included file, of three, and inside a
    subcircuit. The server's reading for an ngspice reader is that body."""
    deck = {
        "deck": DECK,
        "include": ACROSS_INCLUDE,
        "include-first": INCLUDE_FIRST,
        "within-include": WITHIN_INCLUDE,
        "three": THREE,
        "nested": NESTED,
    }[case]
    (tmp_path / "inc.lib").write_text(INCLUDED, encoding="utf-8")
    (tmp_path / "two.lib").write_text(TWO_LIB, encoding="utf-8")
    v_in, output = _ngspice_v_in(deck, mode, tmp_path)
    assert "redefinition of .subckt amp, ignored" in output
    assert v_in == (2000.0 if case == "include-first" else 1000.0)
    path = tmp_path / "server.cir"
    path.write_text(deck, encoding="utf-8")
    if case == "nested":
        found = subckt_definitions(lex_deck(deck).cards, definition_rule(READERS["ngspice"][0]))
        cards = lex_deck(deck).cards
        index = found.in_force[(("outer",), "amp")]
        assert parse_spice_value(cards[index + 1].body.split()[-1]) == v_in
    else:
        assert _server_value(path, mode) == v_in


@pytest.mark.asyncio
async def test_verify_circuit_on_a_qspice_session_says_the_gap(state_no_sim, work_dir: Path):
    """A QSPICE session's verify_circuit reads the deck through LTspice's
    profile, but reports QSPICE's own gap for the definitions, in the syntax
    check and in the quality check's reading (no body flagged)."""
    from spicelib.simulators.qspice_simulator import Qspice

    from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

    state_no_sim.default_simulator = Qspice
    path = work_dir / "d.cir"
    path.write_text(PORTS_DECK, encoding="utf-8", newline="\n")
    args = VerifyCircuitInput.model_validate({"path": str(path), "checks": ["syntax", "quality"]})
    data = (await handle_verify_circuit(args, state_no_sim)).structured_content
    assert data is not None
    found = [f for f in data.get("findings", []) if "subckt" in f["rule_id"]]
    assert [f["rule_id"] for f in found] == ["subckt_defined_twice_unrecorded"], data
    assert "QSPICE" in found[0]["evidence"]["reason"]
    assert not [f for f in data.get("findings", []) if f["rule_id"] == "dangling_node"], data


@pytest.mark.parametrize(("reader", "line"), [("ngspice", 2), ("xvii", 5)])
def test_a_definition_nested_in_one_out_of_force_is_out_of_force_too(reader: str, line: int):
    """A definition inside one the reader keeps out of force is never read,
    even where its own name is defined once."""
    cards = lex_deck(
        "* t\n.subckt a p\nR1 p 0 1\n.ends a\n"
        ".subckt a p\n.subckt inner q\nR2 q 0 1\n.ends inner\n.ends a\n.end\n"
    ).cards
    found = subckt_definitions(cards, _profile(reader).subckt_rule)
    in_force = sorted(cards[i].line_start for i in found.in_force.values())
    assert in_force == ([2] if reader == "ngspice" else [5, 6])
    assert line in in_force
