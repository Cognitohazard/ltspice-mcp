"""The schematic build benchmark, held to its record, and the scorer's definitions.

Every build in `tests/_benchmark_builds.py` is drawn on a blank sheet through
`edit_schematic` and scored by `tests/_build_benchmark.py`. A build must
commit with no refused op, draw the reference circuit with every part,
directive and pin, and have exactly the findings it states. Its scores and the
digest of its sheet after each call are held to
`fixtures/build_benchmark_record.json`, and its final sheet, kept under
`fixtures/ltspice_recorded/inputs/benchmark/`, byte for byte: that is what the
present engine builds, which a change to how ops are applied must build
again. A change made on purpose changes the record and the sheets in the
same commit; record them again with `LTSPICE_MCP_RECORD_BUILD_BENCHMARK=1`,
then record LTspice's export of the sheets again (`docs/TESTING.md`).

The kept sheets are inputs of the LTspice recordings, which hold what both
builds export from each. The scorer's reading of a sheet's nets is held to
that export, and the export to the reference circuit, so the offline
equivalence is evidenced on every sheet of the benchmark.

Like the other records, it reads the suite's symbols alone, so it is the same
on every machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.netlist_graph import canon_net, canon_ref, compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.sheet import Sheet, editor_symbol_resolver, read_sheet
from ltspice_mcp.lib.symbol_library import SymbolResolver
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._benchmark_builds import builds
from tests._build_benchmark import (
    CORPUS,
    Build,
    Built,
    Card,
    Circuit,
    circuit,
    coordinates_written,
    cost,
    drawing,
    equivalence,
    finding_counts,
    netlisted,
    run_build,
)
from tests._proposal_record import write_record
from tests._schematic_fixtures import TESTS
from tests.ltspice_recorder import INPUTS
from tests.test_circuit_asc import BUILD_ARCHETYPES

_RECORD = TESTS / "fixtures" / "build_benchmark_record.json"
_RECORDING = os.environ.get("LTSPICE_MCP_RECORD_BUILD_BENCHMARK") == "1"
_SYMBOLS = TESTS / "fixtures" / "symbols"
_SHEETS = INPUTS / "benchmark"

BUILDS = builds()

pytestmark = pytest.mark.usefixtures("suite_symbols_only")


def score(built: Built, working_dir: Path) -> dict[str, Any]:
    """Everything the record keeps of one build."""
    sheet = read_sheet(built.path.read_bytes(), editor_symbol_resolver(built.path))
    return {
        "cost": cost(built, working_dir),
        "digests": built.digests,
        "drawing": drawing(sheet),
        "equivalence": equivalence(sheet, circuit(built.build.circuit)),
        "findings": finding_counts(sheet, built.path),
    }


def _assert_built(built: Built, said: dict[str, Any]) -> None:
    """What every build must do, whatever the record says."""
    build = built.build
    for reply in built.replies:
        assert reply["outcome"] == "complete", reply
        assert reply["commit_state"] == "committed", reply
        assert reply["failures"] == [], reply["failures"]
        assert reply.get("error") is None, reply.get("error")
    equivalent = said["equivalence"]
    assert equivalent["equivalent"], equivalent
    assert equivalent["values_differ"] == [], equivalent
    assert equivalent["directives_missing"] == [], equivalent
    # Every pin of every part is on a wire or a label, as the editor counts
    # them and as the sheet places them.
    wiring = built.replies[-1]["wiring"]
    sheet = read_sheet(built.path.read_bytes(), editor_symbol_resolver(built.path))
    assert wiring["pins_total"] == sum(len(part.pins) for part in sheet.parts), wiring
    assert wiring["pins_wired"] + wiring["pins_label_only"] == wiring["pins_total"], wiring
    if build.expected_findings is not None:
        assert said["findings"] == dict(build.expected_findings), said["findings"]
    if build.style == "drawn":
        assert equivalent["names_missing"] == [], equivalent


def _recorded() -> dict[str, Any]:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


def _without_reply_size(said: dict[str, Any]) -> dict[str, Any]:
    """``said`` without the size of the replies, which any change of wording moves."""
    return {**said, "cost": {k: v for k, v in said["cost"].items() if k != "reply_bytes"}}


@pytest.mark.skipif(not _RECORDING, reason="set LTSPICE_MCP_RECORD_BUILD_BENCHMARK=1 to record")
async def test_record_the_builds(asc_state: SessionState, work_dir: Path):
    record = {}
    for old in _SHEETS.glob("*.asc"):
        old.unlink()
    for build in BUILDS:
        built = await run_build(asc_state, build, work_dir)
        said = score(built, work_dir)
        _assert_built(built, said)
        record[build.id] = said
        (_SHEETS / built.path.name).write_bytes(built.path.read_bytes())
    write_record(_RECORD, record)


@pytest.mark.skipif(_RECORDING, reason="recording")
@pytest.mark.parametrize("build", BUILDS, ids=lambda build: build.id)
async def test_a_build_is_what_the_record_says(
    build: Build, asc_state: SessionState, work_dir: Path
):
    built = await run_build(asc_state, build, work_dir)
    said = score(built, work_dir)
    _assert_built(built, said)
    assert _without_reply_size(said) == _without_reply_size(_recorded()[build.id])
    assert built.path.read_bytes() == (_SHEETS / built.path.name).read_bytes()


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_the_record_and_the_kept_sheets_are_of_the_builds():
    ids = sorted(build.id for build in BUILDS)
    assert sorted(_recorded()) == ids
    assert sorted(path.stem for path in _SHEETS.glob("*.asc")) == ids


def test_the_symbols_beside_the_kept_sheets_are_the_suites():
    """The recordings were made with these beside the sheets; they must be
    the symbols the builds were drawn with."""
    copies = sorted(_SHEETS.glob("*.asy"))
    assert copies
    for copy in copies:
        assert copy.read_bytes() == (_SYMBOLS / copy.name).read_bytes(), copy.name


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_a_drawn_build_reaches_fewer_pins_by_a_label_alone():
    """The measures tell a drawing from a sheet of labels."""
    record = _recorded()
    drawn = [build for build in BUILDS if build.style == "drawn"]
    assert drawn
    for build in drawn:
        labelled = record[f"{build.circuit}_labelled"]["drawing"]
        mine = record[build.id]["drawing"]
        assert mine["label_only_pins"] < labelled["label_only_pins"], build.id


def test_every_device_class_of_the_build_battery_is_built():
    """A class added to `BUILD_ARCHETYPES` that no build places fails here
    instead of leaving a silent gap."""
    placed = {
        op["symbol"]
        for build in BUILDS
        for call in build.calls
        for op in call
        if op["op"] == "add_component"
    }
    assert {symbol for symbol, _pins in BUILD_ARCHETYPES} <= placed


def test_every_circuit_has_a_labelled_build():
    names = {build.circuit for build in BUILDS}
    assert {build.circuit for build in BUILDS if build.style == "labelled"} == names


# ---------------------------------------------------------------------------
# The scorer's definitions, on sheets small enough to read
# ---------------------------------------------------------------------------


def _sheet(*lines: str) -> Sheet:
    text = "Version 4\nSHEET 1 880 680\n" + "".join(f"{line}\n" for line in lines)
    return read_sheet(text.encode("ascii"), SymbolResolver(local_dir=_SYMBOLS))


def _rc(
    v1: tuple[str, str] = ("in", "0"),
    r1: tuple[str, str] = ("in", "out"),
    r1_value: str = "1k",
    directive: str = ".ac dec 20 10 1meg",
) -> Sheet:
    """The RC low-pass as parts with a label on every pin: V1 at x=0, R1 at 200, C1 at 400."""
    return _sheet(
        "SYMBOL voltage 0 0 R0",
        "SYMATTR InstName V1",
        "SYMATTR Value AC 1",
        "SYMBOL res 200 0 R0",
        "SYMATTR InstName R1",
        f"SYMATTR Value {r1_value}",
        "SYMBOL cap 400 0 R0",
        "SYMATTR InstName C1",
        "SYMATTR Value 100n",
        f"FLAG 0 -32 {v1[0]}",
        f"FLAG 0 32 {v1[1]}",
        f"FLAG 200 -48 {r1[0]}",
        f"FLAG 200 48 {r1[1]}",
        "FLAG 400 -32 out",
        "FLAG 400 32 0",
        f"TEXT 0 120 Left 2 !{directive}",
    )


class TestEquivalence:
    def test_the_reference_drawn_with_labels_is_it(self):
        said = equivalence(_rc(), circuit("rc_lowpass"))
        assert said["equivalent"], said
        assert said["names_missing"] == said["values_differ"] == said["directives_missing"] == []

    def test_a_resistor_either_way_round_is_the_same_circuit(self):
        assert equivalence(_rc(r1=("out", "in")), circuit("rc_lowpass"))["equivalent"]

    def test_a_source_the_other_way_round_is_not(self):
        said = equivalence(_rc(v1=("0", "in")), circuit("rc_lowpass"))
        assert not said["equivalent"]
        assert said["nets_only_drawn"] and said["nets_only_in_reference"]

    def test_ground_is_only_the_label_0(self):
        said = equivalence(_rc(v1=("in", "gnd")), circuit("rc_lowpass"))
        assert said["ground_differs"]
        assert not said["equivalent"]

    def test_a_node_drawn_without_its_name_is_reported_and_still_equivalent(self):
        unnamed = _sheet(
            "SYMBOL voltage 0 0 R0",
            "SYMATTR InstName V1",
            "SYMATTR Value AC 1",
            "SYMBOL res 200 0 R0",
            "SYMATTR InstName R1",
            "SYMATTR Value 1k",
            "SYMBOL cap 400 0 R0",
            "SYMATTR InstName C1",
            "SYMATTR Value 100n",
            "WIRE 0 -32 0 -48",
            "WIRE 0 -48 200 -48",
            "WIRE 200 48 304 48",
            "WIRE 304 48 304 -32",
            "WIRE 304 -32 400 -32",
            "FLAG 0 32 0",
            "FLAG 400 32 0",
            "TEXT 0 120 Left 2 !.ac dec 20 10 1meg",
        )
        said = equivalence(unnamed, circuit("rc_lowpass"))
        assert said["equivalent"], said
        assert said["names_missing"] == ["in", "out"]

    def test_a_node_given_another_name_is_apart_from_it(self):
        said = equivalence(_rc(r1=("in", "x")), circuit("rc_lowpass"))
        assert not said["equivalent"]

    def test_a_value_and_a_directive_are_compared_and_leave_it_equivalent(self):
        said = equivalence(_rc(r1_value="2k", directive=".op"), circuit("rc_lowpass"))
        assert said["equivalent"]
        assert said["values_differ"] == ["R1: '2k' drawn, '1k' in the reference"]
        assert said["directives_missing"] == [".ac dec 20 10 1meg"]

    @pytest.mark.parametrize("nodes", [("d", "g", "s"), ("d", "g", "s", "s")])
    def test_a_three_pin_mosfet_is_a_card_of_three_nodes_or_of_four(self, nodes):
        """The export writes the source again as the bulk; a reference may or may not."""
        sheet = _sheet(
            "SYMBOL nmos 0 0 R0",
            "SYMATTR InstName M1",
            "SYMATTR Value N",
            "FLAG 0 -96 d",
            "FLAG -48 0 g",
            "FLAG 0 96 s",
        )
        reference = Circuit("m", "m", (), (Card("M1", "M", nodes, "N"),), ())
        assert equivalence(sheet, reference)["equivalent"]
        wrong = Circuit("m", "m", (), (Card("M1", "M", ("d", "g", "s", "g"), "N"),), ())
        assert not equivalence(sheet, wrong)["equivalent"]


class TestDrawing:
    def test_a_crossing_is_two_wires_through_each_other_with_nothing_there(self):
        sheet = _sheet("WIRE 0 0 64 0", "WIRE 32 -32 32 32")
        measured = drawing(sheet)
        assert (measured["crossings"], measured["four_way_junctions"]) == (1, 0)

    def test_a_label_on_a_crossing_makes_it_a_four_way_junction(self):
        sheet = _sheet("WIRE 0 0 64 0", "WIRE 32 -32 32 32", "FLAG 32 0 n")
        measured = drawing(sheet)
        assert (measured["crossings"], measured["four_way_junctions"]) == (0, 1)

    def test_four_wire_ends_on_a_point_are_a_four_way_junction(self):
        sheet = _sheet("WIRE 0 0 32 0", "WIRE 32 0 64 0", "WIRE 32 -32 32 0", "WIRE 32 0 32 32")
        assert drawing(sheet)["four_way_junctions"] == 1

    def test_a_bend_is_two_wire_ends_meeting_out_of_line(self):
        assert drawing(_sheet("WIRE 0 0 32 0", "WIRE 32 0 32 32"))["bends"] == 1
        assert drawing(_sheet("WIRE 0 0 32 0", "WIRE 32 0 64 0"))["bends"] == 0
        assert drawing(_sheet("WIRE 0 0 32 0", "WIRE 32 0 32 32", "FLAG 32 0 n"))["bends"] == 0

    def test_length_is_in_grid_steps_and_off_grid_points_are_counted(self):
        measured = drawing(_sheet("WIRE 0 0 64 0", "WIRE 64 0 64 40"))
        assert measured["wire_length"] == 6.5
        assert measured["off_grid_points"] == 1

    def test_pins_reached_only_by_a_label_are_counted_apart_from_ground(self):
        measured = drawing(_rc())
        assert (measured["label_only_pins"], measured["ground_only_pins"]) == (4, 2)


def test_coordinates_written_count_what_locates_something():
    assert coordinates_written({"op": "add_component", "x": 0, "y": 0}) == 2
    assert coordinates_written({"op": "wire_pins", "from_pin": "R1.1", "to_pin": "C1.1"}) == 0
    assert (
        coordinates_written(
            {
                "op": "wire_pins",
                "from_pin": {"x": 0, "y": 0},
                "to_pin": "C1.1",
                "waypoints": [{"x": 0, "y": 16}, {"x": 16, "y": 16}],
            }
        )
        == 6
    )
    assert coordinates_written({"op": "add_net_label", "net": "n", "pin": "R1.1"}) == 0
    assert coordinates_written({"op": "add_net_label", "net": "n", "x": 0, "y": 0}) == 2


# ---------------------------------------------------------------------------
# The scorer and LTspice's netlister
# ---------------------------------------------------------------------------

_EXPORTS = rec.cases_of("build-benchmark-export")


def test_every_kept_sheet_is_exported_in_the_recordings():
    assert sorted(case_id.removeprefix("benchmark/") for case_id in _EXPORTS) == sorted(
        build.id for build in BUILDS
    )


def _exported(build: str, case_id: str) -> dict[str, set[tuple[str, int]]]:
    """The pins LTspice put on each node of a recorded export, by node name."""
    nodes: dict[str, set[tuple[str, int]]] = {}
    for component in parse_netlist_graph(rec.recorded(build, f"{case_id}.net")).components:
        for order, node in enumerate(component.nodes, start=1):
            nodes.setdefault(node, set()).add((canon_ref(component.ref), order))
    return nodes


def _netlisted(build: str, case_id: str, folder: Path) -> list[tuple[frozenset, frozenset]]:
    staged = rec.stage_sheet(build, case_id, folder)
    return netlisted(read_sheet(staged.read_bytes(), SymbolResolver(local_dir=folder)))


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(_EXPORTS)))
def test_the_scorer_nets_a_sheet_as_ltspice_does(build: str, case_id: str, tmp_path: Path):
    """Pin for pin in SpiceOrder, and each named net under its name."""
    exported = _exported(build, case_id)
    mine = _netlisted(build, case_id, tmp_path)
    assert {pins for pins, _names in mine} == {frozenset(pins) for pins in exported.values()}
    for pins, names in mine:
        node = next(node for node, on in exported.items() if on == pins)
        assert not names or node in {canon_net(name) for name in names}, (node, names)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_three_pin_transistor_is_netted_with_the_fourth_node_ltspice_gives_it(
    build: str, tmp_path: Path
):
    """On a sheet where every pin is unconnected: a MOSFET's fourth node is a
    net of its own and a bipolar transistor's is ground. The sheet's other
    parts are left out, since a junction FET's symbol is not in the manifest."""
    transistors = {"m1", "m2", "q1", "q2"}
    exported = {
        frozenset(pins)
        for pins in _exported(build, "export/boilerplate").values()
        if {ref for ref, _order in pins} <= transistors
    }
    mine = {
        pins
        for pins, _names in _netlisted(build, "export/boilerplate", tmp_path)
        if {ref for ref, _order in pins} <= transistors
    }
    assert mine == exported
    assert frozenset({("q1", 4), ("q2", 4)}) in mine


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(_EXPORTS)))
def test_what_ltspice_exports_is_the_reference_circuit(build: str, case_id: str):
    reference = case_id.removeprefix("benchmark/").rsplit("_", 1)[0]
    comparison = compare_graphs(CORPUS / f"{reference}.cir", rec.recorded(build, f"{case_id}.net"))
    assert comparison.equivalent, comparison
