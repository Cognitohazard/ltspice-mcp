"""The server's reading of a schematic, held against the netlist LTspice exported from it.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: sheets
that LTspice 26 and LTspice XVII each turned into a netlist with ``-netlist``.
The server reads the same sheet with its own editor, with the symbols LTspice
resolved, and has to agree with the export on where the pins are, what is
connected, and how a card is spelled.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath

import pytest
from spicelib.editor.asc_editor import AscEditor

from ltspice_mcp.lib import symbol_geometry
from ltspice_mcp.lib.deck_staging import scan_include_references
from ltspice_mcp.lib.encoding import read_spice_text_with_encoding
from ltspice_mcp.lib.lint_rules import UNNAMED_EXPORT_WRITER, deck_generator, export_writer
from ltspice_mcp.lib.netlist_diff import parse_directive, read_deck, structural_delta
from ltspice_mcp.lib.netlist_graph import canon_ref, compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.schematic_ops import (
    collect_component_geometry,
    element_class,
    label_folded_nets,
    make_editor,
    net_partition,
    same_instance_dropped_segments,
    wire_segments_of,
)
from ltspice_mcp.lib.schematic_scene import SymbolResolver, build_scene, layout_issues
from ltspice_mcp.lib.simulator import _in_generation
from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.symbol_geometry import parse_asy_file
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR
from tests.ltspice_recorder import INPUTS

ORIENTATION = rec.cases_of("symbol-orientation") + rec.cases_of("stock-symbol-orientation")
CONNECTIVITY = rec.cases_of("wire-connectivity") + rec.cases_of("net-naming")
SAME_INSTANCE = rec.cases_of("same-instance-wire")
PLACEMENTS = ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270")


@pytest.fixture(autouse=True)
def _symbols_resolve_from_the_sheet() -> Iterator[None]:
    """Each test resolves symbols from beside its own sheet and nowhere else.

    Both symbol caches are keyed by name for the whole process, and other
    modules warm them with the stand-in library, where ``res`` has other pins
    than the build's own.
    """
    editor_cache, editor_paths = dict(AscEditor.symbol_cache), AscEditor.custom_lib_paths
    geometry_cache = dict(symbol_geometry._symbol_cache)
    AscEditor.symbol_cache = {}
    AscEditor.custom_lib_paths = []
    symbol_geometry._symbol_cache.clear()
    try:
        yield
    finally:
        AscEditor.symbol_cache = editor_cache
        AscEditor.custom_lib_paths = editor_paths
        symbol_geometry._symbol_cache.clear()
        symbol_geometry._symbol_cache.update(geometry_cache)


def exported_card(build: str, case_id: str, reference: str):
    """The exported card of the sheet's ``reference``.

    LTspice puts the element letter in front of an instance name, always for a
    subcircuit symbol and otherwise when the name does not start with it, so
    the card is found under the name itself or the name behind one letter.
    """
    cards = rec.export_instances(build, case_id)
    key = canon_ref(reference)
    matches = [name for name in cards if name == key or name[1:] == key]
    assert len(matches) == 1, f"{reference}: {sorted(cards)}"
    return cards[matches[0]]


def placed_pins(sheet: Path) -> dict[str, dict[int, tuple[int, int]]]:
    """Each component's pins by SpiceOrder, as the schematic editor places them."""
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    return {
        row["ref"]: {pin["order"]: (pin["x"], pin["y"]) for pin in row["pins"]}
        for row in collect_component_geometry(editor)
    }


# --------------------------------------------------------------------------
# Where the pins are
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", ORIENTATION)
def test_an_orientation_sheet_labels_every_point_with_its_own_coordinates(case_id: str):
    """What makes the export a statement of position: each label's name is its
    place, so the node LTspice gives a pin says where LTspice found the pin."""
    sheet = (INPUTS / rec.CASES.case(case_id).source).read_text(encoding="utf-8")
    flags = re.findall(r"(?m)^FLAG (-?\d+) (-?\d+) (\S+)$", sheet)
    assert len(flags) > 30
    assert all(name == f"x{x}y{y}" for x, y, name in flags)
    assert re.findall(r"(?m)^SYMBOL \S+ -?\d+ -?\d+ (\S+)$", sheet) == list(PLACEMENTS)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(ORIENTATION)))
class TestPinPositions:
    """All eight placements, for a symbol with four pins in general position
    and for three symbols from the build's own library."""

    def test_the_editor_places_each_pin_where_ltspice_found_it(
        self, build: str, case_id: str, tmp_path: Path
    ):
        pins = placed_pins(rec.stage_sheet(build, case_id, tmp_path))
        assert {reference[1:] for reference in pins} == set(PLACEMENTS)
        for reference, by_order in pins.items():
            nodes = exported_card(build, case_id, reference).nodes
            for order, (x, y) in sorted(by_order.items()):
                assert nodes[order - 1] == f"x{x}y{y}", f"{reference} pin {order}"

    def test_the_renderer_places_each_pin_where_ltspice_found_it(
        self, build: str, case_id: str, tmp_path: Path
    ):
        sheet = rec.stage_sheet(build, case_id, tmp_path)
        scene = build_scene(sheet, SymbolResolver(local_dir=sheet.parent))
        assert not scene.diagnostics
        assert len(scene.symbols) == len(PLACEMENTS)
        for symbol in scene.symbols:
            nodes = exported_card(build, case_id, symbol.reference).nodes
            # A placed symbol's pins are in SpiceOrder, which is node order.
            drawn = [f"x{pin.x}y{pin.y}" for pin in symbol.pins]
            assert drawn == nodes[: len(drawn)], symbol.reference


#: Stand-in symbols written to carry a stock symbol's pins exactly. The rest of
#: ``tests/fixtures/symbols`` are stand-ins with pins of their own.
MATCHES_STOCK = ("e", "g", "nmos4", "npn")


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_stand_in_symbols_that_claim_stock_pins_have_them(build: str):
    stock = rec.manifest(build)["library"]["symbols"]
    for path in sorted((FIXTURES_DIR / "symbols").glob("*.asy")):
        info = parse_asy_file(path)
        ours = [(pin.name, pin.order, pin.x, pin.y) for pin in info.pins]
        theirs = [
            (pin["name"], pin["order"], pin["x"], pin["y"]) for pin in stock[path.stem]["pins"]
        ]
        if path.stem in MATCHES_STOCK:
            assert ours == theirs, path.stem
            assert info.prefix == stock[path.stem]["prefix"], path.stem
        else:
            # Not a claim about LTspice: a test that needs stock geometry must
            # not take it from this file.
            assert ours != theirs, f"{path.stem} now matches stock; list it in MATCHES_STOCK"


# --------------------------------------------------------------------------
# What is connected
# --------------------------------------------------------------------------


def exported_groups(build: str, case_id: str) -> set[frozenset[str]]:
    """The pins LTspice put on each node, as ``REF.order`` sets."""
    nodes: dict[str, set[str]] = {}
    for component in parse_netlist_graph(rec.recorded(build, f"{case_id}.net")).components:
        for order, node in enumerate(component.nodes, start=1):
            nodes.setdefault(node, set()).add(f"{canon_ref(component.ref)}.{order}")
    return {frozenset(pins) for pins in nodes.values()}


def modelled_groups(sheet: Path) -> set[frozenset[str]]:
    """The pins the editor's net partition puts on each node."""
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    node_of = label_folded_nets(net_partition(editor))
    nodes: dict[tuple[int, int], set[str]] = {}
    for row in collect_component_geometry(editor):
        for pin in row["pins"]:
            node = node_of((pin["x"], pin["y"]))
            nodes.setdefault(node, set()).add(f"{canon_ref(row['ref'])}.{pin['order']}")
    return {frozenset(pins) for pins in nodes.values()}


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(CONNECTIVITY)))
def test_the_net_partition_connects_what_ltspice_connects(
    build: str, case_id: str, tmp_path: Path
):
    """T-junctions, a point on a wire's interior, crossings, overlapping and
    diagonal wires, pins that only touch, labels of one name on two wires."""
    sheet = rec.stage_sheet(build, case_id, tmp_path)
    assert modelled_groups(sheet) == exported_groups(build, case_id)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(CONNECTIVITY)))
def test_a_labelled_net_is_named_by_one_of_its_labels_and_ground_wins(
    build: str, case_id: str, tmp_path: Path
):
    sheet = rec.stage_sheet(build, case_id, tmp_path)
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    part = net_partition(editor)
    node_of = label_folded_nets(part)
    labels: dict[tuple[int, int], set[str]] = {}
    for coordinate, texts in part.label_texts.items():
        labels.setdefault(node_of(coordinate), set()).update(texts)
    exported = rec.export_instances(build, case_id)
    for row in collect_component_geometry(editor):
        nodes = exported[canon_ref(row["ref"])].nodes
        for pin in row["pins"]:
            on_net = labels.get(node_of((pin["x"], pin["y"])), set())
            node = nodes[pin["order"] - 1]
            if "0" in on_net:
                assert node == "0", f"{row['ref']}.{pin['order']}"
            elif on_net:
                # Of several labels, the lowest on the sheet names the net, and
                # of those the one furthest right; the order they were placed
                # in does not matter.
                node_labels = {
                    text: coordinate
                    for coordinate, texts in part.label_texts.items()
                    if node_of(coordinate) == node_of((pin["x"], pin["y"]))
                    for text in texts
                }
                winner = max(node_labels, key=lambda text: node_labels[text][::-1])
                assert node == winner, f"{row['ref']}.{pin['order']}"
            else:
                # N001 for a wired net, P001 where pins only touch, NC_01 for
                # a pin on nothing: never a name the sheet gave.
                assert re.fullmatch(r"N\d{3}|P\d{3}|NC_\d{2}", node), node


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SAME_INSTANCE)))
def test_a_wire_between_two_pins_of_one_part_is_dropped_only_when_straight(
    build: str, case_id: str, tmp_path: Path
):
    """The export keeps a part's two pins on separate nodes exactly when the
    editor reports the wire between them as one LTspice discards."""
    sheet = rec.stage_sheet(build, case_id, tmp_path)
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    part = net_partition(editor)
    dropped = same_instance_dropped_segments(part.pin_owners, wire_segments_of(editor))
    first, second = exported_card(build, case_id, "R1").nodes
    assert bool(dropped) == (first != second)
    if dropped:
        assert [(d["ref"], set(d["pins"])) for d in dropped] == [("R1", {"1", "2"})]


#: ``connectivity/bus_label``: the pins both builds put on one node, because a
#: wire labelled as a bus (``D[0:3]``) is on the node of the bus's first member
#: and so is a wire labelled ``D[0]``. The server reads no bus: a label is a
#: name like any other, so it has the two pins on two nets.
JOINED_BY_A_BUS_LABEL_IN_LTSPICE_ONLY = frozenset({"r1.1", "r2.1"})


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_bus_label_is_its_first_member_to_ltspice_and_a_name_to_the_server(
    build: str, tmp_path: Path
):
    """R1 is on the wire labelled ``D[0:3]``, R2 on one labelled ``D[0]`` and
    R3 on one labelled ``D[1]``. Both builds export R1 and R2 on ``D[0]``; the
    editor's net partition keeps them apart and knows R1's net by the label as
    it is written."""
    case_id = "connectivity/bus_label"
    cards = rec.export_instances(build, case_id)
    assert [cards[name].nodes for name in ("r1", "r2", "r3")] == [
        ["D[0]", "0"],
        ["D[0]", "0"],
        ["D[1]", "0"],
    ]
    sheet = rec.stage_sheet(build, case_id, tmp_path)
    exported, modelled = exported_groups(build, case_id), modelled_groups(sheet)
    joined = JOINED_BY_A_BUS_LABEL_IN_LTSPICE_ONLY
    assert joined in exported
    apart = {frozenset({pin}) for pin in joined}
    assert apart <= modelled
    # That one node is all the two disagree on.
    assert exported - {joined} == modelled - apart

    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    part = net_partition(editor)
    node_of = label_folded_nets(part)
    (r1,) = [row for row in collect_component_geometry(editor) if row["ref"] == "R1"]
    (on_the_bus,) = [(pin["x"], pin["y"]) for pin in r1["pins"] if pin["order"] == 1]
    names = {
        text
        for coordinate, texts in part.label_texts.items()
        if node_of(coordinate) == node_of(on_the_bus)
        for text in texts
    }
    assert names == {"D[0:3]"}


# --------------------------------------------------------------------------
# What a sheet holds besides its circuit
# --------------------------------------------------------------------------

#: The sheets of ``sheet-records`` that both builds export.
EXPORTED = ("data_flags", "bus_tap", "blank_line")

#: Sheets both builds export and the schematic editor refuses to open, each
#: with the line its refusal names and what it says of it. spicelib's reader
#: has no branch for either line (``docs/spicelib_bugs.md``, Bug 26), so the
#: editing tools and ``inspect`` cannot read a sheet LTspice reads.
EXPORTED_BY_LTSPICE_AND_REFUSED_BY_THE_EDITOR = {
    "bus_tap": (11, "is a BUSTAP record ('BUSTAP 256 96 256 112')"),
    "blank_line": (5, "is empty"),
}


def exported_bodies(build: str, case_id: str) -> list[str]:
    """The lines of a recorded export that are not comments."""
    return [
        card.body
        for card in rec.export_cards(build, case_id)
        if card.kind not in {"comment", "blank"}
    ]


async def verified(sheet: Path, state) -> dict:
    """``verify_circuit``'s quality check of ``sheet``, with no LTspice in the session."""
    from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

    result = await handle_verify_circuit(
        VerifyCircuitInput.model_validate({"path": str(sheet), "checks": ["quality"]}), state
    )
    data = result.structured_content
    assert data is not None
    return data


@pytest.mark.parametrize("build", rec.BUILDS)
class TestSheetRecords:
    """A data label (``DATAFLAG``), a bus tap (``BUSTAP``), an empty line and a
    keyword no build knows: what each build's export makes of a sheet holding
    one, and what the editor and the drawing make of the same sheet."""

    def test_neither_build_exports_a_sheet_holding_a_keyword_it_does_not_know(self, build: str):
        """What makes an export a statement that the build reads a record.
        LTspice 26 exits as if it had succeeded, having written nothing; XVII
        puts up a message box that quotes the line."""
        entry = rec.entry(build, "export/unknown_record")
        assert (entry["outputs"], entry["written"]) == ({}, [])
        if rec.generation(build) == "xvii":
            assert entry["stopped"]
            assert entry["dialog"] == (
                "LTspice XVII\nAborting:\n\n  Unknown schematic syntax:\n\n  NOSUCHRECORD 96 96"
            )
        else:
            assert (entry["exit_code"], entry["stopped"]) == (0, False)
            assert "dialog" not in entry

    @pytest.mark.parametrize("sheet", EXPORTED)
    def test_both_builds_export_a_data_label_a_bus_tap_and_an_empty_line(
        self, build: str, sheet: str
    ):
        entry = rec.entry(build, f"export/{sheet}")
        assert (entry["exit_code"], entry["stopped"]) == (0, False)
        assert "dialog" not in entry
        assert entry["written"] == [".net"]

    def test_a_data_label_puts_nothing_in_the_netlist(self, build: str):
        """The sheet's two data labels are on the wire between R1 and the port
        ``OUT``. The export is the resistor and nothing else."""
        assert exported_bodies(build, "export/data_flags") == ["R1 OUT 0 1k", ".backanno", ".end"]
        export = rec.export_text(build, "export/data_flags")
        assert "DATAFLAG" not in export
        assert "$*2" not in export

    def test_an_empty_line_changes_nothing_in_the_netlist(self, build: str):
        assert exported_bodies(build, "export/blank_line") == ["R1 a 0 1k", ".backanno", ".end"]

    def test_a_bus_tap_joins_no_nets(self, build: str):
        """R1 is on the bus (the wire labelled ``D[0:3]``), R2 on a tapped wire
        labelled ``D0`` and R3 on a tapped wire with no label. Each is on a
        node of its own: a tap connects nothing, and a wire it leads to is on
        the bus only by a label that names a member of it, which ``D0`` does
        not (``connectivity/bus_label`` has one that does)."""
        cards = rec.export_instances(build, "export/bus_tap")
        assert [cards[name].nodes for name in ("r1", "r2", "r3")] == [
            ["D[0]", "0"],
            ["D0", "0"],
            ["N001", "0"],
        ]

    async def test_an_edit_keeps_the_data_labels_where_the_sheet_has_them(
        self, build: str, state_no_sim, work_dir: Path
    ):
        """Both builds read the sheet's data labels (it exports, where a sheet
        with an unknown keyword does not), so an edit must hand them back. The
        sheet lists them after its net labels and ahead of its first symbol;
        the committed sheet is the same one but for the value that was set."""
        import hashlib

        from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic

        assert rec.entry(build, "export/data_flags")["written"] == [".net"]
        assert rec.entry(build, "export/unknown_record")["written"] == []
        sheet = rec.stage_sheet(build, "export/data_flags", work_dir)
        before = sheet.read_text(encoding="utf-8").splitlines()
        labels = [line for line in before if line.startswith("DATAFLAG")]
        assert labels == ['DATAFLAG 192 96 ""', 'DATAFLAG 240 96 "$*2"']
        result = await handle_edit_schematic(
            EditSchematicInput.model_validate(
                {
                    "target": str(sheet),
                    "expected_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
                    "ops": [{"op": "set_component_value", "reference": "R1", "value": "2k"}],
                }
            ),
            state_no_sim,
        )
        data = result.structured_content
        assert data is not None and data["commit_state"] == "committed"
        written = sheet.read_bytes()
        assert data["sha256"] == hashlib.sha256(written).hexdigest()
        assert written.decode("utf-8").splitlines() == [
            "SYMATTR Value 2k" if line == "SYMATTR Value 1k" else line for line in before
        ]
        assert b"IOPIN 288 96 Out\n" in written

    @pytest.mark.parametrize("sheet", sorted(EXPORTED_BY_LTSPICE_AND_REFUSED_BY_THE_EDITOR))
    def test_the_editor_refuses_a_sheet_both_builds_export_and_names_the_line(
        self, build: str, sheet: str, work_dir: Path
    ):
        from ltspice_mcp.errors import NetlistError

        assert rec.entry(build, f"export/{sheet}")["written"] == [".net"]
        number, what = EXPORTED_BY_LTSPICE_AND_REFUSED_BY_THE_EDITOR[sheet]
        staged = rec.stage_sheet(build, f"export/{sheet}", work_dir)
        assert staged.read_text(encoding="utf-8").splitlines()[number - 1] == (
            "" if sheet == "blank_line" else "BUSTAP 256 96 256 112"
        )
        with pytest.raises(NetlistError) as caught:
            make_editor(staged)
        message = str(caught.value)
        assert f"{sheet}.asc: line {number} {what}, which the schematic editor does not" in message

    def test_the_editor_refuses_the_keyword_neither_build_knows(self, build: str, work_dir: Path):
        """Here the editor and both builds agree: the sheet is not read."""
        from ltspice_mcp.errors import NetlistError

        assert rec.entry(build, "export/unknown_record")["written"] == []
        staged = rec.stage_sheet(build, "export/unknown_record", work_dir)
        with pytest.raises(NetlistError) as caught:
            make_editor(staged)
        assert "line 5 is a NOSUCHRECORD record ('NOSUCHRECORD 96 96')" in str(caught.value)

    async def test_verify_reports_the_bus_taps_its_drawing_reads_past(
        self, build: str, state_no_sim, work_dir: Path
    ):
        assert rec.entry(build, "export/bus_tap")["written"] == [".net"]
        data = await verified(rec.stage_sheet(build, "export/bus_tap", work_dir), state_no_sim)
        (note,) = [o for o in data["observations"] if "BUSTAP" in o]
        assert note.startswith(
            "2 BUSTAP records, the first at line 11, which the drawing does not"
        )

    def test_a_tapped_wire_ends_on_nothing_in_the_drawing_as_it_does_in_the_netlist(
        self, build: str, work_dir: Path
    ):
        """Both builds leave a tapped wire off the bus's node (R3 is on
        ``N001``), so the drawing, which reads past a tap, misses no
        connection: each tapped wire's end at its tap is a wire end on nothing
        there too, and those two are the only loose ends on the sheet."""
        assert rec.export_instances(build, "export/bus_tap")["r3"].nodes[0] == "N001"
        sheet = rec.stage_sheet(build, "export/bus_tap", work_dir)
        issues = layout_issues(build_scene(sheet, SymbolResolver(local_dir=sheet.parent)))
        loose = [issue.coords[0] for issue in issues if issue.kind == "dangling_wire_end"]
        assert sorted(loose) == [(256, 112), (400, 112)]
        assert [issue for issue in issues if issue.kind == "floating_pin"] == []

    @pytest.mark.parametrize("sheet", ["data_flags", "blank_line"])
    async def test_verify_says_nothing_of_a_line_that_puts_nothing_in_the_netlist(
        self, build: str, sheet: str, state_no_sim, work_dir: Path
    ):
        """A data label, a port's direction and an empty line have nothing for
        a check built on the drawing to miss: the export is the circuit without
        them."""
        assert rec.entry(build, f"export/{sheet}")["written"] == [".net"]
        data = await verified(rec.stage_sheet(build, f"export/{sheet}", work_dir), state_no_sim)
        assert [o for o in data["observations"] if "does not read" in o] == []

    async def test_verify_only_observes_a_keyword_that_stops_both_builds(
        self, build: str, state_no_sim, work_dir: Path
    ):
        """Neither build exports this sheet. The drawing reads past the line
        as it reads past a bus tap and reports it the same way, as an
        observation: it does not know which keywords a build knows."""
        assert rec.entry(build, "export/unknown_record")["written"] == []
        staged = rec.stage_sheet(build, "export/unknown_record", work_dir)
        data = await verified(staged, state_no_sim)
        (note,) = [o for o in data["observations"] if "NOSUCHRECORD" in o]
        assert note.startswith("line 5: a NOSUCHRECORD record, which the drawing does not read")
        assert [f for f in data["findings"] if f["severity"] == "error"] == []


# --------------------------------------------------------------------------
# How the export is spelled
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
class TestExportBoilerplate:
    """The lines LTspice adds to every export say nothing about the circuit,
    and the structural diff reads past each of them."""

    def test_the_added_lines_are_not_directives_of_the_circuit(self, build: str):
        directives = [
            parse_directive(card)
            for card in rec.export_cards(build, "export/boilerplate")
            if card.kind in {"model", "directive"}
        ]
        kept = [d for d in directives if d is not None]
        # .backanno and the four standard.* libraries are gone; what is left
        # is the parameterless default model of each device class.
        assert all(d.default_model for d in kept)
        assert {d.model for d in kept} == {"d", "npn", "pnp", "njf", "pjf", "nmos", "pmos"}

    def test_a_hand_written_deck_of_the_same_parts_shows_no_difference(self, build: str):
        cards = [
            card.body
            for card in rec.export_cards(build, "export/boilerplate")
            if card.kind == "instance"
        ]
        by_hand = "* the same parts\n" + "\n".join(cards) + "\n.end\n"
        delta = structural_delta(
            read_deck(by_hand), read_deck(rec.recorded(build, "export/boilerplate.net"))
        )
        assert not any(delta.values()), delta

    def test_the_library_lines_point_into_this_generations_library(self, build: str):
        """XVII keeps its library under Documents\\LTspiceXVII, every later
        build under AppData\\Local\\LTspice: the split the server makes when it
        decides which library directories belong to a run."""
        generation = rec.generation(build)
        libraries = [
            card.body.split(None, 1)[1]
            for card in rec.export_cards(build, "export/boilerplate")
            if card.body.startswith(".lib ")
        ]
        assert sorted(PureWindowsPath(path).name for path in libraries) == [
            "standard.bjt",
            "standard.dio",
            "standard.jft",
            "standard.mos",
        ]
        for path in libraries:
            assert _in_generation(Path(path), generation)
            assert not _in_generation(Path(path), "current" if generation == "xvii" else "xvii")
        root = rec.manifest(build)["library"]["root"]
        expected = (
            "~/Documents/LTspiceXVII/lib"
            if generation == "xvii"
            else "~/AppData/Local/LTspice/lib"
        )
        assert root == expected

    def test_the_header_names_the_build_only_from_ltspice_24_on(self, build: str):
        generator = deck_generator(rec.export_text(build, "export/boilerplate"))
        if rec.generation(build) == "xvii":
            # XVII writes the sheet's path and nothing else above the cards.
            assert generator is None
        else:
            assert generator == rec.manifest(build)["reported_build"]

    def test_every_export_shows_which_generation_wrote_it(self, build: str):
        """XVII names no generator, so its exports are told by their first line,
        the sheet's path, standing alone."""
        exports = sorted(rec.recorded(build, "export/boilerplate.net").parent.glob("*.net"))
        assert len(exports) > 10
        expected = (
            UNNAMED_EXPORT_WRITER
            if rec.generation(build) == "xvii"
            else rec.manifest(build)["reported_build"]
        )
        for path in exports:
            assert export_writer(read_spice_text_with_encoding(path)[0]) == expected, path.name
        # verify_circuit takes the export's writer for the reader of its micro signs.
        assert is_cp1252_ltspice_build(expected) == (rec.generation(build) == "xvii")

    def test_a_bipolar_transistor_is_exported_with_a_grounded_substrate(self, build: str):
        cards = rec.export_instances(build, "export/boilerplate")
        for reference in ("q1", "q2"):
            assert len(cards[reference].nodes) == 4
            assert cards[reference].nodes[3] == "0"


MICRO = "µ"


@pytest.mark.parametrize("build", rec.BUILDS)
class TestExportEncoding:
    """The bytes of an export, for one resistor valued ``1µ`` on a sheet stored
    in each encoding."""

    def value(self, build: str, sheet: str) -> str:
        return rec.export_instances(build, f"export/{sheet}")["r1"].value or ""

    def test_the_export_is_utf8_from_ltspice_24_on_and_cp1252_before(self, build: str):
        path = rec.recorded(build, "export/micro_cp1252.net")
        text, encoding = read_spice_text_with_encoding(path)
        data = path.read_bytes()
        if rec.generation(build) == "xvii":
            assert encoding == "cp1252"
            assert b"1\xb5\r\n" in data
            assert b"\n" not in data.replace(b"\r\n", b"")
        else:
            assert encoding == "utf-8"
            assert b"1\xc2\xb5\n" in data
            assert b"\r" not in data
        assert f"R1 a 0 1{MICRO}" in text

    @pytest.mark.parametrize("sheet", ["micro_cp1252", "micro_cp1252_v41", "micro_utf16le"])
    def test_a_sheet_in_cp1252_or_utf16_exports_a_micro_sign(self, build: str, sheet: str):
        assert self.value(build, sheet) == f"1{MICRO}"

    @pytest.mark.parametrize("sheet", ["micro_utf8_bom", "micro_utf16le_bom"])
    def test_a_sheet_with_a_byte_order_mark_is_not_exported(self, build: str, sheet: str):
        """Neither build takes a byte order mark for part of the file. LTspice
        26 exits as if it had succeeded, with no netlist; XVII puts up a
        message box and waits, which is why an export is run under a bound."""
        entry = rec.entry(build, f"export/{sheet}")
        assert entry["outputs"] == {}
        if rec.generation(build) == "xvii":
            assert entry["stopped"]
            assert entry["dialog"].startswith(
                "LTspice XVII\nAborting:\n\n  Unknown schematic syntax:"
            )
            assert entry["dialog"].endswith("Version 4")
        else:
            assert (entry["exit_code"], entry["stopped"]) == (0, False)

    @pytest.mark.parametrize(
        "sheet",
        ["micro_utf8_bom", "micro_utf16le_bom", "micro_cp1252", "micro_utf8", "micro_utf16le"],
    )
    async def test_verify_reports_a_byte_order_mark_exactly_where_the_build_exports_nothing(
        self, build: str, sheet: str, state_no_sim, work_dir: Path
    ):
        """The drawing reads past a mark; LTspice does not. The quality check
        reports the mark on the sheets the build refused and on no other, with
        no LTspice in the session."""
        from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

        refused = rec.entry(build, f"export/{sheet}")["outputs"] == {}
        staged = rec.stage_sheet(build, f"export/{sheet}", work_dir)
        result = await handle_verify_circuit(
            VerifyCircuitInput.model_validate({"path": str(staged), "checks": ["quality"]}),
            state_no_sim,
        )
        data = result.structured_content
        assert data is not None
        findings = [f for f in data["findings"] if f["rule_id"] == "byte_order_mark"]
        assert [(f["severity"], f["at"]) for f in findings] == (
            [("error", {"file": data["path"], "line": 1})] if refused else []
        )

    def test_the_editor_names_the_utf8_byte_order_mark_it_cannot_read(
        self, build: str, work_dir: Path
    ):
        """spicelib's own reader refuses this sheet too, but blames a missing
        Version line, which the sheet has."""
        from ltspice_mcp.errors import NetlistError

        assert rec.entry(build, "export/micro_utf8_bom")["outputs"] == {}
        staged = rec.stage_sheet(build, "export/micro_utf8_bom", work_dir)
        with pytest.raises(NetlistError, match="UTF-8 byte order mark") as caught:
            make_editor(staged)
        assert "neither LTspice 26 nor LTspice XVII" in str(caught.value)

    async def test_an_edit_writes_a_utf16_sheet_without_the_mark_neither_build_reads(
        self, build: str, state_no_sim, work_dir: Path
    ):
        """Both builds export a UTF-16 LE sheet with no byte order mark and
        neither exports one with a mark, so an edit writes UTF-16 LE without it."""
        import codecs

        from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
        from tests._asc_ops import sha_of

        assert self.value(build, "micro_utf16le") == f"1{MICRO}"
        assert rec.entry(build, "export/micro_utf16le_bom")["outputs"] == {}
        sheet = rec.stage_sheet(build, "export/micro_utf16le_bom", work_dir)
        assert sheet.read_bytes().startswith(codecs.BOM_UTF16_LE)
        result = await handle_edit_schematic(
            EditSchematicInput.model_validate(
                {
                    "target": str(sheet),
                    "expected_sha256": sha_of(sheet),
                    "ops": [
                        {"op": "set_component_value", "reference": "R1", "value": f"2{MICRO}"}
                    ],
                }
            ),
            state_no_sim,
        )
        data = result.structured_content
        assert data is not None and data["commit_state"] == "committed"
        written = sheet.read_bytes()
        assert not written.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
        text = written.decode("utf-16-le")
        assert text.startswith("Version 4\n")
        assert f"SYMATTR Value 2{MICRO}\n" in text
        assert data["sha256"] == sha_of(sheet)

    def test_ltspice_26_reads_a_utf8_sheet_as_cp1252(self, build: str):
        """Neither build reads a sheet as UTF-8. LTspice 26 decodes it as cp1252
        and writes what it saw in UTF-8, so a UTF-8 micro sign comes out as the
        two characters the value-suffix scan calls a mis-decoded micro sign."""
        if rec.generation(build) == "xvii":
            pytest.skip("XVII copies the sheet's bytes into the export unread")
        for sheet in ("micro_utf8", "micro_utf8_v41", "greek_mu_utf8"):
            sites = value_suffix_sites(rec.export_cards(build, f"export/{sheet}"))
            assert [site.misdecoded_micro for site in sites] == [True], sheet
        assert self.value(build, "micro_utf8") == "1Âµ"

    def test_ltspice_xvii_passes_a_utf8_sheets_bytes_through(self, build: str):
        if rec.generation(build) != "xvii":
            pytest.skip("LTspice 26 re-encodes what it read")
        data = rec.recorded(build, "export/micro_utf8.net").read_bytes()
        assert b"R1 a 0 1\xc2\xb5\r\n" in data

    @pytest.mark.parametrize("sheet", ["micro_cp1252", "micro_utf8"])
    async def test_verify_warns_of_a_micro_sign_the_exporting_build_misreads(
        self, build: str, sheet: str, state_no_sim, work_dir: Path
    ):
        """XVII copies a UTF-8 sheet's bytes into its export and decodes the
        export as cp1252, so the micro sign there runs as 1. verify_circuit
        warns of it from the export alone, with no XVII in the session; the
        cp1252 export, and anything LTspice 26 wrote, stay an observation."""
        from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

        deck = work_dir / f"{sheet}.net"
        deck.write_bytes(rec.recorded(build, f"export/{sheet}.net").read_bytes())
        result = await handle_verify_circuit(
            VerifyCircuitInput.model_validate({"path": str(deck), "checks": ["syntax"]}),
            state_no_sim,
        )
        data = result.structured_content
        assert data is not None
        findings = [f for f in data["findings"] if f["rule_id"] == "value_suffix_micro_sign"]
        if rec.generation(build) != "xvii":
            # LTspice 26 re-encodes what it read as cp1252: one micro sign
            # stays one, and a UTF-8 one becomes the two mis-decoded characters.
            assert [f["severity"] for f in findings] == (
                ["observation"] if sheet == "micro_cp1252" else []
            )
        elif sheet == "micro_utf8":
            (finding,) = findings
            assert finding["severity"] == "warning"
            assert finding["evidence"]["reader"] == UNNAMED_EXPORT_WRITER
        else:
            assert [f["severity"] for f in findings] == ["observation"]

    def test_the_setting_that_asks_for_u_writes_u(self, build: str):
        assert self.value(build, "micro_cp1252_as_u") == "1u"

    async def test_an_edit_writes_a_micro_sign_the_way_both_builds_read_a_sheet(
        self, build: str, state_no_sim, work_dir: Path
    ):
        """A micro sign stored as the one cp1252 byte is exported as a micro
        sign by both builds; stored as UTF-8 it is not, by either. So an edit
        that puts the first non-ASCII character into a sheet writes cp1252."""
        import hashlib
        import shutil

        from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic

        assert self.value(build, "micro_cp1252") == f"1{MICRO}"
        assert b"1\xc2\xb5" in (INPUTS / "export/micro_utf8.asc").read_bytes()
        assert self.value(build, "micro_utf8") != f"1{MICRO}" or rec.generation(build) == "xvii"

        sheet = work_dir / "directives.asc"
        shutil.copyfile(INPUTS / "export/directives.asc", sheet)
        shutil.copyfile(INPUTS / "export/res.asy", work_dir / "res.asy")
        assert sheet.read_bytes().isascii()
        result = await handle_edit_schematic(
            EditSchematicInput.model_validate(
                {
                    "target": str(sheet),
                    "expected_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
                    "ops": [
                        {"op": "set_component_value", "reference": "R1", "value": f"2{MICRO}"}
                    ],
                }
            ),
            state_no_sim,
        )
        data = result.structured_content
        assert data is not None
        assert data["outcome"] == "complete", data
        written = sheet.read_bytes()
        assert b"SYMATTR Value 2\xb5\n" in written
        assert b"\xc2\xb5" not in written
        # The token the reply hands back is the digest of those bytes.
        assert data["sha256"] == hashlib.sha256(written).hexdigest()


@pytest.mark.parametrize("build", rec.BUILDS)
class TestExportedNames:
    """How an instance is named on its card."""

    #: ``export/instance_names``'s instances as the sheet names them, the way
    #: a netlist written by hand would.
    WRITTEN = (
        "* the sheet's instances as named\n"
        "R1 NC_01 NC_02 1k\nRLoad NC_03 NC_04 2k\nr3 NC_05 NC_06 3k\n"
        "XU1 NC_07 NC_08 NC_09 NC_10 cell4\n"
        "X2 NC_11 NC_12 NC_13 NC_14 cell4\n"
        "x3 NC_15 NC_16 NC_17 NC_18 cell4\n.end\n"
    )

    @pytest.mark.parametrize("sheet", ["instance_names", "block_symbol"])
    def test_every_spelling_of_a_name_is_one_reference_to_the_comparison(
        self, build: str, sheet: str
    ):
        """LTspice 26 writes ``X§U1`` where XVII writes ``XU1``; both builds
        write ``R§Load``. The netlist comparison reads each as one name."""
        cards = rec.export_instances(build, f"export/{sheet}")
        resistors = ["r1", "r3", "rload"] if sheet == "instance_names" else []
        assert sorted(cards) == [*resistors, "xu1", "xx2", "xx3"]

    @pytest.mark.parametrize("sheet", ["instance_names", "block_symbol"])
    def test_a_subcircuit_symbol_always_gets_its_letter_and_a_resistor_only_when_it_lacks_it(
        self, build: str, sheet: str
    ):
        """A resistor named ``Load`` becomes ``R§Load`` and one named ``r3``
        stays ``r3``. A subcircuit symbol gets an ``X`` in front whatever it is
        called, ``X2`` included, on a library-style symbol and on a block
        symbol alike. LTspice 26 puts the marker after that ``X``; XVII does
        not."""
        references = [
            card.name
            for card in rec.export_cards(build, f"export/{sheet}")
            if card.kind == "instance"
        ]
        if sheet == "instance_names":
            assert references[:3] == ["R1", "R§Load", "r3"]
            references = references[3:]
        if rec.generation(build) == "xvii":
            assert references == ["XU1", "XX2", "Xx3"]
        else:
            assert references == ["X§U1", "X§X2", "X§x3"]

    def test_an_instance_named_as_written_matches_its_export(self, build: str):
        """A netlist naming the sheet's instances as the sheet does (``X2``,
        ``x3``, ``RLoad``) is the export's circuit: the ``X`` LTspice puts in
        front of a subcircuit instance pairs as a rename, not as one removed
        part and one added."""
        export = rec.export_text(build, "export/instance_names")
        result = compare_graphs(self.WRITTEN, export)
        assert (result.added, result.removed) == ([], [])
        # References are compared without the marker, as LTspice 26 names them.
        assert [(r.reference_ref, r.candidate_ref) for r in result.renamed] == [
            ("X2", "XX2"),
            ("x3", "Xx3"),
        ]
        assert result.equivalent

    def test_an_instance_named_as_written_is_a_rename_to_the_structural_diff(self, build: str):
        """The structural diff pairs the same names the equivalence mode does,
        lists them as renamed, and counts no difference for them."""
        from ltspice_mcp.tools.verify import compare_structural

        comparison, _, failure, warnings = compare_structural(
            self.WRITTEN, rec.export_text(build, "export/instance_names")
        )
        assert (failure, warnings) == (None, [])
        assert comparison is not None
        assert (comparison["components_added"], comparison["components_removed"]) == ([], [])
        marker = "X" if rec.generation(build) == "xvii" else "X§"
        assert comparison["components_renamed"] == [
            {"before": "X2", "after": f"{marker}X2"},
            {"before": "x3", "after": f"{marker}x3"},
        ]
        assert comparison["components_changed"] == []
        assert comparison["equivalent"] is True

    def test_a_block_symbol_with_no_sheet_of_its_own_cannot_be_opened(
        self, build: str, tmp_path: Path
    ):
        """LTspice netlists the block as a call to a subcircuit of the symbol's
        name. The editor's loader wants the block's own sheet and stops when
        there is none (``docs/spicelib_bugs.md``); the message names the file."""
        from ltspice_mcp.errors import SymbolResolutionError

        assert rec.entry(build, "export/block_symbol")["exit_code"] == 0
        sheet = rec.stage_sheet(build, "export/block_symbol", tmp_path)
        with pytest.raises(SymbolResolutionError, match=r"probe4\.asc not found"):
            make_editor(sheet)

    def test_the_editor_names_each_parts_element_class(self, build: str, tmp_path: Path):
        sheet = rec.stage_sheet(build, "export/instance_names", tmp_path)
        editor = make_editor(sheet)
        assert isinstance(editor, AscEditor)
        cards = rec.export_instances(build, "export/instance_names")
        for reference in editor.get_components():
            letter = element_class(editor, reference)
            matches = [
                name
                for name in cards
                if name in (reference.casefold(), letter.casefold() + reference.casefold())
            ]
            assert len(matches) == 1, reference
            assert matches[0][0] == letter.casefold()


@pytest.mark.parametrize("build", rec.BUILDS)
class TestExportedAttributes:
    """Where a symbol's attributes land on its card, read back through the
    instance view the validators and the structural diff use."""

    def cards(self, build: str):
        return rec.export_instances(build, "export/attributes")

    def test_subcircuit_parameters_follow_the_subcircuit_name(self, build: str):
        cards = self.cards(build)
        assert (cards["xx1"].model, cards["xx1"].params) == ("sub1", {"gain": "2"})
        # Value, Value2, SpiceLine, SpiceLine2, in that order.
        assert (cards["xx2"].model, cards["xx2"].params) == (
            "sub2",
            {"extra": "1", "a": "1", "b": "2"},
        )

    def test_a_params_marker_is_exported_as_written_and_read_as_a_marker(self, build: str):
        card = self.cards(build)["xx4"]
        assert (card.model, card.params) == ("sub4", {"r": "2k"})
        assert "sub4 params: r=2k" in rec.export_text(build, "export/attributes")

    def test_a_trailing_number_is_a_device_area_not_a_model(self, build: str):
        cards = self.cards(build)
        assert (cards["q1"].model, cards["q1"].value) == ("2N2222", "2")
        assert (cards["q2"].model, cards["q2"].params) == ("2N2222", {"area": "2"})
        assert (cards["d1"].model, cards["d1"].value) == ("1N4148", "3")

    def test_instance_parameters_follow_the_model_or_value(self, build: str):
        cards = self.cards(build)
        assert (cards["m1"].model, cards["m1"].params) == ("NMOS", {"l": "1u", "w": "10u"})
        assert (cards["r1"].value, cards["r1"].params) == ("1k", {"tol": "1", "pwr": "0.1"})


@pytest.mark.parametrize("build", rec.BUILDS)
def test_sheet_directives_reach_the_netlist_as_the_cards_staging_reads(build: str):
    """Directive text is exported verbatim, a two-line directive as two cards
    and a sheet comment as a comment."""
    cards = rec.export_cards(build, "export/directives")
    bodies = [card.body for card in cards if card.kind not in {"comment", "blank"}]
    assert bodies == [
        "R1 a 0 {rval}",
        ".include models.lib",
        ".lib mylib.lib",
        ".lib corners.lib tt",
        ".param rval=1k",
        ".tran 1m",
        ".model sw1 SW(Ron=1)",
        ".options reltol=1e-4",
        ".meas tran vmax MAX V(a)",
        ".backanno",
        ".end",
    ]
    assert "* a comment on the sheet" in rec.export_text(build, "export/directives")
    references = scan_include_references(cards, rec.recorded(build, "export/directives.net"))
    assert [(ref.raw_path, ref.section) for ref in references] == [
        ("models.lib", None),
        ("mylib.lib", None),
        ("corners.lib", "tt"),
    ]
