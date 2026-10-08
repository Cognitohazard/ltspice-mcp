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
from spicelib.utils.file_search import search_file_in_containers

from ltspice_mcp.lib import schematic_ops, symbol_geometry
from ltspice_mcp.lib.asc_document import parse_asc
from ltspice_mcp.lib.connectivity import (
    label_folded_nets,
    same_instance_dropped_segments,
    signature,
)
from ltspice_mcp.lib.deck_staging import scan_include_references
from ltspice_mcp.lib.encoding import read_spice_text_with_encoding
from ltspice_mcp.lib.lint_rules import UNNAMED_EXPORT_WRITER, deck_generator, export_writer
from ltspice_mcp.lib.netlist_diff import parse_directive, read_deck, structural_delta
from ltspice_mcp.lib.netlist_graph import canon_ref, compare_graphs, parse_netlist_graph
from ltspice_mcp.lib.schematic_ops import (
    collect_component_geometry,
    element_class,
    make_editor,
    net_partition,
    wire_segments_of,
)
from ltspice_mcp.lib.schematic_scene import SymbolResolver, build_scene, sheet_view
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.lib.simulator import _in_generation
from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build
from ltspice_mcp.lib.spice_lex_ops import value_suffix_sites
from ltspice_mcp.lib.symbol_geometry import parse_asy_file
from ltspice_mcp.lib.symbol_library import find_symbol
from tests import _ltspice_recorded as rec
from tests._asc_ops import apply_ops, sheet_findings
from tests.conftest import FIXTURES_DIR
from tests.ltspice_recorder import INPUTS

ORIENTATION = rec.cases_of("symbol-orientation") + rec.cases_of("stock-symbol-orientation")
CONNECTIVITY = rec.cases_of("wire-connectivity") + rec.cases_of("net-naming")
SAME_INSTANCE = rec.cases_of("same-instance-wire")
SHARED_POINT = rec.cases_of("pins-of-one-part-on-one-point")
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


def netlisted_groups(sheet: Path) -> set[frozenset[str]]:
    """The pins on each net of the sheet's signature, as ``REF.order`` sets."""
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    pins = [
        ((pin["x"], pin["y"]), (canon_ref(row["ref"]), str(pin["order"])))
        for row in collect_component_geometry(editor)
        for pin in row["pins"]
    ]
    labels = [((int(lbl.coord.X), int(lbl.coord.Y)), lbl.text) for lbl in editor.labels]
    wires = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in wire_segments_of(editor)]
    return {
        frozenset(f"{ref}.{order}" for ref, order in on_net)
        for on_net, _names in signature(pins, labels, wires)
    }


@pytest.mark.parametrize(
    ("build", "case_id"), list(rec.per_build(CONNECTIVITY + SAME_INSTANCE + SHARED_POINT))
)
def test_the_signature_is_the_circuit_ltspice_netlists(build: str, case_id: str, tmp_path: Path):
    """Every joining rule at once, with the two a partition cannot hold: the
    wire between two pins of one part that the export drops, and the pins of
    one part that share a point."""
    sheet = rec.stage_sheet(build, case_id, tmp_path)
    assert netlisted_groups(sheet) == exported_groups(build, case_id)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_parts_pins_on_one_point_are_joined_only_by_what_else_is_there(build: str):
    """What the recording says, read off the export itself."""
    nodes = {
        name: card.nodes
        for name, card in rec.export_instances(
            build, "connectivity/pins_of_one_part_on_one_point"
        ).items()
    }
    # Alone: each pin on a node of its own.
    assert nodes["r1"][0] != nodes["r1"][1]
    # A wire ending there, or a label there: both pins on it.
    assert nodes["r4"] == ["w", "w"]
    assert nodes["r5"] == ["f", "f"]
    # Another part's pin there joins the pin highest in SpiceOrder, whichever
    # order the symbol lists its pins in, and leaves the other alone.
    for stacked, other in (("r2", "r3"), ("r6", "r7")):
        assert nodes[stacked][1] == nodes[other][0]
        assert nodes[stacked][0] not in (nodes[stacked][1], *nodes[other])
    # So do two other parts' pins.
    assert nodes["r8"][1] == nodes["r9"][0] == nodes["r10"][1]
    assert nodes["r8"][0] != nodes["r8"][1]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_both_tools_call_floating_the_pins_ltspice_left_on_nothing(build: str, tmp_path: Path):
    """On the sheet of parts with two pins on one point: a pin is floating
    exactly when the export gives it a node no other pin is on."""
    case_id = "connectivity/pins_of_one_part_on_one_point"
    cards = rec.export_instances(build, case_id)
    on_node: dict[str, list[str]] = {}
    for name, card in cards.items():
        for order, node in enumerate(card.nodes, start=1):
            on_node.setdefault(node, []).append(f"{name}.{order}")
    alone = sorted(pins[0] for node, pins in on_node.items() if len(pins) == 1 and node != "0")
    assert alone == ["r1.1", "r1.2", "r2.1", "r6.1", "r8.1"]

    sheet = rec.stage_sheet(build, case_id, tmp_path)
    editor = make_editor(sheet)
    assert isinstance(editor, AscEditor)
    # The stacked symbols name each pin by its SpiceOrder.
    from_the_editor = sorted(
        f"{found.refs[0]}.{found.facts['pin']}".lower()
        for found in sheet_findings(editor)
        if found.rule == "floating_pin"
    )
    assert from_the_editor == alone

    scene = build_scene(sheet, SymbolResolver(local_dir=sheet.parent))
    from_the_checker = sorted(
        f"{found.refs[0]}.{found.facts['pin']}".lower()
        for found in findings(sheet_view(scene), ["floating_pin"])
    )
    assert from_the_checker == alone


# --------------------------------------------------------------------------
# Where a symbol is found
# --------------------------------------------------------------------------

#: Each recorded search, and whether LTspice 26 and LTspice XVII found the
#: symbol. Beside the sheet the two differ, in opposite ways.
BESIDE_THE_SHEET = {
    # the sheet says ``part``; the symbol is in ``lib`` beside the sheet
    "search/in_subfolder": {"current": False, "xvii": False},
    # the sheet says ``lib\\part``; the symbol is in ``lib`` beside the sheet
    "search/named_subfolder": {"current": True, "xvii": False},
    # the sheet says ``lib\\part``; the symbol is right beside the sheet
    "search_flat/named_folder_absent": {"current": False, "xvii": True},
}
#: The stock ``battery``, which both builds keep in ``Misc``.
IN_THE_LIBRARY = {
    "search/library_folder_bare": {"current": True, "xvii": True},  # ``battery``
    "search/library_folder_named": {"current": True, "xvii": True},  # ``Misc\\battery``
    "search/library_folder_wrong": {"current": True, "xvii": True},  # ``Wrong\\battery``
}
SEARCHES = BESIDE_THE_SHEET | IN_THE_LIBRARY


def _found_by(build: str, case_id: str) -> bool:
    """Whether the build exported the sheet, which it does only with every symbol found."""
    return bool(rec.entry(build, case_id)["outputs"])


def _staged_search(case_id: str, directory: Path) -> tuple[Path, str, list[Path]]:
    """The case's sheet with its symbol where the recording had it; the name the
    sheet uses; and, for a library case, a library that keeps ``battery`` in
    ``Misc`` as both builds' do."""
    sheet = rec.stage_sheet(rec.BUILDS[0], case_id, directory / "sheet")
    name = parse_asc(sheet.read_bytes()).symbols[0].symbol
    libraries: list[Path] = []
    if case_id in IN_THE_LIBRARY:
        library = directory / "library"
        (library / "Misc").mkdir(parents=True)
        (library / "Misc" / "battery.asy").write_text(
            "Version 4\nSYMATTR Prefix V\nPIN 0 16 NONE 0\nPINATTR PinName +\n"
            "PINATTR SpiceOrder 1\nPIN 0 96 NONE 0\nPINATTR PinName -\nPINATTR SpiceOrder 2\n",
            encoding="utf-8",
        )
        libraries.append(library)
    return sheet, name, libraries


def test_every_recorded_search_is_listed():
    recorded = {
        case_id
        for behaviour in (
            "symbol-beside-sheet",
            "symbol-named-with-a-folder-it-is-not-in",
            "symbol-in-library-folder",
        )
        for case_id in rec.cases_of(behaviour)
    }
    assert recorded == set(SEARCHES)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(list(SEARCHES))))
def test_each_build_finds_a_symbol_where_the_table_says(build: str, case_id: str):
    assert _found_by(build, case_id) == SEARCHES[case_id][rec.generation(build)]
    if not _found_by(build, case_id) and rec.generation(build) == "xvii":
        # XVII says which; LTspice 26 only ends with an error.
        assert "Couldn't find symbol(s)" in rec.entry(build, case_id)["dialog"]


@pytest.mark.parametrize("case_id", list(SEARCHES))
class TestTheServerFindsASymbolWhereEitherBuildDoes:
    """A part whose symbol is not found has no pins, so a connection to it goes
    unseen. The search therefore finds what either build would, and nothing
    that neither would."""

    def test_the_rule(self, case_id: str, tmp_path: Path):
        sheet, name, libraries = _staged_search(case_id, tmp_path)
        either = any(_found_by(build, case_id) for build in rec.BUILDS)
        assert (find_symbol(name, sheet.parent, libraries) is not None) == either

    def test_the_editors_pin_geometry(
        self, case_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        sheet, name, libraries = _staged_search(case_id, tmp_path)
        monkeypatch.setattr(AscEditor, "custom_lib_paths", [str(root) for root in libraries])
        either = any(_found_by(build, case_id) for build in rec.BUILDS)
        info = symbol_geometry.get_symbol_info(name, sheet)
        assert (info is not None) == either
        if info is not None:
            assert len(info.pins) == 2

    def test_the_renderer(self, case_id: str, tmp_path: Path):
        sheet, name, libraries = _staged_search(case_id, tmp_path)
        resolver = SymbolResolver(local_dir=sheet.parent, stock_paths=libraries)
        either = any(_found_by(build, case_id) for build in rec.BUILDS)
        assert (resolver.resolve(name) is not None) == either
        assert resolver.resolve(name) == find_symbol(name, sheet.parent, libraries)


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

    def test_a_block_symbol_with_no_sheet_of_its_own_opens_as_ltspice_reads_it(
        self, build: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """LTspice netlists the block as a call to a subcircuit of the symbol's
        name, whatever defines it. spicelib's loader wants the block's own sheet
        and stops when there is none (``docs/spicelib_bugs.md``, Bug 22); the
        server's opens the sheet with each instance's subcircuit unresolved."""
        assert rec.entry(build, "export/block_symbol")["exit_code"] == 0
        sheet = rec.stage_sheet(build, "export/block_symbol", tmp_path)
        # spicelib's own editor still refuses it: drop the workaround once it opens.
        with pytest.raises(FileNotFoundError, match=r"probe4\.asc not found"):
            AscEditor(str(sheet))
        searched: list[str] = []

        def search(filename: str, *containers: str) -> str | None:
            searched.append(filename)
            return search_file_in_containers(filename, *containers)

        monkeypatch.setattr(schematic_ops, "search_file_in_containers", search)
        editor = make_editor(sheet)
        assert sorted(editor.get_components()) == ["U1", "X2", "x3"]
        # A search walks every folder it is given, so the block the sheet
        # places three times is searched for once.
        assert searched == ["probe4.asc"]

    async def test_a_sheet_with_a_block_symbol_of_no_sheet_can_be_edited(
        self, build: str, state_no_sim, work_dir: Path
    ):
        sheet = rec.stage_sheet(build, "export/block_symbol", work_dir)
        data = await apply_ops(
            state_no_sim,
            sheet,
            [
                {"op": "move_component", "reference": "U1", "x": 96, "y": 640},
                {"op": "add_directive", "instruction": ".op", "x": 96, "y": 800},
            ],
        )
        assert data["outcome"] == "complete", data
        written = sheet.read_text(encoding="utf-8")
        assert "SYMBOL probe4 96 640 R0" in written
        assert written.count("SYMBOL probe4 ") == 3
        assert "!.op" in written

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
