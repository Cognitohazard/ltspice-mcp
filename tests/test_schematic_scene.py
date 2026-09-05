"""Scene-graph tests: .asc/.asy parsing, symbol resolution + caching, placement
under the 8-orientation matrix, missing-symbol handling, and degenerate input.

All fixtures are authored inline and written to ``tmp_path`` — nothing here
requires a stock LTspice symbol library to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from ltspice_mcp.lib.schematic_scene import (
    DrawEllipse,
    DrawLine,
    DrawPolyline,
    DrawRect,
    NetFlag,
    PlacedSymbol,
    SymbolResolver,
    Wire,
    build_scene,
    decode_text_lines,
    drawn_text_extent,
    flag_label_anchor,
    font_px,
    ground_polygon,
    parse_symbol,
    resolve_flag_placement,
    text_extent,
    wire_directions_at,
)
from ltspice_mcp.lib.symbol_geometry import (
    _apply_rotation,  # pyright: ignore[reportPrivateUsage]  # shared rotation/mirror transform
)
from tests._schematic_fixtures import (
    BOX2_ALT_ASY,
    BOX2_ASY,
    STACKED_ATTRS_ASY,
)
from tests._schematic_fixtures import (
    asc_with_box2 as _asc_with_box2,
)
from tests._schematic_fixtures import (
    asc_with_stacked_attrs as _asc_with_stacked_attrs,
)
from tests._schematic_fixtures import (
    write_file as _write,
)


class TestAsyParse:
    def test_parses_elements_pins_windows(self, tmp_path: Path) -> None:
        asy = _write(tmp_path / "box2.asy", BOX2_ASY)
        proto = parse_symbol(asy, "box2")
        assert len(proto.rects) == 1
        assert proto.rects[0] == DrawRect(0, 0, 32, 16)
        assert len(proto.lines) == 1
        assert [p.name for p in proto.pins] == ["A", "B"]
        assert proto.pins[1].x == 32 and proto.pins[1].y == 8
        # WINDOW 0 (InstName) and 3 (Value) captured.
        assert proto.window(0) is not None
        assert proto.window(3) is not None
        # bbox encloses rect + pins.
        assert (proto.bbox.x1, proto.bbox.y1, proto.bbox.x2, proto.bbox.y2) == (0, 0, 32, 16)

    def test_degenerate_lines_do_not_crash(self, tmp_path: Path) -> None:
        asy = _write(
            tmp_path / "junk.asy",
            "Version 4\nLINE Normal 0\nRECTANGLE Normal a b c d\nARC Normal 0 0 0 0 0 0 0 0\n"
            "PIN 5\nPIN 1 2 NONE 0\nPINATTR PinName Z\nPINATTR SpiceOrder 1\n",
        )
        proto = parse_symbol(asy, "junk")
        assert proto.lines == () and proto.rects == ()
        # The one well-formed pin survives; the malformed 'PIN 5' is skipped.
        assert [p.name for p in proto.pins] == ["Z"]


class TestDecodeTextLines:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("plain", ["plain"]),
            ("a\\nb", ["a", "b"]),  # backslash-n is a line break
            ("a\\nb\\nc", ["a", "b", "c"]),
            ("C:\\\\new", ["C:\\new"]),  # escaped backslash, NOT a break
            ("a\\\\nb", ["a\\nb"]),  # escaped backslash then literal 'n'
            ("trail\\n", ["trail", ""]),  # trailing break yields an empty line
            ("lone\\", ["lone\\"]),  # dangling backslash kept verbatim
            ("tab\\tx", ["tab\\tx"]),  # unknown escape passed through
            ("", [""]),
        ],
    )
    def test_decoding(self, raw: str, expected: list[str]) -> None:
        assert decode_text_lines(raw) == expected


class TestTextExtent:
    def test_anchor_variants(self) -> None:
        # size 2 -> 14px; advance 0.62; "abcd" width = round(4*14*0.62) = 35.
        start = text_extent(100, 50, ["abcd"], 2, "start")
        assert (start.x1, start.x2) == (100, 135)
        end = text_extent(100, 50, ["abcd"], 2, "end")
        assert (end.x1, end.x2) == (65, 100)
        mid = text_extent(100, 50, ["abcd"], 2, "middle")
        assert (mid.x1, mid.x2) == (83, 118)
        # Vertical: one ascent (14) above the baseline, 0.3 descent below.
        assert (start.y1, start.y2) == (36, 54)

    def test_multiline_grows_downward_and_uses_longest_line(self) -> None:
        box = text_extent(0, 0, ["ab", "abcdef"], 2, "start")
        assert box.x2 == round(6 * 14 * 0.62)  # widest line drives the width
        # second line adds one font height + gap below the first's descent
        assert box.y2 == round(14 * 0.3) + (14 + 2)


class TestFlagPlacement:
    """The verified LTspice rule: a flag glyph continues the wire's travel.

    Directions below are the direction the WIRE extends away from the flag, so
    ``{"up"}`` means the wire arrives from the north.
    """

    def test_wire_directions_at_point(self) -> None:
        wires = [Wire(100, 100, 100, 140), Wire(100, 100, 160, 100), Wire(500, 500, 520, 500)]
        assert wire_directions_at(wires, 100, 100) == {"down", "right"}
        assert wire_directions_at(wires, 100, 140) == {"up"}
        assert wire_directions_at(wires, 7, 7) == set()

    @pytest.mark.parametrize(
        ("wire_side", "orientation", "vertical"),
        [
            ("up", "down", True),  # wire from N -> glyph below, text rotated
            ("down", "up", True),  # wire from S -> glyph above, text rotated
            ("left", "right", False),  # wire from W -> glyph right, horizontal
            ("right", "left", False),  # wire from E -> glyph left, horizontal
        ],
    )
    def test_single_wire_glyph_continues_the_wire(
        self, wire_side: str, orientation: str, vertical: bool
    ) -> None:
        for is_ground in (True, False):
            assert resolve_flag_placement(is_ground, {wire_side}) == (orientation, vertical)

    def test_no_wire_uses_resting_orientation(self) -> None:
        assert resolve_flag_placement(True, set()) == ("down", False)
        assert resolve_flag_placement(False, set()) == ("up", False)

    def test_junction_picks_a_side_no_wire_occupies(self) -> None:
        # Documented fallback, not a verified rule. The reference schematic's
        # single corner sample (wire from W meeting one running S) drew text above,
        # which "first free side" reproduces for a label.
        assert resolve_flag_placement(False, {"left", "down"}) == ("up", False)
        # A ground at that same corner cannot rest downward without sitting on
        # the wire, so it takes the next free side.
        assert resolve_flag_placement(True, {"left", "down"}) == ("up", False)

    def test_flag_mid_vertical_wire_does_not_draw_along_the_wire(self) -> None:
        # Both vertical sides are occupied, so neither glyph may point up/down.
        assert resolve_flag_placement(True, {"up", "down"}) == ("left", False)
        assert resolve_flag_placement(False, {"up", "down"}) == ("right", False)

    def test_flag_mid_horizontal_wire_keeps_its_resting_side(self) -> None:
        # Nothing occupies the vertical axis, so the resting side is still free.
        assert resolve_flag_placement(True, {"left", "right"}) == ("down", False)
        assert resolve_flag_placement(False, {"left", "right"}) == ("up", False)

    def test_wire_directions_include_a_span_the_point_lies_inside(self) -> None:
        # A flag dropped mid-span has wire on both sides of it.
        horizontal = [Wire(100, 100, 200, 100)]
        assert wire_directions_at(horizontal, 150, 100) == {"left", "right"}
        vertical = [Wire(100, 100, 100, 200)]
        assert wire_directions_at(vertical, 100, 150) == {"up", "down"}
        # Endpoints still claim only the one direction they run off in.
        assert wire_directions_at(horizontal, 100, 100) == {"right"}

    @pytest.mark.parametrize(
        ("orientation", "expected"),
        [
            ("down", ((94, 108), (106, 108), (100, 118))),
            ("up", ((94, 108), (106, 108), (100, 98))),
            ("right", ((100, 102), (100, 114), (110, 108))),
            ("left", ((100, 102), (100, 114), (90, 108))),
        ],
    )
    def test_ground_polygon_geometry(self, orientation: str, expected: tuple) -> None:
        flag = NetFlag(100, 108, "0", is_ground=True, orientation=orientation)
        assert ground_polygon(flag) == expected

    @pytest.mark.parametrize(
        ("orientation", "vertical", "expected"),
        [
            # font_px(1) == 11, FLAG_LABEL_DY == 4, px // 3 == 3
            ("up", False, (100, 104, "middle", 0)),
            ("right", False, (104, 111, "start", 0)),
            ("left", False, (96, 111, "end", 0)),
            ("down", False, (100, 123, "middle", 0)),  # horizontal below (junction fallback)
            ("up", True, (103, 104, "start", -90)),  # grows upward, reads bottom-to-top
            ("down", True, (103, 112, "end", -90)),  # grows downward, same reading
        ],
    )
    def test_label_anchor(self, orientation: str, vertical: bool, expected: tuple) -> None:
        flag = NetFlag(
            100, 108, "OUT", is_ground=False, orientation=orientation, text_vertical=vertical
        )
        assert flag_label_anchor(flag) == expected

    @pytest.mark.parametrize(
        ("wire", "flag_xy", "expected"),
        [
            # Mirrors the four reference cases; wire endpoint listed away from the flag.
            ("WIRE 96 32 96 96", (96, 96), "down"),  # from N
            ("WIRE 288 160 288 96", (288, 96), "up"),  # from S
            ("WIRE 416 96 480 96", (480, 96), "right"),  # from W
            ("WIRE 736 96 672 96", (672, 96), "left"),  # from E
        ],
    )
    def test_scene_ground_orientation_matches_probe(
        self, tmp_path: Path, wire: str, flag_xy: tuple[int, int], expected: str
    ) -> None:
        x, y = flag_xy
        asc = _write(tmp_path / "s.asc", f"Version 4\nSHEET 1 880 680\n{wire}\nFLAG {x} {y} 0\n")
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert scene.flags[0].orientation == expected
        assert scene.flags[0].unconnected is False

    def test_scene_label_rotation_matches_probe(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 96 256 96 320\nFLAG 96 320 NORTH\n"
            "WIRE 416 320 480 320\nFLAG 480 320 WEST\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        by_text = {f.text: (f.orientation, f.text_vertical) for f in scene.flags}
        assert by_text["NORTH"] == ("down", True)  # vertical wire -> rotated, below
        assert by_text["WEST"] == ("right", False)  # horizontal wire -> upright, right

    def test_isolated_flag_is_marked_unconnected(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc", "Version 4\nSHEET 1 880 680\nFLAG 96 480 0\nFLAG 288 480 ISO\n"
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert all(f.unconnected for f in scene.flags)
        by_text = {f.text: f.orientation for f in scene.flags}
        assert by_text["0"] == "down"
        assert by_text["ISO"] == "up"

    def test_flag_on_a_symbol_pin_is_not_unconnected(self, tmp_path: Path) -> None:
        # A flag sitting on a pin with no wire is connected; it must not get the
        # unconnected marker (this is the common ground-on-a-pin pattern).
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL box2 100 100 R0\nSYMATTR InstName X1\nFLAG 100 108 0\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert scene.flags[0].unconnected is False

    def test_flag_mid_wire_is_not_unconnected(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nWIRE 100 100 200 100\nFLAG 150 100 MID\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert scene.flags[0].unconnected is False


class TestOrientationMatrix:
    # Hand-computed absolute position of pin B (local 32,8) for a symbol placed
    # at the origin, one per orientation. These are the ground-truth goldens.
    EXPECTED_PIN_B: ClassVar[dict[str, tuple[int, int]]] = {
        "R0": (32, 8),
        "R90": (-8, 32),
        "R180": (-32, -8),
        "R270": (8, -32),
        "M0": (-32, 8),
        "M90": (-8, -32),
        "M180": (32, -8),
        "M270": (8, 32),
    }

    @pytest.mark.parametrize("rotation", list(EXPECTED_PIN_B))
    def test_pin_b_placement(self, tmp_path: Path, rotation: str) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_box2(rotation, 0, 0))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert len(scene.symbols) == 1
        sym = scene.symbols[0]
        assert not sym.missing
        # pins sorted by SpiceOrder: [A, B]
        pin_b = sym.pins[1]
        assert (pin_b.x, pin_b.y) == self.EXPECTED_PIN_B[rotation]

    @pytest.mark.parametrize("rotation", list(EXPECTED_PIN_B))
    def test_placement_matches_shared_transform(self, tmp_path: Path, rotation: str) -> None:
        # Placement must go through symbol_geometry's transform, offset by a
        # non-zero origin — proves it is wired to the shared machinery.
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_box2(rotation, 100, 200))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        pin_b = scene.symbols[0].pins[1]
        rx, ry = _apply_rotation(32, 8, rotation)
        assert (pin_b.x, pin_b.y) == (100 + rx, 200 + ry)


class TestSymbolResolution:
    def test_local_beats_project_collision(self, tmp_path: Path) -> None:
        local = tmp_path / "local"
        project = tmp_path / "project"
        _write(local / "box2.asy", BOX2_ASY)
        _write(project / "box2.asy", BOX2_ALT_ASY)
        r = SymbolResolver(local_dir=local, project_paths=[project])
        assert r.resolve("box2") == local / "box2.asy"
        proto = r.load("box2")
        assert proto is not None
        # Local body has a rectangle; the project alt has a circle.
        assert len(proto.rects) == 1 and len(proto.circles) == 0

    def test_two_dirs_share_name_precedence_follows_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write(a / "box2.asy", BOX2_ASY)
        _write(b / "box2.asy", BOX2_ALT_ASY)
        assert SymbolResolver(local_dir=a, project_paths=[b]).resolve("box2") == a / "box2.asy"
        assert SymbolResolver(local_dir=b, project_paths=[a]).resolve("box2") == b / "box2.asy"

    def test_cache_invalidates_on_rewrite(self, tmp_path: Path) -> None:
        asy = _write(tmp_path / "box2.asy", BOX2_ASY)
        r = SymbolResolver(local_dir=tmp_path)
        first = r.load("box2")
        assert first is not None and len(first.rects) == 1
        # Rewrite the same path with different content (and different size, so
        # the content stamp changes even on a coarse-mtime filesystem).
        _write(asy, BOX2_ALT_ASY)
        second = r.load("box2")
        assert second is not None
        assert len(second.rects) == 0 and len(second.circles) == 1

    def test_caches_within_instance_and_not_across(self, tmp_path: Path) -> None:
        # A second load of the same symbol on one resolver returns the cached
        # object (identity), but a distinct resolver instance has its own cache.
        _write(tmp_path / "box2.asy", BOX2_ASY)
        r1 = SymbolResolver(local_dir=tmp_path)
        a = r1.load("box2")
        b = r1.load("box2")
        assert a is not None and a is b  # served from r1's cache
        r2 = SymbolResolver(local_dir=tmp_path, project_paths=[tmp_path / "extra"])
        assert r1.active_set != r2.active_set
        c = r2.load("box2")
        assert c is not None and c is not a  # r2 parsed independently

    def test_active_set_is_part_of_parse_key(self, tmp_path: Path) -> None:
        # The parse-cache key must carry the active set, not just (path, stamp).
        _write(tmp_path / "box2.asy", BOX2_ASY)
        r = SymbolResolver(local_dir=tmp_path)
        r.load("box2")
        (key,) = r._parse_cache.keys()  # type: ignore[attr-defined]
        _path_key, stamp_key, active_key = key
        assert active_key == r.active_set
        assert isinstance(stamp_key, tuple) and len(stamp_key) == 2

    def test_unresolved_returns_none(self, tmp_path: Path) -> None:
        r = SymbolResolver(local_dir=tmp_path)
        assert r.resolve("nope") is None
        assert r.load("nope") is None


class TestMissingSymbol:
    def test_placeholder_and_diagnostic(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nSYMBOL ghost 100 100 R0\nSYMATTR InstName U9\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert len(scene.symbols) == 1
        sym: PlacedSymbol = scene.symbols[0]
        assert sym.missing is True
        assert sym.resolved_path is None
        # A placeholder box is drawn.
        assert any(isinstance(g, DrawRect) for g in sym.graphics)
        # And exactly one diagnostic naming the symbol.
        assert len(scene.diagnostics) == 1
        assert "ghost" in scene.diagnostics[0]


class TestAscParsing:
    def test_wires_flags_directives(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(
            tmp_path / "s.asc",
            _asc_with_box2("R0", 100, 100) + "WIRE 132 108 200 108\n"
            "FLAG 200 108 OUT\n"
            "FLAG 100 108 0\n"
            "TEXT 40 300 Left 2 ;a comment\n"
            "TEXT 40 340 Left 2 !.tran 1m\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert len(scene.wires) == 1
        assert (scene.wires[0].x1, scene.wires[0].y1) == (132, 108)
        assert len(scene.flags) == 2
        gnd = [f for f in scene.flags if f.is_ground]
        net = [f for f in scene.flags if not f.is_ground]
        assert len(gnd) == 1 and len(net) == 1 and net[0].text == "OUT"
        assert len(scene.directives) == 2
        directive = next(d for d in scene.directives if d.is_directive)
        assert directive.text == ".tran 1m"
        comment = next(d for d in scene.directives if not d.is_directive)
        assert comment.text == "a comment"

    def test_attr_text_placed_at_window(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_box2("R0", 100, 100))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        texts = {t.text: (t.x, t.y) for t in scene.symbols[0].texts}
        # WINDOW 0 (InstName) local (16,-4) + origin (100,100) at R0.
        assert texts["X1"] == (116, 96)
        # WINDOW 3 (Value) local (16,24) + origin (100,100).
        assert texts["5"] == (116, 124)

    def test_sideways_symbol_attributes_do_not_overprint(self, tmp_path: Path) -> None:
        # Field regression: readers of a rendered sheet read a sideways
        # capacitor's "C1" and "100n" as the single string "1001n". The windows
        # are 16 units apart in the symbol's y; a quarter turn moved that gap
        # onto the baseline, where a four-character value is more than twice as
        # wide.
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_stacked_attrs("R90"))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        texts = {t.text: t for t in scene.symbols[0].texts}
        assert texts["C1"].rotation == -90
        assert texts["100n"].rotation == -90
        # Turned glyphs run down the sheet, so the strings occupy separate
        # columns further apart than a glyph is tall — neither can grow into the
        # other however long the value gets.
        assert texts["C1"].y == texts["100n"].y
        assert abs(texts["C1"].x - texts["100n"].x) >= font_px(texts["C1"].size)
        # Horizontally the two used to overlap by half the value's length. The
        # estimated glyph band is deliberately generous above the baseline (a
        # full font height, so the crop never clips), so allow the columns to
        # graze — just not by as much as a character.
        name, value = (drawn_text_extent(texts["C1"]), drawn_text_extent(texts["100n"]))
        overlap = min(name.x2, value.x2) - max(name.x1, value.x1)
        assert overlap < round(font_px(2) * 0.62)

    @pytest.mark.parametrize(
        ("rotation", "expected"),
        [
            # Upright either way up: the text stays horizontal.
            ("R0", (0, "start")),
            ("R180", (0, "start")),
            ("M0", (0, "start")),
            ("M180", (0, "start")),
            # Sideways: text turns with the symbol, and the run follows the
            # symbol's own +x axis so it leaves the body the way it did upright.
            ("R90", (-90, "end")),
            ("M270", (-90, "end")),
            ("R270", (-90, "start")),
            ("M90", (-90, "start")),
        ],
    )
    def test_attr_text_turns_with_the_symbol(
        self, tmp_path: Path, rotation: str, expected: tuple[int, str]
    ) -> None:
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_stacked_attrs(rotation))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        value = next(t for t in scene.symbols[0].texts if t.text == "100n")
        assert (value.rotation, value.anchor) == expected

    @pytest.mark.parametrize(
        ("align", "expected"),
        [
            # Plain token on a turned instance: the author asked for horizontal.
            ("Left", (0, "start")),
            ("Right", (0, "end")),
            # V-prefixed: turned. The run direction comes from the placement, so
            # at R90 both grow down the sheet, but the token still separates the
            # two — Left and Right must not land on the same anchor.
            ("VLeft", (-90, "end")),
            ("VRight", (-90, "start")),
            ("VCenter", (-90, "middle")),
        ],
    )
    def test_explicit_window_justification_decides(
        self, tmp_path: Path, align: str, expected: tuple[int, str]
    ) -> None:
        # A per-instance WINDOW is the author's own choice, so whether the text
        # turns is its call: LTspice writes a V-prefixed token when it means
        # sideways text and a plain one when it does not.
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY)
        asc = _write(
            tmp_path / "s.asc",
            _asc_with_stacked_attrs("R90", windows=f"WINDOW 3 48 0 {align} 2\n"),
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        value = next(t for t in scene.symbols[0].texts if t.text == "100n")
        assert (value.rotation, value.anchor) == expected

    def test_asy_default_window_may_ask_for_turned_text(self, tmp_path: Path) -> None:
        # A .asy default can be V-prefixed too, and then it means the same thing
        # it means in a .asc: turned text, on an upright instance included.
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY.replace("Left 2", "VRight 2"))
        asc = _write(tmp_path / "s.asc", _asc_with_stacked_attrs("R0"))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        value = next(t for t in scene.symbols[0].texts if t.text == "100n")
        assert (value.rotation, value.anchor) == (-90, "end")

    def test_sheet_graphics_parsed(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\n"
            "LINE Normal 0 0 40 0\n"
            "RECTANGLE Normal 0 0 10 10\n"
            "CIRCLE Normal 0 0 8 8\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        kinds = {type(g) for g in scene.sheet_graphics}
        assert DrawLine in kinds and DrawRect in kinds and DrawEllipse in kinds

    def test_degenerate_records_do_not_crash(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "bad.asc",
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL\n"  # no name/coords
            "SYMBOL box2 x y R0\n"  # non-int coords
            "WIRE 1 2\n"  # too few
            "WIRE a b c d\n"  # non-int
            "FLAG 10\n"  # too few
            "LINE Normal 0 0\n"  # too few coords
            "TEXT 1\n"  # too few
            "\n",  # blank
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        assert scene.wires == []
        # 'SYMBOL box2 x y R0' has non-int coords → dropped, not raised.
        assert scene.symbols == []


class TestArcPlacement:
    def test_arc_becomes_polyline(self, tmp_path: Path) -> None:
        asy = _write(
            tmp_path / "arcsym.asy",
            "Version 4\nSymbolType CELL\nARC Normal 0 0 40 40 40 20 20 40\n"
            "PIN 20 20 NONE 0\nPINATTR PinName C\nPINATTR SpiceOrder 1\n",
        )
        proto = parse_symbol(asy, "arcsym")
        assert len(proto.arcs) == 1
        asc = _write(tmp_path / "s.asc", "Version 4\nSHEET 1 880 680\nSYMBOL arcsym 0 0 R0\n")
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        polys = [g for g in scene.symbols[0].graphics if isinstance(g, DrawPolyline)]
        assert len(polys) == 1 and len(polys[0].pts) > 2

    def test_arc_sweep_is_displayed_ccw_major(self, tmp_path: Path) -> None:
        # ARC bbox (0,0)-(40,40): centre (20,20), r=20. Start (40,20) is angle
        # 0, end (20,40) is angle +90deg (screen, y-down). The displayed
        # counter-clockwise sweep is a DECREASING angle → the 270deg MAJOR arc
        # through the upper-left, whose midpoint is (6,6). The wrong (minor)
        # sweep would put the midpoint at the lower-right (34,34). See the stock
        # ind.asy grounding in _arc_polyline's docstring.
        asy = _write(
            tmp_path / "arcsym.asy",
            "Version 4\nSymbolType CELL\nARC Normal 0 0 40 40 40 20 20 40\n",
        )
        assert len(parse_symbol(asy, "arcsym").arcs) == 1
        asc = _write(tmp_path / "s.asc", "Version 4\nSHEET 1 880 680\nSYMBOL arcsym 0 0 R0\n")
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        poly = next(g for g in scene.symbols[0].graphics if isinstance(g, DrawPolyline))
        pts = poly.pts
        assert pts[0] == (40, 20)  # start point
        assert pts[-1] == (20, 40)  # end point
        assert pts[len(pts) // 2] == (6, 6)  # major-arc midpoint, upper-left

    def test_degenerate_arc_skipped(self, tmp_path: Path) -> None:
        asy = _write(
            tmp_path / "flat.asy",
            "Version 4\nSymbolType CELL\nARC Normal 0 10 40 10 0 10 40 10\n",
        )
        proto = parse_symbol(asy, "flat")
        assert len(proto.arcs) == 1  # parsed
        asc = _write(tmp_path / "s.asc", "Version 4\nSHEET 1 880 680\nSYMBOL flat 0 0 R0\n")
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        # Zero-height ellipse → no polyline emitted, no crash.
        assert [g for g in scene.symbols[0].graphics if isinstance(g, DrawPolyline)] == []
