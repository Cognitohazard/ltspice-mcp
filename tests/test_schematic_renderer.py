"""Renderer tests: the SVG is parsed back with ElementTree and element
positions/text are asserted against hand-computed geometry. Also covers crop
bounds, determinism, and missing-symbol styling.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ltspice_mcp.lib.schematic_renderer import render_svg, render_to_file
from ltspice_mcp.lib.schematic_scene import SymbolResolver, build_scene
from ltspice_mcp.render import main as render_main
from tests._schematic_fixtures import (
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

# Reused golden schematic: box2 at (100,100), a wire to a net label, a ground.
GOLDEN_ASC = (
    _asc_with_box2("R0", 100, 100) + "WIRE 132 108 200 108\nFLAG 200 108 OUT\nFLAG 100 108 0\n"
)


def _local(name: str) -> str:
    """Strip the SVG namespace from a qualified tag."""
    return name.rsplit("}", 1)[-1]


def _elements(svg: str, tag: str) -> list[ET.Element]:
    root = ET.fromstring(svg)
    return [e for e in root.iter() if _local(e.tag) == tag]


def _build(tmp_path: Path, asc_text: str) -> str:
    _write(tmp_path / "box2.asy", BOX2_ASY)
    asc = _write(tmp_path / "s.asc", asc_text)
    scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
    return render_svg(scene)


class TestWellFormed:
    def test_parses_as_xml(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        root = ET.fromstring(svg)  # raises on malformed XML
        assert _local(root.tag) == "svg"


class TestSemanticGoldens:
    def test_symbol_rect_position(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        sym_rects = [r for r in _elements(svg, "rect") if r.get("class") == "sym"]
        assert len(sym_rects) == 1
        r = sym_rects[0]
        # box2 rect local (0,0)-(32,16) at origin (100,100), R0.
        assert (r.get("x"), r.get("y"), r.get("width"), r.get("height")) == (
            "100",
            "100",
            "32",
            "16",
        )

    def test_wire_position(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        wires = [ln for ln in _elements(svg, "line") if ln.get("class") == "wire"]
        assert len(wires) == 1
        w = wires[0]
        assert (w.get("x1"), w.get("y1"), w.get("x2"), w.get("y2")) == (
            "132",
            "108",
            "200",
            "108",
        )

    def test_ground_glyph_position(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        polys = [p for p in _elements(svg, "polygon") if p.get("class") == "gnd"]
        assert len(polys) == 1
        # Inverted triangle, top edge through the flag connection point (100,108).
        assert polys[0].get("points") == "94,108 106,108 100,118"

    def test_net_label_text(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        flags = [t for t in _elements(svg, "text") if t.get("class") == "t-flag"]
        assert [t.text for t in flags] == ["OUT"]

    def test_attribute_text_positions(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        attrs = {
            t.text: (t.get("x"), t.get("y"))
            for t in _elements(svg, "text")
            if t.get("class") == "t-attr"
        }
        assert attrs["X1"] == ("116", "96")
        assert attrs["5"] == ("116", "124")

    def test_upright_attribute_text_carries_no_transform(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        attrs = [t for t in _elements(svg, "text") if t.get("class") == "t-attr"]
        assert [t.get("transform") for t in attrs] == [None, None]

    def test_sideways_symbol_turns_its_attribute_text(self, tmp_path: Path) -> None:
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_stacked_attrs("R90"))
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        attrs = {
            t.text: (t.get("x"), t.get("y"), t.get("text-anchor"), t.get("transform"))
            for t in _elements(svg, "text")
            if t.get("class") == "t-attr"
        }
        # Windows (24,-8) and (24,8) turned about the origin, each a quarter turn
        # anticlockwise with the run heading down the sheet.
        assert attrs["C1"] == ("8", "24", "end", "rotate(-90,8,24)")
        assert attrs["100n"] == ("-8", "24", "end", "rotate(-90,-8,24)")

    def test_turned_attribute_text_is_inside_the_crop(self, tmp_path: Path) -> None:
        # The run heads down the sheet, so the crop has to reach past its end
        # rather than past a width to the right.
        _write(tmp_path / "stacked.asy", STACKED_ATTRS_ASY)
        asc = _write(tmp_path / "s.asc", _asc_with_stacked_attrs("R90"))
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        content = scene.content_bbox()
        assert content is not None
        # "100n" from y=24, run = round(4*14*0.62) = 35 downward.
        assert content.y2 >= 24 + 35


class TestFlagRendering:
    """Verified against an LTspice screenshot of the reference flag schematic."""

    @pytest.mark.parametrize(
        ("wire", "flag_xy", "points"),
        [
            ("WIRE 96 32 96 96", (96, 96), "90,96 102,96 96,106"),  # from N -> apex down
            ("WIRE 288 160 288 96", (288, 96), "282,96 294,96 288,86"),  # from S -> apex up
            ("WIRE 416 96 480 96", (480, 96), "480,90 480,102 490,96"),  # from W -> apex right
            ("WIRE 736 96 672 96", (672, 96), "672,90 672,102 662,96"),  # from E -> apex left
        ],
    )
    def test_ground_apex_follows_wire_travel(
        self, tmp_path: Path, wire: str, flag_xy: tuple[int, int], points: str
    ) -> None:
        x, y = flag_xy
        asc = _write(tmp_path / "s.asc", f"Version 4\nSHEET 1 880 680\n{wire}\nFLAG {x} {y} 0\n")
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        poly = next(p for p in _elements(svg, "polygon") if p.get("class") == "gnd")
        assert poly.get("points") == points

    @pytest.mark.parametrize(
        ("wire", "flag", "expect"),
        [
            # vertical wire -> rotated text reading bottom-to-top, away from the wire
            ("WIRE 96 256 96 320", (96, 320, "NORTH"), ("99", "324", "end", "rotate(-90,99,324)")),
            (
                "WIRE 288 384 288 320",
                (288, 320, "SOUTH"),
                ("291", "316", "start", "rotate(-90,291,316)"),
            ),
            # horizontal wire -> upright text on the far side
            ("WIRE 416 320 480 320", (480, 320, "WEST"), ("484", "323", "start", None)),
            ("WIRE 736 320 672 320", (672, 320, "EAST"), ("668", "323", "end", None)),
        ],
    )
    def test_net_label_placement_and_rotation(
        self, tmp_path: Path, wire: str, flag: tuple[int, int, str], expect: tuple
    ) -> None:
        x, y, text = flag
        asc = _write(
            tmp_path / "s.asc", f"Version 4\nSHEET 1 880 680\n{wire}\nFLAG {x} {y} {text}\n"
        )
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        label = next(t for t in _elements(svg, "text") if t.get("class") == "t-flag")
        exp_x, exp_y, exp_anchor, exp_transform = expect
        assert label.text == text
        assert (label.get("x"), label.get("y"), label.get("text-anchor")) == (
            exp_x,
            exp_y,
            exp_anchor,
        )
        assert label.get("transform") == exp_transform

    def test_no_connection_dot_on_net_labels(self, tmp_path: Path) -> None:
        # LTspice draws none; we used to draw a blue dot.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nWIRE 416 320 480 320\nFLAG 480 320 WEST\n",
        )
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        assert [c for c in _elements(svg, "circle") if c.get("class") == "dot"] == []

    def test_unconnected_flags_get_the_marker_square(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nFLAG 96 480 0\nFLAG 288 480 ISOLATED\n",
        )
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        marks = [r for r in _elements(svg, "rect") if r.get("class") == "unconnected"]
        assert len(marks) == 2
        assert {(m.get("x"), m.get("y")) for m in marks} == {("94", "478"), ("286", "478")}
        # The isolated ground still rests apex-down.
        poly = next(p for p in _elements(svg, "polygon") if p.get("class") == "gnd")
        assert poly.get("points") == "90,480 102,480 96,490"

    def test_connected_flag_has_no_marker(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc", "Version 4\nSHEET 1 880 680\nWIRE 96 32 96 96\nFLAG 96 96 0\n"
        )
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        assert [r for r in _elements(svg, "rect") if r.get("class") == "unconnected"] == []

    def test_corner_junction_label_is_horizontal_above(self, tmp_path: Path) -> None:
        # Documented fallback, from the reference schematic's single corner sample.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\n"
            "WIRE 448 480 512 480\nWIRE 512 480 512 544\nFLAG 512 480 CORNER\n",
        )
        svg = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        label = next(t for t in _elements(svg, "text") if t.get("class") == "t-flag")
        assert label.get("transform") is None
        assert (label.get("x"), label.get("y"), label.get("text-anchor")) == (
            "512",
            "476",
            "middle",
        )


class TestMultilineDirective:
    def test_lines_become_tspans_with_spacing(self, tmp_path: Path) -> None:
        # LTspice stores an embedded newline as the two characters backslash-n.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nTEXT 40 300 Left 2 !.tran 1m\\n.save V(out)\\n.end\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        svg = render_svg(scene)
        texts = [t for t in _elements(svg, "text") if t.get("class") == "t-directive"]
        assert len(texts) == 1  # one <text> holding all lines
        spans = [e for e in texts[0] if _local(e.tag) == "tspan"]
        assert [s.text for s in spans] == [".tran 1m", ".save V(out)", ".end"]
        # First line sits on the anchor baseline; each later line drops px+gap.
        assert [s.get("dy") for s in spans] == ["0", "16", "16"]
        assert {s.get("x") for s in spans} == {"40"}
        # The literal escape must not survive into the output.
        assert "\\n" not in svg

    def test_escaped_backslash_is_not_a_line_break(self, tmp_path: Path) -> None:
        # A Windows path is stored with doubled backslashes; splitting naively
        # on backslash-n would cut "C:\\new_models" in half.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\n"
            'TEXT 40 300 Left 2 !.include "C:\\\\new_models\\\\x.lib"\n',
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        svg = render_svg(scene)
        texts = [t for t in _elements(svg, "text") if t.get("class") == "t-directive"]
        spans = [e for e in texts[0] if _local(e.tag) == "tspan"]
        assert len(spans) == 1
        assert spans[0].text == '.include "C:\\new_models\\x.lib"'


class TestReversedRect:
    def test_reversed_corner_rect_renders_positive_extent(self, tmp_path: Path) -> None:
        # A sheet RECTANGLE with corners listed high→low must still render: a
        # naive x2-x1 would be negative and SVG would draw nothing.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nRECTANGLE Normal 140 120 100 80\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        svg = render_svg(scene)
        rects = [r for r in _elements(svg, "rect") if r.get("class") == "sym"]
        assert len(rects) == 1
        r = rects[0]
        assert (r.get("x"), r.get("y"), r.get("width"), r.get("height")) == (
            "100",
            "80",
            "40",
            "40",
        )


class TestCropBounds:
    def test_viewbox_is_content_plus_margin(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", GOLDEN_ASC)
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        content = scene.content_bbox()
        assert content is not None
        # Hand-computed, including estimated glyph extents (advance 0.62,
        # ascent 1.0, descent 0.3 of the font height):
        #   left  94 = ground glyph at x=100 minus its 6-wide half-width
        #   top   82 = "X1" baseline 96 minus one 14px ascent
        #   right 224 = "OUT" sits right of its wire (anchor 204), width
        #               3*11*0.62 = 20
        #   bot  128 = "5" baseline 124 plus a 14*0.3 = 4 descent
        assert (content.x1, content.y1, content.x2, content.y2) == (94, 82, 224, 128)
        svg = render_svg(scene, margin=16)
        root = ET.fromstring(svg)
        # margin 16 → 78 66, 162 x 78
        assert root.get("viewBox") == "78 66 162 78"

    def test_long_attribute_value_is_not_cropped(self, tmp_path: Path) -> None:
        # Field regression: a trailing value such as "100n" ran past the right
        # edge and was clipped, because the crop ignored glyph extents.
        _write(tmp_path / "box2.asy", BOX2_ASY)
        long_value = "100nF_TRAILING"
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nSYMBOL box2 100 100 R0\n"
            f"SYMATTR InstName X1\nSYMATTR Value {long_value}\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        content = scene.content_bbox()
        assert content is not None
        # The value is centred on window 3 at x=116; its estimated half-width is
        # len*14*0.62/2, which reaches well past the symbol body's right edge 132.
        value_right = 116 + (round(len(long_value) * 14 * 0.62)) // 2
        assert content.x2 >= value_right
        assert content.x2 > 132  # strictly beyond the pure-geometry extent
        # And the rendered viewBox actually contains that right edge.
        root = ET.fromstring(render_svg(scene, margin=16))
        vb = [int(v) for v in (root.get("viewBox") or "").split()]
        assert vb[0] + vb[2] >= value_right

    def test_rotated_net_label_is_inside_the_crop(self, tmp_path: Path) -> None:
        # Drives the rotated branch of flag_label_bbox, where a sign error would
        # silently clip the text: the wire arrives from the north, so "NORTH"
        # hangs BELOW the flag at y=320 and the crop must reach past its end.
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nWIRE 96 256 96 320\nFLAG 96 320 NORTH\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        content = scene.content_bbox()
        assert content is not None
        # anchor (99,324) end-anchored; run = round(5*11*0.62) = 34 downward
        # -> y 324..358; glyph band x 99-11 .. 99+3.
        assert (content.x1, content.y1, content.x2, content.y2) == (88, 256, 102, 358)
        root = ET.fromstring(render_svg(scene, margin=16))
        assert root.get("viewBox") == "72 240 46 134"

    def test_empty_scene_has_valid_viewbox(self, tmp_path: Path) -> None:
        asc = _write(tmp_path / "empty.asc", "Version 4\nSHEET 1 880 680\n")
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        svg = render_svg(scene)
        root = ET.fromstring(svg)
        vb = root.get("viewBox")
        assert vb is not None and len(vb.split()) == 4


class TestDeterminism:
    def test_identical_scene_identical_svg(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", GOLDEN_ASC)
        s1 = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        s2 = render_svg(build_scene(asc, SymbolResolver(local_dir=tmp_path)))
        assert s1 == s2

    def test_no_generated_at_marker(self, tmp_path: Path) -> None:
        svg = _build(tmp_path, GOLDEN_ASC)
        # No comment, and no wall-clock/provenance markers that would break
        # byte-for-byte reproducibility across runs.
        assert "<!--" not in svg
        low = svg.lower()
        for marker in ("generated", "created", "timestamp", "datetime"):
            assert marker not in low


class TestMissingSymbolRender:
    def test_placeholder_uses_missing_class(self, tmp_path: Path) -> None:
        asc = _write(
            tmp_path / "s.asc",
            "Version 4\nSHEET 1 880 680\nSYMBOL ghost 100 100 R0\nSYMATTR InstName U9\n",
        )
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        svg = render_svg(scene)
        missing_rects = [r for r in _elements(svg, "rect") if r.get("class") == "missing"]
        assert len(missing_rects) == 1
        labels = [t for t in _elements(svg, "text") if t.get("class") == "t-placeholder"]
        assert [t.text for t in labels] == ["U9"]


class TestRenderToFile:
    def test_writes_svg_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", GOLDEN_ASC)
        scene = build_scene(asc, SymbolResolver(local_dir=tmp_path))
        out = render_to_file(scene, tmp_path / "out.svg")
        assert out.is_file()
        ET.fromstring(out.read_text(encoding="utf-8"))


class TestCli:
    def test_happy_path_writes_and_returns_zero(self, tmp_path: Path) -> None:
        _write(tmp_path / "box2.asy", BOX2_ASY)
        asc = _write(tmp_path / "s.asc", GOLDEN_ASC)
        out = tmp_path / "out.svg"
        assert render_main([str(asc), "-o", str(out)]) == 0
        assert out.is_file()
        ET.fromstring(out.read_text(encoding="utf-8"))

    def test_missing_file_returns_2(self, tmp_path: Path) -> None:
        assert render_main([str(tmp_path / "nope.asc")]) == 2
