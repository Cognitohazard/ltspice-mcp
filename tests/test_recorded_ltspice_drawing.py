"""What each build draws for a sheet, against the scene the server lays out.

The draw cases in ``fixtures/ltspice_recorded`` keep what an image a build wrote
of a sheet draws (``ltspice_recorder.read_emf``), in the image's own units. The
two longest wires of each sheet give the scale to sheet units.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.asc_document import parse_asc
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_scene import (
    NetFlag,
    Scene,
    build_scene,
    drawn_text_extent,
    ground_polygon,
    line_height,
    sheet_view,
)
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.lib.symbol_library import SymbolResolver
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS

#: The text size a build falls back to, which is the size of a text at size
#: index 2, and half the base and the depth of the ground triangle at that size.
DEFAULT_FONT = 28
GROUND_TRIANGLE = (18, 18)
#: A value as the server keeps it, and as both builds draw it.
DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY = {"66u": "66µ", "20u": "20µ", "100u": "100µ"}
#: What a symbol draws of its own that the layout check says it does not read
#: (``verify.LAYOUT_COVERAGE``): the symbol's TEXT line and its pin names.
SYMBOL_TEXT_AND_PIN_NAMES = {"CELL", "IN", "OUT"}
#: Texts at size 0 (``draw/value2_at_origin``), which both builds draw a few
#: units shorter than the scene estimates: the scene sets size 0 at a 10-unit
#: font where a build's text runs as if at about 9.5.
DRAWN_SHORTER_BY_LTSPICE_AT_SIZE_0 = {"AC 22", "Rser=22"}
#: How far a drawn point may lie from the sheet's in sheet units: an image's
#: coordinates are whole numbers, at a scale of a few to the sheet unit.
NEAR = 1


@dataclass(frozen=True)
class _Scale:
    """From an image's units to a sheet's."""

    per_unit: float
    x0: float
    y0: float

    def point(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / self.per_unit, (y - self.y0) / self.per_unit


def _scale(case_id: str, drawn: dict[str, Any]) -> _Scale:
    """The scale of a drawing, from its sheet's two longest wires, level and upright."""
    wires = parse_asc((INPUTS / rec.CASES.case(case_id).source).read_bytes()).wires
    across = max((w for w in wires if w.y1 == w.y2), key=lambda w: abs(w.x2 - w.x1))
    down = max((w for w in wires if w.x1 == w.x2), key=lambda w: abs(w.y2 - w.y1))
    strokes = drawn["strokes"]
    level = max((s for s in strokes if s[1] == s[3]), key=lambda s: abs(s[2] - s[0]))
    upright = max((s for s in strokes if s[0] == s[2]), key=lambda s: abs(s[3] - s[1]))
    per_unit = abs(level[2] - level[0]) / abs(across.x2 - across.x1)
    return _Scale(
        per_unit,
        min(level[0], level[2]) - min(across.x1, across.x2) * per_unit,
        min(upright[1], upright[3]) - min(down.y1, down.y2) * per_unit,
    )


def _scene(build: str, case_id: str, directory: Path) -> Scene:
    sheet = rec.stage_sheet(build, case_id, directory)
    return build_scene(sheet, SymbolResolver(local_dir=sheet.parent))


def _laid_out(build: str, case_id: str, directory: Path) -> list[tuple[str, BBox]]:
    """Each text of a sheet's parts and where the scene lays it out."""
    return [
        (text, box)
        for part in sheet_view(_scene(build, case_id, directory)).parts
        for _x, _y, text, box, _attribute in part.text_extents
    ]


def _triangle(drawn: dict[str, Any], scale: _Scale, flag: NetFlag) -> set[tuple[float, float]]:
    """The corners of the triangle at a flag, from the flag's point: the ends
    of the slanted strokes near it, which only a ground flag draws on these sheets."""
    reach = 30
    corners = set()
    for x1, y1, x2, y2 in drawn["strokes"]:
        ends = [scale.point(x1, y1), scale.point(x2, y2)]
        slanted = x1 != x2 and y1 != y2
        if slanted and all(max(abs(x - flag.x), abs(y - flag.y)) <= reach for x, y in ends):
            corners.update((x - flag.x, y - flag.y) for x, y in ends)
    return corners


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    ("case_id", "font"),
    [("draw/ground_flags", DEFAULT_FONT), ("draw/ground_flags_font_40", 40)],
)
def test_a_ground_flag_is_a_triangle_at_the_text_size(
    build: str, case_id: str, font: int, tmp_path: Path
):
    """Its base lies through the flag's point and it points the way the scene
    turns it, wired on any side or on no wire. Its size follows the setting."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    half_base, depth = (side * font / DEFAULT_FONT for side in GROUND_TRIANGLE)
    wanted = {
        "down": [(-half_base, 0), (half_base, 0), (0, depth)],
        "up": [(-half_base, 0), (half_base, 0), (0, -depth)],
        "right": [(0, -half_base), (0, half_base), (depth, 0)],
        "left": [(0, -half_base), (0, half_base), (-depth, 0)],
    }
    flags = [flag for flag in _scene(build, case_id, tmp_path).flags if flag.is_ground]
    assert sorted(flag.orientation for flag in flags) == ["down", "down", "left", "right", "up"]
    for flag in flags:
        corners = _triangle(drawn, scale, flag)
        assert len(corners) == 3, flag
        for x, y in wanted[flag.orientation]:
            assert any(abs(x - a) <= NEAR and abs(y - b) <= NEAR for a, b in corners), (
                f"{flag}: no corner at ({x}, {y}) among {sorted(corners)}"
            )


#: Every recorded sheet with a ground the scene draws, at the default text size.
#: The sheet at ``SchFontSize=40`` is left out: the scene lays a sheet out at the
#: default size, its text as its grounds.
SHEETS_WITH_GROUNDS = [
    "draw/ground_flags",
    "draw/gnd_and_zero_flags",
    "draw/suite_nmos4_mirrored",
    "draw/suite_nmos_mirror_labelled",
    "draw/suite_two_stage_opamp_drawn",
    "draw/suite_two_stage_opamp_labelled",
    "draw/suite_two_stage_opamp_moved",
]


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SHEETS_WITH_GROUNDS)))
def test_the_scenes_ground_triangle_is_the_one_ltspice_draws(
    build: str, case_id: str, tmp_path: Path
):
    """Every ground triangle the scene lays out has the size and the turn of
    the triangle the build drew there: each corner, taken from the triangle's
    centre, lies on a drawn one. The render draws the scene's triangle and the
    layout check holds text against it, so both cover what LTspice draws.

    Corners are compared from the centre because the scale taken from a
    sheet's wires can place a whole drawing a unit or two off (the suite
    sheets), which moves a triangle without changing its size. A drawn point
    is a whole number of image units, so each corner and the centre may each
    lie half an image unit off along an axis: the tolerance is one image unit
    in sheet units, and never less than ``NEAR``."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    near = max(NEAR, 1 / scale.per_unit)
    grounds = [flag for flag in _scene(build, case_id, tmp_path).flags if flag.is_ground]
    assert grounds
    for flag in grounds:
        corners = _from_centre(_triangle(drawn, scale, flag))
        assert len(corners) == 3, flag
        for x, y in _from_centre(ground_polygon(flag)):
            assert any(abs(x - a) <= near and abs(y - b) <= near for a, b in corners), (
                f"{flag}: no drawn corner at ({x}, {y}) from the centre among {sorted(corners)}"
            )


def _from_centre(
    corners: Iterable[tuple[float, float]],
) -> set[tuple[float, float]]:
    """Each corner less the mean of the corners."""
    points = list(corners)
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return {(x - cx, y - cy) for x, y in points}


def _cell(text: dict[str, Any], scale: _Scale) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = text["bounds"]
    return (*scale.point(x1, y1), *scale.point(x2, y2))


@pytest.mark.parametrize("build", rec.BUILDS)
def test_part_text_lies_where_the_scene_lays_it_out(build: str, tmp_path: Path):
    """At each of the eight placements each name and value starts and ends
    where the scene has it. A drawn box is the text's cell and the scene's is
    the ink it estimates, so they share the edge below the glyphs and the
    cell's other edge lies beyond the ink."""
    case_id = "draw/part_text"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    texts = {text["text"]: text for text in drawn["texts"]}
    laid_out = _laid_out(build, case_id, tmp_path)
    assert len(laid_out) == 16
    for text, box in laid_out:
        _assert_where_drawn(
            text, box, texts[DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY.get(text, text)], scale
        )


def _assert_where_drawn(text: str, box: BBox, shown: dict[str, Any], scale: _Scale) -> None:
    """The scene's box for ``text`` starts and ends where a build drew it. A
    drawn box is the text's cell and the scene's is the ink it estimates, so
    they share the edge below the glyphs and the cell's other edge lies
    beyond the ink. A text at size 0 ends up to a tenth of its length past
    where a build ends it (``DRAWN_SHORTER_BY_LTSPICE_AT_SIZE_0``)."""
    x1, y1, x2, y2 = _cell(shown, scale)
    if shown["escapement"] == 0:
        ours = (box.x1, box.x2, box.y2)
        theirs = (x1, x2, y2)
        assert y1 < box.y1, text
    else:
        ours = (box.y1, box.y2, box.x2)
        theirs = (y1, y2, x2)
        assert x1 < box.x1, text
    if text in DRAWN_SHORTER_BY_LTSPICE_AT_SIZE_0:
        start, end, below = ours
        length = theirs[1] - theirs[0]
        assert theirs[1] < end <= theirs[1] + length / 10, text
        ours = (start, theirs[1], below)
    assert ours == pytest.approx(theirs, abs=2), text


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    ("case_id", "font"),
    [("draw/part_text", DEFAULT_FONT), ("draw/part_text_font_40", 40)],
)
def test_text_is_tahoma_bold_as_high_as_the_setting(build: str, case_id: str, font: int):
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    assert {(text["font"], text["weight"]) for text in drawn["texts"]} == {("Tahoma", 700)}
    for text in drawn["texts"]:
        assert text["height"] / scale.per_unit == pytest.approx(font, abs=1)


def test_the_scenes_line_is_as_high_as_the_default_text():
    assert line_height(2) == DEFAULT_FONT


@pytest.mark.parametrize("build", rec.BUILDS)
def test_value2_is_drawn_only_with_a_visible_window(build: str):
    """V2 carries ``WINDOW 123 32 80 Left 2``, at (192, 32). V1 and R3 have a
    Value2 and no such line, and the symbols give that attribute no window."""
    case_id = "draw/value2"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    texts = {text["text"]: text for text in drawn["texts"]}
    assert set(texts) == {"V1", "1", "V2", "2", "AC 22", "R3", "3k"}
    x1, y1, _x2, y2 = _cell(texts["AC 22"], scale)
    assert x1 == pytest.approx(192 + 32, abs=NEAR)
    assert (y1 + y2) / 2 == pytest.approx(32 + 80, abs=NEAR)


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    "case_id",
    ["draw/value2", "draw/attribute_windows", "draw/value2_at_origin", "draw/attribute_defaults"],
)
def test_the_scene_lays_out_every_attribute_a_build_draws_where_it_draws_it(
    build: str, case_id: str, tmp_path: Path
):
    """Each name, value, Value2, SpiceLine, SpiceLine2 and SpiceModel a build
    draws, and nothing else, each where the build draws it: through a window
    at an offset or at the symbol's origin, at size 2 or size 0, and turned
    with its part, and from the symbol when the part gives no value of its own
    (U1's value and model, U2's model). A value written as "" draws nothing,
    and hides the symbol's (V3's, R3's)."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    laid_out = _laid_out(build, case_id, tmp_path)
    assert Counter(t for t, _box in laid_out) == Counter(t["text"] for t in drawn["texts"])
    for text, box in laid_out:
        # The drawing of that text nearest the box, for a text two parts draw.
        shown = min(
            (t for t in drawn["texts"] if t["text"] == text),
            key=lambda t: abs(_cell(t, scale)[0] - box.x1) + abs(_cell(t, scale)[3] - box.y2),
        )
        _assert_where_drawn(text, box, shown, scale)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_two_attributes_at_one_point_are_drawn_over_each_other(build: str, tmp_path: Path):
    """V1 shows its Value2 and its SpiceLine through windows at the symbol's
    origin, as LTspice writes them on most of its own example sheets: both
    builds draw the two texts over each other, and the layout check says so."""
    case_id = "draw/value2_at_origin"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    cells = {t["text"]: BBox(*map(round, _cell(t, scale))) for t in drawn["texts"]}
    assert cells["AC 11"].overlaps(cells["Rser=11"])
    found = findings(sheet_view(_scene(build, case_id, tmp_path)))
    assert "V1's text 'Rser=11' at (32,32) is drawn across V1's text 'AC 11' at (32,32)" in [
        f.detail for f in found if f.rule == "attribute_text_overlap"
    ]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_what_the_scene_leaves_out_is_a_symbols_own_text_and_its_pin_names(
    build: str, tmp_path: Path
):
    """What the layout check and the render say they do not read
    (``verify.LAYOUT_COVERAGE``): both builds draw the cell's TEXT line and
    its two pin names, and the scene lays out only the part's name."""
    case_id = "draw/symbol_text"
    drawn = {text["text"] for text in rec.drawing(build, case_id)["texts"]}
    laid_out = {text for text, _box in _laid_out(build, case_id, tmp_path)}
    assert laid_out == {"U1"}
    assert drawn - laid_out == SYMBOL_TEXT_AND_PIN_NAMES


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_text_is_drawn_in_the_lines_the_scene_decodes(build: str, tmp_path: Path):
    """Both builds draw a TEXT record's ``\\\\`` as one backslash and break its
    line at ``\\n``, even inside ``C:\\new``, and keep any other backslash as it
    is, in a comment and a directive alike (``text-backslash``): the lines
    ``decode_text_lines`` reads, which the scene lays out."""
    case_id = "draw/text_backslash"
    drawn = sorted(text["text"] for text in rec.drawing(build, case_id)["texts"])
    assert "one\\two" in drawn
    assert {"C:", "ew\\temp", "tail\\nhead", "end\\"} <= set(drawn)
    scene = _scene(build, case_id, tmp_path)
    assert sorted(line for text in scene.directives for line in text.lines) == drawn


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_window_line_short_of_its_justification_or_size_is_read_past(build: str, tmp_path: Path):
    """R2's name has a WINDOW line without its size and R3's one without its
    justification and size: both builds draw each name where R4's lands, at
    the symbol's own window, 160 and 320 to the right of it. R5's symbol has
    both its WINDOW lines cut short, and neither build draws its name or value
    (``window-short``). The scene lays the texts out the same way."""
    case_id = "draw/window_short"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    cells = {text["text"]: _cell(text, scale) for text in drawn["texts"]}
    assert set(cells) == {"R1", "1k", "R2", "2k", "R3", "3k", "R4", "4k"}
    r4 = cells["R4"]
    for name, shift in (("R2", 320), ("R3", 160)):
        x1, y1, x2, y2 = cells[name]
        assert (x1 + shift, y1, x2 + shift, y2) == pytest.approx(r4, abs=NEAR), name
    laid_out = dict(_laid_out(build, case_id, tmp_path))
    assert set(laid_out) == set(cells)
    for name in ("R2", "R3", "R4"):
        x1, _y1, x2, y2 = cells[name]
        box = laid_out[name]
        assert (box.x1, box.x2, box.y2) == pytest.approx((x1, x2, y2), abs=2), name


@pytest.mark.parametrize(
    ("build", "case_id"),
    list(
        rec.per_build([*rec.cases_of("sheet-drawing"), "draw/text_backslash", "draw/window_short"])
    ),
)
def test_nothing_on_these_sheets_is_drawn_in_a_way_the_reader_does_not_read(
    build: str, case_id: str
):
    assert rec.drawing(build, case_id)["other"] == []


# --------------------------------------------------------------------------
# Questions the code answered without a recording
# --------------------------------------------------------------------------


def _drawn_boxes(build: str, case_id: str) -> dict[str, tuple[float, float, float, float, int]]:
    """Each text a build drew, its box in sheet units and its escapement."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    boxes = {}
    for text in drawn["texts"]:
        x0, y0, x1, y1 = text["bounds"]
        (ax, ay), (bx, by) = scale.point(x0, y0), scale.point(x1, y1)
        boxes[text["text"]] = (ax, ay, bx, by, text["escapement"])
    return boxes


#: A port, which both builds draw as an outline around its name: an input's
#: arrow tip on its point and its body running away from it, an output's flat
#: end on its point and its tip away, a two-way port's tips at both ends, and
#: on a wire's end the same along the wire, away from it. The scene holds a
#: port as a net label, its name centred over the point, with no outline.
PORT_DRAWN_AS_AN_OUTLINE_BY_LTSPICE_AND_AS_A_LABEL_BY_THE_SCENE = {
    "pin_in": ((64, 64), (-29, 64)),
    "pin_out": ((240, 64), (350, 64)),
    "pin_io": ((400, 64), (507, 64)),
    "wired_in": ((96, 160), (96, 37)),
}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_port_is_drawn_as_an_outline_around_its_name(build: str, tmp_path: Path):
    case_id = "draw/port_outlines"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    segments = [(scale.point(x1, y1), scale.point(x2, y2)) for x1, y1, x2, y2 in drawn["strokes"]]

    def on_a_stroke(x: float, y: float) -> bool:
        """Whether a stroke passes through (x, y), an end or a side."""
        for (ax, ay), (bx, by) in segments:
            length = math.hypot(bx - ax, by - ay)
            crossed = length and abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / length <= NEAR
            within = min(ax, bx) - NEAR <= x <= max(ax, bx) + NEAR
            if crossed and within and min(ay, by) - NEAR <= y <= max(ay, by) + NEAR:
                return True
        return False

    boxes = _drawn_boxes(build, case_id)
    view = sheet_view(_scene(build, case_id, tmp_path))
    laid_out = {text: (x, y, box) for x, y, text, box in view.text_extents}
    outlines = PORT_DRAWN_AS_AN_OUTLINE_BY_LTSPICE_AND_AS_A_LABEL_BY_THE_SCENE
    for name, (near, far) in outlines.items():
        assert on_a_stroke(*near) and on_a_stroke(*far), (name, near, far)
        x0, y0, x1, y1, _ = boxes[name]
        lo_x, hi_x = sorted((near[0], far[0]))
        lo_y, hi_y = sorted((near[1], far[1]))
        # The name lies along the outline, inside its span.
        if lo_x != hi_x:
            assert lo_x - NEAR <= x0 and x1 <= hi_x + NEAR, name
        else:
            assert lo_y - NEAR <= y0 and y1 <= hi_y + NEAR, name
        x, y, box = laid_out[name]
        assert (x, y) == near
        assert box.y2 == y or box.y2 == near[1], name
    assert (view.texts, view.grounds) == ((), ())
    assert {label[2] for label in view.labels} == set(
        PORT_DRAWN_AS_AN_OUTLINE_BY_LTSPICE_AND_AS_A_LABEL_BY_THE_SCENE
    )


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_window_justified_invisible_is_not_drawn_and_a_moved_one_is(build: str, tmp_path: Path):
    """A window justified ``Invisible`` draws nothing, the name's or the
    value's alone or both, and the scene leaves it out too. A window moved to
    the left of the body, and one moved, turned and given another size, are
    drawn where the scene lays them out, to within the cell's top (a drawn box
    is its cell, a laid-out one its glyphs)."""
    case_id = "draw/windows_hidden_moved"
    drawn = _drawn_boxes(build, case_id)
    assert set(drawn) == {"R1", "1k", "2k", "R3", "R5", "5k", "R6", "6k"}
    laid_out = dict(_laid_out(build, case_id, tmp_path))
    assert set(laid_out) == set(drawn)
    slack = 2
    for text, (x0, y0, x1, y1, escapement) in drawn.items():
        box = laid_out[text]
        if escapement:
            assert (box.y1, box.y2, box.x2) == pytest.approx((y0, y1, x1), abs=slack), text
        else:
            assert (box.x1, box.x2, box.y2) == pytest.approx((x0, x1, y1), abs=slack), text


#: A flag named GND: both builds draw it as they draw a flag named 0, a ground
#: triangle and no text. The scene draws a triangle for 0 only and holds GND as
#: a net label's text, though every reader of nets takes it as ground.
GND_FLAG_DRAWN_AS_A_GROUND_BY_LTSPICE_AND_AS_A_LABEL_BY_THE_SCENE = "GND"


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_flag_named_gnd_is_drawn_as_a_ground(build: str, tmp_path: Path):
    case_id = "draw/gnd_and_zero_flags"
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    assert {text["text"] for text in drawn["texts"]} == {"a", "b"}
    scene = _scene(build, case_id, tmp_path)
    flags = {flag.text: flag for flag in scene.flags}
    for name in ("0", GND_FLAG_DRAWN_AS_A_GROUND_BY_LTSPICE_AND_AS_A_LABEL_BY_THE_SCENE):
        corners = _triangle(drawn, scale, flags[name])
        assert len(corners) == 3, (name, corners)
    view = sheet_view(scene)
    assert [(x, y) for x, y, _corners in view.grounds] == [(64, 96)]
    assert (256, 96, "GND") in view.labels


#: A net label in the middle of an upright wire: both builds draw its text
#: along the wire, turned a quarter, on the wire's left; the scene lays it
#: across the wire, to the right of its point.
LABEL_ON_AN_UPRIGHT_WIRE_DRAWN_ALONG_IT_BY_LTSPICE_AND_ACROSS_IT_BY_THE_SCENE = "mid_upright"


@pytest.mark.parametrize("build", rec.BUILDS)
def test_where_a_net_labels_text_goes_mid_span_and_at_a_corner(build: str, tmp_path: Path):
    """A label in the middle of a level wire and one at the corner of two
    wires: both builds centre the text above the point, and the scene lays it
    there too."""
    case_id = "draw/label_mid_span_and_corner"
    drawn = _drawn_boxes(build, case_id)
    view = sheet_view(_scene(build, case_id, tmp_path))
    laid_out = {text: (x, y, box) for x, y, text, box in view.text_extents}
    for text in ("mid_level", "corner"):
        x0, _y0, x1, y1, escapement = drawn[text]
        x, y, box = laid_out[text]
        assert escapement == 0 and y1 == pytest.approx(y, abs=NEAR)
        assert (x0 + x1) / 2 == pytest.approx(x, abs=NEAR)
        assert (box.x1, box.x2, box.y2) == pytest.approx((x0, x1, y1), abs=NEAR), text
    upright = LABEL_ON_AN_UPRIGHT_WIRE_DRAWN_ALONG_IT_BY_LTSPICE_AND_ACROSS_IT_BY_THE_SCENE
    x0, y0, x1, y1, escapement = drawn[upright]
    x, y, box = laid_out[upright]
    assert escapement == 900 and x1 == pytest.approx(x, abs=NEAR)
    assert (y0 + y1) / 2 == pytest.approx(y, abs=NEAR)
    assert box.x1 == x and box.y1 < y < box.y2


#: Eight of the suite's own sheets drawn by both builds with the suite's
#: stand-in symbols beside them, as the findings snapshot reads them.
SUITE_SHEETS = [c for c in rec.cases_of("sheet-drawing-other-text") if c.startswith("draw/suite_")]
#: A label at a corner where a wire comes in from the left and one from above:
#: both builds centre its text below the point, and the scene puts it to the
#: right, the side its fallback for a corner picks.
CORNER_LABEL_DRAWN_BELOW_BY_LTSPICE_AND_TO_THE_RIGHT_BY_THE_SCENE = {
    ("draw/suite_nmos4_mirrored", "SRC", 240, 144)
}
#: These sheets are wide, so a build draws them at under one image unit to the
#: sheet unit, where its font hinting sets text up to a sixth shorter, and up
#: to four units off its base, than at the size the other draw cases use. A
#: text is where a build draws it when the scene's box and the drawn one share
#: at least this much of their area: every text drawn where the scene puts it
#: shares 0.57 or more, the corner label below shares 0.13, and a label on the
#: other side of its point shares nothing.
SHARED_AREA = 0.5


def _shared(ours: BBox, theirs: BBox) -> float:
    """The area two boxes share, over the area of the two together."""
    x = max(0, min(ours.x2, theirs.x2) - max(ours.x1, theirs.x1))
    y = max(0, min(ours.y2, theirs.y2) - max(ours.y1, theirs.y1))
    both = ours.width * ours.height + theirs.width * theirs.height - x * y
    return x * y / both if both else 0.0


def _glyphs(cell: BBox, ours: BBox) -> BBox:
    """A drawn text's cell cut to the glyphs' height the scene estimates, at
    the cell's base: a cell holds the line's leading too."""
    if ours.width >= ours.height:
        return BBox(cell.x1, cell.y2 - ours.height, cell.x2, cell.y2)
    return BBox(cell.x2 - ours.width, cell.y1, cell.x2, cell.y2)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUITE_SHEETS)))
def test_every_text_on_the_suite_sheets_lies_where_ltspice_draws_it(
    build: str, case_id: str, tmp_path: Path
):
    """Every part text, a value taken from the symbol included, and every net
    label the scene lays out on these sheets is where the build draws it. A
    label on a part's pin with no wire faces out of the part, on the side of
    the symbol's box the pin lies on, turned with the part."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    scene = _scene(build, case_id, tmp_path)
    texts = [(t.text, t.x, t.y, drawn_text_extent(t)) for s in scene.symbols for t in s.texts]
    texts += [(text, x, y, box) for x, y, text, box in scene.sheet_text_extents()]
    away = []
    for text, x, y, ours in texts:
        shown = DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY.get(text, text)
        cells = [BBox(*map(round, _cell(t, scale))) for t in drawn["texts"] if t["text"] == shown]
        assert cells, f"{text} is laid out and not drawn"
        best = max(_shared(ours, _glyphs(cell, ours)) for cell in cells)
        if best < SHARED_AREA:
            away.append((case_id, text, x, y))
    expected = CORNER_LABEL_DRAWN_BELOW_BY_LTSPICE_AND_TO_THE_RIGHT_BY_THE_SCENE
    assert set(away) == {key for key in expected if key[0] == case_id}
    # And the build draws each text as many times as the scene lays it out, a
    # value from the symbol included.
    shown = Counter(DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY.get(t, t) for t, *_ in texts)
    assert Counter(t["text"] for t in drawn["texts"] if t["text"]) == shown


#: The suite sheets with a ground on a part's pin and no wire there.
SUITE_SHEETS_WITH_GROUNDS = [
    "draw/suite_nmos4_mirrored",
    "draw/suite_nmos_mirror_labelled",
    "draw/suite_two_stage_opamp_drawn",
    "draw/suite_two_stage_opamp_labelled",
    "draw/suite_two_stage_opamp_moved",
]


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SUITE_SHEETS_WITH_GROUNDS)))
def test_every_ground_on_the_suite_sheets_points_where_ltspice_draws_it(
    build: str, case_id: str, tmp_path: Path
):
    """A ground on a part's pin with no wire points out of the part, on the
    side its pin faces (``symbol_geometry.pin_side``), as a label's text goes."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    grounds = [flag for flag in _scene(build, case_id, tmp_path).flags if flag.is_ground]
    assert grounds
    for flag in grounds:
        # The apex is the corner whose other two, the base's ends, lie level or
        # upright: within a few units, at these sheets' coarse scale.
        corners = _triangle(drawn, scale, flag)
        ((x, y),) = [
            (cx, cy)
            for cx, cy in corners
            if len(base := sorted(corners - {(cx, cy)})) == 2
            and min(abs(base[0][0] - base[1][0]), abs(base[0][1] - base[1][1])) <= 3
        ]
        way = ("right" if x > 0 else "left") if abs(x) > abs(y) else ("down" if y > 0 else "up")
        assert way == flag.orientation, (flag.x, flag.y)
