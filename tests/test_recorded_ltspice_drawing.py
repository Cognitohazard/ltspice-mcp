"""What each build draws for a sheet, against the scene the server lays out.

The draw cases in ``fixtures/ltspice_recorded`` keep what an image a build wrote
of a sheet draws (``ltspice_recorder.read_emf``), in the image's own units. The
two longest wires of each sheet give the scale to sheet units.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.asc_document import parse_asc
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_scene import (
    GND_DEPTH,
    GND_HALF_W,
    NetFlag,
    Scene,
    build_scene,
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
#: The triangle the server draws and holds text against, which no build drew.
GROUND_TRIANGLE_SMALLER_ON_THE_SERVER_ONLY = (6, 10)
#: A value as the server keeps it, and as both builds draw it.
DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY = {"66u": "66µ"}
#: What a symbol draws of its own that the layout check says it does not read
#: (``verify.LAYOUT_COVERAGE``): the symbol's TEXT line and its pin names.
SYMBOL_TEXT_AND_PIN_NAMES = {"CELL", "IN", "OUT"}
#: Texts at size 0 (``draw/value2_at_origin``), which both builds draw a few
#: units shorter than the scene estimates: the scene sets size 0 at a 10-unit
#: font where a build's text runs as if at about 9.5.
DRAWN_SHORTER_BY_LTSPICE_AT_SIZE_0 = {"AC 22", "Rser=22"}
#: The values U1 and U2 take from their symbol (``draw/attribute_defaults``):
#: both builds draw them, and the scene lays out only a part's own values.
DRAWN_FROM_THE_SYMBOL_BY_LTSPICE_ONLY = {"draw/attribute_defaults": ["DEFMOD", "DEFMOD", "LT1001"]}
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


def test_the_servers_ground_triangle_is_smaller_than_either_build_draws_it():
    """The render draws this triangle and ``attribute_text_overlap`` holds text
    against it, so text over the outer part of a flag's triangle is not reported."""
    assert (GND_HALF_W, GND_DEPTH) == GROUND_TRIANGLE_SMALLER_ON_THE_SERVER_ONLY


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
    with its part. A value written as "" draws nothing, and hides the symbol's
    (V3's, R3's). A value a part takes from its symbol is drawn by both builds
    and not laid out (``DRAWN_FROM_THE_SYMBOL_BY_LTSPICE_ONLY``)."""
    drawn = rec.drawing(build, case_id)
    scale = _scale(case_id, drawn)
    laid_out = _laid_out(build, case_id, tmp_path)
    left_out = Counter(t["text"] for t in drawn["texts"]) - Counter(t for t, _box in laid_out)
    assert sorted(left_out.elements()) == DRAWN_FROM_THE_SYMBOL_BY_LTSPICE_ONLY.get(case_id, [])
    assert not Counter(t for t, _box in laid_out) - Counter(t["text"] for t in drawn["texts"])
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
