"""What each build draws for a sheet, against the scene the server lays out.

The draw cases in ``fixtures/ltspice_recorded`` keep what an image a build wrote
of a sheet draws (``ltspice_recorder.read_emf``), in the image's own units. The
two longest wires of each sheet give the scale to sheet units.
"""

from __future__ import annotations

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
#: A Value2 with a visible window, which both builds draw and the scene does
#: not hold, so the render does not show it and no rule sees it.
DRAWN_BY_LTSPICE_AND_LEFT_OUT_OF_THE_SCENE = {"AC 22"}
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
        for _x, _y, text, box in part.text_extents
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
        shown = texts[DRAWN_WITH_A_MICRO_SIGN_BY_LTSPICE_ONLY.get(text, text)]
        x1, y1, x2, y2 = _cell(shown, scale)
        if shown["escapement"] == 0:
            ours = (box.x1, box.x2, box.y2)
            theirs = (x1, x2, y2)
            assert y1 < box.y1, text
        else:
            ours = (box.y1, box.y2, box.x2)
            theirs = (y1, y2, x2)
            assert x1 < box.x1, text
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
def test_the_scene_holds_every_part_text_a_build_draws_but_a_value2(build: str, tmp_path: Path):
    case_id = "draw/value2"
    drawn = {text["text"] for text in rec.drawing(build, case_id)["texts"]}
    laid_out = {text for text, _box in _laid_out(build, case_id, tmp_path)}
    assert drawn - laid_out == DRAWN_BY_LTSPICE_AND_LEFT_OUT_OF_THE_SCENE
    assert laid_out - drawn == set()


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(rec.cases_of("sheet-drawing"))))
def test_nothing_on_these_sheets_is_drawn_in_a_way_the_reader_does_not_read(
    build: str, case_id: str
):
    assert rec.drawing(build, case_id)["other"] == []
