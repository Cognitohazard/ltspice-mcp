"""Where LTspice draws schematic text, against where the scene lays it out.

``tests/fixtures/ltspice_screenshots`` holds sheets that place text every way
LTspice justifies it, at every placement and size, and each LTspice build's own
drawing of them (``docs/TESTING.md``, "LTspice screenshots"). This reads from
each picture where LTspice drew every text relative to the point it is placed
by, and checks the scene puts the same text on the same sides of that point, as
long and as thick.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_scene import (
    build_scene,
    drawn_text_extent,
    flag_label_bbox,
    font_px,
    text_extent,
)
from ltspice_mcp.lib.symbol_library import SymbolResolver
from tests._schematic_fixtures import SCREENSHOT_SHEETS

PIL = pytest.importorskip("PIL.Image")

SHOTS = SCREENSHOT_SHEETS
SHEETS = sorted(path.stem for path in SHOTS.glob("*.asc"))
BUILDS = ("ltspice26", "ltspice17")

# Colours LTspice draws with: the sheet, wires (the crosses marking each
# text's point among them), and symbol bodies. Every other colour is text.
_BACKGROUND = (192, 192, 192)
_WIRE = (0, 0, 255)
_BODY = (0, 0, 127)
# How close to a text's edge its point may lie and still be at that edge.
_EDGE_PX = 2


@dataclass(frozen=True)
class _Text:
    text: str
    point: tuple[float, float]
    box: BBox


def _components(points: set[tuple[int, int]], cell: int) -> list[list[tuple[int, int]]]:
    """``points`` in groups joined through squares of side ``cell`` touching
    at an edge or a corner: with ``cell`` 1, the pixels that touch."""
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in points:
        cells.setdefault((x // cell, y // cell), []).append((x, y))
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for start in sorted(cells):
        if start in seen:
            continue
        seen.add(start)
        stack, group = [start], []
        while stack:
            cx, cy = stack.pop()
            group.extend(cells[cx, cy])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    near = (cx + dx, cy + dy)
                    if near in cells and near not in seen:
                        seen.add(near)
                        stack.append(near)
        out.append(group)
    return out


def _box(group: list[tuple[int, int]]) -> BBox:
    box = BBox.from_points(group)
    assert box is not None
    return box


def _crosses(wire: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Where the arms of each plus-shaped run of wire colour meet."""
    found = []
    for group in _components(wire, 1):
        box = _box(group)
        wide, tall = box.x2 - box.x1 + 1, box.y2 - box.y1 + 1
        if min(wide, tall) < 5 or abs(wide - tall) > 2:
            continue
        ((x, down), *_) = Counter(x for x, _ in group).most_common(1)
        ((y, across), *_) = Counter(y for _, y in group).most_common(1)
        if down == tall and across == wide and len(group) == wide + tall - 1:
            found.append((x, y))
    return found


def _merge_across_bodies(boxes: list[BBox], em: float) -> list[BBox]:
    """Join the pieces of a line of text that a symbol's body, or a gap LTspice
    leaves for an underscore it does not draw, splits: pieces no thicker than a
    line, in line with each other and up to an em and a half apart."""
    merged = list(boxes)
    joined = True
    while joined:
        joined = False
        for i, a in enumerate(merged):
            for j in range(i + 1, len(merged)):
                b = merged[j]
                level = (
                    min(a.y2, b.y2) - max(a.y1, b.y1) >= 0.6 * min(a.height, b.height)
                    and max(a.height, b.height) <= 2 * em
                    and max(a.x1, b.x1) - min(a.x2, b.x2) <= 1.5 * em
                )
                upright = (
                    min(a.x2, b.x2) - max(a.x1, b.x1) >= 0.6 * min(a.width, b.width)
                    and max(a.width, b.width) <= 2 * em
                    and max(a.y1, b.y1) - min(a.y2, b.y2) <= 1.5 * em
                )
                if level or upright:
                    merged[i] = BBox(
                        min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2)
                    )
                    del merged[j]
                    joined = True
                    break
            if joined:
                break
    return merged


def _calibration(sheet: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """The two crosses each sheet draws beyond its content, first in the file."""
    middles = []
    for match in re.finditer(
        r"^WIRE (-?\d+) (-?\d+) (-?\d+) (-?\d+)\r?$", sheet.read_text(), re.M
    ):
        x1, y1, x2, y2 = map(int, match.groups())
        if y1 == y2:
            middles.append(((x1 + x2) // 2, y1))
    return middles[0], middles[1]


def _ours(sheet: Path) -> list[_Text]:
    scene = build_scene(sheet, SymbolResolver(local_dir=SHOTS))
    texts = [
        _Text(t.text, (t.x, t.y), drawn_text_extent(t)) for sym in scene.symbols for t in sym.texts
    ]
    for d in scene.directives:
        layout = d.layout
        assert layout is not None
        texts.append(_Text(d.lines[0], (d.x, d.y), text_extent(d.x, d.y, d.lines, d.size, layout)))
    texts += [
        _Text(flag.text, (flag.x, flag.y), flag_label_bbox(flag))
        for flag in scene.flags
        if not flag.is_ground
    ]
    return texts


def _ltspices(sheet: Path, picture: Path, ours: list[_Text]) -> tuple[list[_Text], float]:
    """Each of ``ours``' texts as LTspice drew it, in sheet units, and how many
    sheet units a pixel is."""
    image = PIL.open(picture).convert("RGB")
    pixels = image.load()
    assert pixels is not None
    wire: set[tuple[int, int]] = set()
    ink: set[tuple[int, int]] = set()
    for x in range(image.width):
        for y in range(image.height):
            colour = pixels[x, y]
            if colour == _WIRE:
                wire.add((x, y))
            elif colour not in (_BACKGROUND, _BODY):
                ink.add((x, y))
    crosses = _crosses(wire)
    low_px, high_px = min(crosses, key=sum), max(crosses, key=sum)
    low, high = _calibration(sheet)
    scale_x = (high_px[0] - low_px[0]) / (high[0] - low[0])
    scale_y = (high_px[1] - low_px[1]) / (high[1] - low[1])
    assert scale_x == pytest.approx(scale_y, rel=0.02), "the calibration crosses were misread"
    scale = (scale_x + scale_y) / 2
    # The letters and the lines of one text lie closer together than half an
    # em of the default size, and different texts further apart. A speck of a
    # few pixels, such as the middle of a pin's square, is not text.
    em = font_px(2) * scale
    ink = {point for group in _components(ink, 1) if len(group) >= 4 for point in group}
    pieces = _components(ink, max(1, round(em / 2)))
    glyphs = _merge_across_bodies([_box(group) for group in pieces if len(group) >= 15], em)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return low_px[0] + (x - low[0]) * scale, low_px[1] + (y - low[1]) * scale

    def to_sheet(x: float, y: float) -> tuple[float, float]:
        return low[0] + (x - low_px[0]) / scale, low[1] + (y - low_px[1]) / scale

    drawn = []
    for text in ours:
        px, py = to_px(*text.point)
        nearest = min(
            glyphs,
            key=lambda b: max(b.x1 - px, 0, px - b.x2) ** 2 + max(b.y1 - py, 0, py - b.y2) ** 2,
        )
        x1, y1 = to_sheet(nearest.x1 - 0.5, nearest.y1 - 0.5)
        x2, y2 = to_sheet(nearest.x2 + 0.5, nearest.y2 + 0.5)
        drawn.append(
            _Text(text.text, text.point, BBox(round(x1), round(y1), round(x2), round(y2)))
        )
    return drawn, 1 / scale


def _sides(text: _Text, edge: float) -> tuple[str, str]:
    """Where a text lies from its point along x and along y: wholly after it
    (``+``), wholly before it (``-``), or across it (``~``)."""

    def side(low: float, high: float, at: float) -> str:
        if low >= at - edge:
            return "+"
        if high <= at + edge:
            return "-"
        return "~"

    (x, y), box = text.point, text.box
    return side(box.x1, box.x2, x), side(box.y1, box.y2, y)


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("sheet", SHEETS)
def test_text_lies_where_ltspice_draws_it(sheet: str, build: str) -> None:
    ours = _ours(SHOTS / f"{sheet}.asc")
    assert ours, "the sheet places no text"
    drawn, unit = _ltspices(SHOTS / f"{sheet}.asc", SHOTS / build / f"{sheet}.png", ours)
    edge = _EDGE_PX * unit
    wrong = []
    for mine, theirs in zip(ours, drawn, strict=True):
        if _sides(mine, 0.5) != _sides(theirs, edge):
            wrong.append(f"{mine.text}: ours {_sides(mine, 0.5)}, LTspice {_sides(theirs, edge)}")
            continue
        # Length along its run and thickness across it, as the font's metrics
        # give them and as LTspice drew them, to a pixel or two of rounding.
        our_long, our_thick = sorted((mine.box.width, mine.box.height), reverse=True)
        their_long, their_thick = sorted((theirs.box.width, theirs.box.height), reverse=True)
        for what, ours_span, their_span, share in (
            ("long", our_long, their_long, 0.1),
            ("thick", our_thick, their_thick, 0.3),
        ):
            if abs(ours_span - their_span) > share * their_span + 3 * unit:
                wrong.append(f"{mine.text}: ours {ours_span} units {what}, LTspice {their_span}")
    assert not wrong, "\n".join(wrong)
