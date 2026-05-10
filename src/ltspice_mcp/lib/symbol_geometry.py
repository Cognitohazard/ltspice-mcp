"""Symbol geometry: pin positions, bounding boxes, and rotation transforms.

Reads .asy symbol files from LTspice's library directories to extract pin
positions and bounding boxes, then applies rotation/mirror transforms to
compute absolute coordinates for placed components.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from spicelib import AscEditor

from ltspice_mcp.lib.geometry import BBox

logger = logging.getLogger(__name__)

# Rotation transforms applied to (x, y) relative to symbol origin.
# LTspice coordinate system: x increases right, y increases down.
_TRANSFORMS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    # (x', y') = (a*x + b*y, c*x + d*y)  →  stored as ((a, b), (c, d))
    "R0": ((1, 0), (0, 1)),
    "R90": ((0, -1), (1, 0)),
    "R180": ((-1, 0), (0, -1)),
    "R270": ((0, 1), (-1, 0)),
    "M0": ((-1, 0), (0, 1)),
    "M90": ((0, -1), (-1, 0)),
    "M180": ((1, 0), (0, -1)),
    "M270": ((0, 1), (1, 0)),
}


@dataclass(frozen=True)
class PinInfo:
    """A symbol pin with name, SPICE order, and position."""

    name: str
    order: int
    x: int
    y: int

    def to_dict(self) -> dict:
        return {"name": self.name, "order": self.order, "x": self.x, "y": self.y}


# .asy graphic primitives. Bbox-relevant fields only — line style ("Normal",
# "Dotted", ...) is parsed but discarded since nothing downstream uses it.


@dataclass(frozen=True)
class LineEl:
    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class RectEl:
    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class CircleEl:
    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class ArcEl:
    """Arc described by its underlying ellipse's bounding rectangle.

    LTspice's ARC syntax is ``ARC <style> x1 y1 x2 y2 sx sy ex ey`` where
    (x1,y1)-(x2,y2) is the ellipse bbox and (sx,sy)/(ex,ey) are the start
    and end points (which lie on the arc, hence inside the bbox). For the
    bbox of the arc itself we therefore only need the four bbox corners.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


Element = LineEl | RectEl | CircleEl | ArcEl


def _parse_shape(line: str) -> Element | None:
    """Parse one graphic-primitive line. Returns ``None`` for non-shape lines."""
    parts = line.split()
    if len(parts) < 6:
        return None
    kw = parts[0]
    try:
        x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    except ValueError:
        return None
    if kw == "LINE":
        return LineEl(x1, y1, x2, y2)
    if kw == "RECTANGLE":
        return RectEl(x1, y1, x2, y2)
    if kw == "CIRCLE":
        return CircleEl(x1, y1, x2, y2)
    if kw == "ARC":
        return ArcEl(x1, y1, x2, y2)
    return None


def bbox_from_elements(
    elements: Sequence[Element], extra_points: Sequence[tuple[int, int]] | None = None
) -> BBox | None:
    """Smallest BBox enclosing all element points and any extras (e.g. pins)."""
    pts: list[tuple[int, int]] = []
    for e in elements:
        pts.extend(e.points())
    if extra_points:
        pts.extend(extra_points)
    return BBox.from_points(pts)


@dataclass(frozen=True)
class SymbolInfo:
    """Parsed symbol metadata: pins, bounding box, description.

    The bounding box is in the symbol's local coordinate space. LTspice
    symbols are typically centered around the origin, so ``bbox.x1`` and
    ``bbox.y1`` are usually negative.
    """

    name: str
    description: str
    pins: tuple[PinInfo, ...]
    bbox: BBox

    def to_dict(self) -> dict:
        return {
            "symbol": self.name,
            "description": self.description,
            "pins": [p.to_dict() for p in self.pins],
            "bounding_box": self.bbox.to_origin_size_dict(),
        }


def _apply_rotation(x: int, y: int, rotation: str) -> tuple[int, int]:
    """Apply rotation/mirror transform to a point relative to symbol origin."""
    (a, b), (c, d) = _TRANSFORMS[rotation]
    return (a * x + b * y, c * x + d * y)


_DIRECTION_NAMES = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}


def _pin_direction(
    px: int, py: int, bbox_x: int, bbox_y: int, bbox_w: int, bbox_h: int, rotation: str
) -> str:
    """Determine which direction a pin's lead extends for external wiring.

    Computed from the pin's position relative to the bounding box center,
    then transformed by the rotation.
    """
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2
    dx = px - cx
    dy = py - cy

    # Determine primary axis (which edge the pin is closest to)
    if abs(dx) / max(bbox_w, 1) >= abs(dy) / max(bbox_h, 1):
        raw = (1 if dx > 0 else -1, 0)
    else:
        raw = (0, 1 if dy > 0 else -1)

    # Apply rotation to direction vector
    rx, ry = _apply_rotation(raw[0], raw[1], rotation)
    return _DIRECTION_NAMES.get((rx, ry), "unknown")


def _find_asy_file(symbol: str) -> Path | None:
    """Find a .asy symbol file in AscEditor's configured library paths."""
    search_paths: list[str] = []
    if hasattr(AscEditor, "custom_lib_paths") and AscEditor.custom_lib_paths:
        search_paths.extend(AscEditor.custom_lib_paths)
    if hasattr(AscEditor, "simulator_lib_paths") and AscEditor.simulator_lib_paths:
        search_paths.extend(AscEditor.simulator_lib_paths)

    for lib_path in search_paths:
        candidate = Path(lib_path) / f"{symbol}.asy"
        if candidate.exists():
            return candidate
        # Search subdirectories (LTspice organizes symbols in folders)
        for match in Path(lib_path).rglob(f"{symbol}.asy"):
            return match

    return None


def parse_asy_file(asy_path: Path) -> SymbolInfo:
    """Parse a .asy symbol file to extract pins, bounding box, and description."""
    lines = asy_path.read_text().splitlines()

    pins: list[PinInfo] = []
    description = ""
    elements: list[Element] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # PIN x y ...  followed by zero or more PINATTR lines
        if line.startswith("PIN "):
            parts = line.split()
            px, py = int(parts[1]), int(parts[2])
            pin_name = ""
            pin_order = 0
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("PINATTR"):
                attr_line = lines[j].strip()
                if attr_line.startswith("PINATTR PinName"):
                    pin_name = (
                        attr_line.split(None, 2)[2] if len(attr_line.split(None, 2)) > 2 else ""
                    )
                elif attr_line.startswith("PINATTR SpiceOrder"):
                    pin_order = int(attr_line.split()[-1])
                j += 1
            pins.append(PinInfo(name=pin_name, order=pin_order, x=px, y=py))
            i = j
            continue

        # Graphic primitives — typed parse contributes via .points()
        shape = _parse_shape(line)
        if shape is not None:
            elements.append(shape)

        if line.startswith("SYMATTR Description"):
            description = line.split(None, 2)[2] if len(line.split(None, 2)) > 2 else ""

        i += 1

    pin_points = [(p.x, p.y) for p in pins]
    bbox = bbox_from_elements(elements, extra_points=pin_points) or BBox(0, 0, 0, 0)

    pins.sort(key=lambda p: p.order)
    return SymbolInfo(
        name=asy_path.stem,
        description=description,
        pins=tuple(pins),
        bbox=bbox,
    )


# Cache parsed symbols (and resolution misses) to avoid re-walking .asy paths.
_symbol_cache: dict[str, SymbolInfo | None] = {}


def get_symbol_info(symbol: str) -> SymbolInfo | None:
    """Get symbol info by name. Returns ``None`` if symbol file not found.

    Negative results are cached too — without that, every reference to a
    missing symbol re-walks the entire library search path via ``rglob``.
    """
    if symbol in _symbol_cache:
        return _symbol_cache[symbol]

    asy_path = _find_asy_file(symbol)
    if asy_path is None:
        _symbol_cache[symbol] = None
        return None

    info = parse_asy_file(asy_path)
    _symbol_cache[symbol] = info
    return info


def compute_placed_geometry(
    symbol_info: SymbolInfo, origin_x: int, origin_y: int, rotation: str = "R0"
) -> dict:
    """Compute absolute pin positions and bounding box for a placed component.

    Returns dict with 'pins' (list of {name, order, x, y, dir}) and
    'bounding_box' ({x, y, width, height}) in absolute schematic coordinates.
    """
    bb = symbol_info.bbox

    placed_pins = []
    for pin in symbol_info.pins:
        rx, ry = _apply_rotation(pin.x, pin.y, rotation)
        placed_pins.append(
            {
                "name": pin.name,
                "order": pin.order,
                "x": origin_x + rx,
                "y": origin_y + ry,
                "dir": _pin_direction(pin.x, pin.y, bb.x1, bb.y1, bb.width, bb.height, rotation),
            }
        )

    # Transform the four corners of the local bbox, then take the AABB of the result.
    corners = [(bb.x1, bb.y1), (bb.x2, bb.y1), (bb.x1, bb.y2), (bb.x2, bb.y2)]
    transformed = [
        (origin_x + tx, origin_y + ty)
        for tx, ty in (_apply_rotation(cx, cy, rotation) for cx, cy in corners)
    ]
    placed = BBox.from_points(transformed) or BBox(origin_x, origin_y, origin_x, origin_y)
    return {"pins": placed_pins, "bounding_box": placed.to_origin_size_dict()}
