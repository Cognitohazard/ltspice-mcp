"""Symbol geometry: pin positions, bounding boxes, and rotation transforms.

Reads .asy symbol files from LTspice's library directories to extract pin
positions and bounding boxes, then applies rotation/mirror transforms to
compute absolute coordinates for placed components.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from spicelib import AscEditor

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


@dataclass(frozen=True)
class SymbolInfo:
    """Parsed symbol metadata: pins, bounding box, description.

    The bounding box is described in the symbol's local coordinate space.
    LTspice symbols are typically centered around the origin, so ``bbox_x``
    and ``bbox_y`` are usually negative.
    """

    name: str
    description: str
    pins: tuple[PinInfo, ...]
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    def to_dict(self) -> dict:
        return {
            "symbol": self.name,
            "description": self.description,
            "pins": [p.to_dict() for p in self.pins],
            "bounding_box": {
                "x": self.bbox_x,
                "y": self.bbox_y,
                "width": self.bbox_width,
                "height": self.bbox_height,
            },
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
    content = asy_path.read_text()
    lines = content.splitlines()

    pins: list[PinInfo] = []
    description = ""
    all_x: list[int] = []
    all_y: list[int] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # PIN x y ...
        if line.startswith("PIN "):
            parts = line.split()
            px, py = int(parts[1]), int(parts[2])
            # Read PINATTR lines following the PIN
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
            all_x.append(px)
            all_y.append(py)
            i = j
            continue

        # LINE Normal x1 y1 x2 y2
        if line.startswith("LINE "):
            parts = line.split()
            if len(parts) >= 6:
                x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

        # RECTANGLE / CIRCLE / ARC — also contribute to bounding box
        if line.startswith("RECTANGLE ") or line.startswith("CIRCLE ") or line.startswith("ARC "):
            coords = re.findall(r"-?\d+", line)
            for ci in range(0, len(coords) - 1, 2):
                all_x.append(int(coords[ci]))
                all_y.append(int(coords[ci + 1]))

        # SYMATTR Description
        if line.startswith("SYMATTR Description"):
            description = line.split(None, 2)[2] if len(line.split(None, 2)) > 2 else ""

        i += 1

    # Compute bounding box from all geometry points (preserve min coords;
    # symbols can have negative coordinates since they're typically centered).
    if all_x and all_y:
        bbox_x = min(all_x)
        bbox_y = min(all_y)
        bbox_width = max(all_x) - bbox_x
        bbox_height = max(all_y) - bbox_y
    else:
        bbox_x = 0
        bbox_y = 0
        bbox_width = 0
        bbox_height = 0

    symbol_name = asy_path.stem
    pins.sort(key=lambda p: p.order)
    return SymbolInfo(
        name=symbol_name,
        description=description,
        pins=tuple(pins),
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )


# Cache parsed symbols to avoid re-reading .asy files
_symbol_cache: dict[str, SymbolInfo] = {}


def get_symbol_info(symbol: str) -> SymbolInfo | None:
    """Get symbol info by name. Returns None if symbol file not found."""
    if symbol in _symbol_cache:
        return _symbol_cache[symbol]

    asy_path = _find_asy_file(symbol)
    if asy_path is None:
        return None

    info = parse_asy_file(asy_path)
    _symbol_cache[symbol] = info
    return info


def compute_placed_geometry(
    symbol_info: SymbolInfo, origin_x: int, origin_y: int, rotation: str = "R0"
) -> dict:
    """Compute absolute pin positions and bounding box for a placed component.

    Returns dict with 'pins' (list of {name, order, x, y}) and
    'bounding_box' ({x, y, width, height}) in absolute schematic coordinates.
    """
    # Transform pins
    placed_pins = []
    bx, by = symbol_info.bbox_x, symbol_info.bbox_y
    bw, bh = symbol_info.bbox_width, symbol_info.bbox_height
    for pin in symbol_info.pins:
        rx, ry = _apply_rotation(pin.x, pin.y, rotation)
        placed_pins.append(
            {
                "name": pin.name,
                "order": pin.order,
                "x": origin_x + rx,
                "y": origin_y + ry,
                "dir": _pin_direction(pin.x, pin.y, bx, by, bw, bh, rotation),
            }
        )

    # Transform the four corners of the symbol's local bounding box.
    # Symbols are typically centered around the origin, so bbox_x/bbox_y
    # are usually negative.
    corners = [
        (bx, by),
        (bx + bw, by),
        (bx, by + bh),
        (bx + bw, by + bh),
    ]
    transformed = [_apply_rotation(cx, cy, rotation) for cx, cy in corners]
    tx = [c[0] for c in transformed]
    ty = [c[1] for c in transformed]
    bbox = {
        "x": origin_x + min(tx),
        "y": origin_y + min(ty),
        "width": max(tx) - min(tx),
        "height": max(ty) - min(ty),
    }

    return {"pins": placed_pins, "bounding_box": bbox}
