"""Symbol geometry: pin positions, bounding boxes, and rotation transforms.

Reads a .asy symbol file for its pins and bounding box, and applies the
rotation and mirror transforms that place them on a sheet. Which file a
symbol's name means is the session's library's to say
(``symbol_library.SymbolLibrary``).
"""

from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_file import PinInfo, SymbolFile, read_symbol
from ltspice_mcp.lib.symbol_library import FoundSymbol

# Rotation transforms applied to (x, y) relative to symbol origin.
# LTspice coordinate system: x increases right, y increases down.
# A mirrored placement 'M<deg>' is the rotation 'R<deg>' followed by negating x,
# which is the order LTspice applies them in: M90 takes (x, y) to (y, x). The
# opposite order agrees for M0 and M180 but swaps M90 with M270.
_TRANSFORMS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    # (x', y') = (a*x + b*y, c*x + d*y)  →  stored as ((a, b), (c, d))
    "R0": ((1, 0), (0, 1)),
    "R90": ((0, -1), (1, 0)),
    "R180": ((-1, 0), (0, -1)),
    "R270": ((0, 1), (-1, 0)),
    "M0": ((-1, 0), (0, 1)),
    "M90": ((0, 1), (1, 0)),
    "M180": ((1, 0), (0, -1)),
    "M270": ((0, -1), (-1, 0)),
}


@dataclass(frozen=True)
class SymbolInfo:
    """Parsed symbol metadata: pins, bounding box, type and attributes.

    The bounding box is in the symbol's local coordinate space. LTspice
    symbols are typically centered around the origin, so ``bbox.x1`` and
    ``bbox.y1`` are usually negative. ``attributes`` is every ``SYMATTR`` the
    symbol carries, in file order (``Prefix``, ``Description``, ``SpiceModel``,
    ``Value``, ``SpiceLine``, ``ModelFile``...), and ``symbol_type`` its
    ``SymbolType`` (``CELL``, ``BLOCK``), empty when it states none.
    """

    name: str
    pins: tuple[PinInfo, ...]
    bbox: BBox
    symbol_type: str = ""
    attributes: tuple[tuple[str, str], ...] = ()

    def attribute(self, name: str) -> str:
        """The value of the symbol's last ``SYMATTR name``, "" when it has none."""
        return dict(self.attributes).get(name, "")

    @property
    def description(self) -> str:
        return self.attribute("Description")

    @property
    def prefix(self) -> str:
        """``SYMATTR Prefix`` (``R``, ``QN``, ``MN``, ``X``...): its first letter
        is the element class LTspice netlists the part as, whatever the
        instance is named."""
        return self.attribute("Prefix")


def _apply_rotation(x: int, y: int, rotation: str) -> tuple[int, int]:
    """Apply rotation/mirror transform to a point relative to symbol origin."""
    (a, b), (c, d) = _TRANSFORMS[rotation]
    return (a * x + b * y, c * x + d * y)


def place_point(x: int, y: int, origin_x: int, origin_y: int, rotation: str) -> tuple[int, int]:
    """A symbol-local point placed on the sheet at ``origin`` with ``rotation``."""
    rx, ry = _apply_rotation(x, y, rotation)
    return (origin_x + rx, origin_y + ry)


def place_box(box: BBox, origin_x: int, origin_y: int, rotation: str) -> BBox:
    """A symbol-local box placed on the sheet: the box around its four placed corners.

    LTspice's turns and mirrors keep a box upright, so this is exact.
    """
    corners = [(box.x1, box.y1), (box.x2, box.y1), (box.x1, box.y2), (box.x2, box.y2)]
    placed = BBox.from_points(
        [place_point(x, y, origin_x, origin_y, rotation) for x, y in corners]
    )
    assert placed is not None  # four corners are never empty
    return placed


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


def parse_asy_file(asy_path: Path) -> SymbolInfo:
    """Parse a .asy symbol file to extract pins, bounding box, and description.

    Reads via ``read_spice_text`` (BOM/UTF-16/cp1252 fallback), NOT a bare
    UTF-8 ``read_text``: hundreds of real LTspice vendor symbols carry cp1252
    bytes (``µ``/``°``/``±``/``©`` in description fields), and a strict-UTF-8
    read raises ``UnicodeDecodeError`` on them — which previously escaped
    ``add_component`` as an opaque "Internal error".

    Raises ``ValueError`` for a symbol with a pin line that does not read: a
    pin left out is a terminal no wire could be checked against, so the symbol
    is refused whole.
    """
    return symbol_info_of(asy_path.stem, read_symbol(read_spice_text(asy_path)))


def symbol_info_of(name: str, symbol: SymbolFile) -> SymbolInfo:
    """What the placement of a part needs of a symbol file read as ``name``.

    Raises ``ValueError`` for a symbol with a pin line that does not read: a
    pin left out is a terminal no wire could be checked against, so the
    symbol is unusable whole.
    """
    if symbol.unread_pins:
        raise ValueError(f"{name}: unreadable pin line {symbol.unread_pins[0]!r}")
    return SymbolInfo(
        name=name,
        pins=symbol.pins,
        bbox=symbol.bbox,
        symbol_type=symbol.symbol_type,
        attributes=symbol.attrs,
    )


def usable_symbol_info(found: FoundSymbol | None) -> SymbolInfo | None:
    """What the placement of a part needs of a symbol a resolver found; ``None``
    for one not found, and for one that cannot be used (``FoundSymbol.problem``)."""
    if found is None or found.problem:
        return None
    return symbol_info_of(found.name, found.file)


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

    placed = place_box(bb, origin_x, origin_y, rotation)
    return {"pins": placed_pins, "bounding_box": placed.to_origin_size_dict()}
