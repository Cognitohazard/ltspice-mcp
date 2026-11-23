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


def pin_side(box: BBox, x: int, y: int, rotation: str = "R0") -> str | None:
    """The side a part faces out on at its symbol-local pin ``(x, y)``, turned
    by ``rotation``: the edge of the symbol's box ``box`` the pin lies on, the
    top or bottom at a corner; ``None`` for a pin inside the box.

    Both builds draw a net label or a ground on a part's pin with no wire out
    of the part on this side: labels on every placement of three symbols
    (``draw/suite_nmos4_orientations``, ``draw/suite_npn_orientations``,
    ``draw/suite_res_orientations``), grounds on the transistors' pins of
    ``draw/suite_two_stage_opamp_drawn`` and ``draw/suite_nmos_mirror_labelled``.
    So a wire to the pin should leave on it too.

    A box with no height has no top or bottom, and one with no width no sides,
    so a symbol drawn as one level line faces left and right at its ends. No
    recording shows such a symbol.
    """
    if box.height and y == box.y1:
        facing = (0, -1)
    elif box.height and y == box.y2:
        facing = (0, 1)
    elif box.width and x == box.x1:
        facing = (-1, 0)
    elif box.width and x == box.x2:
        facing = (1, 0)
    else:
        return None
    return _DIRECTION_NAMES[_apply_rotation(*facing, rotation)]


def _pin_direction(px: int, py: int, box: BBox, rotation: str) -> str:
    """Which way a wire to the pin at ``(px, py)`` leaves the part: the side
    the part faces out on there (``pin_side``), else, for a pin inside the
    box, the edge it lies nearest, relative to the box's size.

    A pin inside the box still gets a direction, because a caller wiring it
    needs one. No recording shows how LTspice draws a label on such a pin, so
    the scene keeps a label's resting side there instead
    (``test_a_pin_inside_its_symbols_box_has_a_wiring_direction_and_no_label_side``).
    """
    side = pin_side(box, px, py, rotation)
    if side is not None:
        return side
    dx = px - (box.x1 + box.width / 2)
    dy = py - (box.y1 + box.height / 2)
    if abs(dx) / max(box.width, 1) >= abs(dy) / max(box.height, 1):
        raw = (1 if dx > 0 else -1, 0)
    else:
        raw = (0, 1 if dy > 0 else -1)
    return _DIRECTION_NAMES[_apply_rotation(*raw, rotation)]


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
                "dir": _pin_direction(pin.x, pin.y, bb, rotation),
            }
        )

    placed = place_box(bb, origin_x, origin_y, rotation)
    return {"pins": placed_pins, "bounding_box": placed.to_origin_size_dict()}
