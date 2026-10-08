"""Symbol geometry: pin positions, bounding boxes, and rotation transforms.

Reads .asy symbol files from LTspice's library directories to extract pin
positions and bounding boxes, then applies rotation/mirror transforms to
compute absolute coordinates for placed components.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from spicelib import AscEditor

from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_file import PinInfo, read_symbol
from ltspice_mcp.lib.symbol_library import find_beside, find_symbol

logger = logging.getLogger(__name__)

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


def library_roots() -> list[Path]:
    """The libraries a symbol is looked for in, in order: the paths the session
    configured for spicelib's editor, then the simulator's own.

    They are class attributes of ``AscEditor`` for as long as it opens sheets,
    so this reads them at each call.
    """
    roots = [*(AscEditor.custom_lib_paths or ()), *(AscEditor.simulator_lib_paths or ())]
    return [Path(root) for root in roots]


def _find_asy_file(symbol: str) -> Path | None:
    """Find a .asy symbol file in the libraries (``library_roots``).

    The search is ``symbol_library.find_symbol``, the rule the renderer's
    lookup follows too.
    """
    return find_symbol(symbol, None, library_roots())


def parse_asy_file(asy_path: Path) -> SymbolInfo:
    """Parse a .asy symbol file to extract pins, bounding box, and description.

    Reads via ``read_spice_text`` (BOM/UTF-16/cp1252 fallback), NOT a bare
    UTF-8 ``read_text``: hundreds of real LTspice vendor symbols carry cp1252
    bytes (``µ``/``°``/``±``/``©`` in description fields), and a strict-UTF-8
    read raises ``UnicodeDecodeError`` on them — which previously escaped
    ``add_component`` as an opaque "Internal error".

    Raises ``ValueError`` for a symbol with a pin line that does not read: a
    pin left out is a terminal no wire could be checked against, so the symbol
    is refused whole (``_parse_or_none`` turns that into an unusable symbol).
    """
    symbol = read_symbol(read_spice_text(asy_path))
    if symbol.unread_pins:
        raise ValueError(f"{asy_path.name}: unreadable pin line {symbol.unread_pins[0]!r}")
    return SymbolInfo(
        name=asy_path.stem,
        pins=symbol.pins,
        bbox=symbol.bbox,
        symbol_type=symbol.symbol_type,
        attributes=symbol.attrs,
    )


# Library symbols by name, misses included, so a repeat never re-walks the paths.
_symbol_cache: dict[str, SymbolInfo | None] = {}

# Symbols saved beside a schematic, by file and content stamp: two sheets in
# different folders may each carry their own same-named symbol, and a sheet's
# own symbol is the one a user redraws mid-session.
_local_symbol_cache: FileCache[SymbolInfo | None] = FileCache(maxsize=256)


def _parse_or_none(asy_path: Path) -> SymbolInfo | None:
    try:
        return parse_asy_file(asy_path)
    except Exception:
        # A malformed/binary .asy (or an unhandled encoding) must not crash the
        # caller — callers treat None as "unusable symbol" and degrade cleanly
        # (add_component → a clear NetlistError; overlap scan → skip it).
        logger.warning("Failed to parse symbol file %s", asy_path, exc_info=True)
        return None


def get_symbol_info(symbol: str, asc_path: Path | None) -> SymbolInfo | None:
    """Get symbol info by name. Returns ``None`` if symbol file not found.

    ``asc_path`` is the schematic the symbol is placed on, or ``None`` for a
    library-only lookup. Its folder is searched first, as LTspice does, so a
    symbol saved beside the sheet wins over a same-named library one. That
    search is ``symbol_library.find_beside``: the place the name says, or the
    bare name right beside the sheet, and never a walk.
    ``schematic_ops.symbol_info_for`` is the per-request memo placed
    components go through.

    Library results are cached by name, negative ones too — without that,
    every reference to a missing symbol re-walks the entire library search
    path via ``rglob``.
    """
    if asc_path is not None:
        local = find_beside(asc_path.parent, symbol)
        if local is not None:
            return _local_symbol_cache.get(local, _parse_or_none)

    if symbol not in _symbol_cache:
        asy_path = _find_asy_file(symbol)
        _symbol_cache[symbol] = _parse_or_none(asy_path) if asy_path is not None else None
    return _symbol_cache[symbol]


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
