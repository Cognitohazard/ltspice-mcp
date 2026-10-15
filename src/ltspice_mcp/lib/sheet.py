"""A schematic sheet as a read-only value: its records and the parts they place.

``read_sheet`` parses an ``.asc`` file's bytes with ``asc_document`` and looks
up each part's symbol with a ``SymbolResolver``. A ``Sheet`` holds the parsed
document and, for each part, where its pins and boxes are. Readers that do not
edit a sheet use it instead of spicelib's editor.

Nothing here writes a file or needs the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from ltspice_mcp.lib.asc_document import AscDocument, Symbol, parse_asc, parse_asc_text
from ltspice_mcp.lib.connectivity import NetPartition, partition
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_geometry import (
    SymbolInfo,
    _apply_rotation,  # pyright: ignore[reportPrivateUsage]  # shared rotation/mirror transform
    compute_placed_geometry,
)
from ltspice_mcp.lib.symbol_library import FoundSymbol, SymbolResolver

Point = tuple[int, int]


@dataclass(frozen=True)
class PlacedPin:
    """A pin of a placed part, in sheet coordinates."""

    name: str
    #: The pin's SpiceOrder; 0 when the symbol gives it none.
    order: int
    x: int
    y: int
    #: The side of the part the pin points away from, as ``compute_placed_geometry`` names it.
    direction: str


@dataclass(frozen=True)
class SheetPart:
    """A placed part.

    ``symbol`` is the symbol file the part's name resolved to, or ``None`` when
    none was found or it could not be read. ``unreadable`` is the first pin line
    of a found symbol that does not parse. Such a symbol is unusable, as it is to
    the schematic editor, so the part then has no pins and no boxes either.
    """

    record: Symbol
    #: The 1-based line of the part's ``SYMBOL`` record.
    line: int
    symbol: FoundSymbol | None
    unreadable: str = ""
    pins: tuple[PlacedPin, ...] = ()
    #: The part's extent including its pins, and the extent of its drawing alone.
    box: BBox | None = None
    body: BBox | None = None

    @property
    def reference(self) -> str:
        """The instance name, or ``""`` when the part has none."""
        return self.attribute("InstName") or ""

    @property
    def symbol_name(self) -> str:
        """The symbol name as the sheet spells it."""
        return self.record.symbol

    @property
    def at(self) -> Point:
        return (self.record.x, self.record.y)

    @property
    def rotation(self) -> str:
        return self.record.rotation

    @property
    def usable(self) -> bool:
        """True when the symbol was found and every pin line of it parses."""
        return self.symbol is not None and not self.unreadable

    def attribute(self, name: str) -> str | None:
        """The value of attribute ``name``; the last one when the part repeats it."""
        found = [value for key, value in self.record.attrs if key == name]
        return found[-1] if found else None


@dataclass(frozen=True)
class Sheet:
    """An ``.asc`` sheet: its parsed document and the parts it places, in file order."""

    document: AscDocument
    parts: tuple[SheetPart, ...]

    @property
    def wires(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple((w.x1, w.y1, w.x2, w.y2) for w in self.document.wires)

    @property
    def labels(self) -> tuple[tuple[int, int, str], ...]:
        """Each net label and ground flag as ``(x, y, name)``."""
        return tuple((f.x, f.y, f.name) for f in self.document.flags)

    def parts_named(self, reference: str) -> tuple[SheetPart, ...]:
        """Every part with instance name ``reference``; more than one when the sheet repeats it."""
        return tuple(part for part in self.parts if part.reference == reference)

    def partition(self) -> NetPartition:
        """The sheet's nets, grouped by ``connectivity.partition``."""
        return partition(
            [
                ((pin.x, pin.y), (part.reference, pin.name))
                for part in self.parts
                for pin in part.pins
            ],
            [((x, y), name) for x, y, name in self.labels],
            [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in self.wires],
        )


def _placed_box(box: BBox, x: int, y: int, rotation: str) -> BBox:
    corners = [(box.x1, box.y1), (box.x2, box.y1), (box.x1, box.y2), (box.x2, box.y2)]
    placed = BBox.from_points(
        [(x + dx, y + dy) for dx, dy in (_apply_rotation(cx, cy, rotation) for cx, cy in corners)]
    )
    assert placed is not None  # four corners are never empty
    return placed


def _placed(record: Symbol, line: int, found: FoundSymbol | None) -> SheetPart:
    if found is None:
        return SheetPart(record, line, None)
    if found.file.unread_pins:
        return SheetPart(record, line, found, unreadable=found.file.unread_pins[0])
    info = SymbolInfo(name=found.name, pins=found.file.pins, bbox=found.file.bbox)
    geometry = compute_placed_geometry(info, record.x, record.y, record.rotation)
    pins = tuple(
        PlacedPin(pin["name"], pin["order"], pin["x"], pin["y"], pin["dir"])
        for pin in geometry["pins"]
    )
    body = found.file.body
    return SheetPart(
        record,
        line,
        found,
        pins=pins,
        box=_placed_box(found.file.bbox, record.x, record.y, record.rotation),
        body=_placed_box(body, record.x, record.y, record.rotation) if body is not None else None,
    )


def sheet_of(document: AscDocument, resolver: SymbolResolver) -> Sheet:
    """The sheet ``document`` describes, with each part's symbol found by ``resolver``."""
    parts: list[SheetPart] = []
    line = 1
    for record in document.records:
        if isinstance(record, Symbol):
            parts.append(_placed(record, line, resolver.load(record.symbol)))
        line += len(record.source or ())
    return Sheet(document, tuple(parts))


def read_sheet_text(text: str, resolver: SymbolResolver) -> Sheet:
    """The sheet whose ``.asc`` text, already decoded, is ``text``."""
    return sheet_of(parse_asc_text(text), resolver)


def read_sheet(data: bytes, resolver: SymbolResolver) -> Sheet:
    """The sheet whose ``.asc`` bytes are ``data``.

    Raises ``NetlistError`` only for bytes that are not text (``parse_asc``).
    """
    return sheet_of(parse_asc(data), resolver)
