"""A schematic sheet as a read-only value: its records and the parts they place.

``read_sheet`` reads an ``.asc`` file's bytes with ``asc_document`` and finds
each part's symbol with a ``SymbolResolver``. A ``Sheet`` holds the parsed
document and, for each part, where its pins and boxes are. The scene draws it,
and the readers that do not edit a sheet (``inspect``) read it instead of
spicelib's editor.

Nothing here writes a file or needs the event loop.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.asc_document import AscDocument, Opaque, Record, Symbol, parse_asc_text
from ltspice_mcp.lib.connectivity import NetPartition, partition
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_geometry import (
    compute_placed_geometry,
    place_box,
    usable_symbol_info,
)
from ltspice_mcp.lib.symbol_library import FoundSymbol, SymbolResolver

Point = tuple[int, int]

# The one record a sheet holds that nothing draws or connects through: a data
# label's expression (recorded as export/data_flags: the export is the circuit
# without it). Any other line the records do not read is reported, a bus tap
# among them: LTspice draws one, though it connects nothing in the netlist
# (export/bus_tap).
NOTHING_TO_READ = frozenset({"DATAFLAG"})


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

    def as_dict(self) -> dict[str, object]:
        """The pin as ``compute_placed_geometry`` and the tools' replies give one."""
        return {
            "name": self.name,
            "order": self.order,
            "x": self.x,
            "y": self.y,
            "dir": self.direction,
        }


#: The attributes whose text LTspice writes onto a part's netlist line, in the
#: order it writes them, after the nodes and any SpiceModel (recorded:
#: ``export-attributes``). A ``KEY=VALUE`` in any of them is a parameter of
#: the part.
PARAMETER_ATTRIBUTES = ("Value", "Value2", "SpiceLine", "SpiceLine2")


@dataclass(frozen=True)
class SheetPart:
    """A placed part.

    ``symbol`` is the symbol file the part's name resolved to, or ``None`` when
    none was found. A found symbol with a ``problem`` is unusable, as it is to
    the schematic editor; the part then has no pins and no boxes.
    """

    record: Symbol
    symbol: FoundSymbol | None
    pins: tuple[PlacedPin, ...] = ()
    #: The part's extent including its pins, and the extent of its drawing alone.
    box: BBox | None = None
    body: BBox | None = None

    @property
    def reference(self) -> str:
        """The instance name, or ``""`` when the part has none."""
        return self.record.reference

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
        """True when the symbol was found and can be used."""
        return self.symbol is not None and not self.symbol.problem

    def attribute(self, name: str) -> str | None:
        """The value of attribute ``name``; the last one when the part repeats it."""
        return self.record.attr(name)

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        """Each of :data:`PARAMETER_ATTRIBUTES` the part gives, as ``(name, text)``."""
        return tuple(
            (name, text) for name in PARAMETER_ATTRIBUTES if (text := self.attribute(name))
        )


def _numbered(document: AscDocument) -> Iterator[tuple[int, Record]]:
    """Each record with the 1-based line it starts on."""
    line = 1
    for record in document.records:
        yield line, record
        line += len(record.source or ())


def find_pin(
    pins: Sequence[PlacedPin], reference: str, symbol: str, pin_name: str, problem: str = ""
) -> Point:
    """Where pin ``pin_name`` of part ``reference`` is.

    The name is matched against the pin names first, case-insensitively. Only
    when no pin carries it, and it is all digits, is it read as the pin's
    1-based SpiceOrder, the terminal number a netlist uses: some symbols name
    their pins '1', '2' in an order that need not be their SpiceOrder.
    ``problem`` says why the part has no usable symbol, when it has none.
    Raises ``NetlistError`` for that, and for a pin that is not there, naming
    the pins there are.
    """
    if problem:
        raise NetlistError(f"Cannot resolve pins for '{reference}': symbol '{symbol}' {problem}.")
    for pin in pins:
        if pin.name.upper() == pin_name.upper():
            return pin.x, pin.y
    # A pin with no SpiceOrder line reads as order 0, which is no ordinal.
    ordinal = int(pin_name) if pin_name.isascii() and pin_name.isdigit() else 0
    if ordinal:
        for pin in pins:
            if pin.order == ordinal:
                return pin.x, pin.y
    available = [f"{p.name} ({p.order})" if p.order else p.name for p in pins]
    raise NetlistError(
        f"Pin '{pin_name}' not found on {reference} ({symbol}). Available: "
        f"{', '.join(available)} (pin name, with its SpiceOrder in parentheses)."
    )


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

    def unread_lines(self) -> dict[str, list[int]]:
        """Each keyword of a line the records do not read, with the 1-based lines it is on.

        A line no record type reads, and a line under a part that does not read
        as a window or an attribute. ``NOTHING_TO_READ`` and blank lines are left
        out. What reads the sheet does not see what such a line holds, so it
        reports it.
        """
        unread: dict[str, list[int]] = {}
        for line, record in _numbered(self.document):
            if isinstance(record, Opaque):
                if record.keyword and record.keyword not in NOTHING_TO_READ:
                    unread.setdefault(record.keyword, []).append(line)
            elif isinstance(record, Symbol) and record.unread:
                pending = list(record.unread)
                for offset, text in enumerate((record.source or ())[1:], 1):
                    if pending and text.rstrip("\r\n") == pending[0]:
                        keyword = pending.pop(0).split(None, 1)[0]
                        unread.setdefault(keyword, []).append(line + offset)
        return unread

    def unread_notes(self, reader: str, one: str, many: str) -> list[str]:
        """A note for each keyword of a line the records do not read, as ``reader`` says it.

        ``one`` and ``many`` say what not reading it leaves out, for one line and
        for several.
        """
        notes: list[str] = []
        for keyword, lines in self.unread_lines().items():
            if len(lines) == 1:
                notes.append(
                    f"line {lines[0]}: a {keyword} record, which {reader} does not read; {one}"
                )
            else:
                notes.append(
                    f"{len(lines)} {keyword} records, the first at line {lines[0]}, which "
                    f"{reader} does not read; {many}"
                )
        return notes

    def parts_named(self, reference: str) -> tuple[SheetPart, ...]:
        """Every part with instance name ``reference``; more than one when the sheet repeats it."""
        return tuple(part for part in self.parts if part.reference == reference)

    def pin_at(self, spec: str) -> Point:
        """Where the pin ``spec`` (``'R1.A'``, or ``'X1.2'`` by SpiceOrder) is, by ``find_pin``.

        Raises ``NetlistError`` as ``find_pin`` does, for a part that is not
        there, and for a reference two parts share, naming where each is.
        """
        if "." not in spec:
            raise NetlistError(
                f"Invalid pin reference '{spec}'. Use 'Reference.Pin', the pin by "
                "name or 1-based SpiceOrder (e.g., 'M1.D', 'X1.2'), or 'net:name' "
                "(e.g., 'net:VDD')."
            )
        reference, pin_name = spec.rsplit(".", 1)
        named: list[SheetPart] = list(self.parts_named(reference))
        if not named:
            available = sorted({part.reference for part in self.parts if part.reference})
            raise NetlistError(
                f"Component '{reference}' not found. Available: {', '.join(available)}"
            )
        if len(named) > 1:
            places = ", ".join(f"({part.at[0]},{part.at[1]})" for part in named)
            raise NetlistError(
                f"'{reference}' names {len(named)} parts, placed at {places}, so a pin of "
                f"'{reference}' could be on either. Give the pin's coordinate instead."
            )
        part = named[0]
        problem = "not found" if part.symbol is None else part.symbol.problem
        return find_pin(part.pins, reference, part.symbol_name, pin_name, problem)

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


def _placed(record: Symbol, found: FoundSymbol | None) -> SheetPart:
    info = usable_symbol_info(found)
    if found is None or info is None:
        return SheetPart(record, found)
    geometry = compute_placed_geometry(info, record.x, record.y, record.rotation)
    pins = tuple(
        PlacedPin(pin["name"], pin["order"], pin["x"], pin["y"], pin["dir"])
        for pin in geometry["pins"]
    )
    box = geometry["bounding_box"]
    body = found.file.body
    return SheetPart(
        record,
        found,
        pins=pins,
        box=BBox.from_origin_size(box["x"], box["y"], box["width"], box["height"]),
        body=place_box(body, record.x, record.y, record.rotation) if body is not None else None,
    )


def sheet_of(document: AscDocument, resolver: SymbolResolver) -> Sheet:
    """The sheet ``document`` describes, with each part's symbol found by ``resolver``."""
    found: dict[str, FoundSymbol | None] = {}
    parts: list[SheetPart] = []
    for record in document.symbols:
        if record.symbol not in found:
            found[record.symbol] = resolver.load(record.symbol)
        parts.append(_placed(record, found[record.symbol]))
    return Sheet(document, tuple(parts))


def read_sheet_text(text: str, resolver: SymbolResolver) -> Sheet:
    """The sheet whose ``.asc`` text, already decoded, is ``text``."""
    return sheet_of(parse_asc_text(text), resolver)


def read_sheet(data: bytes, resolver: SymbolResolver) -> Sheet:
    """The sheet whose ``.asc`` bytes are ``data``, decoded as a sheet's text always is
    (``encoding.decode_spice_bytes``)."""
    return read_sheet_text(decode_spice_bytes(data), resolver)


def read_sheet_file(path: Path, resolver: SymbolResolver) -> tuple[Sheet, str]:
    """The sheet at ``path``, with the SHA-256 of the bytes it was read from.

    Its symbols are found with ``resolver``, the session's for that sheet
    (``SymbolLibrary.for_sheet``). One read, so the digest is of exactly what
    the sheet holds. Blocking: call it off the event loop.
    """
    data = path.read_bytes()
    return read_sheet(data, resolver), hashlib.sha256(data).hexdigest()
