"""An LTspice schematic (``.asc``) as records, read and written without loss.

``parse_asc`` turns a sheet's bytes into an :class:`AscDocument`: its records
in file order, with the encoding, byte order mark and line ending it was
written in. ``AscDocument.to_bytes`` gives the bytes back. A sheet nothing
changed comes back byte for byte, and one that was edited differs only in the
records the edit touched.

Two rules make that hold:

- **A record keeps the lines it was read from**, and those lines are written
  back for as long as they still read as that record. Only a changed or new
  record is formatted. How a record was changed does not matter: the lines are
  read again at write time and compared with the record, so a stale copy of
  them is never written.
- **A line with no record type here is kept as text** (:class:`Opaque`): a
  blank line, a record LTspice writes that this module has no type for, a line
  of a known kind that does not read as one. Nothing is refused and nothing is
  dropped. ``AscDocument.opaque`` lists them, so a caller that must understand
  every wire and part before it edits can tell when it does not.

A formatted record is the line spicelib's ``AscEditor`` writes for it, and the
line LTspice 26 and LTspice XVII each write for it when they save a sheet (the
``save`` cases; ``docs/TESTING.md``, "Recorded LTspice behaviour"). A new
record goes after the last record of its kind, so a sheet built from blank
comes out in the order both builds save one in: wires, flags, symbols, text,
drawn lines, then the other shapes. New text is written with LF line endings,
as both builds write every sheet, unless the sheet uses another.

This module reads no symbol file and knows no geometry beyond the numbers on
each line. Nothing in it depends on the event loop. The schematic editor does
not write through it yet: ``edit_schematic`` still commits through spicelib's
editor (``docs/design/schematic_engine.md``, section 6).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, Self, cast

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.encoding import (
    bom_of,
    decode_spice_bytes_strictly,
    encode_spice_text_strictly,
)

#: The eight placements of a symbol, as a ``SYMBOL`` line names them.
ROTATIONS: tuple[str, ...] = ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270")

#: The alignments a ``TEXT`` or ``WINDOW`` line names; a leading ``V`` turns the
#: text a quarter turn.
_ALIGNMENTS = frozenset({"left", "right", "top", "bottom", "center", "invisible"})

ShapeKind = Literal["LINE", "RECTANGLE", "CIRCLE", "ARC"]
_SHAPE_COORDS: dict[str, int] = {"LINE": 4, "RECTANGLE": 4, "CIRCLE": 4, "ARC": 8}

_INT = re.compile(r"-?[0-9]+")
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")
_TEXT = re.compile(
    r"\s*TEXT\s+(-?[0-9]+)\s+(-?[0-9]+)\s+(\S+)\s+([0-9]+)\s*([!;])(.*)",
    re.DOTALL,
)


@dataclass(frozen=True)
class _Record:
    #: The lines the record was read from, line endings included; ``None`` for
    #: one made in memory. Not part of the record's value.
    source: tuple[str, ...] | None = field(default=None, compare=False, repr=False, kw_only=True)

    _rank: ClassVar[int | None] = None

    @property
    def rank(self) -> int | None:
        """Where the record's kind stands in a formatted sheet: the header, wires,
        flags, symbols, text, drawn lines, then the other shapes. ``None`` for a
        line kept as text, which has no place of its own."""
        return self._rank

    def changed(self, **fields: Any) -> Self:
        """A copy with ``fields`` replaced, no longer tied to the lines it was read from."""
        return replace(self, source=None, **fields)

    def format(self) -> tuple[str, ...]:
        """The record as lines, without line endings."""
        raise NotImplementedError


@dataclass(frozen=True)
class Version(_Record):
    """``Version 4``: the file format the sheet declares."""

    _rank: ClassVar[int | None] = 0

    version: str

    def format(self) -> tuple[str, ...]:
        return (f"Version {self.version}",)


@dataclass(frozen=True)
class SheetHeader(_Record):
    """``SHEET 1 880 680``: the sheet number and its extent."""

    _rank: ClassVar[int | None] = 0

    number: int
    width: int
    height: int

    def format(self) -> tuple[str, ...]:
        return (f"SHEET {self.number} {self.width} {self.height}",)


@dataclass(frozen=True)
class Wire(_Record):
    """``WIRE x1 y1 x2 y2``."""

    _rank: ClassVar[int | None] = 1

    x1: int
    y1: int
    x2: int
    y2: int

    def format(self) -> tuple[str, ...]:
        return (f"WIRE {self.x1} {self.y1} {self.x2} {self.y2}",)


@dataclass(frozen=True)
class Flag(_Record):
    """``FLAG x y name``: a net label, or ground when the name is ``0``.

    ``port`` is the direction on the ``IOPIN`` line that follows the flag of a
    hierarchical port (``In``, ``Out``, ``BiDir``), or ``None`` for a plain label.
    """

    _rank: ClassVar[int | None] = 2

    x: int
    y: int
    name: str
    port: str | None = None

    def format(self) -> tuple[str, ...]:
        flag = f"FLAG {self.x} {self.y} {self.name}"
        if self.port is None:
            return (flag,)
        return (flag, f"IOPIN {self.x} {self.y} {self.port}")


@dataclass(frozen=True)
class Window:
    """``WINDOW n x y align size``: where one of a symbol's attributes is drawn.

    A line of a :class:`Symbol`, not a record of its own. ``number`` picks the
    attribute (0 the instance name, 3 the value); the position is in the
    symbol's own coordinates.
    """

    number: int
    x: int
    y: int
    align: str
    size: int

    def format(self) -> str:
        return f"WINDOW {self.number} {self.x} {self.y} {self.align} {self.size}"


@dataclass(frozen=True)
class Symbol(_Record):
    """A placed part: its ``SYMBOL`` line and the ``WINDOW`` and ``SYMATTR`` lines under it.

    ``attrs`` keeps the attributes in file order as ``(name, value)`` pairs. A
    formatted symbol writes its windows, then ``InstName``, then the rest in
    that order.

    ``unread`` holds each line under the symbol that did not read as a window
    or an attribute, as it was written. The part is still a part: it has its
    place and its name, an edit can address it, and those lines are written
    back after the ones that read.
    """

    _rank: ClassVar[int | None] = 3

    symbol: str
    x: int
    y: int
    rotation: str
    windows: tuple[Window, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()
    unread: tuple[str, ...] = ()

    @property
    def reference(self) -> str:
        """The instance name, or ``""`` for a symbol that has none."""
        return self.attr("InstName") or ""

    def attr(self, name: str) -> str | None:
        """The value of attribute ``name``, or ``None`` when the symbol has no such line."""
        return next((value for key, value in self.attrs if key == name), None)

    def with_attr(self, name: str, value: str) -> Symbol:
        """A copy with attribute ``name`` set: replaced where it is, else added last."""
        if self.attr(name) is None:
            return self.changed(attrs=(*self.attrs, (name, value)))
        return self.changed(
            attrs=tuple((key, value if key == name else old) for key, old in self.attrs)
        )

    def without_attr(self, name: str) -> Symbol:
        """A copy with no ``name`` attribute line."""
        return self.changed(attrs=tuple(pair for pair in self.attrs if pair[0] != name))

    def format(self) -> tuple[str, ...]:
        lines = [f"SYMBOL {self.symbol} {self.x} {self.y} {self.rotation}"]
        lines += [window.format() for window in self.windows]
        ordered = sorted(self.attrs, key=lambda pair: pair[0] != "InstName")
        # An attribute with no value is the keyword and name alone, with no
        # trailing space: the form this module reads back as an empty value.
        lines += [f"SYMATTR {name} {value}".rstrip() for name, value in ordered]
        lines += self.unread
        return tuple(lines)


@dataclass(frozen=True)
class Text(_Record):
    """``TEXT x y align size !body``: a directive (``!``) or a comment (``;``).

    ``body`` is the text as stored, with LTspice's ``\\n`` for a line break.
    """

    _rank: ClassVar[int | None] = 4

    x: int
    y: int
    align: str
    size: int
    kind: Literal["!", ";"]
    body: str

    @property
    def is_directive(self) -> bool:
        return self.kind == "!"

    def format(self) -> tuple[str, ...]:
        return (f"TEXT {self.x} {self.y} {self.align} {self.size} {self.kind}{self.body}",)


@dataclass(frozen=True)
class Shape(_Record):
    """A drawn line, rectangle, circle or arc: annotation, with no electrical meaning.

    ``coords`` is the record's numbers in order (two points; four for an arc).
    ``style`` is the trailing line-style number, absent for a solid line.
    """

    kind: ShapeKind
    weight: str
    coords: tuple[int, ...]
    style: int | None = None

    @property
    def rank(self) -> int:
        return 5 if self.kind == "LINE" else 6

    def format(self) -> tuple[str, ...]:
        numbers = " ".join(str(c) for c in self.coords)
        tail = "" if self.style is None else f" {self.style}"
        return (f"{self.kind} {self.weight} {numbers}{tail}",)


@dataclass(frozen=True)
class Opaque(_Record):
    """One line kept as text because no record type here reads it."""

    text: str

    @property
    def keyword(self) -> str:
        """The line's first word (``DATAFLAG``, ``BUSTAP``), or ``""`` for a blank line."""
        return _keyword(self.text)

    def format(self) -> tuple[str, ...]:
        return (self.text,)


Record = Version | SheetHeader | Wire | Flag | Symbol | Text | Shape | Opaque


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _body(line: str) -> str:
    """``line`` without its line ending."""
    return line.rstrip("\r\n")


def _keyword(line: str) -> str:
    words = line.split(None, 1)
    return words[0] if words else ""


def _ints(words: Sequence[str]) -> tuple[int, ...] | None:
    """``words`` as integers, or ``None`` unless every one is a plain decimal."""
    if not all(_INT.fullmatch(word) for word in words):
        return None
    return tuple(int(word) for word in words)


def _alignment(word: str) -> bool:
    return word.lower().removeprefix("v") in _ALIGNMENTS


def _read_version(body: str) -> Version | None:
    words = body.split()
    return Version(words[1]) if len(words) == 2 else None


def _read_sheet(body: str) -> SheetHeader | None:
    words = body.split()
    numbers = _ints(words[1:]) if len(words) == 4 else None
    return SheetHeader(*numbers) if numbers is not None else None


def _read_wire(body: str) -> Wire | None:
    words = body.split()
    numbers = _ints(words[1:]) if len(words) == 5 else None
    return Wire(*numbers) if numbers is not None else None


def _read_text(body: str) -> Text | None:
    match = _TEXT.fullmatch(body)
    if match is None or not _alignment(match[3]):
        return None
    kind: Literal["!", ";"] = "!" if match[5] == "!" else ";"
    return Text(int(match[1]), int(match[2]), match[3], int(match[4]), kind, match[6])


def _read_shape(body: str) -> Shape | None:
    words = body.split()
    kind = words[0]
    count = _SHAPE_COORDS[kind]
    if len(words) not in (2 + count, 3 + count):
        return None
    numbers = _ints(words[2:])
    if numbers is None:
        return None
    style = numbers[count] if len(numbers) > count else None
    return Shape(cast(ShapeKind, kind), words[1], numbers[:count], style)


def _read_window(body: str) -> Window | None:
    words = body.split()
    if len(words) != 6 or not _alignment(words[4]):
        return None
    numbers = _ints([words[1], words[2], words[3], words[5]])
    if numbers is None:
        return None
    return Window(numbers[0], numbers[1], numbers[2], words[4], numbers[3])


def _read_flag(lines: Sequence[str]) -> Flag | None:
    """A flag from its ``FLAG`` line, with the ``IOPIN`` line after it when there is one."""
    words = _body(lines[0]).split(None, 3)
    at = _ints(words[1:3]) if len(words) == 4 else None
    if at is None:
        return None
    port: str | None = None
    if len(lines) == 2:
        pin = _body(lines[1]).split()
        if len(pin) != 4 or _ints(pin[1:3]) != at:
            return None
        port = pin[3]
    return Flag(at[0], at[1], words[3].strip(), port)


def _read_symbol(lines: Sequence[str]) -> Symbol | None:
    """A symbol from its ``SYMBOL`` line and the lines under it.

    ``None`` when the ``SYMBOL`` line itself does not read: without a place and
    an orientation there is no part to speak of. A line under it that does not
    read is carried on the symbol as written.
    """
    words = _body(lines[0]).split()
    if len(words) < 5 or words[-1] not in ROTATIONS:
        return None
    at = _ints(words[-3:-1])
    if at is None:
        return None
    windows: list[Window] = []
    attrs: list[tuple[str, str]] = []
    unread: list[str] = []
    for line in lines[1:]:
        body = _body(line)
        if _keyword(body) == "WINDOW":
            window = _read_window(body)
            if window is None:
                unread.append(body)
            else:
                windows.append(window)
            continue
        # LTspice writes an attribute with no value as the keyword and name
        # alone, which reads here as the empty value.
        parts = body.split(None, 2)
        if len(parts) < 2:
            unread.append(body)
        else:
            attrs.append((parts[1], parts[2].strip() if len(parts) > 2 else ""))
    return Symbol(
        " ".join(words[1:-3]),
        at[0],
        at[1],
        words[-1],
        tuple(windows),
        tuple(attrs),
        tuple(unread),
    )


_ONE_LINE: dict[str, Callable[[str], Record | None]] = {
    "Version": _read_version,
    "SHEET": _read_sheet,
    "WIRE": _read_wire,
    "TEXT": _read_text,
    **dict.fromkeys(_SHAPE_COORDS, _read_shape),
}


def _read(lines: Sequence[str]) -> list[Record]:
    """``lines`` (with their line endings) as records, each holding the lines it came from."""
    records: list[Record] = []
    index = 0
    while index < len(lines):
        keyword = _keyword(_body(lines[index]))
        end = index + 1
        record: Record | None = None
        if keyword == "SYMBOL":
            while end < len(lines) and _keyword(_body(lines[end])) in ("WINDOW", "SYMATTR"):
                end += 1
            record = _read_symbol(lines[index:end])
        elif keyword == "FLAG":
            if end < len(lines) and _keyword(_body(lines[end])) == "IOPIN":
                record = _read_flag(lines[index : end + 1])
            if record is not None:
                end += 1
            else:
                record = _read_flag(lines[index:end])
        elif keyword in _ONE_LINE:
            record = _ONE_LINE[keyword](_body(lines[index]))
        block = tuple(lines[index:end])
        if record is None:
            records.extend(Opaque(_body(line), source=(line,)) for line in block)
        else:
            records.append(replace(record, source=block))
        index = end
    return records


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AscDocument:
    """A sheet's records in file order, and how its bytes spell them.

    ``encoding`` is the codec name the decoder gave, ``bom`` the byte order
    mark the file began with, and ``newline`` the line ending a formatted
    record is written with (the sheet's most common one). The document is a
    value: every change returns a new one and leaves this one as it was.
    """

    records: tuple[Record, ...]
    encoding: str = "utf-8"
    bom: bytes = b""
    newline: str = "\n"

    @classmethod
    def blank(cls, width: int = 880, height: int = 680) -> AscDocument:
        """An empty sheet (880x680 is LTspice's default extent)."""
        return cls((Version("4"), SheetHeader(1, width, height)))

    # -- reading ----------------------------------------------------------

    @property
    def version(self) -> str | None:
        return next((r.version for r in self.records if isinstance(r, Version)), None)

    @property
    def sheet(self) -> SheetHeader | None:
        return next((r for r in self.records if isinstance(r, SheetHeader)), None)

    @property
    def wires(self) -> tuple[Wire, ...]:
        return tuple(r for r in self.records if isinstance(r, Wire))

    @property
    def flags(self) -> tuple[Flag, ...]:
        return tuple(r for r in self.records if isinstance(r, Flag))

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return tuple(r for r in self.records if isinstance(r, Symbol))

    @property
    def texts(self) -> tuple[Text, ...]:
        return tuple(r for r in self.records if isinstance(r, Text))

    @property
    def shapes(self) -> tuple[Shape, ...]:
        return tuple(r for r in self.records if isinstance(r, Shape))

    @property
    def opaque(self) -> tuple[Opaque, ...]:
        """The lines no record type here reads, blank lines included."""
        return tuple(r for r in self.records if isinstance(r, Opaque))

    def symbol(self, reference: str) -> Symbol | None:
        """The first symbol whose instance name is ``reference``."""
        return next((s for s in self.symbols if s.reference == reference), None)

    # -- changing ---------------------------------------------------------

    def added(self, *records: Record) -> AscDocument:
        """A copy with each of ``records`` placed after the last record of its kind.

        A kind the sheet does not have yet goes after the kinds that precede it
        in a formatted sheet, and a line kept as text goes last.
        """
        out = list(self.records)
        for record in records:
            rank = record.rank
            at = len(out)
            if rank is not None:
                # After the last record whose kind stands at or before this one's.
                at = next(
                    (
                        index + 1
                        for index in range(len(out) - 1, -1, -1)
                        if (there := out[index].rank) is not None and there <= rank
                    ),
                    0,
                )
            out.insert(at, record)
        return replace(self, records=tuple(out))

    def replaced(self, old: Record, new: Record) -> AscDocument:
        """A copy with ``new`` where ``old`` is. ``old`` is one of this document's own records."""
        at = self._index(old)
        return replace(self, records=(*self.records[:at], new, *self.records[at + 1 :]))

    def removed(self, *records: Record) -> AscDocument:
        """A copy without ``records``, each one of this document's own records."""
        gone = {self._index(record) for record in records}
        return replace(
            self, records=tuple(r for index, r in enumerate(self.records) if index not in gone)
        )

    def _index(self, record: Record) -> int:
        # By identity: two wires drawn twice are equal as values and still two
        # records, and a caller means the one it was handed.
        for index, existing in enumerate(self.records):
            if existing is record:
                return index
        raise ValueError(f"{record!r} is not a record of this document")

    # -- writing ----------------------------------------------------------

    def to_text(self) -> str:
        """The sheet's text: each record's own lines where they still read as it, else formatted."""
        lines: list[str] = []
        for record in self.records:
            # Read again here, so that however a record was changed, lines that
            # no longer describe it are never written.
            if record.source is not None and _read(record.source) == [record]:
                lines.extend(record.source)
            else:
                lines.extend(line + self.newline for line in record.format())
        # A sheet whose last line had no line ending keeps it that way; a line
        # that is no longer last needs one.
        for index, line in enumerate(lines[:-1]):
            if not line.endswith(("\n", "\r")):
                lines[index] = line + self.newline
        return "".join(lines)

    def to_bytes(self) -> bytes:
        """The sheet in its own encoding, with the byte order mark it was read with.

        Raises ``NetlistError`` naming the character when an edit added one the
        encoding cannot spell; ``with_encoding`` picks another.
        """
        text = self.to_text()
        try:
            return self.bom + encode_spice_text_strictly(text, self.encoding)
        except UnicodeEncodeError as exc:
            raise NetlistError(
                f"the sheet is {self.encoding} text, which has no spelling for "
                f"{text[exc.start : exc.end]!r}"
            ) from exc

    def with_encoding(self, encoding: str, bom: bytes = b"") -> AscDocument:
        """A copy that is written in ``encoding``, a codec name a decoder here gives."""
        return replace(self, encoding=encoding, bom=bom)


def parse_asc(data: bytes) -> AscDocument:
    """Read a sheet's bytes. Raises ``NetlistError`` only for bytes that are not text.

    Every line becomes a record or is kept as :class:`Opaque` text, so a sheet
    this module only half understands still reads, and writes back unchanged.
    """
    try:
        text, encoding = decode_spice_bytes_strictly(data)
    except UnicodeError as exc:
        raise NetlistError(f"the sheet is not readable text: {exc}") from exc
    lines = _LINE.findall(text)
    endings = Counter(line[len(_body(line)) :] for line in lines)
    endings.pop("", None)
    newline = endings.most_common(1)[0][0] if endings else "\n"
    return AscDocument(tuple(_read(lines)), encoding=encoding, bom=bom_of(data), newline=newline)
