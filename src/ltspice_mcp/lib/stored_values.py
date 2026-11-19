"""Which recorded LTspice builds read a stored value as another number than the server.

The server reads a deck's bytes once (``encoding.decode_spice_bytes_with_encoding``)
and reads a micro sign (``µ`` or ``μ``) at a value's scale suffix as micro: a
run through the server spells it ``u`` before the simulator sees it, so that
is the number the run uses. A person who opens the same file in LTspice gets
whatever that build makes of the stored bytes, which the recordings show
differs by build and by encoding (``deck-encoding`` in ``inputs/cases.toml``):

- LTspice 26 reads a micro sign stored as cp1252, UTF-8 (with and without a
  byte order mark) or UTF-16, and a Greek mu stored as UTF-8, as micro: the
  same text the server reads.
- LTspice XVII reads a micro sign stored as cp1252 or UTF-16 as micro, but a
  micro sign or a Greek mu stored as UTF-8 (with or without a mark) as the
  two cp1252 characters of its bytes (``1Âµ``), neither of them a scale, so
  it takes the bare number (``deck/micro_utf8``, ``deck/micro_utf8_bom``,
  ``deck/greek_mu_utf8``).

This module is the one answer to "which recorded builds read this stored
value as another number than the server does". It holds no tool's rules: a
reply that gives a number or a comparison read from a stored deck asks it
(``readings_of_cards``, ``readings_of_file``) and states each reading it
returns (``OtherReading.statement``). An encoding or a sign no recording shows
for a build is returned as not established for that build, never assumed; an
absent reading means the build reads the server's text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.encoding import decode_spice_bytes_with_encoding, decode_windows_1252
from ltspice_mcp.lib.format import fold_micro_sign, parse_spice_value
from ltspice_mcp.lib.reply_cuts import CUT, NEVER
from ltspice_mcp.lib.spice_lex import LexResult, SpiceCard, lex_deck
from ltspice_mcp.lib.spice_lex_ops import ValueSuffixSite, value_suffix_sites
from ltspice_mcp.lib.spice_lex_views import read_instance

#: The code every statement of a reading carries.
CODE = "value_read_otherwise"

#: The recorded builds, by the name a statement gives them.
BUILDS: tuple[str, ...] = ("LTspice 26", "LTspice XVII")

#: The recorded build that reads an 8-bit deck as cp1252: the one an LTspice
#: before 24 (``simulator_build.is_cp1252_ltspice_build``) is read as.
CP1252_BUILD = "LTspice XVII"

#: How a build decodes a sign stored in a codec: ``same`` (the text the
#: server reads), or ``cp1252`` (each byte of the stored UTF-8 as cp1252's
#: character). A pair not listed is not established for that build. The
#: codec is the name ``decode_spice_bytes_with_encoding`` gives; UTF-16 with
#: and without a mark both decode as ``utf-16-le``.
_Decoding = Literal["same", "cp1252"]
_RECORDED: dict[tuple[str, str], dict[str, tuple[_Decoding, tuple[str, ...]]]] = {
    ("µ", "cp1252"): {
        "LTspice 26": ("same", ("deck/micro_cp1252",)),
        "LTspice XVII": ("same", ("deck/micro_cp1252",)),
    },
    ("µ", "utf-8"): {
        "LTspice 26": ("same", ("deck/micro_utf8",)),
        "LTspice XVII": ("cp1252", ("deck/micro_utf8",)),
    },
    ("µ", "utf-8-sig"): {
        "LTspice 26": ("same", ("deck/micro_utf8_bom",)),
        "LTspice XVII": ("cp1252", ("deck/micro_utf8_bom",)),
    },
    ("µ", "utf-16-le"): {
        "LTspice 26": ("same", ("deck/micro_utf16le_bom", "deck/micro_utf16le")),
        "LTspice XVII": ("same", ("deck/micro_utf16le_bom", "deck/micro_utf16le")),
    },
    ("μ", "utf-8"): {
        "LTspice 26": ("same", ("deck/greek_mu_utf8",)),
        "LTspice XVII": ("cp1252", ("deck/greek_mu_utf8",)),
    },
}

#: The recordings that show the number a build takes from a value whose
#: suffix it reads as letters that are no scale: the bare number
#: (``deck/micro_utf8`` for the micro sign read as ``Âµ``, ``deck/suffix_unknown``
#: for letters after a number). Both are instance values outside braces.
_BARE_NUMBER_EVIDENCE = ("deck/micro_utf8", "deck/suffix_unknown")


@dataclass(frozen=True)
class OtherReading:
    """One stored value one recorded build reads otherwise than the server,
    or whose reading by that build no recording establishes.

    ``build_reads`` is the text the build reads in place of ``value`` and
    ``build_number`` the number it takes; either is None where no recording
    shows it. ``evidence`` names the recordings that show what is stated.
    """

    file: str | None
    line: int
    value: str
    encoding: str
    server_number: float | None
    build: str
    build_reads: str | None
    build_number: float | None
    evidence: tuple[str, ...]
    #: Where on its card the value's sign sits, which tells two values of
    #: one line apart.
    offset: int = 0

    @property
    def established(self) -> bool:
        return self.build_number is not None

    def statement(self) -> dict[str, Any]:
        """The reading as a reply states it, under :data:`CODE`."""
        stated: dict[str, Any] = {
            "code": CODE,
            "value": self.value,
            "line": self.line,
            "encoding": self.encoding,
            "build": self.build,
            "established": self.established,
            "evidence": list(self.evidence),
        }
        if self.file is not None:
            stated["file"] = self.file
        if self.server_number is not None:
            stated["server_number"] = self.server_number
        if self.build_reads is not None:
            stated["build_reads"] = self.build_reads
        if self.build_number is not None:
            stated["build_number"] = self.build_number
        stated["detail"] = self._detail()
        return stated

    def _detail(self) -> str:
        server = (
            " (the server reads no number from it)"
            if self.server_number is None
            else f" (the server reads {self.server_number:g})"
        )
        ascii_spelling = fold_micro_sign(self.value)
        if self.build_number is not None:
            return (
                f"{self.build} reads the stored bytes of {self.value}{server} in this "
                f"{self.encoding} file as {self.build_reads!s} and takes "
                f"{self.build_number:g}. Write {ascii_spelling}: 'u' is micro to every build."
            )
        if self.build_reads is not None:
            return (
                f"{self.build} reads the stored bytes of {self.value}{server} in this "
                f"{self.encoding} file as {self.build_reads}; no recording shows the number "
                f"it takes there. Write {ascii_spelling}: 'u' is micro to every build."
            )
        return (
            f"No recording shows how {self.build} reads {self.value}{server} stored as "
            f"{self.encoding}, so the number it takes is not established. Write "
            f"{ascii_spelling}: 'u' is micro to every build."
        )


#: The JSON Schema of one statement, for a reply's output schema.
STATEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "A value read from a stored file that a recorded LTspice build reads as "
        "another number than the server does, or whose reading by that build no "
        "recording establishes (established false)."
    ),
    "properties": {
        "code": {"type": "string", "const": CODE},
        "value": {"type": "string", "description": "The value as the server reads it."},
        "file": {"type": "string"},
        "line": {"type": "integer"},
        "encoding": {"type": "string", "description": "The codec the server decoded the file as."},
        "server_number": {"type": "number"},
        "build": {"type": "string"},
        "build_reads": {"type": "string", "description": "The text the build reads instead."},
        "build_number": {"type": "number", "description": "The number the build takes."},
        "established": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
        "detail": {"type": "string"},
    },
    "required": ["code", "value", "line", "encoding", "build", "established", "detail"],
}

#: A reply's list of statements, sent whole: a statement left out would read
#: as a value every build reads alike.
STATEMENTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    CUT: NEVER,
    "description": (
        "Each value read from a stored file that a recorded LTspice build reads as another "
        "number than this reply gives. Absent when every value read is read alike."
    ),
    "items": STATEMENT_SCHEMA,
}


#: The elements whose own value the bare-number recordings show.
_VALUE_ELEMENTS = frozenset("RCLVI")


def _is_the_value(site: ValueSuffixSite) -> bool:
    """The site is an instance's own value, written alone outside braces: the
    position the recordings show a build's bare number for."""
    if site.card.kind != "instance" or site.in_braces or site.after_equals:
        return False
    # The recordings show a two-terminal part's value; the word after a
    # transistor's model is its area, a position none shows.
    if (site.card.name or "")[:1].upper() not in _VALUE_ELEMENTS:
        return False
    line = read_instance(site.card)
    return line is not None and line.value == site.token


def _server_number(site: ValueSuffixSite) -> float | None:
    try:
        return parse_spice_value(fold_micro_sign(site.token))
    except ValueError:
        return None


def _reading(
    site: ValueSuffixSite, encoding: str, build: str, file: str | None
) -> OtherReading | None:
    common = {
        "file": file,
        "line": site.line,
        "value": site.token,
        "encoding": encoding,
        "server_number": _server_number(site),
        "build": build,
        "offset": site.offset,
    }
    recorded = _RECORDED.get((site.suffix, encoding), {}).get(build)
    if recorded is None:
        settle = (
            f"a deck holding {site.suffix} stored as {encoding}, recorded on {build}",
        )
        return OtherReading(build_reads=None, build_number=None, evidence=settle, **common)
    decoding, evidence = recorded
    if decoding == "same":
        return None
    reads = decode_windows_1252(site.token.encode("utf-8"))
    if not _is_the_value(site):
        # The bare number is recorded for an instance's own value outside
        # braces; in an instance parameter, a source function, an expression
        # or a directive no recording shows what the build makes of the letters.
        return OtherReading(build_reads=reads, build_number=None, evidence=evidence, **common)
    return OtherReading(
        build_reads=reads,
        build_number=float(site.number),
        evidence=tuple(dict.fromkeys((*evidence, *_BARE_NUMBER_EVIDENCE))),
        **common,
    )


def readings_of_sites(
    sites: Iterable[ValueSuffixSite], encoding: str, *, file: str | Path | None = None
) -> tuple[OtherReading, ...]:
    """Every reading of a micro sign among ``sites`` that a recorded build
    makes otherwise than the server, or that no recording establishes for a
    build, in a file the server decoded as ``encoding``."""
    where = None if file is None else str(file)
    found: list[OtherReading] = []
    for site in sites:
        if not site.micro:
            continue
        for build in BUILDS:
            reading = _reading(site, encoding, build, where)
            if reading is not None:
                found.append(reading)
    return tuple(found)


def readings_of_cards(
    cards: Iterable[SpiceCard], encoding: str, *, file: str | Path | None = None
) -> tuple[OtherReading, ...]:
    """:func:`readings_of_sites` for every value in ``cards``: the cards a
    simulator reads (``lex_deck``: a root deck's title dropped), from a file
    the server decoded as ``encoding``."""
    return readings_of_sites(value_suffix_sites(cards), encoding, file=file)


@dataclass(frozen=True)
class StoredDeck:
    """A stored deck as the server reads it: its text, the codec that decoded
    it, the cards a simulator reads, and the readings of its values a
    recorded build makes otherwise (:func:`readings_of_cards`)."""

    text: str
    encoding: str
    lexed: LexResult
    readings: tuple[OtherReading, ...]


def read_stored_bytes(
    content: bytes, *, root: bool, file: str | Path | None = None
) -> StoredDeck:
    """A stored file's bytes read as a deck (``root``, whose first line is its
    title) or as an included file, with the readings of its values."""
    text, encoding = decode_spice_bytes_with_encoding(content)
    lexed = lex_deck(text, title=root)
    return StoredDeck(text, encoding, lexed, readings_of_cards(lexed.cards, encoding, file=file))


def read_stored_deck(path: Path, *, root: bool = True) -> StoredDeck:
    """:func:`read_stored_bytes` for the file at ``path``."""
    return read_stored_bytes(path.read_bytes(), root=root, file=path)


def readings_of_bytes(
    content: bytes, *, root: bool, file: str | Path | None = None
) -> tuple[OtherReading, ...]:
    """The readings alone of :func:`read_stored_bytes`."""
    return read_stored_bytes(content, root=root, file=file).readings


def readings_of_file(path: Path, *, root: bool = True) -> tuple[OtherReading, ...]:
    """The readings alone of :func:`read_stored_deck`."""
    return read_stored_deck(path, root=root).readings


def statements(readings: Sequence[OtherReading]) -> list[dict[str, Any]]:
    """Each reading as a reply states it, in order, once."""
    seen: set[tuple[Any, ...]] = set()
    stated: list[dict[str, Any]] = []
    for reading in readings:
        key = (reading.file, reading.line, reading.offset, reading.value, reading.build)
        if key not in seen:
            seen.add(key)
            stated.append(reading.statement())
    return stated


#: Why a sheet read as the server's export of it is not checked: a build's
#: reading of the export's bytes is not its reading of the sheet's, and no
#: reply here reads the sheet's own value bytes.
SHEET_NOT_CHECKED = (
    "read as the server's export of the sheet, whose micro signs a recorded build "
    "reads otherwise; how a build reads the values stored in the sheet itself was "
    "not checked. Write 'u' for micro in the sheet: every build reads it alike."
)

#: A reply's list of stored files whose values were not checked, sent whole:
#: one entry per sheet a reply reads as its export.
NOT_CHECKED_SCHEMA: dict[str, Any] = {
    "type": "array",
    CUT: NEVER,
    "description": (
        "Each sheet read as the server's export of it whose values were not checked "
        "for another build's reading; absent when every file read was checked."
    ),
    "items": {
        "type": "object",
        "properties": {"file": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["file", "reason"],
    },
}


def holds_micro_sign(text: str) -> bool:
    """Whether ``text``, a deck with no stored bytes here (a sheet's export
    handed over as text), holds a micro sign at a value's suffix: a value
    whose reading by another build this module cannot check without the bytes."""
    return any(site.micro for site in value_suffix_sites(lex_deck(text).cards))
