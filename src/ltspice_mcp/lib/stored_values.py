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

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.encoding import (
    decode_spice_bytes_with_encoding,
    decode_windows_1252,
    latin1_reading,
)
from ltspice_mcp.lib.format import fold_micro_sign, parse_spice_value
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS, NEVER
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

    def example(self) -> dict[str, Any]:
        """The value as a summary's example names it."""
        named: dict[str, Any] = {"value": self.value, "line": self.line}
        if self.server_number is not None:
            named["server_number"] = self.server_number
        if self.build_reads is not None:
            named["build_reads"] = self.build_reads
        if self.build_number is not None:
            named["build_number"] = self.build_number
        return named


#: How many values a summary names; ``count`` says how many there are.
EXAMPLES = 3

_EXAMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "The value as the server reads it."},
        "line": {"type": "integer"},
        "server_number": {"type": "number"},
        "build_reads": {"type": "string", "description": "The text the build reads instead."},
        "build_number": {"type": "number", "description": "The number the build takes."},
    },
    "required": ["value", "line"],
}

#: The JSON Schema of one summary, for a reply's output schema.
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "The values of one stored file that one LTspice build this session runs reads as "
        "another number than the server does (reading established), or whose number no "
        "recording establishes for that build (reading not_established)."
    ),
    "properties": {
        "code": {"type": "string", "const": CODE},
        "file": {"type": "string"},
        "encoding": {"type": "string", "description": "The codec the server decoded the file as."},
        "build": {"type": "string"},
        "reading": {
            "type": "string",
            "enum": ["established", "not_established"],
            "description": (
                "established: the recordings show the number the build takes; "
                "not_established: no recording shows it, so it is not claimed."
            ),
        },
        "count": {"type": "integer", "description": "How many values this summary covers."},
        "examples": {
            "type": "array",
            "items": _EXAMPLE_SCHEMA,
            CUT: NEVER,
            "description": f"At most {EXAMPLES} of the values, in file order.",
        },
        "evidence": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
        "detail": {"type": "string"},
        "listed_by": {
            "type": "object",
            "description": "The call that lists every value this summary counts.",
            "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["tool", "arguments"],
        },
    },
    "required": [
        "code",
        "file",
        "encoding",
        "build",
        "reading",
        "count",
        "examples",
        "evidence",
        "detail",
        "listed_by",
    ],
}

#: A reply's list of summaries: one per file, build and kind of reading, so
#: its length does not grow with the number of values.
SUMMARIES_SCHEMA: dict[str, Any] = {
    "type": "array",
    CUT: ITEMS,
    "description": (
        "Per stored file and LTspice build this session runs, the values that build reads "
        "as another number than this reply gives. Absent when every value read is read alike."
    ),
    "items": SUMMARY_SCHEMA,
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


def _as_8_bit(token: str) -> str:
    """``token``'s UTF-8 bytes as a build that reads 8-bit text reads them
    (``encoding.latin1_reading``, the one answer to how a build reads them)."""
    return latin1_reading(decode_windows_1252(token.encode("utf-8")))


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
        settle = (f"a deck holding {site.suffix} stored as {encoding}, recorded on {build}",)
        return OtherReading(build_reads=None, build_number=None, evidence=settle, **common)
    decoding, evidence = recorded
    if decoding == "same":
        return None
    reads = _as_8_bit(site.token)
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
    #: The file read; None for bytes with no file.
    path: Path | None = None


def read_stored_bytes(content: bytes, *, root: bool, file: str | Path | None = None) -> StoredDeck:
    """A stored file's bytes read as a deck (``root``, whose first line is its
    title) or as an included file, with the readings of its values."""
    text, encoding = decode_spice_bytes_with_encoding(content)
    lexed = lex_deck(text, title=root)
    return StoredDeck(text, encoding, lexed, readings_of_cards(lexed.cards, encoding, file=file))


def read_stored_deck(path: Path, *, root: bool = True) -> StoredDeck:
    """:func:`read_stored_bytes` for the file at ``path``."""
    return replace(read_stored_bytes(path.read_bytes(), root=root, file=path), path=path)


def readings_of_bytes(
    content: bytes, *, root: bool, file: str | Path | None = None
) -> tuple[OtherReading, ...]:
    """The readings alone of :func:`read_stored_bytes`."""
    return read_stored_bytes(content, root=root, file=file).readings


def readings_of_file(path: Path, *, root: bool = True) -> tuple[OtherReading, ...]:
    """The readings alone of :func:`read_stored_deck`."""
    return read_stored_deck(path, root=root).readings


def _detail(group: Sequence[OtherReading]) -> str:
    first = group[0]
    sign = next((c for c in first.value if c in "µμ"), "µ")
    advice = "Write 'u' for micro: every build reads it alike."
    if first.build_number is not None:
        server = (
            ""
            if first.server_number is None
            else f" where the server reads {first.server_number:g}"
        )
        return (
            f"{first.build} reads {len(group)} value(s) of this {first.encoding} file as "
            f"letters that are no scale and takes the bare number: {first.value} as "
            f"{first.build_reads}, {first.build_number:g}{server}. {advice}"
        )
    if first.build_reads is not None:
        return (
            f"{first.build} reads {len(group)} value(s) of this {first.encoding} file as "
            f"letters that are no scale ({first.value} as {first.build_reads}); no recording "
            f"shows the number it takes where they stand. {advice}"
        )
    return (
        f"No recording shows how {first.build} reads a {sign} stored as {first.encoding}, so "
        f"the number it takes for {len(group)} value(s) is not established. {advice}"
    )


def _listed_by(file: str) -> dict[str, Any]:
    if file.casefold().endswith(".asc"):
        return {
            "tool": "inspect",
            "arguments": {"queries": [{"kind": "components", "path": file}]},
        }
    return {"tool": "verify_circuit", "arguments": {"path": file, "checks": ["syntax"]}}


def summaries(readings: Iterable[OtherReading], builds: Collection[str]) -> list[dict[str, Any]]:
    """The readings of the LTspice ``builds`` a session runs, as a reply
    states them: one summary per file, build and kind of reading (established
    or not), with the count, at most :data:`EXAMPLES` of the values, and the
    call that lists every one. A reading by a build the session does not run
    is not stated: it is no number that session's user meets."""
    groups: dict[tuple[str, str, str, bool], list[OtherReading]] = {}
    seen: set[tuple[Any, ...]] = set()
    for reading in readings:
        key = (reading.file, reading.line, reading.offset, reading.value, reading.build)
        if reading.build not in builds or key in seen or reading.file is None:
            continue
        seen.add(key)
        group = (reading.file, reading.encoding, reading.build, reading.established)
        groups.setdefault(group, []).append(reading)
    stated: list[dict[str, Any]] = []
    for (file, encoding, build, established), group in groups.items():
        stated.append(
            {
                "code": CODE,
                "file": file,
                "encoding": encoding,
                "build": build,
                "reading": "established" if established else "not_established",
                "count": len(group),
                "examples": [reading.example() for reading in group[:EXAMPLES]],
                "evidence": sorted({item for reading in group for item in reading.evidence}),
                "detail": _detail(group),
                "listed_by": _listed_by(file),
            }
        )
    return stated


# Sheets -----------------------------------------------------------------------

#: How a build reads a sign stored in a sheet (``.asc``) of a codec, from its
#: export (``export-encoding`` in the recordings): ``same`` (the sign), or
#: ``cp1252`` (each byte of the stored UTF-8 as its own character). Unlike a
#: deck, both builds read a UTF-8 sheet byte by byte: LTspice 26's export of
#: ``export/micro_utf8`` writes ``1Âµ``, XVII's writes the two bytes back.
_SHEET_RECORDED: dict[tuple[str, str], dict[str, tuple[_Decoding, tuple[str, ...]]]] = {
    ("µ", "cp1252"): {
        build: ("same", ("export/micro_cp1252", "export/micro_cp1252_v41")) for build in BUILDS
    },
    ("µ", "utf-8"): {
        build: ("cp1252", ("export/micro_utf8", "export/micro_utf8_v41")) for build in BUILDS
    },
    ("µ", "utf-16-le"): {build: ("same", ("export/micro_utf16le",)) for build in BUILDS},
    ("μ", "utf-8"): {build: ("cp1252", ("export/greek_mu_utf8",)) for build in BUILDS},
}

#: The number a build takes from a sheet value it reads as letters: shown
#: only for LTspice XVII, which reads its own export's bytes as a deck of
#: those letters (``deck/micro_utf8``); no recording shows LTspice 26 running
#: such letters.
_SHEET_BARE_NUMBER = {"LTspice XVII": ("deck/micro_utf8", "deck/suffix_unknown")}


def readings_of_sheet(
    export_cards: Iterable[SpiceCard], sheet: Path
) -> tuple[OtherReading, ...] | None:
    """The readings of a sheet's values, for a reply that read the sheet as
    the server's export of it (``export_cards``, as the server decoded them).

    A value counts where the server read the export's sign as micro and the
    sheet itself holds that value with the sign: an export can spell a
    sheet's ``u`` as a micro sign, which says nothing of the sheet. Each such
    value is read as the recordings show each build reads the sheet's stored
    bytes. None when the sheet cannot be read, which the caller states."""
    micro = [site for site in value_suffix_sites(export_cards) if site.micro]
    if not micro:
        # The server read no value as micro: every build reads it alike.
        return ()
    try:
        text, encoding = decode_spice_bytes_with_encoding(sheet.read_bytes())
    except OSError:
        return None
    lines = text.split("\n")
    found: list[OtherReading] = []
    for site in micro:
        if site.token not in text:
            continue
        line = next((i for i, written in enumerate(lines, 1) if site.token in written), 1)
        for build in BUILDS:
            recorded = _SHEET_RECORDED.get((site.suffix, encoding), {}).get(build)
            common: dict[str, Any] = {
                "file": str(sheet),
                "line": line,
                "value": site.token,
                "encoding": encoding,
                "server_number": _server_number(site),
                "build": build,
                "offset": site.offset,
            }
            if recorded is None:
                settle = (
                    f"a sheet holding {site.suffix} stored as {encoding}, exported by {build}",
                )
                found.append(
                    OtherReading(build_reads=None, build_number=None, evidence=settle, **common)
                )
                continue
            decoding, evidence = recorded
            if decoding == "same":
                continue
            bare = _SHEET_BARE_NUMBER.get(build)
            number = float(site.number) if bare is not None and _is_the_value(site) else None
            found.append(
                OtherReading(
                    build_reads=_as_8_bit(site.token),
                    build_number=number,
                    evidence=tuple(dict.fromkeys((*evidence, *(bare or ()))))
                    if number is not None
                    else evidence,
                    **common,
                )
            )
    return tuple(found)


#: Why a sheet's values were not checked: the sheet could not be read.
SHEET_UNREAD = "the sheet could not be read, so how a build reads its values was not checked"

#: A reply's list of stored files whose values were not checked, sent whole:
#: one entry per side a reply reads.
NOT_CHECKED_SCHEMA: dict[str, Any] = {
    "type": "array",
    CUT: NEVER,
    "description": (
        "Each stored file whose values were not checked for another build's reading, "
        "with why; absent when every file read was checked."
    ),
    "items": {
        "type": "object",
        "properties": {"file": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["file", "reason"],
    },
}
