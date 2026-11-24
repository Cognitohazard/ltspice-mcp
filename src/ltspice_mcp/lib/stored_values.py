"""Which recorded LTspice builds read a stored value as another number than the server.

The server reads a deck's bytes once (``encoding.decode_spice_bytes_with_encoding``)
and reads a micro sign (``µ`` or ``μ``) at a value's scale suffix as micro: a
run through the server spells it ``u`` before the simulator sees it, so that
is the number the run uses. A person who opens the same file in LTspice gets
whatever that build makes of the stored bytes, which the recordings show
differs by build and by encoding (``deck-encoding`` in ``inputs/cases.toml``):

- LTspice 26 reads a micro sign stored as cp1252, UTF-8 (with and without a
  byte order mark) or UTF-16, and a Greek mu stored as UTF-8 (with and
  without a mark) or UTF-16, as micro: the same text the server reads.
- LTspice XVII reads a micro sign stored as cp1252 or UTF-16, and a Greek mu
  stored as UTF-16, as micro, but a micro sign or a Greek mu stored as UTF-8
  (with or without a mark) as the two cp1252 characters of its bytes
  (``1Âµ``), neither of them a scale, so it takes the bare number
  (``deck/micro_utf8``, ``deck/micro_utf8_bom``, ``deck/greek_mu_utf8``,
  ``deck/number_grid_utf8_bom``, ``deck/number_grid_utf16le``).

This module is the one answer to "which recorded builds read this stored
value as another number than the server does". It holds no tool's rules: a
reply that gives a number or a comparison read from a stored deck asks it
(``readings_of_cards``, ``readings_of_file``) and states each reading it
returns (``OtherReading.statement``). An encoding or a sign no recording shows
for a build is returned as not established for that build, never assumed; an
absent reading means the build reads the server's text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any, Literal

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import NETLIST_SUFFIXES
from ltspice_mcp.lib.asc_document import AscDocument, Symbol, Text, parse_asc
from ltspice_mcp.lib.encoding import (
    decode_spice_bytes_with_encoding,
    decode_windows_1252,
    latin1_reading,
)
from ltspice_mcp.lib.format import cap_note, fold_micro_sign, parse_spice_value
from ltspice_mcp.lib.reply_cuts import CUT, NEVER
from ltspice_mcp.lib.spice_lex import LexResult, SpiceCard, lex_deck
from ltspice_mcp.lib.spice_lex_ops import (
    ValueSuffixSite,
    exported_part,
    value_suffix_sites,
)
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
    ("μ", "utf-8-sig"): {
        "LTspice 26": ("same", ("deck/number_grid_utf8_bom",)),
        "LTspice XVII": ("cp1252", ("deck/number_grid_utf8_bom",)),
    },
    ("μ", "utf-16-le"): {
        "LTspice 26": ("same", ("deck/number_grid_suffixes", "deck/number_grid_utf16le")),
        "LTspice XVII": ("same", ("deck/number_grid_suffixes", "deck/number_grid_utf16le")),
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


def value_readings(
    readings: Iterable[OtherReading], builds: SessionBuilds
) -> list[dict[str, Any]]:
    """Each reading by a build the session may run, as a value's own record
    (what ``verify_circuit``'s syntax check gives each value): the value, its
    line, the build, the kind of reading, what the build reads and takes, and
    the note an unidentified build carries."""
    stated: list[dict[str, Any]] = []
    for reading in readings:
        if reading.build not in builds:
            continue
        record = {
            **reading.example(),
            "build": reading.build,
            "reading": "established" if reading.established else "not_established",
            "evidence": list(reading.evidence),
        }
        note = builds[reading.build]
        if note is not None:
            record["note"] = note
        stated.append(record)
    return stated


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

#: How many files a summary names; ``files`` says how many there are.
FILES = 3

#: The note a summary carries for a build the session may run: an LTspice
#: whose build could not be identified is not known to read alike.
UNIDENTIFIED = (
    "the session's LTspice build was not identified, so what this recorded build does is stated"
)

#: Which values a summary is about, by kind of reading: the rule needs no list.
RULES = {
    True: (
        "every value of these files whose scale suffix is a micro sign, written as a "
        "two-terminal part's own value (R, C, L, V or I) outside braces"
    ),
    False: (
        "every value of these files whose scale suffix is a micro sign, where no recording "
        "shows the number this build takes: in another position (braces, a parameter, a "
        "source function, a device's area, a directive) or in an encoding not recorded "
        "for this build"
    ),
}

_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file": {"type": "string"},
        "encoding": {"type": "string", "description": "The codec the server decoded the file as."},
        "count": {"type": "integer", "description": "How many of the file's values."},
        "examples": {
            "type": "array",
            "items": _EXAMPLE_SCHEMA,
            CUT: NEVER,
            "description": f"At most {EXAMPLES} of the values, in file order.",
        },
        "listed_by": {
            "type": "object",
            "description": (
                "A call that lists each of the file's values with this build's reading; "
                "absent where no call does."
            ),
            "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["tool", "arguments"],
        },
    },
    "required": ["file", "encoding", "count", "examples"],
}

#: The JSON Schema of one summary, for a reply's output schema.
SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "The values of the stored files read that one LTspice build reads as another "
        "number than the server does (reading established), or whose number no recording "
        "establishes for that build (reading not_established)."
    ),
    "properties": {
        "code": {"type": "string", "const": CODE},
        "build": {"type": "string"},
        "reading": {
            "type": "string",
            "enum": ["established", "not_established"],
            "description": (
                "established: the recordings show the number the build takes; "
                "not_established: no recording shows it, so it is not claimed."
            ),
        },
        "note": {"type": "string", "description": "Why the build is stated though unidentified."},
        "rule": {"type": "string", "description": "Which values the summary is about."},
        "count": {"type": "integer", "description": "How many values, in every file."},
        "files": {"type": "integer", "description": "How many files hold them."},
        "by_file": {
            "type": "array",
            "items": _FILE_SCHEMA,
            CUT: NEVER,
            "description": f"At most {FILES} of the files, in the order read.",
        },
        "files_not_listed": {"type": "integer"},
        "evidence": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
        "detail": {"type": "string"},
    },
    "required": [
        "code",
        "build",
        "reading",
        "rule",
        "count",
        "files",
        "by_file",
        "evidence",
        "detail",
    ],
}

#: A reply's list of summaries: one per build and kind of reading, at most
#: four, whatever the number of files and values.
SUMMARIES_SCHEMA: dict[str, Any] = {
    "type": "array",
    CUT: NEVER,
    "description": (
        "Per LTspice build this session may run and kind of reading, the values of the "
        "stored files read that build reads as another number than this reply gives. "
        "Absent when every value read is read alike."
    ),
    "items": SUMMARY_SCHEMA,
}

#: The LTspice builds a session may run, each with the note its summaries
#: carry: None for a build the session's LTspice was identified as.
SessionBuilds = Mapping[str, "str | None"]


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


def _detail(build: str, established: bool, by_file: dict[str, list[OtherReading]]) -> str:
    count = sum(len(group) for group in by_file.values())
    names = [PurePath(file).name for file in by_file]
    shown = ", ".join(names[:FILES])
    files = f"{len(names)} file(s) ({shown}{', ...' if len(names) > FILES else ''})"
    first = next(iter(by_file.values()))[0]
    advice = "Write 'u' for micro: every build reads it alike."
    if established:
        server = (
            ""
            if first.server_number is None
            else f" where the server reads {first.server_number:g}"
        )
        said = (
            f"{build} reads {count} value(s) in {files} as letters that are no scale and "
            f"takes the bare number: {first.value} as {first.build_reads}, "
            f"{first.build_number:g}{server}."
        )
    elif first.build_reads is not None:
        said = (
            f"{build} reads {count} value(s) in {files} as letters that are no scale "
            f"({first.value} as {first.build_reads}); no recording shows the number it "
            "takes where they stand."
        )
    else:
        said = (
            f"No recording shows how {build} reads {count} value(s) in {files}, stored in "
            "an encoding not recorded for it, so the number it takes is not established."
        )
    covered = "the rule names every one"
    if len(names) > FILES:
        said += f" Files: {cap_note(FILES, len(names), 'files', covered)}."
    return f"{said} {advice}"


def _listed_by(file: str) -> dict[str, Any] | None:
    """The call that lists each of ``file``'s values with each build's reading:
    ``verify_circuit``'s syntax check, for a netlist it checks. None for any
    other file (an included library, a sheet), which no call lists so."""
    if PurePath(file).suffix.lower() not in NETLIST_SUFFIXES:
        return None
    return {"tool": "verify_circuit", "arguments": {"path": file, "checks": ["syntax"]}}


def _file_entry(file: str, group: Sequence[OtherReading]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": file,
        "encoding": group[0].encoding,
        "count": len(group),
        "examples": [reading.example() for reading in group[:EXAMPLES]],
    }
    listed_by = _listed_by(file)
    if listed_by is not None:
        entry["listed_by"] = listed_by
    return entry


def summaries(readings: Iterable[OtherReading], builds: SessionBuilds) -> list[dict[str, Any]]:
    """The readings of the LTspice ``builds`` a session may run, as a reply
    states them: one summary per build and kind of reading (established or
    not), with the rule that picks its values, the totals, and at most
    :data:`FILES` files with their counts and :data:`EXAMPLES` examples. So a
    summary's size does not grow with the number of files or values, and no
    value is outside it. A reading by a build the session does not run is not
    stated: it is no number that session's user meets."""
    groups: dict[tuple[str, bool], dict[str, list[OtherReading]]] = {}
    seen: set[tuple[Any, ...]] = set()
    for reading in readings:
        key = (reading.file, reading.line, reading.offset, reading.value, reading.build)
        if reading.build not in builds or key in seen or reading.file is None:
            continue
        seen.add(key)
        by_file = groups.setdefault((reading.build, reading.established), {})
        by_file.setdefault(reading.file, []).append(reading)
    stated: list[dict[str, Any]] = []
    for (build, established), by_file in sorted(groups.items(), key=lambda item: item[0]):
        summary: dict[str, Any] = {
            "code": CODE,
            "build": build,
            "reading": "established" if established else "not_established",
            "rule": RULES[established],
            "count": sum(len(group) for group in by_file.values()),
            "files": len(by_file),
            "by_file": [_file_entry(file, group) for file, group in list(by_file.items())[:FILES]],
            "evidence": sorted(
                {
                    item
                    for group in by_file.values()
                    for reading in group
                    for item in reading.evidence
                }
            ),
            "detail": _detail(build, established, by_file),
        }
        note = builds[build]
        if note is not None:
            summary["note"] = note
        if len(by_file) > FILES:
            summary["files_not_listed"] = len(by_file) - FILES
        stated.append(summary)
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


#: The attributes of a part whose text is a value or part of one.
_VALUE_ATTRIBUTES = frozenset({"Value", "Value2", "SpiceLine", "SpiceLine2"})


@dataclass(frozen=True)
class _SheetText:
    """One value attribute of a part, or one directive, with its line."""

    reference: str | None  # the part's InstName; None for a directive
    text: str
    line: int


def _sheet_texts(document: AscDocument) -> list[_SheetText]:
    """Each part's value attributes and each directive of a sheet, as the
    sheet parser reads them, with the line each stands on. A comment is no
    value and is not among them."""
    found: list[_SheetText] = []
    line = 1
    for record in document.records:
        lines = record.source or ()
        if isinstance(record, Symbol):
            # The attribute's text is the parser's; only its line is looked up,
            # among the lines the record was read from.
            for name, value in record.attrs:
                if name not in _VALUE_ATTRIBUTES:
                    continue
                at = next(
                    (
                        offset
                        for offset, written in enumerate(lines)
                        if written.split(None, 2)[:2] == ["SYMATTR", name]
                    ),
                    0,
                )
                found.append(_SheetText(record.reference, value, line + at))
        elif isinstance(record, Text) and record.is_directive:
            found.append(_SheetText(None, record.body, line))
        line += len(lines)
    return found


def _holds(text: str, token: str) -> bool:
    """``token`` is a whole value in ``text``, not part of a longer one."""
    return re.search(rf"(?<![\w.]){re.escape(token)}(?!\w)", text) is not None


def _sheet_line(site: ValueSuffixSite, texts: Sequence[_SheetText]) -> int | None:
    """The line of the sheet text the export's value came from, where that text
    holds the value with its sign: the part's own value attribute for an
    element (named as the export names it, ``R1`` or ``R§Load``), a directive
    for a directive. None where the sheet holds no such text (its part says
    ``u``, which an export may spell as a micro sign)."""
    name = site.card.name or ""
    if site.card.kind == "instance":
        # Every name the exported part answers to, the sheet's own among them
        # (``R§Load`` is ``Load`` on the sheet): spice_lex_ops' answer.
        part = exported_part(name, {})
        names = {n.casefold() for n in (name, *(part.names() if part else ()))}
        candidates = [
            t for t in texts if t.reference is not None and t.reference.casefold() in names
        ]
    else:
        candidates = [t for t in texts if t.reference is None]
    return next((t.line for t in candidates if _holds(t.text, site.token)), None)


def readings_of_sheet(
    export_cards: Iterable[SpiceCard], sheet: Path
) -> tuple[OtherReading, ...] | None:
    """The readings of a sheet's values, for a reply that read the sheet as
    the server's export of it (``export_cards``, as the server decoded them).

    A value counts where the server read the export's sign as micro and the
    sheet's own text for it, read by the sheet parser (the part's value
    attribute, or the directive), holds that value with the sign: an export
    can spell a sheet's ``u`` as a micro sign, which says nothing of the
    sheet. Each such value is read as the recordings show each build reads
    the sheet's stored bytes, at that text's line. None when the sheet cannot
    be read, which the caller states."""
    micro = [site for site in value_suffix_sites(export_cards) if site.micro]
    if not micro:
        # The server read no value as micro: every build reads it alike.
        return ()
    try:
        document = parse_asc(sheet.read_bytes())
    except (OSError, NetlistError):
        return None
    texts = _sheet_texts(document)
    encoding = document.encoding
    found: list[OtherReading] = []
    for site in micro:
        line = _sheet_line(site, texts)
        if line is None:
            continue
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
