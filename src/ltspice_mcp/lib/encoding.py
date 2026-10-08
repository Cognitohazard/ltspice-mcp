"""Encoding detection for SPICE library / netlist files.

LTspice's bundled ``lib/cmp/standard.{mos,bjt,...}`` files are UTF-16 LE
— sometimes with a BOM, sometimes without. Earlier versions of this
codebase assumed UTF-8 (Python's default for ``Path.read_text``) and
silently produced empty parses for those files. The detection here is
shared by ``library_parser`` and ``spice_lex.cards_from_path``.

Resolution order:

1. BOM sniff (UTF-32 LE/BE, UTF-16 LE/BE, UTF-8 with BOM).
2. Heuristic null-byte scan for UTF-16 LE/BE without BOM (the
   LTspice 26+ ``standard.bjt`` / ``standard.mos`` shape).
3. UTF-8 strict — if the bytes are clean UTF-8 (including pure ASCII),
   decode as-is. This branch is the common case for hand-edited
   netlists.
4. Windows-1252 for anything else. Windows-edited LTspice files often
   carry a single non-ASCII character (degree sign, mu, en-dash) in a
   comment without any BOM, and a deck saved in a double-byte code page
   (Japanese, Chinese) is 8-bit text too. Every byte has a character
   here, so this step never fails and the text encodes back to the bytes
   it came from.

Nothing in an 8-bit file is replaced. A rewritten copy of a deck must keep
every byte the rewrite did not touch: LTspice reads those bytes one at a
time, so a comment or an include path in a code page this module cannot
name is still exactly what the simulator was given.
"""

from __future__ import annotations

import codecs
from pathlib import Path

# Order matters: UTF-32 BOMs start with the same two bytes as UTF-16,
# so check the longer ones first.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)


def leading_byte_order_mark(data: bytes) -> str | None:
    """The codec named by the byte order mark ``data`` starts with, or None."""
    return next((encoding for bom, encoding in _BOM_ENCODINGS if data.startswith(bom)), None)


#: The byte order marks neither LTspice 26 nor LTspice XVII reads a sheet
#: behind, by the codec each names, with the name a message gives it
#: (``export/micro_utf8_bom`` and ``export/micro_utf16le_bom`` in the recordings).
_MARKS_LTSPICE_REFUSES = {"utf-8-sig": "UTF-8", "utf-16-le": "UTF-16 LE"}


def refused_sheet_mark(data: bytes) -> str | None:
    """The name of the mark ``data`` starts with, if LTspice refuses a sheet behind it."""
    return _MARKS_LTSPICE_REFUSES.get(leading_byte_order_mark(data) or "")


def refused_sheet_mark_note(name: str) -> str:
    """What a refusal or a finding says of a sheet that starts with the mark ``name``."""
    return (
        f"starts with a {name} byte order mark; neither LTspice 26 nor LTspice XVII reads "
        'a sheet that does (26 exports nothing from it, XVII stops on "Unknown schematic '
        'syntax"). Save it without the mark.'
    )


def detect_utf16_endianness(probe: bytes) -> str | None:
    """Return ``"utf-16-le"`` / ``"utf-16-be"`` if every other byte is null.

    ASCII text in UTF-16 LE produces a null byte at every odd position;
    UTF-16 BE puts the null at every even position. Counting nulls in a
    small head-of-file probe disambiguates ASCII-in-UTF-16 from real
    binary (which has mixed null distributions).
    """
    if len(probe) < 4 or len(probe) % 2:
        probe = probe[: (len(probe) // 2) * 2]
        if not probe:
            return None
    odd_nulls = sum(1 for i in range(1, len(probe), 2) if probe[i] == 0)
    even_nulls = sum(1 for i in range(0, len(probe), 2) if probe[i] == 0)
    half = len(probe) // 2
    if odd_nulls > 0.8 * half and even_nulls < 0.2 * half:
        return "utf-16-le"
    if even_nulls > 0.8 * half and odd_nulls < 0.2 * half:
        return "utf-16-be"
    return None


#: The five bytes cp1252 gives no character. Windows reads each as the control
#: character of the same number, and so do LTspice 26 and LTspice XVII
#: (``deck/bytes_in_node_names`` in the recordings).
_CP1252_UNDEFINED = (0x81, 0x8D, 0x8F, 0x90, 0x9D)

#: cp1252 with those five filled in: a character for every byte, and a byte
#: for each of those characters, so text decoded with it encodes back exactly.
_WINDOWS_1252 = "".join(
    chr(byte) if byte in _CP1252_UNDEFINED else bytes([byte]).decode("cp1252")
    for byte in range(256)
)
_WINDOWS_1252_BYTES = codecs.charmap_build(_WINDOWS_1252)
#: The same table as the characters that differ from Latin-1, which is how a
#: byte string is turned into text without a codec of its own.
_WINDOWS_1252_OVER_LATIN_1 = {
    byte: char for byte, char in enumerate(_WINDOWS_1252) if char != chr(byte)
}


_LATIN_1_OVER_WINDOWS_1252 = {ord(char): byte for byte, char in _WINDOWS_1252_OVER_LATIN_1.items()}


def latin1_reading(text: str) -> str:
    """``text`` decoded from an 8-bit file, as LTspice reads the same bytes.

    Both recorded builds read an 8-bit deck as Latin-1: every byte from 0x80
    to 0x9F is the control character of the same number, where the table this
    module decodes with shows cp1252's character (``€``, curly quotes,
    dashes). Every other character is the same either way.
    """
    return text.translate(_LATIN_1_OVER_WINDOWS_1252)


def _decode(raw: bytes, errors: str) -> tuple[str, str]:
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(encoding, errors=errors), encoding
    encoding = detect_utf16_endianness(raw[:256])
    if encoding is not None:
        return raw.decode(encoding, errors=errors), encoding
    # UTF-8 strict for clean ASCII and well-formed UTF-8 (no replacement).
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    # Anything else is 8-bit text. The degree signs, mus and en-dashes that
    # Windows-edited LTspice files put in comments read as themselves, and
    # text in a code page this cannot name (cp932, cp936) keeps its bytes.
    return decode_windows_1252(raw), "cp1252"


def decode_windows_1252(raw: bytes) -> str:
    """Every byte as Windows-1252 reads it, the five it leaves undefined as the
    control character of the same number.

    This is how LTspice reads an 8-bit sheet or deck whatever its bytes were
    written as, so it is also the reading to compare with what LTspice shows:
    a micro sign stored as UTF-8 is two characters there.
    """
    return raw.decode("latin-1").translate(_WINDOWS_1252_OVER_LATIN_1)


def decode_spice_bytes_with_encoding(raw: bytes) -> tuple[str, str]:
    """Decode a SPICE-text byte string and name the codec that decoded it.

    The name matters where a character's meaning depends on the reader: a
    micro sign stored as UTF-8 is two characters to a cp1252 reader. An 8-bit
    file is named ``cp1252`` whatever code page it was written in, and
    ``encode_spice_text`` gives its bytes back. Only a UTF-16 or UTF-32 file
    can lose anything, where it is malformed.
    """
    return _decode(raw, "replace")


def decode_spice_bytes_strictly(raw: bytes) -> tuple[str, str]:
    """``decode_spice_bytes_with_encoding`` that refuses instead of replacing.

    Raises ``UnicodeError`` for a UTF-16 or UTF-32 file that is malformed. No
    8-bit file is refused: every byte of one has a character.
    """
    return _decode(raw, "strict")


#: The codecs a rewritten deck keeps. Each is ASCII-compatible, so every
#: directive keeps its bytes and the byte-level edits made to a deck at run time
#: (``runner_base``'s injected ``.options``/``write`` lines) still land.
_KEPT_CODECS = frozenset({"utf-8", "utf-8-sig", "cp1252"})


def rewrite_codec(encoding: str) -> str:
    """The codec a rewritten copy of a deck read as ``encoding`` is written in.

    ``encoding`` is the name ``decode_spice_bytes_with_encoding`` gave. The
    copy keeps it, so the characters a rewrite does not touch keep their
    bytes: a ``§`` in a cp1252 deck stays the one byte A7 LTspice XVII reads,
    rather than becoming the two UTF-8 bytes it reads as ``Â§``. A UTF-16 or
    UTF-32 deck is rewritten as UTF-8, which spells every character it held and
    is what every simulator here reads.
    """
    return encoding if encoding in _KEPT_CODECS else "utf-8"


def bom_of(raw: bytes) -> bytes:
    """The byte order mark ``raw`` starts with, or ``b""``.

    The decoders here drop the mark, so a caller that writes a file back as it
    was read keeps it from this.
    """
    return next((bom for bom, _ in _BOM_ENCODINGS if raw.startswith(bom)), b"")


def encode_spice_text_strictly(text: str, codec: str) -> bytes:
    """``text`` in ``codec``, a name a decoder here gave, without a byte order mark.

    Raises ``UnicodeEncodeError`` for a character the codec cannot spell.
    ``cp1252`` is the whole-byte table the decoder reads 8-bit files with, so
    text it decoded comes back as its bytes; ``utf-8-sig`` is written as UTF-8,
    the mark being the caller's to keep (``bom_of``).
    """
    if codec == "cp1252":
        return codecs.charmap_encode(text, "strict", _WINDOWS_1252_BYTES)[0]
    return text.encode("utf-8" if codec == "utf-8-sig" else codec)


def encode_spice_text(text: str, codec: str) -> bytes:
    """``text`` in ``codec`` (a ``rewrite_codec`` name), else as UTF-8.

    UTF-8 is the fallback for text the codec cannot spell, such as a rewritten
    include path naming a character outside cp1252: it is the encoding LTspice
    24 and later write and read. ``cp1252`` here is the whole-byte table the
    decoder reads 8-bit files with, so text it decoded comes back as its bytes.
    """
    try:
        if codec == "cp1252":
            return codecs.charmap_encode(text, "strict", _WINDOWS_1252_BYTES)[0]
        return text.encode(codec)
    except UnicodeEncodeError:
        return text.encode("utf-8")


def decode_spice_bytes(raw: bytes) -> str:
    """Decode a SPICE-text byte string with BOM sniffing + UTF-16 heuristic."""
    return decode_spice_bytes_with_encoding(raw)[0]


def read_spice_text(path: Path) -> str:
    """Read a SPICE library/netlist file and return its decoded text."""
    return decode_spice_bytes(path.read_bytes())


def read_spice_text_with_encoding(path: Path) -> tuple[str, str]:
    """``read_spice_text`` plus the name of the codec that decoded the file."""
    return decode_spice_bytes_with_encoding(path.read_bytes())
