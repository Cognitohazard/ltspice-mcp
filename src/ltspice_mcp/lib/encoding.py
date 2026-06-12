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
4. CP1252 strict — Windows-edited LTspice files often carry a single
   non-ASCII character (degree sign, mu, en-dash) in a comment without
   any BOM. Trying CP1252 strictly before the lossy UTF-8 fallback
   preserves those characters instead of replacing them with U+FFFD.
5. UTF-8 with ``errors="replace"`` as the last-resort catch-all.
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


def decode_spice_bytes(raw: bytes) -> str:
    """Decode a SPICE-text byte string with BOM sniffing + UTF-16 heuristic."""
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(encoding, errors="replace")
    encoding = detect_utf16_endianness(raw[:256])
    if encoding is not None:
        return raw.decode(encoding, errors="replace")
    # UTF-8 strict for clean ASCII and well-formed UTF-8 (no replacement).
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # CP1252 strict — preserves degree signs / mu / en-dashes that
    # Windows-edited LTspice files put in comments without a BOM.
    # cp1252 is a strict superset of Latin-1 for the printable range,
    # so this also handles ISO-8859-1 inputs.
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        pass
    return raw.decode("utf-8", errors="replace")


def read_spice_text(path: Path) -> str:
    """Read a SPICE library/netlist file and return its decoded text."""
    return decode_spice_bytes(path.read_bytes())
