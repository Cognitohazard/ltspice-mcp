"""Tests for ``lib/encoding.py``.

The BOM/UTF-16 heuristic is shared between ``library_parser`` and
``spice_lex.cards_from_path`` and is load-bearing for LTspice's
bundled ``standard.bjt`` / ``standard.mos`` (UTF-16 LE without a BOM).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.encoding import (
    decode_spice_bytes,
    decode_spice_bytes_strictly,
    decode_spice_bytes_with_encoding,
    detect_utf16_endianness,
    encode_spice_text,
    read_spice_text,
    rewrite_codec,
)


class TestDecodeSpiceBytes:
    def test_utf8_no_bom(self) -> None:
        text = ".MODEL Q NPN(BF=200)\n"
        assert decode_spice_bytes(text.encode("utf-8")) == text

    def test_utf8_bom_stripped(self) -> None:
        text = ".PARAM Vdd=5\n"
        assert decode_spice_bytes(b"\xef\xbb\xbf" + text.encode("utf-8")) == text

    def test_utf16_le_bom_stripped(self) -> None:
        text = ".MODEL FOO NMOS(VTO=0.7)\n"
        assert decode_spice_bytes(b"\xff\xfe" + text.encode("utf-16-le")) == text

    def test_utf16_be_bom_stripped(self) -> None:
        text = ".MODEL FOO NMOS(VTO=0.7)\n"
        assert decode_spice_bytes(b"\xfe\xff" + text.encode("utf-16-be")) == text

    def test_utf32_le_bom_stripped(self) -> None:
        text = ".PARAM x=1\n"
        assert decode_spice_bytes(b"\xff\xfe\x00\x00" + text.encode("utf-32-le")) == text

    def test_utf16_le_no_bom_via_heuristic(self) -> None:
        # LTspice's bundled standard.{mos,bjt} files are UTF-16 LE
        # without a BOM. ASCII text in UTF-16 LE has a null byte at
        # every odd position; the heuristic catches that.
        text = "* LTspice standard library\n.MODEL 2N3904 NPN(BF=300 IS=1e-14)\n"
        encoded = text.encode("utf-16-le")
        assert decode_spice_bytes(encoded) == text

    def test_utf16_be_no_bom_via_heuristic(self) -> None:
        text = "* test\n.MODEL Q NPN\n"
        encoded = text.encode("utf-16-be")
        assert decode_spice_bytes(encoded) == text

    def test_plain_ascii_falls_through_to_utf8(self) -> None:
        text = "R1 a b 1k\n"
        assert decode_spice_bytes(text.encode("ascii")) == text

    def test_cp1252_degree_sign_preserved(self) -> None:
        # Windows-edited LTspice files often have a single non-ASCII
        # char (degree, mu, en-dash) without a BOM. cp1252 strict-decode
        # preserves them instead of replacing with U+FFFD.
        text = "* °C operating point\n"
        raw = text.encode("cp1252")
        # No BOM, no UTF-16 null pattern — would have fallen to utf-8
        # errors="replace" before. Now: cp1252 strict succeeds.
        assert decode_spice_bytes(raw) == text
        assert "�" not in decode_spice_bytes(raw)

    def test_cp1252_mu_sign_preserved(self) -> None:
        text = "C1 a b 10µF\n"
        raw = text.encode("cp1252")
        assert decode_spice_bytes(raw) == text

    def test_bytes_no_codec_defines_still_decode(self) -> None:
        # 81 has no character in cp1252 and the file is not UTF-8. It must
        # not raise, and nothing in it is replaced.
        raw = b".MODEL Q NPN\n\x80\x81\xfe\n"  # \xfe alone is not a UTF-16 BOM
        out = decode_spice_bytes(raw)
        assert ".MODEL Q NPN" in out
        assert "\ufffd" not in out


#: Text as a Windows of that language saves it, each holding at least one of
#: the five bytes cp1252 gives no character: the Japanese ideographic comma is
#: 81 41, and the kanji around it lead with 8D, 8F and 90.
LEGACY_TEXT = [
    ("cp932", "\u30d5\u30a3\u30eb\u30bf\u3001\u62b5\u6297\u6570\u5024"),
    ("cp932", "\u9ad8\u5c0f\u65b0"),
    ("cp936", "\u6ee4\u6ce2\u5668\u4e02"),
    ("cp949", "\u3131\u314f\uac02"),
]
CP1252_UNDEFINED = {0x81, 0x8D, 0x8F, 0x90, 0x9D}


class TestEightBitTextKeepsItsBytes:
    """A deck in a code page this module cannot name is still 8-bit text to
    LTspice, which reads it a byte at a time. Decoding it loses nothing, so a
    rewritten copy holds every byte the rewrite did not touch."""

    @pytest.mark.parametrize(("code_page", "words"), LEGACY_TEXT)
    def test_text_in_a_double_byte_code_page_comes_back_as_written(
        self, code_page: str, words: str
    ) -> None:
        raw = f'* {words}\n.include "C:\\{words}\\m.lib"\nR1 a 0 1k\n.end\n'.encode(code_page)
        assert set(raw) & CP1252_UNDEFINED, "the sample must hold a byte cp1252 lacks"
        text, encoding = decode_spice_bytes_with_encoding(raw)
        assert "\ufffd" not in text
        assert encode_spice_text(text, rewrite_codec(encoding)) == raw
        # The cards the deck holds are read as they were.
        assert "R1 a 0 1k\n" in text

    def test_every_byte_has_a_character_and_comes_back(self) -> None:
        # 0xC0 first: a UTF-8 lead byte with nothing after it, so no part of
        # this can be taken for UTF-8.
        raw = b"* \xc0 " + bytes(range(0x80, 0x100)) + b"\n"
        text, encoding = decode_spice_bytes_with_encoding(raw)
        assert encoding == "cp1252"
        assert len(text) == len(raw)
        assert encode_spice_text(text, rewrite_codec(encoding)) == raw

    def test_the_bytes_cp1252_defines_read_as_cp1252(self) -> None:
        raw = b"* \x96 \xb5 \xa7 \xb0\n"
        assert decode_spice_bytes(raw) == "* \u2013 \u00b5 \u00a7 \u00b0\n"

    def test_no_byte_becomes_a_line_break(self) -> None:
        # Read as Latin-1, 85 is the next-line control, which splits a line
        # for str.splitlines and so for the lexer. Here it is an ellipsis.
        text = decode_spice_bytes(b"* \x81 a\x85b\nR1 a 0 1k\n")
        assert text.splitlines() == ["* \x81 a\u2026b", "R1 a 0 1k"]

    def test_a_strict_read_takes_any_eight_bit_file(self) -> None:
        raw = "* \u62b5\u6297\nR1 a 0 1k\n".encode("cp932")
        assert decode_spice_bytes_strictly(raw) == decode_spice_bytes_with_encoding(raw)

    def test_a_strict_read_refuses_malformed_utf16(self) -> None:
        raw = "R1 a 0 1k\n".encode("utf-16") + b"\x00\xd8"  # a lone high surrogate
        assert "\ufffd" in decode_spice_bytes(raw)
        with pytest.raises(UnicodeError):
            decode_spice_bytes_strictly(raw)

    def test_a_strict_read_strips_a_byte_order_mark(self) -> None:
        for codec in ("utf-16", "utf-8-sig"):
            assert decode_spice_bytes_strictly("R1 a 0 1k\n".encode(codec))[0] == "R1 a 0 1k\n"


class TestDetectUtf16Endianness:
    def test_recognises_utf16_le_ascii(self) -> None:
        probe = "abc".encode("utf-16-le")
        assert detect_utf16_endianness(probe) == "utf-16-le"

    def test_recognises_utf16_be_ascii(self) -> None:
        probe = "abc".encode("utf-16-be")
        assert detect_utf16_endianness(probe) == "utf-16-be"

    def test_returns_none_for_real_utf8(self) -> None:
        assert detect_utf16_endianness(b".MODEL Q NPN(BF=200)") is None

    def test_returns_none_for_short_input(self) -> None:
        assert detect_utf16_endianness(b"") is None
        assert detect_utf16_endianness(b"\x00") is None

    def test_returns_none_for_mixed_null_distribution(self) -> None:
        # Real binary blob with mixed nulls — heuristic must NOT
        # claim it's UTF-16.
        probe = bytes(range(256))[::2] + bytes(range(256))[1::2]
        assert detect_utf16_endianness(probe) is None


class TestReadSpiceText:
    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        text = ".MODEL Q NPN\n"
        p = tmp_path / "x.lib"
        p.write_text(text, encoding="utf-8", newline="\n")
        assert read_spice_text(p) == text

    def test_reads_utf16_le_no_bom_file(self, tmp_path: Path) -> None:
        # Mirrors the LTspice standard-library shape.
        text = "* LTspice stock\n.MODEL 2N3904 NPN(BF=300)\n"
        p = tmp_path / "standard.bjt"
        p.write_bytes(text.encode("utf-16-le"))
        assert read_spice_text(p) == text


class TestReadCircuitEncodingZoo:
    """``read_circuit`` used to crash on UTF-8-BOM, UTF-16-BE-no-BOM,
    and unclosed-``.SUBCKT`` files because spicelib's ``SpiceEditor`` was
    in the read path. The fix routes ``.cir/.net`` reads through
    ``services.extract_netlist_info`` which uses ``read_spice_text`` +
    ``cards_from_path``.
    """

    def _write_extra(self, path: Path, prefix: bytes, encoding: str) -> None:
        body = "* probe\nR1 in out 1k\n.tran 1u\n.end\n"
        path.write_bytes(prefix + body.encode(encoding))

    def test_utf8_bom_does_not_crash(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "utf8bom.cir"
        self._write_extra(cir, b"\xef\xbb\xbf", "utf-8")
        info = extract_netlist_info(cir)
        assert info["type"] == "netlist"
        refs = [c["reference"] for c in info["components"]]
        assert "R1" in refs

    def test_utf16le_no_bom(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "utf16le.cir"
        body = "* probe\nR1 in out 1k\n.tran 1u\n.end\n"
        cir.write_bytes(body.encode("utf-16-le"))
        info = extract_netlist_info(cir)
        # content must be properly decoded — no NUL interleavings
        assert "\x00" not in info["content"]
        refs = [c["reference"] for c in info["components"]]
        assert "R1" in refs

    def test_unclosed_subckt_warns_not_crashes(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.services import extract_netlist_info

        cir = tmp_path / "trunc.cir"
        cir.write_text(
            ".subckt amp in out\nR1 in mid 1k\nR2 mid out 1k\n* missing .ENDS\nV1 vdd 0 5\n.end\n"
        )
        info = extract_netlist_info(cir)
        assert "warnings" in info
        assert any("unclosed .subckt" in w.lower() for w in info["warnings"])


class TestRewriteCodec:
    """A rewritten deck is written back in the codec it was read with."""

    def test_ascii_compatible_codecs_are_kept(self) -> None:
        text = "R§1 a 0 1k\n"
        for codec in ("utf-8", "utf-8-sig", "cp1252"):
            raw = text.encode(codec)
            text, encoding = decode_spice_bytes_with_encoding(raw)
            assert rewrite_codec(encoding) == codec
            assert encode_spice_text(text, codec) == raw

    def test_utf16_and_utf32_are_rewritten_as_utf8(self) -> None:
        for raw in ("R1 a 0 1k\n".encode(codec) for codec in ("utf-16", "utf-16-le", "utf-32")):
            assert rewrite_codec(decode_spice_bytes_with_encoding(raw)[1]) == "utf-8"

    def test_text_the_codec_cannot_spell_is_written_as_utf8(self) -> None:
        text = '.include "/stage/日本/core.inc"\n'
        assert encode_spice_text(text, "cp1252") == text.encode("utf-8")
