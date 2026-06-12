"""Tests for ``lib/encoding.py``.

The BOM/UTF-16 heuristic is shared between ``library_parser`` and
``spice_lex.cards_from_path`` and is load-bearing for LTspice's
bundled ``standard.bjt`` / ``standard.mos`` (UTF-16 LE without a BOM).
"""

from __future__ import annotations

from pathlib import Path

from ltspice_mcp.lib.encoding import (
    decode_spice_bytes,
    detect_utf16_endianness,
    read_spice_text,
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

    def test_invalid_utf8_replaces_rather_than_raises(self) -> None:
        # Mixed-encoding garbage must not raise — fall through to
        # utf-8 with errors="replace".
        raw = b".MODEL Q NPN\n\x80\x81\xfe\n"  # \xfe alone is not a UTF-16 BOM
        # Should decode without raising; the bad bytes become U+FFFD.
        out = decode_spice_bytes(raw)
        assert ".MODEL Q NPN" in out


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
        p.write_text(text, encoding="utf-8")
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
