"""Tests for library_parser: continuation merging, param extraction, model/subckt parsing."""

from pathlib import Path

from ltspice_mcp.lib.library_parser import (
    _extract_parameters,
    _merge_continuation_lines,
    parse_library_file,
)


class TestMergeContinuationLines:
    def test_basic(self):
        lines = [".MODEL Q1 NPN", "+ BF=200 IS=1e-14"]
        result = _merge_continuation_lines(lines)
        assert len(result) == 1
        assert "BF=200" in result[0]
        assert result[0].startswith(".MODEL")

    def test_strips_comments(self):
        lines = [
            "* This is a comment",
            ".MODEL Q1 NPN ; inline comment",
            "+ BF=200 $ another comment",
        ]
        result = _merge_continuation_lines(lines)
        assert len(result) == 1
        assert "comment" not in result[0]
        assert "BF=200" in result[0]

    def test_empty_input(self):
        assert _merge_continuation_lines([]) == []


class TestExtractParameters:
    def test_basic(self):
        params = _extract_parameters("BF=200 IS=1e-14")
        assert params == {"BF": "200", "IS": "1e-14"}

    def test_limit(self):
        text = "A=1 B=2 C=3 D=4 E=5 F=6 G=7"
        params = _extract_parameters(text, limit=5)
        assert len(params) == 5

    def test_empty(self):
        assert _extract_parameters("") == {}


class TestParseLibraryFile:
    def test_parse_model_entry(self, tmp_path: Path):
        lib = tmp_path / "models.lib"
        lib.write_text(".MODEL 2N2222 NPN (BF=200 IS=1e-14 VAF=100)\n")

        index = parse_library_file(lib)
        assert len(index.models) == 1
        m = index.models[0]
        assert m.name == "2N2222"
        assert m.model_type == ".MODEL"
        assert m.parameters["BF"] == "200"
        assert m.parameters["IS"] == "1e-14"
        assert m.source_path == lib

    def test_parse_model_no_space_before_paren(self, tmp_path: Path):
        # Regression: SPICE allows '.MODEL Q1 NPN(BF=200)' with no space.
        # Previously the regex captured 'NPN(BF=200)' as the type and lost
        # the parameters entirely.
        lib = tmp_path / "models.lib"
        lib.write_text(".MODEL Q1 NPN(BF=200 IS=1e-14)\n")
        index = parse_library_file(lib)
        assert len(index.models) == 1
        m = index.models[0]
        assert m.name == "Q1"
        assert m.parameters["BF"] == "200"
        assert m.parameters["IS"] == "1e-14"

    def test_parse_nested_subckt(self, tmp_path: Path):
        # Regression: a .SUBCKT containing another .SUBCKT was previously
        # truncated by the inner .ENDS, losing both the inner subckt and the
        # rest of the outer body.
        lib = tmp_path / "nested.lib"
        lib.write_text(
            ".SUBCKT outer in out\n"
            ".SUBCKT inner a b\n"
            "R1 a b 1k\n"
            ".ENDS\n"
            "X1 in out inner\n"
            ".ENDS\n"
            ".MODEL D1 D(IS=1e-14)\n"
        )
        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "outer" in names
        assert "D1" in names  # must be reachable after the nested .ENDS

        outer = next(m for m in index.models if m.name == "outer")
        # outer's body includes the nested .SUBCKT/.ENDS pair plus X1 and the
        # final .ENDS — i.e. all 6 lines from .SUBCKT outer through .ENDS.
        assert outer.line_count == 6

    def test_parse_subckt_with_params_keyword(self, tmp_path: Path):
        # Regression: '.SUBCKT name node1 node2 PARAMS: key=value' previously
        # parsed PARAMS: and key=value as additional node names.
        lib = tmp_path / "params.lib"
        lib.write_text(".SUBCKT myamp in out vcc PARAMS: gain=10 offset=0\nR1 in out 1k\n.ENDS\n")
        index = parse_library_file(lib)
        m = index.models[0]
        assert m.parameters == {"node1": "in", "node2": "out", "node3": "vcc"}

    def test_parse_subckt_with_ends(self, tmp_path: Path):
        lib = tmp_path / "sub.lib"
        lib.write_text(".SUBCKT opamp in+ in- out vcc vee\nR1 in+ in- 1Meg\n.ENDS\n")

        index = parse_library_file(lib)
        assert len(index.models) == 1
        m = index.models[0]
        assert m.name == "opamp"
        assert m.model_type == ".SUBCKT"
        assert m.parameters["node1"] == "in+"
        assert m.parameters["node2"] == "in-"
        assert m.line_count == 3

    def test_parse_subckt_missing_ends(self, tmp_path: Path):
        lib = tmp_path / "broken.lib"
        lib.write_text(".SUBCKT incomplete in out\nR1 in out 1k\n")

        index = parse_library_file(lib)
        assert len(index.models) == 0  # skipped due to missing .ENDS

    def test_parse_subckt_implicit_params_with_spaces(self, tmp_path: Path):
        # Whitespace around `=` in implicit PARAMS form: ``gain = 10``
        # must be classified as a param default, not a third port.
        lib = tmp_path / "spaced.lib"
        lib.write_text(".SUBCKT amp in out gain = 10\nR1 in out 1k\n.ENDS\n")
        index = parse_library_file(lib)
        assert len(index.models) == 1
        m = index.models[0]
        assert m.parameters == {"node1": "in", "node2": "out"}
        assert "node3" not in m.parameters  # `gain` is not a port

    def test_parse_mixed_file(self, tmp_path: Path):
        lib = tmp_path / "mixed.lib"
        lib.write_text(
            ".MODEL D1N4148 D(IS=2.52e-9 RS=0.568)\n"
            ".SUBCKT LM741 in+ in- out vcc vee\n"
            "R1 in+ in- 2Meg\n"
            ".ENDS\n"
        )

        index = parse_library_file(lib)
        assert len(index.models) == 2
        types = {m.model_type for m in index.models}
        assert types == {".MODEL", ".SUBCKT"}

    def test_search_pagination(self, tmp_path: Path):
        lib = tmp_path / "many.lib"
        lines = [f".MODEL M{i:02d} NPN(BF={100 + i})\n" for i in range(10)]
        lib.write_text("".join(lines))

        index = parse_library_file(lib)
        assert len(index.models) == 10

        page, total = index.search("M", offset=3, limit=4)
        assert total == 10
        assert len(page) == 4

    def test_get_model_case_insensitive(self, tmp_path: Path):
        lib = tmp_path / "case.lib"
        lib.write_text(".MODEL 2N2222 NPN(BF=200)\n")

        index = parse_library_file(lib)
        result = index.get_model("2n2222")
        assert result is not None
        assert result.name == "2N2222"

    def test_utf16_le_with_bom(self, tmp_path: Path):
        """LTspice's bundled ``lib/cmp/standard.{mos,bjt,...}`` files are
        UTF-16 LE with a BOM. Earlier the parser used ``Path.read_text``
        which defaulted to UTF-8 and produced empty model lists for these
        files — so ``find_model(include_builtin=True)`` couldn't find any
        of LTspice's stock parts."""
        import codecs

        lib = tmp_path / "standard.mos"
        text = ".MODEL 2N7000 NMOS(VTO=2.0 KP=0.05)\n.MODEL BSS84 PMOS(VTO=-2.0)\n"
        lib.write_bytes(codecs.BOM_UTF16_LE + text.encode("utf-16-le"))

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "2N7000" in names
        assert "BSS84" in names

    def test_utf16_be_with_bom(self, tmp_path: Path):
        import codecs

        lib = tmp_path / "be.mos"
        text = ".MODEL FOO NMOS(VTO=1.0)\n"
        lib.write_bytes(codecs.BOM_UTF16_BE + text.encode("utf-16-be"))

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "FOO" in names

    def test_utf8_bom_skipped(self, tmp_path: Path):
        import codecs

        lib = tmp_path / "u8.lib"
        text = ".MODEL UTF8MODEL NPN(BF=300)\n"
        lib.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "UTF8MODEL" in names

    def test_utf16_le_without_bom(self, tmp_path: Path):
        """Some LTspice 26+ installs ship ``standard.{mos,bjt}`` as UTF-16 LE
        WITHOUT a BOM. Bug H: ``load_library`` returned 0 models for those
        files. Heuristic null-byte detection picks them up now."""
        lib = tmp_path / "no_bom.mos"
        text = ".MODEL NMOS_NB NMOS(VTO=2.0 KP=0.05)\n.MODEL PMOS_NB PMOS(VTO=-1.5)\n"
        lib.write_bytes(text.encode("utf-16-le"))

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "NMOS_NB" in names
        assert "PMOS_NB" in names

    def test_utf16_be_without_bom(self, tmp_path: Path):
        lib = tmp_path / "no_bom_be.mos"
        text = ".MODEL BE_MODEL NMOS(VTO=2.0)\n"
        lib.write_bytes(text.encode("utf-16-be"))

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "BE_MODEL" in names

    def test_utf8_ascii_without_bom_still_works(self, tmp_path: Path):
        """Pure ASCII / UTF-8 libraries (most third-party .lib files) must
        not be misclassified as UTF-16 by the heuristic."""
        lib = tmp_path / "ascii.lib"
        # ASCII text — no null bytes anywhere.
        lib.write_bytes(b".MODEL ASCII_ONE NPN(BF=100)\n")

        index = parse_library_file(lib)
        names = {m.name for m in index.models}
        assert "ASCII_ONE" in names
