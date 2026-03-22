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
        # Space before '(' is required — regex group(2) captures \S+ which
        # would swallow the paren if it's adjacent to the type name.
        lib.write_text(".MODEL 2N2222 NPN (BF=200 IS=1e-14 VAF=100)\n")

        index = parse_library_file(lib)
        assert len(index.models) == 1
        m = index.models[0]
        assert m.name == "2N2222"
        assert m.model_type == ".MODEL"
        assert m.parameters["BF"] == "200"
        assert m.parameters["IS"] == "1e-14"
        assert m.source_path == lib

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
