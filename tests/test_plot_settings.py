"""The plot settings file module: what it writes, and what it makes of a file.

How LTspice itself writes and reads the file is held against the recordings in
``test_recorded_ltspice_plot_settings.py``; this module covers the writer's own
form, the reader's handling of sections it does not change, the refusals, and
that each server-written file the recorder hands LTspice is exactly what the
writer produces today.
"""

from __future__ import annotations

import codecs
from typing import get_args

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.plot_settings import (
    DEFAULT_SCALES,
    SECTION_NAMES,
    PlotAnalysis,
    PlotPane,
    PlotSection,
    PlotSettings,
    XScale,
    YScale,
    check_trace,
    decode_plot_settings,
    parse_plot_settings,
    read_plot_settings,
    render_plot_settings,
    scale_names,
    scales_of,
    with_panes,
    write_plot_settings,
)
from tests.ltspice_recorder import INPUTS


def pane(
    *traces: str,
    analysis: PlotAnalysis = "tran",
    x: XScale | None = None,
    y: YScale | None = None,
) -> PlotPane:
    return PlotPane(traces=traces, scales=scales_of(analysis, x, y))


def section_of(settings: PlotSettings, name: str) -> PlotSection:
    found = settings.section(name)
    assert found is not None, f"no [{name}] section"
    return found


def written(*sections: tuple[PlotAnalysis, list[PlotPane]]) -> bytes:
    settings = PlotSettings()
    for analysis, panes in sections:
        settings = with_panes(settings, analysis, panes)
    return write_plot_settings(settings)


TWO_PANES = [pane("V(out)"), pane("V(in)", "I(R1)")]

#: What the server wrote for each file the recorder hands LTspice to read
#: (``inputs/plot``, the plot-settings-read cases).
SERVER_WRITTEN: dict[str, list[tuple[PlotAnalysis, list[PlotPane]]]] = {
    "plot/two_panes.plt": [("tran", TWO_PANES)],
    "plot/math.plt": [("tran", [pane("V(in)-V(out)", "V(out)*I(R1)")])],
    "plot/log_y.plt": [("tran", [pane("V(out)", y="log")])],
    "plot/ac.plt": [("ac", [pane("V(out)", analysis="ac")])],
    "plot/two_sections.plt": [
        ("tran", [pane("V(out)")]),
        ("ac", [pane("V(out)", analysis="ac")]),
    ],
}


class TestWhatTheWriterWrites:
    def test_utf16_le_without_a_byte_order_mark_and_lf_line_ends(self):
        data = written(("tran", TWO_PANES))
        assert not data.startswith(codecs.BOM_UTF16_LE)
        text = data.decode("utf-16-le")
        assert text.startswith("[Transient Analysis]\n{\n")
        assert "\r" not in text
        assert text.endswith("}\n")

    def test_a_section_lists_its_panes_bottom_first(self):
        text = written(("tran", TWO_PANES)).decode("utf-16-le")
        assert text.index('"V(in)"') < text.index('"V(out)"')
        assert "   Npanes: 2\n" in text

    def test_every_trace_gets_zero_for_its_id_and_its_axis(self):
        text = written(("tran", TWO_PANES)).decode("utf-16-le")
        assert '      traces: 2 {0,0,"V(in)"} {0,0,"I(R1)"}\n' in text

    def test_every_pane_has_a_log_line(self):
        text = written(("ac", [pane("V(out)", analysis="ac"), pane("V(in)", analysis="ac")]))
        assert text.decode("utf-16-le").count("      Log: 1 2 0\n") == 2

    def test_no_axis_range_is_written(self):
        text = written(("tran", TWO_PANES)).decode("utf-16-le")
        assert "X:" not in text and "Y[0]" not in text

    def test_what_it_writes_reads_back_as_the_same_panes(self):
        settings = read_plot_settings(
            written(("tran", TWO_PANES), ("ac", [pane("V(out)", analysis="ac")]))
        )
        assert [s.name for s in settings.sections] == ["Transient Analysis", "AC Analysis"]
        assert settings.sections[0].panes == tuple(TWO_PANES)
        assert settings.sections[1].panes == (pane("V(out)", analysis="ac"),)

    @pytest.mark.parametrize("name", sorted(SERVER_WRITTEN))
    def test_each_file_handed_to_ltspice_is_what_the_writer_writes_now(self, name: str):
        """A recording of how a build reads a server-written file is evidence for
        the writer only while the file it read is the writer's output."""
        assert (INPUTS / name).read_bytes() == written(*SERVER_WRITTEN[name])

    def test_the_file_with_a_spaced_trace_is_the_writers_form_with_spaces(self):
        """The writer refuses the spaces this file has (see TestTraces), so the
        file is its form for the same expression written without them."""
        text = written(("tran", [pane("V(in)-V(out)")])).decode("utf-16-le")
        spaced = text.replace("V(in)-V(out)", "V(in) - V(out)").encode("utf-16-le")
        assert (INPUTS / "plot/spaced.plt").read_bytes() == spaced

    def test_the_file_with_a_grid_line_is_the_writers_form_with_one_in_each_pane(self):
        """The writer writes no GridStyle line, so the file is its form for the
        same panes with one after each Log line, as LTspice XVII saves a pane it
        made with the waveform grid on."""
        text = written(*SERVER_WRITTEN["plot/two_panes.plt"]).decode("utf-16-le")
        grid = text.replace("      Log: 0 0 0\n", "      Log: 0 0 0\n      GridStyle: 1\n")
        assert grid.count("GridStyle") == 2
        assert (INPUTS / "plot/grid.plt").read_bytes() == grid.encode("utf-16-le")

    def test_the_utf8_file_handed_to_ltspice_is_the_same_text(self):
        text = written(*SERVER_WRITTEN["plot/two_panes.plt"]).decode("utf-16-le")
        assert (INPUTS / "plot/two_panes_utf8.plt").read_bytes() == text.encode("utf-8")


class TestScales:
    def test_a_pane_gets_the_analysis_default_unless_a_scale_is_named(self):
        assert scales_of("tran", None, None) == DEFAULT_SCALES["tran"] == (0, 0, 0)
        assert scales_of("ac", None, None) == DEFAULT_SCALES["ac"] == (1, 2, 0)
        assert scales_of("tran", None, "log") == (0, 1, 0)
        assert scales_of("ac", "linear", "linear") == (0, 0, 0)

    def test_a_log_line_reads_back_as_the_scales_that_wrote_it(self):
        for analysis in SECTION_NAMES:
            for x in get_args(XScale):
                for y in get_args(YScale):
                    assert scale_names(scales_of(analysis, x, y)) == {"x_scale": x, "y_scale": y}

    def test_a_pane_without_a_log_line_names_no_scale(self):
        assert scale_names(None) == {}


class TestReplacingASection:
    def test_the_other_sections_keep_their_text_and_place(self):
        source = (
            '[AC Analysis]\n{\n   Npanes: 1\n   {\n      traces: 1 {524290,0,"V(out)"}\n'
            "      X: ('K',0,10,0,100000)\n      Log: 1 2 0\n   }\n}\n"
            '[Transient Analysis]\n{\n   Npanes: 1\n   {\n      traces: 1 {0,0,"V(a)"}\n'
            "      Log: 0 0 0\n   }\n}\n"
        )
        changed = with_panes(parse_plot_settings(source), "tran", [pane("V(b)")])
        text = render_plot_settings(changed)
        assert text.startswith(source[: source.index("[Transient")])
        assert section_of(parse_plot_settings(text), "Transient Analysis").panes == (pane("V(b)"),)

    def test_a_new_section_goes_last(self):
        changed = with_panes(
            read_plot_settings(written(("tran", TWO_PANES))), "ac", [pane("V(out)", analysis="ac")]
        )
        assert [s.name for s in changed.sections] == ["Transient Analysis", "AC Analysis"]

    def test_no_panes_removes_the_section(self):
        source = read_plot_settings(
            written(("tran", TWO_PANES), ("ac", [pane("V(out)", analysis="ac")]))
        )
        changed = with_panes(source, "tran", [])
        assert [s.name for s in changed.sections] == ["AC Analysis"]
        assert with_panes(changed, "ac", []).sections == ()

    def test_a_pane_without_a_trace_is_refused(self):
        with pytest.raises(NetlistError, match="at least one trace"):
            with_panes(PlotSettings(), "tran", [PlotPane(traces=())])


class TestTraces:
    @pytest.mark.parametrize(
        "expression", ["V(out)", "V(in)-V(out)", "V(out)*I(R1)", "Ix(U1:OUT)", "V(µout)"]
    )
    def test_an_expression_is_written_as_given(self, expression: str):
        text = written(("tran", [pane(expression)])).decode("utf-16-le")
        assert f'{{0,0,"{expression}"}}' in text

    @pytest.mark.parametrize(
        "expression", ['V("out")', "V(out){1}", "V(a)\nV(b)", "V(a)\tV(b)", " "]
    )
    def test_an_expression_a_file_cannot_carry_is_refused(self, expression: str):
        with pytest.raises(NetlistError):
            check_trace(expression)
        with pytest.raises(NetlistError):
            with_panes(PlotSettings(), "tran", [pane(expression)])

    def test_whitespace_is_refused_with_the_spelling_that_is_read_whole(self):
        with pytest.raises(NetlistError, match="'V\\(in\\)-V\\(out\\)'"):
            check_trace("V(in) - V(out)")


class TestReading:
    def test_utf8_with_or_without_a_byte_order_mark(self):
        text = written(("tran", TWO_PANES)).decode("utf-16-le")
        for data in (text.encode("utf-8"), codecs.BOM_UTF8 + text.encode("utf-8")):
            assert decode_plot_settings(data) == text

    def test_utf16_with_a_byte_order_mark(self):
        data = written(("tran", TWO_PANES))
        assert decode_plot_settings(codecs.BOM_UTF16_LE + data) == data.decode("utf-16-le")

    def test_an_empty_file_has_no_sections(self):
        assert parse_plot_settings("").sections == ()
        assert parse_plot_settings("\n \n").sections == ()

    def test_text_outside_a_section_is_refused(self):
        text = written(("tran", TWO_PANES)).decode("utf-16-le")
        with pytest.raises(NetlistError, match="outside its sections"):
            parse_plot_settings(text + "孛慲獮")

    def test_a_brace_never_closed_is_refused(self):
        with pytest.raises(NetlistError, match="never closed"):
            parse_plot_settings("[Transient Analysis]\n{\n   Npanes: 1\n   {\n")

    def test_a_brace_inside_a_quoted_expression_does_not_end_the_section(self):
        settings = parse_plot_settings(
            '[Transient Analysis]\n{\n   Npanes: 1\n   {\n      traces: 1 {0,0,"V(a)}"}\n   }\n}\n'
        )
        assert section_of(settings, "Transient Analysis").panes[0].traces == ("V(a)}",)
