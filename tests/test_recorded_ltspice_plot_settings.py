"""The plot settings file, held to what LTspice 26 and LTspice XVII did with one.

Two behaviours are recorded (``inputs/cases.toml``): ``plot-settings``, the
file each build's waveform window saves for panes it made itself, and
``plot-settings-read``, what each build shows for a file the server wrote,
which is what it saves straight back. Every case runs an RC sheet in the window
with the file beside it, as a person opening the sheet does.

A read case whose file a build rejected has no saved file, or one holding other
panes, so every check of it fails rather than skips.
"""

from __future__ import annotations

import codecs

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.plot_settings import (
    DEFAULT_SCALES,
    SECTION_NAMES,
    PlotPane,
    check_trace,
    encode_plot_settings,
    read_plot_settings,
)
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS

WRITTEN = rec.cases_of("plot-settings")
READ = rec.cases_of("plot-settings-read")
TRAN = SECTION_NAMES["tran"]
AC = SECTION_NAMES["ac"]
CURRENT = next(build for build in rec.BUILDS if rec.generation(build) == "current")
XVII = next(build for build in rec.BUILDS if rec.generation(build) == "xvii")


def saved(build: str, case_id: str) -> bytes:
    return rec.recorded(build, f"{case_id}.plt").read_bytes()


def handed(case_id: str) -> bytes:
    """The file the server wrote that the case put beside the sheet."""
    return (INPUTS / rec.CASES.case(case_id).plot).read_bytes()


def panes(data: bytes, section: str) -> tuple[PlotPane, ...]:
    found = read_plot_settings(data).section(section)
    assert found is not None, f"no [{section}] section"
    return found.panes


def traces(data: bytes, section: str) -> list[tuple[str, ...]]:
    return [pane.traces for pane in panes(data, section)]


# --------------------------------------------------------------------------
# What a build writes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "case_id"),
    [pair for pair in rec.per_build(WRITTEN + READ) if pair[1] != "plot/read_utf8"],
)
def test_each_build_writes_its_own_encoding_with_lf_line_ends(build: str, case_id: str):
    data = saved(build, case_id)
    assert not data.startswith((codecs.BOM_UTF8, codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    if rec.generation(build) == "xvii":
        text = data.decode("utf-16-le")
    else:
        assert b"\0" not in data
        text = data.decode("utf-8")
    assert "\r" not in text
    assert text.startswith("[") and text.endswith("}\n")


def test_the_server_writes_the_form_ltspice_xvii_writes():
    """UTF-16 LE without a byte order mark, LF line ends: XVII's own form."""
    own = saved(XVII, "plot/one_trace")
    assert encode_plot_settings(own.decode("utf-16-le")) == own


@pytest.mark.parametrize("build", rec.BUILDS)
class TestWhatABuildWritesForPanesItMade:
    def test_one_trace(self, build: str):
        assert panes(saved(build, "plot/one_trace"), TRAN) == (
            PlotPane(traces=("V(out)",), scales=DEFAULT_SCALES["tran"]),
        )

    def test_an_ac_pane_is_log_frequency_and_decibel_magnitude(self, build: str):
        assert panes(saved(build, "plot/ac"), AC) == (
            PlotPane(traces=("V(out)",), scales=DEFAULT_SCALES["ac"]),
        )

    def test_math_expressions_are_kept_as_typed(self, build: str):
        assert panes(saved(build, "plot/math"), TRAN) == (
            PlotPane(traces=("V(in)-V(out)", "V(out)*I(R1)"), scales=DEFAULT_SCALES["tran"]),
        )

    def test_a_pane_made_with_the_waveform_grid_on_has_a_grid_line(self, build: str):
        assert panes(saved(build, "plot/math_grid"), TRAN) == (
            PlotPane(("V(in)-V(out)", "V(out)*I(R1)"), scales=DEFAULT_SCALES["tran"], grid=1),
        )
        assert panes(saved(build, "plot/ac_grid"), AC) == (
            PlotPane(("V(out)",), scales=DEFAULT_SCALES["ac"], grid=1),
        )

    def test_a_micro_sign_in_a_unit_is_in_the_builds_own_encoding(self, build: str):
        data = saved(build, "plot/math")
        micro = "µ".encode("utf-16-le" if rec.generation(build) == "xvii" else "utf-8")
        assert micro in data


def test_the_pane_ltspice_26_adds_below_is_listed_first():
    """The first pane listed is the bottom one.

    The case puts V(out) in the window's one pane, adds a pane *below* it with
    LTspice 26's Add Plot Pane Below Active Pane, and gives the new pane V(in)
    and I(R1). Read top first, the file is V(out) over V(in) and I(R1).
    """
    assert traces(saved(CURRENT, "plot/pane_below"), TRAN) == [("V(out)",), ("V(in)", "I(R1)")]


def test_the_pane_ltspice_xvii_adds_is_above_the_active_one():
    """XVII has one Add Plot Pane, with the command id LTspice 26 gives Add Plot
    Pane Above Active Pane. Read with the bottom-first order LTspice 26's own
    file shows, XVII's new pane is on top, where that command puts it."""
    assert traces(saved(XVII, "plot/pane_added"), TRAN) == [("V(in)", "I(R1)"), ("V(out)",)]


# --------------------------------------------------------------------------
# What a build makes of a file the server wrote
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(READ)))
def test_no_build_stops_on_a_file_the_server_wrote(build: str, case_id: str):
    entry = rec.entry(build, case_id)
    assert "dialog" not in entry, entry["dialog"]
    assert f"{case_id}.plt" in entry["outputs"], "the build saved no plot settings"


@pytest.mark.parametrize(
    ("build", "case_id"),
    list(
        rec.per_build(
            [
                "plot/read_two_panes",
                "plot/read_math",
                "plot/read_log_y",
                "plot/read_ac",
                "plot/read_grid",
                "plot/read_ac_grid",
            ]
        )
    ),
)
def test_a_build_shows_the_panes_traces_and_scales_written(build: str, case_id: str):
    written = read_plot_settings(handed(case_id))
    shown = read_plot_settings(saved(build, case_id))
    assert [s.name for s in shown.sections] == [s.name for s in written.sections]
    for section in written.sections:
        assert panes(saved(build, case_id), section.name) == section.panes


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_pane_read_without_a_grid_line_gets_no_grid_from_the_setting(build: str):
    """With the waveform grid on, a build saves a file the server wrote exactly
    as it does on its default: a pane read without a GridStyle line has none,
    so a pane written here shows no grid whatever the person's setting is."""
    assert saved(build, "plot/read_two_panes_grid") == saved(build, "plot/read_two_panes")


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_build_keeps_the_section_it_was_not_running(build: str):
    """A transient run saves its own section and writes the AC one back as it was."""
    data = saved(build, "plot/read_two_sections")
    written = read_plot_settings(handed("plot/read_two_sections"))
    shown = read_plot_settings(data)
    assert [s.name for s in shown.sections] == [TRAN, AC]
    assert panes(data, TRAN) == panes(handed("plot/read_two_sections"), TRAN)
    shown_ac, written_ac = shown.section(AC), written.section(AC)
    assert shown_ac is not None and written_ac is not None
    assert shown_ac.body == written_ac.body


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_trace_is_read_only_up_to_its_first_space(build: str):
    """Why the writer refuses whitespace in a trace."""
    assert traces(handed("plot/read_spaced"), TRAN) == [("V(in) - V(out)",)]
    assert traces(saved(build, "plot/read_spaced"), TRAN) == [("V(in)",)]
    with pytest.raises(NetlistError, match="first space"):
        check_trace("V(in) - V(out)")


def test_ltspice_26_reads_a_file_in_utf8():
    written = handed("plot/read_utf8")
    assert written.decode("utf-8").startswith("[Transient Analysis]")
    assert panes(saved(CURRENT, "plot/read_utf8"), TRAN) == panes(written, TRAN)


def test_ltspice_xvii_reads_a_file_in_utf8_and_appends_it_to_its_own_when_it_saves():
    """XVII shows the panes of a UTF-8 file, then saves its own UTF-16 section
    followed by the old file's bytes. That is the file the server would leave
    behind by writing UTF-8, and why it writes UTF-16; the reader refuses one."""
    written = handed("plot/read_utf8")
    data = saved(XVII, "plot/read_utf8")
    own = data[: data.index("}\n}\n".encode("utf-16-le")) + 8]
    assert panes(own, TRAN) == panes(written, TRAN)
    assert data[len(own) :].startswith(written)
    with pytest.raises(NetlistError, match="outside its sections"):
        read_plot_settings(data)
