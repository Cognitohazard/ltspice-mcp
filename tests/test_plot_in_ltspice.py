"""plot_waveform(in_ltspice=True): a run opened in the user's LTspice window.

A results file opened in LTspice shows an empty plot unless a plot settings
file beside it names the traces. The server writes that file
(``lib/plot_settings.py``, held to both builds' recordings by
``test_recorded_ltspice_plot_settings.py``) and then asks a running LTspice,
through the bridge it ships, to open the results. These go through the real
handler against the stand-in bridge (``tests/fake_ltspice_bridge.py``, replayed
against a recording of LTspice 26.1.1 in ``test_ltspice_bridge.py``).

A run of a sheet is opened from the sheet instead, by the sheet's own Visible
Traces command, which is what makes LTspice tie the plot to the sheet. The
stand-in for the window's frame is ``FakeFrame``; what the real one does is in
the bridge recording.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import ltspice_window, plot_settings
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.lib.plot_settings import (
    DEFAULT_SCALES,
    SECTION_NAMES,
    PlotPane,
    PlotSettings,
    analysis_of,
    holds_only_panes,
    read_plot_settings,
    with_panes,
    write_plot_settings,
)
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from tests import _ltspice_recorded as rec
from tests._ltspice_window import (
    PID,
    STARTED_PID,
    FakeFrame,
    FakeStart,
    a_window,
    as_ltspice_reads,
    fake_command,
    put_windows,
    read_world,
    write_world,
)
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    make_experiment_job,
    stage_recorded_fixture,
)

TRAN = SECTION_NAMES["tran"]
# What each build's own waveform window saved for panes it made.
SAVED_BY_A_BUILD = list(rec.per_build(rec.cases_of("plot-settings")))


def read(path: Path) -> bytes:
    return path.read_bytes()


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def panes_of(path: Path, section: str = TRAN) -> tuple[PlotPane, ...]:
    found = read_plot_settings(path.read_bytes()).section(section)
    assert found is not None, f"no [{section}] section in {path.name}"
    return found.panes


def was_written(path: str) -> bool:
    return Path(path).is_file()


def one_window(state: SessionState, world: Path) -> None:
    put_windows(state, world, [a_window()])


def holding(sheet: Path, **entry: Any) -> dict[str, Any]:
    """A window that has ``sheet`` open, holding it as the file reads."""
    return a_window({str(sheet): as_ltspice_reads(sheet)}, **entry)


#: What stands beside a sheet as its results before a newer run is shown.
EARLIER = b"the results of an earlier run"


async def plot(state: SessionState, **arguments: Any) -> dict[str, Any]:
    result = await handle_plot_waveform(PlotWaveformInput(**arguments), state)
    assert result.structured_content is not None
    return result.structured_content


# ---------------------------------------------------------------------------
# Whose file it is
# ---------------------------------------------------------------------------


class TestWhoseSettings:
    def test_a_file_written_from_panes_is_one_written_here(self):
        written = with_panes(
            PlotSettings(),
            "ac",
            [
                PlotPane(("V(out)", "V(in)"), DEFAULT_SCALES["ac"]),
                PlotPane(("I(R1)",), DEFAULT_SCALES["ac"]),
            ],
        )
        assert holds_only_panes(write_plot_settings(written))

    @pytest.mark.parametrize(("build", "case_id"), SAVED_BY_A_BUILD)
    def test_a_file_a_build_saved_is_not(self, build: str, case_id: str):
        assert not holds_only_panes(rec.recorded(build, f"{case_id}.plt").read_bytes())

    def test_a_file_that_is_not_plot_settings_is_not(self):
        assert not holds_only_panes(b"something else that was beside the results")

    def test_a_section_is_found_by_the_name_a_raw_file_gives_its_plot(self):
        assert analysis_of("Transient Analysis") == "tran"
        assert analysis_of("AC Analysis") == "ac"
        assert analysis_of("DC transfer characteristic") == "dc"
        assert analysis_of("Noise Spectral Density - (V/Hz½ or A/Hz½)") == "noise"
        assert analysis_of("Operating Point") is None


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


async def test_the_run_is_opened_in_ltspice_with_its_traces_named(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"] == {
        "shown": True,
        "results": str(raw),
        "plot_settings": str(raw.with_suffix(".plt")),
        "panes": [["V(out)"]],
        "pid": PID,
        "version": "26.1.1",
    }
    assert panes_of(raw.with_suffix(".plt")) == (PlotPane(("V(out)",), DEFAULT_SCALES["tran"]),)
    window = read_world(world)["windows"][0]
    assert window["shown"] == window["panes"] == [str(raw)]
    assert "closed" not in window
    # Asked for in LTspice, the chart is not opened in a browser as well.
    assert data["opened"] is False
    # The chart and its numbers are made all the same.
    assert was_written(data["path"])
    assert data["traces"]


REOPENED = (
    "LTspice already had a plot of these results open; it was closed and reopened to "
    "show the current results, and traces added to it by hand were lost."
)


def another_run_of_the_same_name(raw: Path, directory: Path) -> Path:
    """The results of another run of the deck ``raw`` came from, where a job
    keeps them: under the same name, in a directory of its own."""
    directory.mkdir(parents=True)
    return Path(shutil.copyfile(raw, directory / raw.name))


async def test_results_already_open_are_closed_and_opened_again_to_draw_the_traces(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice reads a results file and the plot settings beside it once, as
    it opens them, so a pane that is open would go on drawing the traces asked
    for last time. It closes without a question, so it is closed and opened."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window(panes=[str(raw)])])

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["reopened"] is True
    assert shown["note"] == REOPENED
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(raw)]
    # Opened after the settings were written, which is when LTspice reads them.
    assert window["shown"] == window["panes"] == [str(raw)]


async def test_the_results_of_another_run_of_that_name_are_left_open(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """Two runs of one deck leave results files of one name, and a pane is
    titled with the name: the file is found by its path, in front or not."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    other = another_run_of_the_same_name(raw, work_dir / "runs" / "earlier")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window(panes=[str(other)], plot_in_front=True)])

    first = (await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True))[
        "ltspice"
    ]

    assert first["shown"] is True and "reopened" not in first
    window = read_world(world)["windows"][0]
    assert "closed" not in window
    assert window["panes"] == [str(raw), str(other)]

    # With both open and the other one in front, the one asked for is the one closed.
    put_windows(state_no_sim, world, [a_window(panes=[str(other), str(raw)], plot_in_front=True)])

    again = (await plot(state_no_sim, path=str(raw), signals=["V(in)"], in_ltspice=True))[
        "ltspice"
    ]

    assert again["reopened"] is True
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(raw)]
    assert window["panes"] == [str(raw), str(other)]


async def test_results_open_on_their_own_that_will_not_close_are_reported(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ltspice_window, "_CLOSED_S", 0.3)  # timing: the wait under test
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window(panes=[str(raw)])], frame_keeps_panes=True)

    shown = (await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True))[
        "ltspice"
    ]

    assert shown["shown"] is False
    assert shown["reason"] == (
        "LTspice has ltspice_tran_rc.raw open and did not close it within 0.3 s, so it "
        "goes on showing the results it read; close that plot there and ask again"
    )


async def test_nothing_is_closed_while_ltspice_waits_on_a_box_of_its_own(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """A pane closed from outside then would go from under whatever the box
    is about, so it is left, and the reply says what is in the way."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window(panes=[str(raw)])], box_up=True)

    shown = (await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True))[
        "ltspice"
    ]

    assert shown["shown"] is False
    assert shown["reason"] == (
        "LTspice is waiting for an answer to a box of its own, so nothing was closed in "
        "it; answer that box and ask again"
    )
    assert "closed" not in read_world(world)["windows"][0]


async def test_named_panels_become_panes(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    one_window(state_no_sim, tmp_path / "world.json")

    data = await plot(
        state_no_sim,
        path=str(raw),
        panels=[["V(out)"], ["V(in)"]],
        in_ltspice=True,
        open=False,
    )

    assert data["ltspice"]["panes"] == [["V(out)"], ["V(in)"]]
    # Top first, as the window shows them.
    assert [pane.traces for pane in panes_of(raw.with_suffix(".plt"))] == [("V(out)",), ("V(in)",)]


@pytest.mark.parametrize(("build", "case_id"), SAVED_BY_A_BUILD)
async def test_settings_saved_from_ltspice_are_left_as_they_are(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, build: str, case_id: str
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    saved = rec.recorded(build, f"{case_id}.plt").read_bytes()
    raw.with_suffix(".plt").write_bytes(saved)
    one_window(state_no_sim, tmp_path / "world.json")

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] is None
    assert "was saved from LTspice and is left as it is" in data["ltspice"]["note"]
    assert read(raw.with_suffix(".plt")) == saved
    # What the window draws in place of the trace asked for: the transient
    # panes the saved file holds, which is none when it was saved from a run
    # of another kind.
    section = read_plot_settings(saved).section(TRAN)
    held = plot_settings.pane_specs(section.panes if section is not None else ())
    assert data["ltspice"]["saved_panes"] == held
    if case_id == "plot/one_trace":
        assert held == [{"traces": ["V(out)"], "x_scale": "linear", "y_scale": "linear"}]
    # There is no sheet here, so no edit that could replace them is named.
    assert "set_plot_panes" not in data["ltspice"]["note"]


async def test_settings_written_before_are_replaced(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    one_window(state_no_sim, tmp_path / "world.json")
    await plot(state_no_sim, path=str(raw), signals=["V(in)"], in_ltspice=True)

    await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert [pane.traces for pane in panes_of(raw.with_suffix(".plt"))] == [("V(out)",)]


async def test_the_panes_of_another_analysis_are_kept(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """The file beside a results file is the one beside a sheet of that name,
    where set_plot_panes may have put an AC run's panes: showing a transient
    run replaces the transient section and no other."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    ac_panes = (PlotPane(("V(out)", "V(in)"), DEFAULT_SCALES["ac"]),)
    write(raw.with_suffix(".plt"), write_plot_settings(with_panes(PlotSettings(), "ac", ac_panes)))
    one_window(state_no_sim, tmp_path / "world.json")

    await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert panes_of(raw.with_suffix(".plt"), SECTION_NAMES["ac"]) == ac_panes
    assert [pane.traces for pane in panes_of(raw.with_suffix(".plt"))] == [("V(out)",)]


async def test_an_ac_run_is_given_the_scales_ltspice_gives_one(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    one_window(state_no_sim, tmp_path / "world.json")

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert panes_of(raw.with_suffix(".plt"), SECTION_NAMES["ac"]) == (
        PlotPane(("V(out)",), DEFAULT_SCALES["ac"]),
    )


@pytest.mark.parametrize(
    ("fixture", "analysis", "signal"),
    [("ltspice_dc_div", "dc", "V(out)"), ("ltspice_noise_rc", "noise", "v(onoise)")],
)
async def test_a_dc_sweep_and_a_noise_run_open_with_their_traces_too(
    state_no_sim: SessionState,
    work_dir: Path,
    tmp_path: Path,
    fixture: str,
    analysis: Any,
    signal: str,
):
    raw = stage_recorded_fixture(work_dir, fixture)
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, path=str(raw), signals=[signal], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] == str(raw.with_suffix(".plt"))
    # Under the section LTspice keeps that analysis's panes in, on its scales.
    assert panes_of(raw.with_suffix(".plt"), SECTION_NAMES[analysis]) == (
        PlotPane((signal,), DEFAULT_SCALES[analysis]),
    )
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]


async def test_a_run_whose_plot_settings_are_not_recorded_is_opened_without_any(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """A results file whose plot name is not one LTspice gives a run: this one
    is ngspice's, whose noise plot has a name of its own."""
    raw = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, path=str(raw), dialect="ngspice", in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] is None
    assert "is not recorded" in data["ltspice"]["note"]
    assert "saved_panes" not in data["ltspice"]
    assert not raw.with_suffix(".plt").exists()
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]


async def test_with_no_window_open_nothing_is_started_and_the_traces_are_still_written(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [])

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is False
    assert data["ltspice"]["reason"] == "no LTspice window is open, and none is started for this"
    assert data["ltspice"]["plot_settings"] == str(raw.with_suffix(".plt"))
    assert "Opened by hand in LTspice" in data["ltspice"]["note"]
    assert [pane.traces for pane in panes_of(raw.with_suffix(".plt"))] == [("V(out)",)]
    assert was_written(data["path"])


async def test_where_windows_cannot_be_reached_it_says_so_and_writes_nothing(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.open_windows = OpenWindows(None, unavailable="needs the server on Windows itself")

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"] == {
        "shown": False,
        "results": str(raw),
        "plot_settings": None,
        "panes": [["V(out)"]],
        "reason": "LTspice windows cannot be reached here: needs the server on Windows itself",
    }
    assert not raw.with_suffix(".plt").exists()


async def test_without_the_argument_ltspice_is_not_asked(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], open=False)

    assert "ltspice" not in data
    assert not raw.with_suffix(".plt").exists()
    assert "shown" not in read_world(world)["windows"][0]


# ---------------------------------------------------------------------------
# A sheet's run, opened from the sheet
# ---------------------------------------------------------------------------

BY_HAND = (
    "The results are beside the sheet: View > Visible Traces on it in LTspice opens "
    "them, and a click on a net then plots it."
)
TIED = "The plot is tied to the sheet: a click on a net there plots it."
SHEET_REOPENED = (
    "LTspice opened the sheet before it had results and only looks for results when it "
    "opens a sheet, so the sheet was closed and reopened."
)
ALONE = (
    "The plot is not tied to the sheet, so clicking a net there does not plot it: "
    "LTspice opened the sheet before it had results and has unsaved changes to it, so "
    "the sheet was not closed and reopened. After the user saves or closes it in "
    "LTspice, showing the run again ties the plot."
)


def a_sheet(directory: Path, name: str = "amp") -> Path:
    sheet = directory / f"{name}.asc"
    shutil.copyfile(FIXTURES_DIR / "Draft1.asc", sheet)
    return sheet


def with_its_plot_open(sheet: Path, **entry: Any) -> dict[str, Any]:
    """A window that has ``sheet`` open and the results beside it opened from it."""
    results = str(sheet.with_suffix(".raw"))
    return holding(sheet, panes=[results], tied=[results], with_results=[str(sheet)], **entry)


def a_job_of(state: SessionState, sheet: Path, work_dir: Path) -> Path:
    """A finished job that ran ``sheet``, its results where a job keeps them:
    under another name, in another directory. Returns the results file."""
    kept = work_dir / "runs" / "shown"
    kept.mkdir(parents=True)
    raw = stage_recorded_fixture(kept, "ltspice_tran_rc")
    job = make_experiment_job(state, job_id="shown", raw=raw)
    job.cases[0].circuit_path = sheet
    return raw


async def plot_the_job(state: SessionState) -> dict[str, Any]:
    """Plot the job ``a_job_of`` made, asking for it to be shown in LTspice."""
    return await plot(state, job_id="shown", signals=["V(out)"], in_ltspice=True)


def exists(path: Path) -> bool:
    return path.exists()


async def test_a_sheets_run_is_opened_from_the_sheet(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), EARLIER)
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot_the_job(state_no_sim)

    assert data["ltspice"] == {
        "shown": True,
        "sheet": str(sheet),
        "results": str(sheet.with_suffix(".raw")),
        "plot_settings": str(sheet.with_suffix(".plt")),
        "panes": [["V(out)"]],
        "tied": True,
        "pid": PID,
        "version": "26.1.1",
        "note": TIED,
    }
    # The job's results and log are the sheet's now, and its own are as they were.
    assert read(sheet.with_suffix(".raw")) == read(raw)
    assert read(sheet.with_suffix(".log")) == read(raw.with_suffix(".log"))
    assert [pane.traces for pane in panes_of(sheet.with_suffix(".plt"))] == [("V(out)",)]
    assert not exists(raw.with_suffix(".plt"))
    # The sheet was opened and put in front, and its own command opened the
    # results: never the bridge's way, which leaves a plot tied to nothing.
    window = read_world(world)["windows"][0]
    assert window["active"] == str(sheet)
    assert window["commands"] == ["Visible Traces"]
    assert window["panes"] == window["tied"] == [str(sheet.with_suffix(".raw"))]
    assert "shown" not in window
    # The results were beside the sheet before LTspice opened it, which is
    # when it looks for them.
    assert window["with_results"] == [str(sheet)]


async def test_results_beside_the_sheet_of_their_name_are_opened_from_it(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """What a run in LTspice leaves: nothing is copied, and the plot is tied."""
    sheet = a_sheet(work_dir, "ltspice_tran_rc")
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["sheet"] == str(sheet)
    assert data["ltspice"]["results"] == str(raw)
    assert read_world(world)["windows"][0]["commands"] == ["Visible Traces"]


async def test_a_plot_already_open_is_closed_and_opened_again_with_the_newer_results(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice does not read a results file again while its pane is open, and
    with the pane open its command asks which traces to show. The pane closes
    without a question, so it is closed, and the newer results opened."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), EARLIER)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [with_its_plot_open(sheet)])

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["reopened"] is True
    assert shown["note"] == f"{TIED} {REOPENED}"
    assert read(sheet.with_suffix(".raw")) == read(raw)
    window = read_world(world)["windows"][0]
    # The plot alone: the sheet it was opened from stays as it is.
    assert window["closed"] == [str(sheet.with_suffix(".raw"))]
    assert window["commands"] == ["Visible Traces"]
    assert window["panes"] == [str(sheet.with_suffix(".raw"))]
    assert "sheet_reopened" not in shown


@pytest.mark.usefixtures("quick_to_tell")
async def test_results_of_that_name_open_on_their_own_do_not_pass_for_the_sheets(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """A netlist of the sheet's name ran too, and its results are open on
    their own: they are another file, so they stay, and what is waited for is
    the pane the sheet's command opens."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), EARLIER)
    other = another_run_of_the_same_name(sheet.with_suffix(".raw"), work_dir / "runs" / "netlist")
    world = tmp_path / "world.json"
    put_windows(
        state_no_sim,
        world,
        [holding(sheet, panes=[str(other)], plot_in_front=True, with_results=[str(sheet)])],
    )

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is True and shown["tied"] is True
    assert "reopened" not in shown
    window = read_world(world)["windows"][0]
    assert "closed" not in window
    assert window["panes"] == [str(sheet.with_suffix(".raw")), str(other)]

    # Nor is a sheet that knows of no results taken for one that opened them:
    # it is closed and opened again all the same.
    put_windows(state_no_sim, world, [holding(sheet, panes=[str(other)], plot_in_front=True)])

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["sheet_reopened"] is True
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(sheet)]
    assert window["panes"] == [str(sheet.with_suffix(".raw")), str(other)]
    # Its menu says it knows of none, so it is not sent the command until it is reopened.
    assert window["commands"] == ["Visible Traces"]


@pytest.mark.usefixtures("quick_to_tell")
async def test_where_the_menu_cannot_be_read_the_command_is_tried_first(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """Without the menu to say whether the sheet knows of results, it is sent
    the command where it stands, and opened again when nothing comes of it."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [holding(sheet)], menu_unread=True)

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert (shown["tied"], shown["sheet_reopened"]) == (True, True)
    assert read_world(world)["windows"][0]["commands"] == ["Visible Traces", "Visible Traces"]


async def test_a_plot_that_was_not_open_is_not_said_to_be_opened_again(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is True
    assert "reopened" not in shown
    assert "closed" not in read_world(world)["windows"][0]


async def test_a_plot_that_will_not_close_leaves_the_results_beside_the_sheet_alone(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ltspice_window, "_CLOSED_S", 0.3)  # timing: the wait under test
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), EARLIER)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [with_its_plot_open(sheet)], frame_keeps_panes=True)

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == (
        "LTspice has amp.raw open and did not close it within 0.3 s, so it goes on "
        "showing the results it read; close that plot there and ask again"
    )
    assert "note" not in shown and "sheet" not in shown
    assert read(sheet.with_suffix(".raw")) == EARLIER
    assert not exists(sheet.with_suffix(".plt"))
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(sheet.with_suffix(".raw"))]
    assert "commands" not in window
    # The chart and its numbers are made all the same.
    assert was_written(data["path"])


async def test_a_build_without_the_command_leaves_the_results_for_the_person_to_open(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    put_windows(state_no_sim, tmp_path / "world.json", [a_window()], frame_has_no_command=True)

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == "this LTspice build's sheet menu has no 'Visible Traces' command"
    assert shown["note"] == BY_HAND
    assert read(sheet.with_suffix(".raw")) == read(raw)
    assert shown["plot_settings"] == str(sheet.with_suffix(".plt"))


async def test_a_window_that_does_not_open_them_is_reported(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ltspice_window, "_OPENED_S", 0.3)  # timing: the wait under test
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    put_windows(state_no_sim, tmp_path / "world.json", [a_window()], frame_ignores=True)

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == "LTspice did not open amp.raw within 0.3 s"
    assert shown["note"] == BY_HAND


@pytest.fixture
def quick_to_tell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sheet that knows of no results is given up on sooner than in use."""
    monkeypatch.setattr(ltspice_window, "_LOOKED_S", 0.2)  # timing: the wait under test


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_that_was_open_before_it_had_results_is_closed_and_opened_again(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice looks for a sheet's results as it opens the sheet. One it opened
    with none beside it does nothing with the command, whatever is put there
    afterwards, so the sheet is closed and opened again, which a sheet nobody
    changed does without a question."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [holding(sheet)])

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["tied"] is True
    assert shown["sheet_reopened"] is True
    assert shown["note"] == f"{TIED} {SHEET_REOPENED}"
    # It holds the file now, so there is no copy of the window's to compare.
    assert "differs_from_file" not in shown
    assert read(sheet.with_suffix(".raw")) == read(raw)
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(sheet)]
    assert "answers" not in window
    # Only once opened again: with no results beside it, it could know of none.
    assert window["commands"] == ["Visible Traces"]
    assert window["with_results"] == [str(sheet)]
    assert window["panes"] == [str(sheet.with_suffix(".raw"))]


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_the_window_was_handed_is_closed_past_ltspices_question(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice counts a sheet it was handed through the bridge as changed,
    which every sheet this server edited while it was open is, and asks
    whether to save it. The window holds what the file does, so the answer is
    No, and neither is any different for it."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    on_disk = read(sheet)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [holding(sheet, unsaved=[str(sheet)])])

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is True
    assert shown["tied"] is True
    assert shown["sheet_reopened"] is True
    window = read_world(world)["windows"][0]
    assert window["answers"] == ["no"]
    assert "asking" not in window and not window["unsaved"]
    assert window["designs"] == {str(sheet): as_ltspice_reads(sheet)}
    assert read(sheet) == on_disk


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_with_changes_nobody_saved_is_left_open_and_the_plot_stands_alone(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """Closing it would lose what someone did there, so it is not asked to
    close at all. The results are shown all the same, on their own, and the
    reply says the plot is not tied and what the window holds that the file
    does not."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    held = as_ltspice_reads(sheet).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window({str(sheet): held}, unsaved=[str(sheet)])])

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    beside = sheet.with_suffix(".raw")
    assert shown["shown"] is True
    assert shown["tied"] is False
    assert shown["sheet"] == str(sheet) and shown["results"] == str(beside)
    assert shown["note"] == ALONE
    assert shown["differs_from_file"] is True
    assert "sheet_reopened" not in shown
    assert read(beside) == read(raw)
    window = read_world(world)["windows"][0]
    assert "closed" not in window and "answers" not in window
    assert window["designs"] == {str(sheet): held}
    # Opened the bridge's way, which ties a plot to nothing.
    assert window["shown"] == window["panes"] == [str(beside)]
    assert "tied" not in window

    # Saved there, the window holds what the file does, and the next showing
    # ties the plot: the pane that stood alone is closed for it.
    write(sheet, held.encode("cp1252"))
    put_windows(
        state_no_sim,
        world,
        [a_window({str(sheet): held}, unsaved=[str(sheet)], panes=[str(beside)])],
    )

    again = (await plot_the_job(state_no_sim))["ltspice"]

    assert (again["tied"], again["reopened"], again["sheet_reopened"]) == (True, True, True)
    window = read_world(world)["windows"][0]
    assert window["closed"] == [str(beside), str(sheet)]
    assert window["panes"] == window["tied"] == [str(beside)]


class FrameWhoseSheetChangesAsItAsks(FakeFrame):
    """A window someone changes the sheet in between its being read and its
    being told to close: by the time LTspice asks, it holds ``text``."""

    def __init__(self, world: Path, sheet: Path, text: str) -> None:
        super().__init__(world)
        self._sheet = str(sheet)
        self._text = text

    def close(self, pane: int) -> None:
        super().close(pane)
        changed = read_world(self._world)
        for entry in changed["windows"]:
            if entry.get("asking") == self._sheet:
                entry["designs"][self._sheet] = self._text
        self._keep(changed)


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_changed_before_ltspice_asks_is_not_closed(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """What the window holds is read again while LTspice asks, when nobody
    can change it any more. It is not the file's, so the answer is Cancel."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    changed = as_ltspice_reads(sheet).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    world = tmp_path / "world.json"
    write_world(world, [holding(sheet, unsaved=[str(sheet)])])
    state_no_sim.open_windows = OpenWindows(
        fake_command(world),
        timeout=LIVENESS_S,
        frame=FrameWhoseSheetChangesAsItAsks(world, sheet, changed),
    )

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is True
    assert shown["tied"] is False
    assert shown["differs_from_file"] is True
    window = read_world(world)["windows"][0]
    assert window["answers"] == ["cancel"]
    assert window["designs"] == {str(sheet): changed}


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_that_will_not_close_leaves_the_results_for_the_person_to_open(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ltspice_window, "_CLOSED_S", 0.3)  # timing: the wait under test
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    put_windows(state_no_sim, tmp_path / "world.json", [holding(sheet)], frame_keeps_panes=True)

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is False
    assert shown["reason"] == "LTspice did not close amp.asc within 0.3 s"
    assert shown["note"] == BY_HAND


@pytest.mark.usefixtures("quick_to_tell")
async def test_a_sheet_is_not_closed_while_ltspice_waits_on_a_box_of_its_own(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [holding(sheet)], box_up=True)

    shown = (await plot_the_job(state_no_sim))["ltspice"]

    assert shown["shown"] is False
    assert shown["reason"].startswith("LTspice is waiting for an answer to a box of its own")
    assert shown["note"] == BY_HAND
    assert "closed" not in read_world(world)["windows"][0]


async def test_a_sheet_ltspice_knows_has_results_is_shown_them_where_it_stands(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """A sheet already open that had results when it was opened, or was run
    there: the file is replaced and the command opens what it holds now."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), EARLIER)
    world = tmp_path / "world.json"
    put_windows(
        state_no_sim,
        world,
        [holding(sheet, with_results=[str(sheet)])],
    )

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["differs_from_file"] is False
    assert "sheet_reopened" not in shown
    assert read(sheet.with_suffix(".raw")) == read(raw)
    window = read_world(world)["windows"][0]
    assert window["panes"] == [str(sheet.with_suffix(".raw"))]
    assert "closed" not in window


async def test_with_no_window_nothing_is_put_beside_the_sheet(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    put_windows(state_no_sim, tmp_path / "world.json", [])

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown == {
        "shown": False,
        "results": str(raw),
        "plot_settings": None,
        "panes": [["V(out)"]],
        "reason": "no LTspice window is open, and none is started for this",
    }
    assert not exists(sheet.with_suffix(".raw"))
    assert not exists(sheet.with_suffix(".plt"))


async def test_a_window_whose_copy_of_the_sheet_differs_says_so(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """The run was of the file. A window with changes nobody saved shows
    another sheet beside the plot, and the reply says which."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    held = as_ltspice_reads(sheet).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    put_windows(
        state_no_sim,
        tmp_path / "world.json",
        [a_window({str(sheet): held}, with_results=[str(sheet)])],
    )

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["differs_from_file"] is True
    # The resistor whose value changed, named as an entry of the sheet is.
    assert shown["difference"] == (
        "only in the window: SYMBOL res 128 112 R90. only in the file: SYMBOL res 128 112 R90"
    )


async def test_a_sheet_outside_the_sandbox_has_nothing_written_beside_it(
    state_no_sim: SessionState,
    work_dir: Path,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    """The run is still shown, on its own, from where the job keeps it."""
    sheet = a_sheet(tmp_path_factory.mktemp("elsewhere"))
    raw = a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert "sheet" not in shown
    assert shown["results"] == str(raw)
    assert not exists(sheet.with_suffix(".raw"))
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]


async def test_a_sheets_saved_traces_are_named_with_the_edit_that_replaces_them(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """The person saved the sheet's plot settings from LTspice. They are left as
    they are, and the reply says which traces the window draws and how to put
    others there, since a caller asked for different ones."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    saved = rec.recorded(SAVED_BY_A_BUILD[0][0], "plot/math.plt").read_bytes()
    write(sheet.with_suffix(".plt"), saved)
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["panes"] == [["V(out)"]]
    # In the form the edit takes its panes, so they can be passed to it with others.
    assert shown["saved_panes"] == [
        {"traces": ["V(in)-V(out)", "V(out)*I(R1)"], "x_scale": "linear", "y_scale": "linear"}
    ]
    assert shown["plot_settings"] is None
    assert shown["note"] == (
        f"{TIED} amp.plt was "
        "saved from LTspice and is left as it is, so the window shows the traces saved "
        "in it. saved_panes lists them as the set_plot_panes op of edit_schematic takes "
        "panes. To draw others, replace them with that op on the sheet; the axis ranges "
        "saved with them go too."
    )
    assert read(sheet.with_suffix(".plt")) == saved


def lock_is_held(target: Path) -> bool:
    """Whether the lock an edit takes for ``target`` is someone's right now."""
    try:
        with file_lock(Store.circuit_lock(target), timeout=0):
            return False
    except TimeoutError:
        return True


class FrameThatLooksAtALock(FakeFrame):
    """Notes whether ``target``'s lock is held each time LTspice is sent a command."""

    def __init__(self, world: Path, target: Path) -> None:
        super().__init__(world)
        self._target = target
        self.held: list[bool] = []

    def send(self, pid: int, label: str) -> None:
        self.held.append(lock_is_held(self._target))
        super().send(pid, label)


async def test_a_sheets_plot_settings_are_locked_for_their_write_and_no_longer(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A sheet's plot settings are an edit's to write too (``set_plot_panes``),
    under a lock an edit waits ten seconds for. Being started and opening the
    results can take LTspice longer than that, so the lock is held while the
    file is written and is free again by the time LTspice is asked."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    settings = sheet.with_suffix(".plt")
    world = tmp_path / "world.json"
    write_world(world, [a_window()])
    frame = FrameThatLooksAtALock(world, settings)
    state_no_sim.open_windows = OpenWindows(fake_command(world), timeout=LIVENESS_S, frame=frame)
    held_for_the_write: list[bool] = []
    write_beside = plot_settings.write_beside

    def looked_at(
        results: Path, plot_name: str, panes: list[list[str]]
    ) -> plot_settings.Kept | None:
        held_for_the_write.append(lock_is_held(settings))
        return write_beside(results, plot_name, panes)

    monkeypatch.setattr(plot_settings, "write_beside", looked_at)

    data = await plot_the_job(state_no_sim)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] == str(settings)
    assert held_for_the_write == [True]
    assert frame.held == [False]


# ---------------------------------------------------------------------------
# No window open: LTspice is started for it
# ---------------------------------------------------------------------------


async def test_with_no_window_open_ltspice_is_started_for_a_run(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    start = FakeStart(world)
    put_windows(state_no_sim, world, [], start=start)

    data = await plot(state_no_sim, path=str(raw), signals=["V(out)"], in_ltspice=True)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["started"] is True
    assert shown["pid"] == STARTED_PID
    assert start.calls == 1
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]


async def test_a_started_ltspice_opens_a_sheets_run_from_the_sheet(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice is started with no document, so the sheet is opened in it only
    once the results are beside it, which is when LTspice looks for them."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [], start=FakeStart(world))

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["started"] is True
    assert shown["sheet"] == str(sheet)
    assert read(sheet.with_suffix(".raw")) == read(raw)
    window = read_world(world)["windows"][0]
    assert window["with_results"] == [str(sheet)]
    assert window["commands"] == ["Visible Traces"]


async def test_an_ltspice_that_cannot_be_started_leaves_a_sheets_results_alone(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [], start=FakeStart(world, error=OSError("access is denied")))

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == "LTspice could not be started: access is denied"
    assert not exists(sheet.with_suffix(".raw"))
