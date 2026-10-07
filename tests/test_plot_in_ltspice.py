"""plot_waveform(in_ltspice=True): a run opened in the user's LTspice window.

A results file opened in LTspice shows an empty plot unless a plot settings
file beside it names the traces. The server writes that file
(``lib/plot_settings.py``, held to both builds' recordings by
``test_recorded_ltspice_plot_settings.py``) and then asks a running LTspice,
through the bridge it ships, to open the results. These go through the real
handler against the stand-in bridge (``tests/fake_ltspice_bridge.py``, replayed
against a recording of LTspice 26.1.1 in ``test_ltspice_bridge.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from tests import _ltspice_recorded as rec
from tests._ltspice_window import PID, a_window, put_windows, read_world
from tests.conftest import stage_recorded_fixture

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
        assert analysis_of("DC transfer characteristic") is None


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


async def test_the_run_is_opened_in_ltspice_with_its_traces_named(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"] == {
        "shown": True,
        "results": str(raw),
        "plot_settings": str(raw.with_suffix(".plt")),
        "panes": [["V(out)"]],
        "pid": PID,
        "version": "26.1.1",
        "note": (
            "A results file LTspice already had open keeps the traces it was showing; "
            "close it there and ask again to change them."
        ),
    }
    assert panes_of(raw.with_suffix(".plt")) == (PlotPane(("V(out)",), DEFAULT_SCALES["tran"]),)
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]
    # Asked for in LTspice, the chart is not opened in a browser as well.
    assert data["opened"] is False
    # The chart and its numbers are made all the same.
    assert was_written(data["path"])
    assert data["traces"]


async def test_named_panels_become_panes(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    one_window(state_no_sim, tmp_path / "world.json")

    data = await plot(
        state_no_sim,
        raw_file=str(raw),
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

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] is None
    assert "was saved from LTspice and is left as it is" in data["ltspice"]["note"]
    assert read(raw.with_suffix(".plt")) == saved


async def test_settings_written_before_are_replaced(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    one_window(state_no_sim, tmp_path / "world.json")
    await plot(state_no_sim, raw_file=str(raw), signals=["V(in)"], in_ltspice=True)

    await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

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

    await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert panes_of(raw.with_suffix(".plt"), SECTION_NAMES["ac"]) == ac_panes
    assert [pane.traces for pane in panes_of(raw.with_suffix(".plt"))] == [("V(out)",)]


async def test_an_ac_run_is_given_the_scales_ltspice_gives_one(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    one_window(state_no_sim, tmp_path / "world.json")

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert panes_of(raw.with_suffix(".plt"), SECTION_NAMES["ac"]) == (
        PlotPane(("V(out)",), DEFAULT_SCALES["ac"]),
    )


async def test_a_run_whose_plot_settings_are_not_recorded_is_opened_without_any(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_dc_div")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot(state_no_sim, raw_file=str(raw), in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] is None
    assert "is not recorded" in data["ltspice"]["note"]
    assert not raw.with_suffix(".plt").exists()
    assert read_world(world)["windows"][0]["shown"] == [str(raw)]


async def test_with_no_window_open_nothing_is_started_and_the_traces_are_still_written(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [])

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

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

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

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

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], open=False)

    assert "ltspice" not in data
    assert not raw.with_suffix(".plt").exists()
    assert "shown" not in read_world(world)["windows"][0]
