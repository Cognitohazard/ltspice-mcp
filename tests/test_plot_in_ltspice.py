"""plot_waveform(in_ltspice=True): a run opened in the user's LTspice window.

A results file opened in LTspice shows an empty plot unless a plot settings
file beside it names the traces. The server writes that file and then asks a
running LTspice, through the bridge it ships, to open the results. These go
through the real handler against the stand-in bridge
(``tests/fake_ltspice_bridge.py``, replayed against a recording of LTspice
26.1.1 in ``test_ltspice_bridge.py``). That LTspice then draws the traces was
looked at, on 26.1.1, and cannot be asked of it: see ``lib/plot_settings.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import plot_settings
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from tests.conftest import LIVENESS_S, stage_recorded_fixture
from tests.ltspice_bridge_recorder import FIXTURES, load_manifest, recorded_builds
from tests.test_ltspice_bridge import fake_command, write_world

PID = 4242
ONE_PANE = (
    "[Transient Analysis]\r\n"
    "{\r\n"
    "   Npanes: 1\r\n"
    "   {\r\n"
    '      traces: 1 {524290,0,"V(out)"}\r\n'
    "   }\r\n"
    "}\r\n"
)
# What LTspice writes when a person saves the settings: the axes and the grid too.
SAVED_FROM_LTSPICE = (
    "[Transient Analysis]\r\n"
    "{\r\n"
    "   Npanes: 1\r\n"
    "   {\r\n"
    '      traces: 1 {524290,0,"V(in)"}\r\n'
    "      X: ('m',0,0,0.001,0.01)\r\n"
    "      Y[0]: (' ',1,0,0.5,5)\r\n"
    "      Log: 0 0 0\r\n"
    "      GridStyle: 1\r\n"
    "   }\r\n"
    "}\r\n"
)


def read(path: Path) -> bytes:
    return path.read_bytes()


def was_written(path: str) -> bool:
    return Path(path).is_file()


def world_of(world: Path) -> dict[str, Any]:
    return json.loads(world.read_text(encoding="utf-8"))


def a_window(state: SessionState, world: Path, **extra: Any) -> None:
    write_world(world, [{"pid": PID, "version": "26.1.1", "designs": {}}], **extra)
    state.open_windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)


async def plot(state: SessionState, **arguments: Any) -> dict[str, Any]:
    result = await handle_plot_waveform(PlotWaveformInput(**arguments), state)
    assert result.structured_content is not None
    return result.structured_content


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


class TestPlotSettings:
    def test_one_pane(self):
        assert (
            plot_settings.settings_text("Transient Analysis", [["V(out)"]]).replace("\n", "\r\n")
            == ONE_PANE
        )

    def test_panes_are_separated_and_each_numbers_its_own_traces(self):
        assert plot_settings.settings_text("AC Analysis", [["V(out)", "V(in)"], ["I(R1)"]]) == (
            "[AC Analysis]\n"
            "{\n"
            "   Npanes: 2\n"
            "   {\n"
            '      traces: 2 {524290,0,"V(out)"} {524291,0,"V(in)"}\n'
            "   },\n"
            "   {\n"
            '      traces: 1 {524290,0,"I(R1)"}\n'
            "   }\n"
            "}\n"
        )

    def test_a_file_holding_only_traces_is_one_written_here(self):
        assert plot_settings.holds_only_traces(ONE_PANE.encode("ascii"))
        two = plot_settings.settings_text("AC Analysis", [["V(a,b)", "2*V(out)"], ["I(R1)"]])
        assert plot_settings.holds_only_traces(two.encode("ascii"))

    def test_a_file_saved_from_ltspice_is_not(self):
        assert not plot_settings.holds_only_traces(SAVED_FROM_LTSPICE.encode("ascii"))
        assert not plot_settings.holds_only_traces(b"")
        assert not plot_settings.holds_only_traces("[x]".encode("utf-16"))

    def test_a_noise_runs_section_is_written_as_ltspice_writes_it(self, tmp_path: Path):
        results = tmp_path / "noise.raw"
        name = "Noise Spectral Density - (V/Hz½ or A/Hz½)"
        assert plot_settings.write_beside(results, name, [["V(onoise)"]]) is None
        assert b"(V/Hz\xbd or A/Hz\xbd)]\r\n" in read(tmp_path / "noise.plt")

    def test_a_trace_name_that_cannot_be_written_is_refused(self, tmp_path: Path):
        with pytest.raises(plot_settings.PlotSettingsError, match="cannot be written"):
            plot_settings.settings_text("Transient Analysis", [['V("out")']])
        with pytest.raises(plot_settings.PlotSettingsError, match="cannot"):
            plot_settings.write_beside(tmp_path / "a.raw", "Transient Analysis", [["V(节点)"]])
        assert not (tmp_path / "a.plt").exists()

    def test_a_pane_with_no_trace_is_refused(self):
        with pytest.raises(plot_settings.PlotSettingsError):
            plot_settings.settings_text("Transient Analysis", [["V(out)"], []])

    @pytest.mark.parametrize("build", recorded_builds())
    def test_the_file_is_in_the_shape_of_the_ones_ltspice_ships(self, build: str):
        """The section names and the two words written here are among those in
        the plot settings files installed with LTspice's own examples."""
        shipped = load_manifest(FIXTURES / build)["plot_settings"]
        assert shipped["files"] > 10
        assert {"Transient Analysis", "AC Analysis", "DC transfer characteristic"} <= set(
            shipped["sections"]
        )
        assert {"Npanes", "traces"} <= set(shipped["keys"])
        # What marks a file as saved from a window, and so as not written here.
        assert {"X", "Y", "GridStyle"} <= set(shipped["keys"])


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


async def test_the_run_is_opened_in_ltspice_with_its_traces_named(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    a_window(state_no_sim, world)

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
    assert read(raw.with_suffix(".plt")) == ONE_PANE.encode("ascii")
    assert world_of(world)["windows"][0]["shown"] == [str(raw)]
    # Asked for in LTspice, the chart is not opened in a browser as well.
    assert data["opened"] is False
    # The chart and its numbers are made all the same.
    assert was_written(data["path"])
    assert data["traces"]


async def test_named_panels_become_panes(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    a_window(state_no_sim, tmp_path / "world.json")

    data = await plot(
        state_no_sim,
        raw_file=str(raw),
        panels=[["V(out)"], ["V(in)"]],
        in_ltspice=True,
        open=False,
    )

    assert data["ltspice"]["panes"] == [["V(out)"], ["V(in)"]]
    assert b"   Npanes: 2\r\n" in read(raw.with_suffix(".plt"))


async def test_settings_saved_from_ltspice_are_left_as_they_are(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.with_suffix(".plt").write_bytes(SAVED_FROM_LTSPICE.encode("ascii"))
    a_window(state_no_sim, tmp_path / "world.json")

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["plot_settings"] is None
    assert "was saved from LTspice and is left as it is" in data["ltspice"]["note"]
    assert read(raw.with_suffix(".plt")) == SAVED_FROM_LTSPICE.encode("ascii")


async def test_settings_written_before_are_replaced(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    a_window(state_no_sim, tmp_path / "world.json")
    await plot(state_no_sim, raw_file=str(raw), signals=["V(in)"], in_ltspice=True)

    await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert read(raw.with_suffix(".plt")) == ONE_PANE.encode("ascii")


async def test_with_no_window_open_nothing_is_started_and_the_traces_are_still_written(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    world = tmp_path / "world.json"
    write_world(world, [])
    state_no_sim.open_windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is False
    assert data["ltspice"]["reason"] == "no LTspice window is open, and none is started for this"
    assert data["ltspice"]["plot_settings"] == str(raw.with_suffix(".plt"))
    assert "Opened by hand in LTspice" in data["ltspice"]["note"]
    assert read(raw.with_suffix(".plt")) == ONE_PANE.encode("ascii")
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
    a_window(state_no_sim, world)

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], open=False)

    assert "ltspice" not in data
    assert not raw.with_suffix(".plt").exists()
    assert "shown" not in world_of(world)["windows"][0]
