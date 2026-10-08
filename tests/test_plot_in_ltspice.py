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


# ---------------------------------------------------------------------------
# A sheet's run, opened from the sheet
# ---------------------------------------------------------------------------

BY_HAND = (
    "The results are beside the sheet: View > Visible Traces on it in LTspice opens "
    "them, and a click on a net then plots it."
)


def a_sheet(directory: Path, name: str = "amp") -> Path:
    sheet = directory / f"{name}.asc"
    shutil.copyfile(FIXTURES_DIR / "Draft1.asc", sheet)
    return sheet


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
    write(sheet.with_suffix(".raw"), b"the results of an earlier run")
    world = tmp_path / "world.json"
    one_window(state_no_sim, world)

    data = await plot_the_job(state_no_sim)

    assert data["ltspice"] == {
        "shown": True,
        "sheet": str(sheet),
        "results": str(sheet.with_suffix(".raw")),
        "plot_settings": str(sheet.with_suffix(".plt")),
        "panes": [["V(out)"]],
        "pid": PID,
        "version": "26.1.1",
        "note": "The plot is tied to the sheet: a click on a net there plots it.",
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
    assert window["panes"] == ["amp.raw"]
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

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["sheet"] == str(sheet)
    assert data["ltspice"]["results"] == str(raw)
    assert read_world(world)["windows"][0]["commands"] == ["Visible Traces"]


async def test_results_the_window_already_has_open_are_left_alone(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """LTspice does not read a results file again, and with them open its
    command asks which traces to show: nothing is replaced and nothing sent."""
    sheet = a_sheet(work_dir)
    a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), b"the results of an earlier run")
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window(panes=["amp.raw"])])

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == (
        "LTspice already has amp.raw open, and goes on showing the results it read; "
        "close that plot there and ask again"
    )
    assert "note" not in shown and "sheet" not in shown
    assert read(sheet.with_suffix(".raw")) == b"the results of an earlier run"
    assert not exists(sheet.with_suffix(".plt"))
    assert "commands" not in read_world(world)["windows"][0]
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


async def test_a_sheet_that_was_open_before_it_had_results_has_to_be_opened_again(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """LTspice looks for a sheet's results as it opens the sheet. One it opened
    with none beside it does nothing with the command, whatever is put there
    afterwards, so the reply says what will make it look again."""
    monkeypatch.setattr(ltspice_window, "_OPENED_S", 0.3)  # timing: the wait under test
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    world = tmp_path / "world.json"
    put_windows(state_no_sim, world, [a_window({str(sheet): as_ltspice_reads(sheet)})])

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is False
    assert shown["reason"] == (
        "LTspice had amp.asc open before these results were beside it, and looks for a "
        "sheet's results only as it opens the sheet"
    )
    assert shown["note"] == (
        "Close amp.asc in LTspice and ask again: it is then opened with these results, "
        "which are beside it now."
    )
    assert read(sheet.with_suffix(".raw")) == read(raw)
    assert read_world(world)["windows"][0]["commands"] == ["Visible Traces"]


async def test_a_sheet_ltspice_knows_has_results_is_shown_them_where_it_stands(
    state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """A sheet already open that had results when it was opened, or was run
    there: the file is replaced and the command opens what it holds now."""
    sheet = a_sheet(work_dir)
    raw = a_job_of(state_no_sim, sheet, work_dir)
    write(sheet.with_suffix(".raw"), b"the results of an earlier run")
    world = tmp_path / "world.json"
    put_windows(
        state_no_sim,
        world,
        [a_window({str(sheet): as_ltspice_reads(sheet)}, with_results=[str(sheet)])],
    )

    data = await plot_the_job(state_no_sim)

    shown = data["ltspice"]
    assert shown["shown"] is True
    assert shown["differs_from_file"] is False
    assert read(sheet.with_suffix(".raw")) == read(raw)
    assert read_world(world)["windows"][0]["panes"] == ["amp.raw"]


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

    def looked_at(results: Path, plot_name: str, panes: list[list[str]]) -> str | None:
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

    data = await plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], in_ltspice=True)

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
