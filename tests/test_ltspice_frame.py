"""The frame of an LTspice window: its panes, and a command sent to it by label.

What a real frame does is in the bridge recording
(``tests/fixtures/ltspice_bridge_recorded``, the facts noted after "a results
file that is not there"), made by sending a real window the command on a
desktop of its own. These hold the code to that recording, and the stand-in
frame the handler tests run (``FakeFrame``) to the same facts.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from ltspice_mcp.lib import hidden_desktop, ltspice_frame, pe_menu
from ltspice_mcp.lib.ltspice_bridge import BridgeSession
from ltspice_mcp.lib.ltspice_frame import VISIBLE_TRACES, FrameError, LtspiceFrame, menu_command
from tests._ltspice_window import PID, FakeFrame, a_window, fake_command, write_world
from tests.conftest import LIVENESS_S
from tests.ltspice_bridge_recorder import FIXTURES, load_manifest, observed, recorded_builds

BUILDS = recorded_builds()
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="a window's frame is asked through Windows"
)


# ---------------------------------------------------------------------------
# The command, by its label
# ---------------------------------------------------------------------------


def a_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, menus: list) -> Path:
    """A file standing for a build whose menus are ``menus``."""
    program = tmp_path / "LTspice.exe"
    program.write_bytes(b"a build")
    monkeypatch.setattr(pe_menu, "menus", lambda _executable: menus)
    ltspice_frame._SHEET_COMMANDS.clear()
    return program


WAVEFORM_MENU = [(None, "&File"), (57603, "&Save Plot Settings"), (40001, "&Visible Traces")]
SHEET_MENU = [
    (None, "&File"),
    (None, "&Hierarchy"),
    (None, "&View"),
    (32791, "&Visible Traces"),
    (32845, "Visible Traces"),
]


def test_the_command_is_the_sheet_menus_and_the_first_of_that_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The waveform viewer's menu has an item of the same label; the sheet's
    menu is the one with a Hierarchy menu in it."""
    program = a_program(tmp_path, monkeypatch, [WAVEFORM_MENU, SHEET_MENU])
    assert menu_command(program, VISIBLE_TRACES) == 32791


def test_a_build_whose_sheet_menu_lacks_the_label_has_no_such_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    program = a_program(tmp_path, monkeypatch, [WAVEFORM_MENU, [(None, "&Hierarchy")]])
    assert menu_command(program, VISIBLE_TRACES) is None
    assert menu_command(a_program(tmp_path, monkeypatch, [WAVEFORM_MENU]), VISIBLE_TRACES) is None


@pytest.mark.parametrize("build", BUILDS)
def test_the_recorded_build_gives_the_label_a_command(build: str):
    recorded = load_manifest(FIXTURES / build)["sheet_commands"]
    assert set(recorded) == {VISIBLE_TRACES}
    assert isinstance(recorded[VISIBLE_TRACES], int)


# ---------------------------------------------------------------------------
# A frame
# ---------------------------------------------------------------------------


class Windows:
    """A desktop's windows, as the calls a frame makes would find them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.classes: dict[int, str] = {}
        self.texts: dict[int, str] = {}
        self.children: dict[int, list[int]] = {}
        self.posted: list[tuple[int, int]] = []
        monkeypatch.setattr(hidden_desktop, "window_class", lambda w: self.classes.get(w, ""))
        monkeypatch.setattr(hidden_desktop, "window_text", lambda w: self.texts.get(w, ""))
        monkeypatch.setattr(hidden_desktop, "child_windows", lambda w: self.children.get(w, []))
        monkeypatch.setattr(
            hidden_desktop, "post_command", lambda w, command: self.posted.append((w, command))
        )

    def frame(self, window: int, panes: dict[int, str]) -> None:
        self.classes[window] = "Afx:00007FF6:8"
        self.texts[window] = "LTspice - [amp.asc]"
        self.children[window] = list(panes)
        self.texts.update(panes)


@windows_only
def test_the_panes_are_the_titles_inside_the_frame(monkeypatch: pytest.MonkeyPatch):
    windows = Windows(monkeypatch)
    windows.texts[5] = "Tooltip"
    windows.frame(7, {71: "amp.asc", 72: "amp.raw", 73: "", 74: "amp.raw"})
    frame = LtspiceFrame(lambda _pid: [5, 7])
    assert frame.panes(PID) == ["amp.asc", "amp.raw"]


@windows_only
def test_a_process_with_no_frame_cannot_be_asked(monkeypatch: pytest.MonkeyPatch):
    Windows(monkeypatch)
    frame = LtspiceFrame(lambda _pid: [5])
    with pytest.raises(FrameError, match="has no window on this desktop"):
        frame.panes(PID)
    with pytest.raises(FrameError, match="has no window on this desktop"):
        frame.send(PID, VISIBLE_TRACES)


@pytest.mark.skipif(sys.platform == "win32", reason="off Windows there is no window to ask")
def test_off_windows_no_frame_is_asked():
    with pytest.raises(FrameError, match="reached through Windows"):
        LtspiceFrame(lambda _pid: [7]).panes(PID)


# ---------------------------------------------------------------------------
# The stand-in frame, held to what the real one was recorded doing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", BUILDS)
def test_the_stand_in_frame_opens_results_when_ltspice_was_recorded_opening_them(
    build: str, tmp_path: Path
):
    facts = observed(build)
    open_sheet, placed = tmp_path / "open.asc", tmp_path / "placed.asc"
    for sheet in (open_sheet, placed):
        sheet.write_text("Version 4.1\n", encoding="utf-8")
    world = tmp_path / "world.json"
    # A window that opened one sheet when there were no results beside it.
    write_world(world, [a_window({str(open_sheet): "Version 4.1\n"}, active=str(open_sheet))])
    frame = FakeFrame(world)

    panes = frame.panes(PID)
    assert [open_sheet.name in panes, "open.raw" in panes] == facts[
        "the frame has a pane titled for an open sheet, and none for results it has not opened"
    ]

    shutil.copyfile(open_sheet, open_sheet.with_suffix(".raw"))
    frame.send(PID, VISIBLE_TRACES)
    assert ("open.raw" not in frame.panes(PID)) is facts[
        "results put beside a sheet that was already open are not opened by its command"
    ]

    shutil.copyfile(placed, placed.with_suffix(".raw"))
    with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
        session.attach(PID)
        session.open_design(str(placed))
        session.bring_to_front(str(placed))
    frame.send(PID, VISIBLE_TRACES)
    assert ("placed.raw" in frame.panes(PID)) is facts[
        "results beside a sheet when it is opened are opened by its command"
    ]
