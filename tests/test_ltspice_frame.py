"""The frame of an LTspice window: its panes, a command sent to it by label,
a pane told to close, and the question LTspice asks before it closes a sheet.

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
from ltspice_mcp.lib.ltspice_frame import (
    VISIBLE_TRACES,
    FrameError,
    LtspiceFrame,
    SaveQuestion,
    menu_command,
)
from tests._ltspice_window import PID, FakeFrame, a_window, fake_command, read_world, write_world
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
        self.closed: list[int] = []
        monkeypatch.setattr(hidden_desktop, "post_close", self.closed.append)

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
def test_a_pane_is_the_window_of_its_title_whose_class_begins_as_the_frames_does(
    monkeypatch: pytest.MonkeyPatch,
):
    """A pane's title is on more than one window inside the frame, and two
    files of one name have a pane each: Windows lists them from the front."""
    windows = Windows(monkeypatch)
    windows.frame(7, {71: "amp.asc", 72: "amp.raw", 73: "AMP.RAW", 74: "other.raw", 75: "amp.raw"})
    pane_class = "Afx:00007FF6:b"
    windows.classes.update({72: "Static", 73: pane_class, 74: pane_class, 75: pane_class})
    frame = LtspiceFrame(lambda _pid: [7])
    assert frame.panes_titled(PID, "amp.raw") == [73, 75]
    assert frame.pane(PID, "amp.raw") == 73
    assert frame.pane(PID, "gone.raw") is None

    frame.close(73)
    assert windows.closed == [73]


BUTTON, WORDS, BOX = "Button", "Static", hidden_desktop.DIALOG_CLASS


def a_box(windows: Windows, box: int, parts: list[str]) -> None:
    """A box with ``parts`` in it: buttons, and last the words it says."""
    windows.classes[box] = BOX
    windows.children[box] = [box * 10 + place for place in range(len(parts))]
    for child, text in zip(windows.children[box], parts, strict=True):
        windows.texts[child] = text
        windows.classes[child] = BUTTON
    windows.classes[windows.children[box][-1]] = WORDS


@windows_only
@pytest.mark.parametrize("build", BUILDS)
def test_the_question_ltspice_was_recorded_asking_is_read_for_the_document_it_names(
    build: str, monkeypatch: pytest.MonkeyPatch
):
    (asked,) = observed(build)[
        "a sheet the window was handed is asked about before it closes, in these words"
    ]
    windows = Windows(monkeypatch)
    windows.frame(7, {71: "closing.asc"})
    a_box(windows, 9, asked)
    frame = LtspiceFrame(lambda _pid: [7, 9])

    question = frame.save_question(PID)

    assert question is not None
    assert question == SaveQuestion(9, "C:\\recording\\closing.asc")
    # Each answer is the box's own button for it, and neither saves.
    frame.answer(question, discard=True)
    frame.answer(question, discard=False)
    assert windows.posted == [(9, 7), (9, 2)]


@windows_only
def test_a_box_that_asks_anything_else_is_not_the_question(monkeypatch: pytest.MonkeyPatch):
    windows = Windows(monkeypatch)
    # The words of the question on a pane, and a box that says something else.
    windows.frame(7, {71: "Save changes to C:\\work\\amp.asc?"})
    windows.classes[71] = WORDS
    a_box(windows, 9, ["OK", "The file could not be read."])
    assert LtspiceFrame(lambda _pid: [7, 9]).save_question(PID) is None


@windows_only
def test_a_frame_takes_input_unless_windows_says_it_does_not(monkeypatch: pytest.MonkeyPatch):
    windows = Windows(monkeypatch)
    windows.frame(7, {71: "amp.asc"})
    off: set[int] = set()
    gone: set[int] = set()
    monkeypatch.setattr(hidden_desktop, "takes_input", lambda window: window not in off)
    monkeypatch.setattr(hidden_desktop, "window_exists", lambda window: window not in gone)
    frame = LtspiceFrame(lambda _pid: [7])
    assert frame.takes_input(PID) and frame.is_open(71)
    off.add(7)
    gone.add(71)
    assert not frame.takes_input(PID) and not frame.is_open(71)


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

    pane = frame.pane(PID, "placed.raw")
    assert pane is not None
    frame.close(pane)
    assert (not frame.is_open(pane) and frame.save_question(PID) is None) is facts[
        "a results pane told to close is closed, and LTspice asks nothing"
    ]
    assert (placed.name in frame.panes(PID)) is facts["the sheet it was opened from stays open"]
    frame.send(PID, VISIBLE_TRACES)
    # Opened again, the recorded pane held the newer file: so it was open again.
    held = facts[
        "the precision of the results the pane held, and of those it holds once opened again"
    ]
    assert (frame.pane(PID, "placed.raw") is not None) is (held == ["single", "double"])


@pytest.mark.parametrize("build", BUILDS)
def test_the_stand_in_finds_the_pane_in_front_as_ltspice_was_recorded_listing_them(
    build: str, tmp_path: Path
):
    """Two runs of one deck leave results of one name, and two folders can
    hold a sheet of one name: the pane found first is the one put in front."""
    facts = observed(build)
    sheets = [tmp_path / "amp.asc", tmp_path / "elsewhere" / "amp.asc"]
    runs = [tmp_path / "runs" / run / "run.raw" for run in ("one", "other")]
    for file in (*sheets, *runs):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Version 4.1\n", encoding="utf-8")
    world = tmp_path / "world.json"
    write_world(world, [a_window({str(sheet): "Version 4.1\n" for sheet in sheets})])
    frame = FakeFrame(world)

    with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
        session.attach(PID)
        for run in runs:
            session.show_results(str(run))
        in_front: list[str] = []
        found: list[int | None] = []
        for run in (runs[0], runs[1], runs[0]):
            session.bring_to_front(str(run))
            in_front.append(Path(session.results_in_front() or "").parent.name)
            found.append(frame.pane(PID, "run.raw"))
        assert in_front == facts["the results put in front, of two of one name"]
        assert (found[0] == found[2] != found[1]) is facts[
            "of two results panes of one title, the first found is the one in front"
        ]
        assert found[2] is not None
        frame.close(found[2])
        # One that is not open is not put in front: the other run's stay there.
        session.bring_to_front(str(runs[0]))
        assert (
            Path(session.results_in_front() or "").parent.name
            == (facts["with that one closed, the results left open are the other run's"])
        )

        found = []
        for sheet in (sheets[0], sheets[1], sheets[0]):
            session.bring_to_front(str(sheet))
            found.append(frame.pane(PID, "amp.asc"))
        assert (found[0] == found[2] != found[1]) is facts[
            "of two sheets of one title, the first pane found is the one in front"
        ]


@pytest.mark.parametrize("build", BUILDS)
def test_the_stand_in_closes_a_sheet_as_ltspice_was_recorded_closing_one(
    build: str, tmp_path: Path
):
    facts = observed(build)
    sheet = tmp_path / "closing.asc"
    sheet.write_text("Version 4.1\n", encoding="utf-8")
    world = tmp_path / "world.json"
    # A window that opened the sheet when there were no results beside it.
    write_world(world, [a_window({str(sheet): "Version 4.1\n"}, active=str(sheet))])
    frame = FakeFrame(world)
    plot = sheet.with_suffix(".raw")
    shutil.copyfile(sheet, plot)
    on_disk = sheet.read_bytes()

    def the_sheet() -> int:
        pane = frame.pane(PID, sheet.name)
        assert pane is not None
        return pane

    assert frame.takes_input(PID) is facts["the frame takes input while LTspice asks nothing"]
    pane = the_sheet()
    frame.close(pane)
    assert [not frame.is_open(pane), frame.save_question(PID) is None] == facts[
        "a sheet nobody changed, told to close, is closed, and LTspice asks nothing"
    ]

    with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
        session.attach(PID)
        session.open_design(str(sheet))
        frame.send(PID, VISIBLE_TRACES)
        assert (frame.pane(PID, plot.name) is not None) is facts[
            "opened again, it opens the results put beside it while it was open"
        ]
        pane = the_sheet()
        frame.close(pane)
        assert [not frame.is_open(pane), frame.pane(PID, plot.name) is None] == facts[
            "the plot opened from a sheet closes with it"
        ]

        session.open_design(str(sheet))
        held = session.design_text(str(sheet))
        session.replace_design_text(str(sheet), held + "TEXT 0 0 Left 2 ;a change\n")
        session.replace_design_text(str(sheet), held)
        pane = the_sheet()
        frame.close(pane)
        question = frame.save_question(PID)
        assert question is not None and frame.is_open(pane)
        assert (Path(question.path) == sheet) is facts[
            "the document the question names is the sheet"
        ]
        assert frame.takes_input(PID) is facts["the frame takes input while LTspice asks that"]
        # The bridge still reads the sheet while LTspice asks, as it was recorded doing.
        assert session.design_text(str(sheet)) == held

        frame.answer(question, discard=False)
        assert [
            frame.is_open(pane),
            session.design_text(str(sheet)) == held,
            frame.takes_input(PID),
        ] == facts[
            "answered Cancel, the sheet stays open, holds what it held, and the frame takes input"
        ]
        assert not frame.is_open(question.box)

        frame.close(pane)
        question = frame.save_question(PID)
        assert question is not None
        frame.answer(question, discard=True)
        assert [not frame.is_open(pane), sheet.read_bytes() == on_disk] == facts[
            "answered No, the sheet closes and its file is as it was"
        ]
    assert read_world(world)["windows"][0]["answers"] == ["cancel", "no"]
