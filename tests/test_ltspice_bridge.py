"""The client for LTspice's own MCP bridge, and what it models about a window.

Everything LTspice-specific here is held to recordings of LTspice 26.1.1
(``tests/fixtures/ltspice_bridge_recorded``, made by
``tests/ltspice_bridge_recorder.py``): how a window's copy of a sheet differs
from the file it read, and what the bridge answers to each call the server
makes. The stand-in bridge the rest of the suite uses is replayed against the
same recording, so a test that passes against it passes for a recorded reason.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import hidden_desktop, ltspice_window, windows_job
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.ltspice_bridge import (
    BRIDGE_NAME,
    BridgeError,
    BridgeSession,
    Instance,
    bridge_command,
)
from ltspice_mcp.lib.ltspice_window import (
    NoWindow,
    OpenDesign,
    OpenSheet,
    OpenWindows,
    WindowsUnavailable,
    content_difference,
    sheet_content,
)
from ltspice_mcp.lib.ltspice_windows import start_in_view
from tests._ltspice_window import (
    STARTED_PID,
    FakeStart,
    a_window,
    fake_command,
    read_world,
    write_world,
)
from tests.conftest import LIVENESS_S
from tests.ltspice_bridge_recorder import (
    FIXTURES,
    INPUTS,
    NEUTRAL_DIR,
    NEUTRAL_PID,
    input_names,
    load_conversation,
    load_manifest,
    observed,
    recorded_builds,
    sha256_bytes,
)
from tests.test_hidden_desktop import own_desktop, windows_only

BUILDS = recorded_builds()
#: The calls the server makes. The recorder also runs a sheet in the window, which it never does.
SERVER_CALLS = {
    "status",
    "attach",
    "list_open_designs",
    "get_active_design_path",
    "get_design_content",
    "set_design_content",
    "get_raw_info",
    "bring_to_front",
    "open_design",
}


def recorded_sheet(build: str, name: str) -> str:
    return (FIXTURES / build / "sheets" / f"{name}.asc").read_text(encoding="utf-8")


def input_sheet(name: str) -> str:
    return decode_spice_bytes((INPUTS / f"{name}.asc").read_bytes())


# ---------------------------------------------------------------------------
# The recordings themselves
# ---------------------------------------------------------------------------


def test_a_build_with_a_bridge_is_recorded():
    assert BUILDS, "no bridge recording is committed"


@pytest.mark.parametrize("build", BUILDS)
class TestRecording:
    def test_every_recorded_file_is_present_and_unchanged(self, build: str):
        directory = FIXTURES / build
        manifest = load_manifest(directory)
        on_disk = {
            path.relative_to(directory).as_posix(): sha256_bytes(path.read_bytes())
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        assert on_disk == manifest["files"]

    def test_every_sheet_was_recorded_from_the_input_that_is_committed(self, build: str):
        manifest = load_manifest(FIXTURES / build)
        assert manifest["inputs"] == {
            name: sha256_bytes((INPUTS / f"{name}.asc").read_bytes()) for name in input_names()
        }

    def test_what_the_recording_observed(self, build: str):
        """The facts the server relies on, as LTspice showed them."""
        assert observed(build) == {
            # What OpenWindows relies on when it starts LTspice.
            "an LTspice started with no document is offered to the bridge as a window": True,
            "it has no document open and none in front": [[], None],
            "the window then reads back exactly what it was given": True,
            "the file is as it was": True,
            "after the file is rewritten the window still holds its own copy": True,
            # The window held 2k and the file 3k when the run was made.
            "a run in the window is of the window's copy and not of the file": "2k",
            "the run did not write the sheet": True,
            "what the run left beside the sheet": [".asc", ".log", ".net", ".op.raw", ".raw"],
            "the results file is then the one LTspice has in front": True,
            # What the window's frame shows and does (lib/ltspice_frame.py).
            "the frame has a pane titled for an open sheet, and none for results "
            "it has not opened": [True, False],
            "results put beside a sheet that was already open are not opened by its command": True,
            "results beside a sheet when it is opened are opened by its command": True,
            "it asks nothing on the way": True,
            "with those results open the same command asks which traces to show": (
                "Select Visible Waveforms"
            ),
            "no LTspice was started in its place": True,
        }


# ---------------------------------------------------------------------------
# A window's copy of a sheet against the file it read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", BUILDS)
class TestSheetAsAWindowHoldsIt:
    @pytest.mark.parametrize("name", input_names())
    def test_a_sheet_ltspice_only_opened_has_the_files_content(self, build: str, name: str):
        assert content_difference(input_sheet(name), recorded_sheet(build, name)) is None

    @pytest.mark.parametrize(
        ("name", "in_file", "in_window"),
        [
            ("older_version", "Version 4\n", "Version 4.1\n"),
            ("symbol_attributes", 'SYMATTR Value2 ""\n', "SYMATTR Value 4.7µ\n"),
            (
                "text_off_grid",
                "TEXT -20 532 Left 2 ;x-20 y532\n",
                "TEXT -24 536 Left 2 ;x-20 y532\n",
            ),
            ("micro_sign", "SYMATTR Value 1µ\n", "SYMATTR Value 1µ\n"),
        ],
    )
    def test_each_input_shows_the_rewriting_it_is_there_for(
        self, build: str, name: str, in_file: str, in_window: str
    ):
        """Without this a recording that came back unchanged would pass the test above."""
        assert in_file in input_sheet(name)
        assert in_window in recorded_sheet(build, name)
        if name != "micro_sign":
            assert input_sheet(name) != recorded_sheet(build, name)

    def test_wires_come_back_in_another_order(self, build: str):
        wires = [
            line for line in input_sheet("older_version").split("\n") if line.startswith("WIRE")
        ]
        held = [
            line
            for line in recorded_sheet(build, "older_version").split("\n")
            if line.startswith("WIRE")
        ]
        assert wires != held
        assert sorted(wires) == sorted(held)

    def test_text_goes_to_the_nearest_grid_point_and_a_tie_away_from_zero(self, build: str):
        moved = {
            line.split(";")[1]: tuple(line.split()[1:3])
            for line in recorded_sheet(build, "text_off_grid").split("\n")
            if line.startswith("TEXT") and ";" in line
        }
        assert moved["x-418 y468"] == ("-416", "472")
        assert moved["x20 y500"] == ("24", "504")
        assert moved["x-20 y532"] == ("-24", "536")
        assert moved["x64 y-100"] == ("64", "-104")
        assert moved["x51 y596"] == ("48", "600")


class TestContentDifference:
    SHEET = input_sheet("older_version")

    def test_a_changed_value_is_named_on_both_sides(self):
        changed = self.SHEET.replace("SYMATTR Value 1k", "SYMATTR Value 2k")
        said = content_difference(self.SHEET, changed)
        assert said is not None
        assert said.summary() == (
            "only in the window: SYMBOL res 128 112 R90. only in the file: SYMBOL res 128 112 R90"
        )
        # In full, each side has the part with the value it holds.
        (in_window,), (in_file,) = said.only_in_window, said.only_in_file
        assert in_window.split("\n")[0] == in_file.split("\n")[0] == "SYMBOL res 128 112 R90"
        assert "SYMATTR Value 2k" in in_window.split("\n")
        assert "SYMATTR Value 1k" in in_file.split("\n")

    def test_an_added_wire_is_named(self):
        said = content_difference(self.SHEET, self.SHEET + "WIRE 336 128 240 128\n")
        assert said is not None
        assert said.summary() == "only in the window: WIRE 336 128 240 128"
        assert (said.only_in_window, said.only_in_file) == (("WIRE 336 128 240 128",), ())

    def test_a_long_difference_is_cut_and_counted(self):
        extra = "".join(f"WIRE {x} 0 {x} 16\n" for x in range(0, 160, 16))
        said = content_difference(self.SHEET, self.SHEET + extra)
        assert said is not None
        assert said.summary(limit=2) == (
            "only in the window: WIRE 0 0 0 16; WIRE 112 0 112 16; and 8 more"
        )
        # The sentence is cut; the difference itself holds every entry.
        assert len(said.only_in_window) == 10

    def test_line_endings_do_not_count(self):
        assert content_difference(self.SHEET, self.SHEET.replace("\n", "\r\n")) is None

    def test_a_moved_component_counts(self):
        moved = self.SHEET.replace("SYMBOL cap 224 144 R0", "SYMBOL cap 224 160 R0")
        assert content_difference(self.SHEET, moved) is not None

    def test_an_attribute_keeps_to_its_own_symbol(self):
        """Two symbols swapping values is a change, though the lines are the same set."""
        swapped = self.SHEET.replace("SYMATTR Value 1k", "SYMATTR Value TMP")
        swapped = swapped.replace("SYMATTR Value 100n", "SYMATTR Value 1k")
        swapped = swapped.replace("SYMATTR Value TMP", "SYMATTR Value 100n")
        assert sorted(swapped.split("\n")) == sorted(self.SHEET.split("\n"))
        assert content_difference(self.SHEET, swapped) is not None

    def test_content_is_one_entry_per_symbol_and_per_other_line(self):
        content = sheet_content(self.SHEET)
        assert len([entry for entry in content if entry.startswith("SYMBOL")]) == 3
        assert not any(
            entry.startswith(("Version", "SHEET", "SYMATTR", "WINDOW")) for entry in content
        )


# ---------------------------------------------------------------------------
# The stand-in bridge against the recorded conversation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", BUILDS)
def test_the_stand_in_answers_as_ltspice_was_recorded_answering(build: str, tmp_path: Path):
    world = tmp_path / "world.json"
    # LTspice was started with the first sheet and asked to open the others.
    started_with, *opened_later = input_names()
    window = {
        "pid": NEUTRAL_PID,
        "version": next(
            step["reply"]["instances"][0]["version"]
            for step in load_conversation(FIXTURES / build)
            if step["note"] == "one window is open"
        ),
        "designs": {f"{NEUTRAL_DIR}\\{started_with}.asc": recorded_sheet(build, started_with)},
    }
    files = {f"{NEUTRAL_DIR}\\{name}.asc": recorded_sheet(build, name) for name in opened_later}
    # The sheet the recorder puts results beside and then opens.
    files[f"{NEUTRAL_DIR}\\placed.asc"] = recorded_sheet(build, "older_version")
    steps = [
        step for step in load_conversation(FIXTURES / build) if step.get("call") in SERVER_CALLS
    ]
    # The recorder's three stretches: nothing running, a window, the window gone.
    first_with_window = next(
        i for i, step in enumerate(steps) if step["note"] == "one window is open"
    )
    first_after = next(
        i for i, step in enumerate(steps) if step["note"] == "the window has closed"
    )

    def replay(session: BridgeSession, step: dict[str, Any]) -> None:
        if "error" in step:
            with pytest.raises(BridgeError) as refused:
                session.call(step["call"], **step["arguments"])
            assert str(refused.value) == step["error"], step["note"]
            return
        reply: Any = session.call(step["call"], **step["arguments"])
        if isinstance(reply.get("text"), str):
            reply["text"] = f"<{len(reply['text'].splitlines())} lines>"
        assert reply == step["reply"], step["note"]

    write_world(world, [])
    with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
        for step in steps[:first_with_window]:
            replay(session, step)
    write_world(world, [window], results=[f"{NEUTRAL_DIR}\\older_version.raw"], files=files)
    with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
        for step in steps[first_with_window:first_after]:
            replay(session, step)
        write_world(world, [])
        for step in steps[first_after:]:
            replay(session, step)


# ---------------------------------------------------------------------------
# A session
# ---------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path: Path) -> Path:
    path = tmp_path / "world.json"
    write_world(
        path,
        [{"pid": 4242, "version": "26.1.1", "designs": {"C:\\work\\a.asc": "Version 4.1\n"}}],
    )
    return path


class TestSession:
    def test_it_lists_the_windows_and_reads_one(self, world: Path):
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            assert session.instances() == [Instance(pid=4242, mode="gui", version="26.1.1")]
            session.attach(4242)
            assert session.open_designs() == ["C:\\work\\a.asc"]
            assert session.design_text("C:\\work\\a.asc") == "Version 4.1\n"

    def test_it_says_which_document_is_in_front(self, world: Path, tmp_path: Path):
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            session.attach(4242)
            assert session.active_design() == "C:\\work\\a.asc"

    def test_a_window_with_no_document_has_none_in_front(self, world: Path):
        write_world(world, [{"pid": 4242, "version": "26.1.1", "designs": {}}])
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            session.attach(4242)
            assert session.active_design() is None

    def test_replacing_a_sheet_says_whether_it_changed(self, world: Path):
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            session.attach(4242)
            assert session.replace_design_text("C:\\work\\a.asc", "Version 4.1\n") is False
            assert session.replace_design_text("C:\\work\\a.asc", "Version 4.1\nWIRE 0 0 16 0\n")
            assert session.design_text("C:\\work\\a.asc") == "Version 4.1\nWIRE 0 0 16 0\n"

    def test_text_with_a_nul_character_is_never_sent(self, world: Path):
        """LTspice 26.1.1 stops answering for good when it is handed such text
        (seen with a UTF-16 sheet read as an 8-bit one), and in a window that
        takes every open document with it."""
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            session.attach(4242)
            with pytest.raises(BridgeError, match="NUL character"):
                session.replace_design_text("C:\\work\\a.asc", "Version 4" + chr(0) + ".1\n")
            # By name too: the refusal is where every call passes.
            with pytest.raises(BridgeError, match="NUL character"):
                session.call("set_design_content", path="C:\\work\\a.asc", text=chr(0))
            assert session.design_text("C:\\work\\a.asc") == "Version 4.1\n"

    def test_a_window_that_is_not_there_cannot_be_attached_to(self, world: Path):
        with (
            BridgeSession(fake_command(world), timeout=LIVENESS_S) as session,
            pytest.raises(BridgeError, match="no live LTspice instance with that pid"),
        ):
            session.attach(1)

    def test_a_sheet_no_window_has_open_is_refused(self, world: Path):
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            session.attach(4242)
            with pytest.raises(BridgeError, match="document not found"):
                session.design_text("C:\\work\\other.asc")

    def test_a_bridge_that_stops_answering_is_ended_at_the_deadline(self, world: Path):
        write_world(world, [], silent_on="status")
        started: list[BridgeSession] = []

        def ask() -> None:
            with BridgeSession(  # timing: the deadline under test
                fake_command(world), timeout=0.5
            ) as session:
                started.append(session)
                session.instances()

        # The stand-in never answers, so what ends the wait is the deadline. It
        # is the whole session's, its start included, so on a loaded machine it
        # can fall before the request that is never answered: the same end.
        with pytest.raises(BridgeError, match=r"did not answer within 0\.5 s and was ended"):
            ask()
        for session in started:
            assert session._process.wait(timeout=LIVENESS_S) is not None

    def test_a_bridge_that_exits_is_reported(self, tmp_path: Path):
        with pytest.raises(BridgeError, match="closed before answering initialize"):
            BridgeSession([sys.executable, "-c", "pass"], timeout=LIVENESS_S)

    def test_a_bridge_that_cannot_be_started_is_reported(self, tmp_path: Path):
        with pytest.raises(BridgeError, match="could not be started"):
            BridgeSession([str(tmp_path / "absent.exe")], timeout=LIVENESS_S)

    def test_the_bridge_is_gone_once_the_session_is_closed(self, world: Path):
        session = BridgeSession(fake_command(world), timeout=LIVENESS_S)
        session.close()
        assert session._process.wait(timeout=LIVENESS_S) is not None


@windows_only
class TestWhereTheBridgeRuns:
    """The bridge starts an LTspice of its own when it has none to talk to. On
    the server's hidden desktop whatever it starts has its windows there too,
    where they cannot take the keyboard focus (the opt-in tier does that with
    the bridge itself: ``TestSheetOpenInAWindow``)."""

    def test_it_runs_on_the_servers_hidden_desktop(self, world: Path):
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            where = session.call("where_am_i")["desktop"]
        desktop = hidden_desktop.shared()
        assert desktop is not None
        assert where == desktop.name
        assert where != own_desktop()

    def test_with_that_desktop_turned_off_it_runs_on_the_callers(self, world: Path):
        hidden_desktop.configure(enabled=False)
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            assert session.call("where_am_i")["desktop"] == own_desktop()


class TestBridgeCommand:
    def test_a_build_with_no_bridge_beside_it_has_none(self, tmp_path: Path):
        assert bridge_command(tmp_path / "LTspice.exe") is None

    def test_the_command_names_an_ltspice_that_does_not_exist(self, tmp_path: Path):
        """Which makes the bridge's launch of one fail (the recording's 'a request
        that needs LTspice, with none running'). The second line of two: where
        the bridge runs is the first (``TestWhereTheBridgeRuns``)."""
        (tmp_path / BRIDGE_NAME).write_bytes(b"")
        command = bridge_command(tmp_path / "LTspice.exe")
        assert command is not None
        assert command[:2] == [str(tmp_path / BRIDGE_NAME), "--ltspice-path"]
        assert not Path(command[2]).exists()


# ---------------------------------------------------------------------------
# The windows a session keeps in step
# ---------------------------------------------------------------------------


class TestOpenWindows:
    def test_it_finds_the_window_holding_a_file(self, tmp_path: Path):
        sheet = tmp_path / "amp.asc"
        other = tmp_path / "other.asc"
        world = tmp_path / "world.json"
        write_world(
            world,
            [
                {"pid": 7, "version": "26.1.1", "designs": {str(other): "A\n"}},
                {"pid": 8, "version": "26.1.1", "designs": {str(sheet): "B\n", str(other): "A\n"}},
            ],
        )
        windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)
        assert windows.holding(sheet) == [
            OpenSheet(pid=8, version="26.1.1", path=str(sheet), text="B\n")
        ]
        assert [held.pid for held in windows.holding(other)] == [7, 8]
        assert windows.holding(tmp_path / "unopened.asc") == []

    @pytest.mark.skipif(sys.platform != "win32", reason="a Windows path is case-blind")
    def test_a_path_matches_in_any_case_and_either_separator(self, tmp_path: Path):
        sheet = tmp_path / "Amp.asc"
        world = tmp_path / "world.json"
        spelled = str(sheet).upper().replace("\\", "/")
        write_world(world, [{"pid": 8, "version": "26.1.1", "designs": {spelled: "B\n"}}])
        held = OpenWindows(fake_command(world), timeout=LIVENESS_S).holding(sheet)
        assert [found.path for found in held] == [spelled]

    def test_it_lists_every_document_and_reads_only_the_ones_asked_for(self, tmp_path: Path):
        sheet, deck = tmp_path / "amp.asc", tmp_path / "amp.net"
        world = tmp_path / "world.json"
        write_world(
            world,
            [
                {
                    "pid": 8,
                    "version": "26.1.1",
                    "designs": {str(sheet): "B\n", str(deck): "* deck\n"},
                    "active": str(sheet),
                },
                {"pid": 9, "version": "26.1.1", "designs": {}},
            ],
        )
        windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)
        count, designs = windows.designs(lambda spelled: spelled.endswith(".asc"))
        assert count == 2
        assert designs == [
            OpenDesign(pid=8, version="26.1.1", path=str(sheet), active=True, text="B\n"),
            OpenDesign(pid=8, version="26.1.1", path=str(deck), active=False, text=None),
        ]

    def test_with_no_bridge_every_question_says_why_it_cannot_be_asked(self, tmp_path: Path):
        windows = OpenWindows(None, unavailable="because")
        assert not windows.available
        assert windows.unavailable == "because"
        sheet = tmp_path / "amp.asc"
        for ask in (
            windows.designs,
            lambda: windows.holding(sheet),
            lambda: windows.open_sheet(sheet),
            lambda: windows.show_results(sheet.with_suffix(".raw")),
        ):
            with pytest.raises(
                WindowsUnavailable, match="LTspice windows cannot be reached here: because"
            ):
                ask()

    def test_showing_a_sheet_replaces_what_that_window_holds(self, tmp_path: Path):
        sheet = tmp_path / "amp.asc"
        world = tmp_path / "world.json"
        write_world(world, [{"pid": 8, "version": "26.1.1", "designs": {str(sheet): "B\n"}}])
        windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)
        assert windows.show_each(windows.holding(sheet), "C\n") == [None]
        assert read_world(world)["windows"][0]["designs"][str(sheet)] == "C\n"

    def test_a_window_that_closed_in_between_is_an_answer_not_an_error(self, tmp_path: Path):
        sheet = tmp_path / "amp.asc"
        world = tmp_path / "world.json"
        write_world(world, [{"pid": 8, "version": "26.1.1", "designs": {str(sheet): "B\n"}}])
        windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)
        held = windows.holding(sheet)
        write_world(world, [])
        assert windows.show_each(held, "C\n") == ["no live LTspice instance with that pid"]

    def test_the_setting_turns_it_off(self):
        windows = OpenWindows.detect(["C:\\anywhere\\LTspice.exe"], enabled=False)
        assert not windows.available
        assert windows.unavailable == "turned off by [schematic] sync_open_window"

    @pytest.mark.skipif(sys.platform == "win32", reason="off Windows there is no bridge to run")
    def test_off_windows_there_is_nothing_to_ask(self, tmp_path: Path):
        (tmp_path / BRIDGE_NAME).write_bytes(b"")
        windows = OpenWindows.detect([str(tmp_path / "LTspice.exe")])
        assert windows.unavailable == "needs the server to run on Windows itself"

    @pytest.mark.skipif(sys.platform != "win32", reason="the bridge is a Windows program")
    def test_on_windows_it_is_the_bridge_beside_a_detected_ltspice(self, tmp_path: Path):
        assert not OpenWindows.detect([str(tmp_path / "LTspice.exe")]).available
        (tmp_path / BRIDGE_NAME).write_bytes(b"")
        assert OpenWindows.detect([str(tmp_path / "LTspice.exe")]).available


# ---------------------------------------------------------------------------
# Starting LTspice for a caller asked to show something in it
# ---------------------------------------------------------------------------


class TestStartingLtspice:
    """Each of the three ways of showing something starts LTspice when no
    window is open; these go through the one that opens a results file."""

    def windows(self, world: Path, start: FakeStart | None) -> OpenWindows:
        return OpenWindows(fake_command(world), timeout=LIVENESS_S, start=start)

    def results(self, tmp_path: Path) -> Path:
        results = tmp_path / "run.raw"
        results.write_bytes(b"results")
        return results

    def test_with_no_window_open_it_is_started_and_its_window_waited_for(self, tmp_path: Path):
        world = tmp_path / "world.json"
        write_world(world, [])
        start = FakeStart(world)
        windows = self.windows(world, start)
        assert windows.starts_ltspice
        shown = windows.show_results(self.results(tmp_path))
        assert shown.started
        assert shown.window.pid == STARTED_PID
        assert start.calls == 1
        # It is shown in the window that was started, which had nothing open.
        (window,) = read_world(world)["windows"]
        assert window["shown"] == [str(tmp_path / "run.raw")]
        assert window["designs"] == {}

    @pytest.mark.parametrize("build", BUILDS)
    def test_the_stand_in_start_leaves_what_a_started_ltspice_was_recorded_leaving(
        self, build: str, tmp_path: Path
    ):
        facts = observed(build)
        assert facts["an LTspice started with no document is offered to the bridge as a window"]
        world = tmp_path / "world.json"
        write_world(world, [])
        FakeStart(world)()
        with BridgeSession(fake_command(world), timeout=LIVENESS_S) as session:
            assert [found.pid for found in session.instances()] == [STARTED_PID]
            session.attach(STARTED_PID)
            assert [session.open_designs(), session.active_design()] == facts[
                "it has no document open and none in front"
            ]

    def test_with_a_window_open_nothing_is_started(self, tmp_path: Path):
        world = tmp_path / "world.json"
        write_world(world, [a_window()])
        start = FakeStart(world)
        shown = self.windows(world, start).show_results(self.results(tmp_path))
        assert not shown.started
        assert start.calls == 0

    def test_where_starting_is_not_allowed_nothing_is_started(self, tmp_path: Path):
        world = tmp_path / "world.json"
        write_world(world, [])
        windows = self.windows(world, None)
        assert not windows.starts_ltspice
        with pytest.raises(NoWindow, match="no LTspice window is open, and none is started"):
            windows.show_results(self.results(tmp_path))
        assert read_world(world)["windows"] == []

    def test_a_start_that_fails_is_reported(self, tmp_path: Path):
        world = tmp_path / "world.json"
        write_world(world, [])
        windows = self.windows(world, FakeStart(world, error=OSError("access is denied")))
        with pytest.raises(BridgeError, match="LTspice could not be started: access is denied"):
            windows.show_results(self.results(tmp_path))

    def test_an_ltspice_that_opens_no_window_is_given_up_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(ltspice_window, "_STARTED_S", 0.3)  # timing: the wait under test
        world = tmp_path / "world.json"
        write_world(world, [])
        windows = self.windows(world, FakeStart(world, opens_window=False))
        with pytest.raises(BridgeError, match=r"did not open a window within 0\.3 s"):
            windows.show_results(self.results(tmp_path))

    def test_with_no_bridge_there_is_nothing_to_start_for(self, tmp_path: Path):
        started: list[bool] = []
        windows = OpenWindows(None, unavailable="because", start=lambda: started.append(True))
        assert not windows.starts_ltspice
        with pytest.raises(WindowsUnavailable):
            windows.show_results(self.results(tmp_path))
        assert not started

    @pytest.mark.skipif(sys.platform != "win32", reason="the bridge is a Windows program")
    def test_the_setting_decides_whether_the_detected_ltspice_is_started(self, tmp_path: Path):
        (tmp_path / BRIDGE_NAME).write_bytes(b"")
        executable = str(tmp_path / "LTspice.exe")
        assert not OpenWindows.detect([executable]).starts_ltspice
        assert OpenWindows.detect([executable], start_ltspice=True).starts_ltspice
        assert not OpenWindows.detect(
            [executable], enabled=False, start_ltspice=True
        ).starts_ltspice

    @pytest.mark.skipif(sys.platform != "win32", reason="a window is started through Windows")
    def test_the_start_is_of_the_program_alone_and_outside_the_servers_job(self, tmp_path: Path):
        """No document and no settings file are named, so LTspice opens as it
        does for the person; and it is asked for outside the server's job
        where that job lets a process leave, so that it does not end with a
        session. Nothing is started: the start is handed what would be."""
        started: list[tuple[list[str], int]] = []

        class Started:
            returncode: int | None = None

        def spawn(command: list[str], *, creationflags: int, **_streams: object) -> Started:
            started.append((command, creationflags))
            return Started()

        start_in_view(tmp_path / "LTspice.exe", spawn=spawn)
        apart = 0x00000008 | 0x00000200
        assert started == [
            ([str(tmp_path / "LTspice.exe")], apart | windows_job.detached_creation_flags())
        ]

    @pytest.mark.skipif(sys.platform == "win32", reason="off Windows there is no window to start")
    def test_off_windows_nothing_is_started(self, tmp_path: Path):
        with pytest.raises(OSError, match="started through Windows"):
            start_in_view(tmp_path / "LTspice.exe")
