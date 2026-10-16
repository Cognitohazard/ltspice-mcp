"""Records what LTspice's own MCP bridge does, for the tests that model it.

``lib/ltspice_bridge.py`` and ``lib/ltspice_window.py`` encode how a running
LTspice answers through ``ltspice-mcp-bridge.exe``: what a window hands back
for a sheet it has open, that replacing it leaves the file alone, that it
never reads the file again, that a run in the window is of the window's copy,
and that a bridge told where LTspice is not cannot start one.
``lib/ltspice_frame.py`` encodes what the window's own frame does when it is
asked directly: that its panes are titled with their files' names, that a
sheet's Visible Traces command opens the results put beside the sheet, that
with those results open the same command asks which traces to show, that a
results pane goes on holding what it read when its file is replaced, and that
it closes without a question, a trace added by hand or not. The same holds of
results opened on their own and the plot settings beside them, and of two
panes of one title the first found is the one in front. A sheet nobody
changed closes without a question too, and opened again finds the results put
beside it meanwhile; one the window was handed is asked about first, in a box
that can be answered while the bridge still reads the sheet. And
``OpenWindows`` relies on an LTspice it starts with no document being offered
to the bridge as a window, with nothing open in it. Each of those is recorded
here from an installed build, under
``tests/fixtures/ltspice_bridge_recorded/<build>/``:

- ``sheets/<name>.asc``: the window's copy of ``inputs/<name>.asc``, as UTF-8;
- ``conversation.json``: every call made and what came back, in order;
- ``reader/<build>/<case>.json``: what LTspice's own reader says of a results
  file the main recordings hold (below);
- ``manifest.json``: the build, and the digest of each input and recording.

The bridge can also read a results file: its samples, how a stepped run
divides, and the parameter values of each step. That is LTspice reading its
own format, and the server's reader (``lib/raw_parser.py``) is one this
project keeps up itself. So every results file in the main recordings' ``raw``
group, from each build that wrote one, is handed to LTspice's reader, and what
it says is kept for the server's reading to be held to
(``tests/test_ltspice_reader_agreement.py``). A file it refuses is kept as its
refusal.

It is a recorder of its own because it needs a window: the main recorder
(``tests/ltspice_recorder.py``) runs a build once per input and reads the
files it leaves. This one starts LTspice with its window on a desktop of its
own, where it cannot take the keyboard focus, and talks to it with the
server's own client, whose bridge runs on the server's desktop: another one,
so every recording is also of a bridge reaching a window on a desktop that is
not its own, as it does for a person's.

LTspice XVII has no bridge, so only builds from 26.1 on are recorded.

    uv run python scripts/record_ltspice_bridge.py
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import plot_settings
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.guide import split_front_matter
from ltspice_mcp.lib.hidden_desktop import (
    DIALOG_CLASS,
    HiddenDesktop,
    child_windows,
    post_command,
    window_class,
    window_text,
)
from ltspice_mcp.lib.ltspice_bridge import BridgeError, BridgeSession, bridge_command
from ltspice_mcp.lib.ltspice_frame import (
    VISIBLE_TRACES,
    LtspiceFrame,
    SaveQuestion,
    menu_command,
)
from ltspice_mcp.lib.ltspice_window import file_difference
from tests import ltspice_recorder
from tests.ltspice_recorder import (
    MANIFEST,
    NEUTRAL_DIR,
    Build,
    RecorderError,
    add_trace_by_hand,
    assert_private,
    click_all_over,
    discover_builds,
    drawn_traces,
    neutral_settings,
    private_strings,
    sha256_bytes,
    unavailable_reason,
)
from tests.ltspice_recorder import load_manifest as load_manifest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ltspice_bridge_recorded"
INPUTS = FIXTURES / "inputs"
CONVERSATION = "conversation.json"
READER = "reader"
MANIFEST_SCHEMA = 1

NEUTRAL_PID = 1000
NEUTRAL_PORT = 50000
NOT_A_PROCESS = 999999
#: The sheet the write and read-back steps are made on.
EDITED = "older_version"
_STARTED_S = 60.0
#: More than any results file in the main recordings holds, so each is read whole.
_READER_MAX_VALUES = 2_000_000
_RECORDING_S = 600.0
"""How long the session that records a window may take in all: a cap on a
recording that has stopped, where the server's own sessions are a handful of
calls and are capped at seconds."""


def input_names() -> list[str]:
    return sorted(path.stem for path in INPUTS.glob("*.asc"))


def recorded_builds() -> list[str]:
    return ltspice_recorder.recorded_builds(FIXTURES)


def load_conversation(directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / CONVERSATION).read_text(encoding="utf-8"))


def observed(build: str) -> dict[str, Any]:
    """What the recording of ``build`` noted LTspice doing, by the note that says what."""
    return {
        step["note"]: step["observed"]
        for step in load_conversation(FIXTURES / build)
        if "observed" in step
    }


class _Scrub:
    """Replaces what differs between two recordings of one build: the run
    directory, the process id and the port."""

    def __init__(self, work: Path) -> None:
        spelled = re.escape(str(work)).replace(r"\\", r"[\\/]")
        self._work = re.compile(spelled, re.IGNORECASE)
        self.pid: int | None = None
        self.port: int | None = None

    def text(self, value: str) -> str:
        return self._work.sub(lambda _match: NEUTRAL_DIR, value)

    def __call__(self, value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: self(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [self(item, key) for item in value]
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, int) and not isinstance(value, bool):
            if key.lower().endswith("pid") and value == self.pid:
                return NEUTRAL_PID
            if key == "port" and value == self.port:
                return NEUTRAL_PORT
        return value


class _Recording:
    """The conversation so far: each call as it was made and what it returned."""

    def __init__(self, scrub: _Scrub) -> None:
        self.steps: list[dict[str, Any]] = []
        self._scrub = scrub

    def call(
        self, session: BridgeSession, note: str, name: str, **arguments: Any
    ) -> dict[str, Any]:
        """Make one call and record it. A refusal is recorded and returned as ``{}``."""
        step: dict[str, Any] = {"note": note, "call": name, "arguments": self._scrub(arguments)}
        try:
            reply = session.call(name, **arguments)
        except BridgeError as error:
            step["error"] = self._scrub.text(str(error))
            self.steps.append(step)
            return {}
        kept = dict(reply)
        if isinstance(kept.get("text"), str):
            # The sheet itself is recorded beside this file, once.
            kept["text"] = f"<{len(kept['text'].splitlines())} lines>"
        if name == "get_raw_info":
            # The server opens a results file with this call and reads nothing
            # of what it describes, which is the main recorder's subject.
            kept = {"path": kept.get("path")}
        listed = kept.get("instances")
        if isinstance(listed, list):
            # Only the LTspice this recording started. Another bridge on the
            # machine (an assistant's own LTspice server, say) may run one for
            # itself, which comes and goes as windows here open and close.
            kept["instances"] = [
                row
                for row in listed
                if isinstance(row, dict) and row.get("pid") == self._scrub.pid
            ]
        step["reply"] = self._scrub(kept)
        self.steps.append(step)
        return reply

    def fact(self, note: str, value: Any) -> None:
        self.steps.append({"note": note, "observed": value})


def wait_for_window(command: Sequence[str], pid: int, sheet: Path) -> int:
    """The port of the window ``pid`` once it has ``sheet`` open.

    LTspice offers a window to the bridge before it has finished opening the
    sheet it was started with, and says so nowhere but to the bridge, so this
    asks until the sheet is listed.
    """
    deadline = time.monotonic() + _STARTED_S
    while time.monotonic() < deadline:
        with contextlib.suppress(BridgeError), BridgeSession(command) as session:
            for row in session.call("status").get("instances", []):
                if row.get("pid") != pid or row.get("mode") != "gui":
                    continue
                session.attach(pid)
                if any(Path(spelled) == sheet for spelled in session.open_designs()):
                    return int(row["port"])
        time.sleep(0.25)  # timing: between two looks; what is waited for is the listing
    raise RecorderError(f"LTspice process {pid} never offered its window to the bridge")


def _run_in_window(session: BridgeSession, recording: _Recording, sheet: Path) -> Path:
    """Run ``sheet`` in the window, which is how a results file comes to be."""
    recording.call(session, "run the sheet in the window", "start_simulation", path=str(sheet))
    deadline = time.monotonic() + _STARTED_S
    while time.monotonic() < deadline:
        running = session.call("is_simulation_running", path=str(sheet))
        if str(running.get("result")).lower() == "false":
            break
        time.sleep(0.25)  # timing: between two looks; what is waited for is the run's end
    results = sheet.with_suffix(".raw")
    deadline = time.monotonic() + _STARTED_S
    while not results.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)  # timing: between two looks; what is waited for is the file
    if not results.is_file():
        raise RecorderError(f"running {sheet.name} in the window left no results file")
    return results


_IDCANCEL = 2


def _until(condition: Callable[[], Any], doing: str) -> Any:
    deadline = time.monotonic() + _STARTED_S
    while time.monotonic() < deadline:
        found = condition()
        if found:
            return found
        time.sleep(0.1)  # timing: between two looks; what is waited for is the window
    raise RecorderError(f"LTspice's window never got as far as {doing}")


def _started_with_no_document(
    recording: _Recording,
    command: Sequence[str],
    desktop: HiddenDesktop,
    build: Build,
    ini: Path,
) -> None:
    """What the bridge finds of an LTspice started with no document, which is
    how the server starts one for a caller asked to show something in it."""
    started = desktop.start([str(build.exe), "-ini", str(ini)])

    def windows() -> list[int]:
        with contextlib.suppress(BridgeError), BridgeSession(command) as session:
            return [
                int(row["pid"])
                for row in session.call("status").get("instances", [])
                if row.get("mode") == "gui"
            ]
        return [started.pid]  # not answered: no word on whether it is there

    try:
        _until(lambda: started.pid in windows(), "offering a window with no document")
        recording.fact(
            "an LTspice started with no document is offered to the bridge as a window", True
        )
        with BridgeSession(command) as session:
            session.attach(started.pid)
            recording.fact(
                "it has no document open and none in front",
                [session.open_designs(), session.active_design()],
            )
    finally:
        started.kill()
        started.wait(timeout=30)
        started.close()
    _until(lambda: started.pid not in windows(), "being gone from the bridge's list")


def _close_and_wait(frame: LtspiceFrame, pane: int, doing: str) -> None:
    frame.close(pane)
    _until(lambda: not frame.is_open(pane), doing)


def _close_all(frame: LtspiceFrame, pid: int, title: str) -> None:
    """Close every pane titled ``title`` and wait for them all to go."""
    for pane in frame.panes_titled(pid, title):
        frame.close(pane)
    _until(lambda: not frame.panes_titled(pid, title), f"closing every pane titled {title}")


def _dialogs(desktop: HiddenDesktop, pid: int) -> list[int]:
    """The dialogs ``pid`` has up that ask something: a waveform viewer keeps
    untitled ones of its own, for its cursors."""
    return [w for w in desktop.windows(pid) if window_class(w) == DIALOG_CLASS and window_text(w)]


def _frame_facts(
    recording: _Recording,
    session: BridgeSession,
    desktop: HiddenDesktop,
    pid: int,
    open_sheet: Path,
    results: Path,
) -> Path:
    """What the window's frame shows, and when a sheet's Visible Traces
    command opens the results beside the sheet.

    ``results`` are a run's, copied beside a sheet under the sheet's name,
    which is how a job's results come to stand there: once beside
    ``open_sheet``, which the window opened with none, and once beside a sheet
    it has yet to open. Returns that second sheet, left open with its results
    open beside it.
    """
    frame = LtspiceFrame(desktop.windows)
    dialogs = functools.partial(_dialogs, desktop, pid)

    late = open_sheet.with_suffix(".raw")
    before = frame.panes(pid)
    recording.fact(
        "the frame has a pane titled for an open sheet, and none for results it has not opened",
        [open_sheet.name in before, late.name in before],
    )
    shutil.copyfile(results, late)
    recording.call(session, "put the open sheet in front", "bring_to_front", path=str(open_sheet))
    frame.send(pid, VISIBLE_TRACES)
    time.sleep(3.0)  # timing: nothing is waited for; what is recorded is that nothing came
    recording.fact(
        "results put beside a sheet that was already open are not opened by its command",
        late.name not in frame.panes(pid) and not dialogs(),
    )

    placed = open_sheet.with_name("placed.asc")
    shutil.copyfile(INPUTS / f"{EDITED}.asc", placed)
    shutil.copyfile(results, placed.with_suffix(".raw"))
    recording.call(
        session, "open a sheet that has results beside it", "open_design", path=str(placed)
    )
    recording.call(session, "put it in front", "bring_to_front", path=str(placed))
    frame.send(pid, VISIBLE_TRACES)
    _until(
        lambda: placed.with_suffix(".raw").name in frame.panes(pid),
        "opening the results beside the sheet",
    )
    recording.fact("results beside a sheet when it is opened are opened by its command", True)
    recording.fact("it asks nothing on the way", not dialogs())
    recording.call(session, "put the sheet in front again", "bring_to_front", path=str(placed))
    frame.send(pid, VISIBLE_TRACES)
    (asked,) = _until(dialogs, "asking which traces to show")
    recording.fact(
        "with those results open the same command asks which traces to show",
        window_text(asked),
    )
    post_command(asked, _IDCANCEL)
    _until(lambda: not dialogs(), "closing the dialog")

    return placed


def _results_pane_facts(
    recording: _Recording,
    session: BridgeSession,
    desktop: HiddenDesktop,
    build: Build,
    pid: int,
    sheet: Path,
    newer: Path,
) -> None:
    """What a results pane does when its file is replaced and when it is closed.

    ``sheet`` is open with the results beside it open in a pane, opened from
    it. ``newer`` are another run's, in double precision where those are in
    single, which is how the two are told apart.
    """
    frame = LtspiceFrame(desktop.windows)
    plot = sheet.with_suffix(".raw")

    def held() -> list[str]:
        """What the results pane in front holds: how many points, in which precision."""
        info = session.call("get_raw_info")
        return [str(info["n_points"]), str(info["precision"])]

    def closes() -> list[bool]:
        """Tell the pane to close: whether it has gone, and whether nothing was asked."""
        pane = frame.pane(pid, plot.name)
        assert pane is not None
        frame.close(pane)
        _until(
            lambda: not frame.is_open(pane) or _dialogs(desktop, pid), "closing the results pane"
        )
        return [not frame.is_open(pane), not _dialogs(desktop, pid)]

    recording.call(session, "put the results in front", "bring_to_front", path=str(plot))
    read_first = held()
    shutil.copyfile(newer, plot)
    recording.fact(
        "a results pane goes on holding what it read when its file is replaced",
        held() == read_first,
    )
    recording.fact(
        "a results pane told to close is closed, and LTspice asks nothing", all(closes())
    )
    recording.fact("the sheet it was opened from stays open", sheet.name in frame.panes(pid))
    recording.call(session, "put the sheet in front", "bring_to_front", path=str(sheet))
    frame.send(pid, VISIBLE_TRACES)
    _until(lambda: frame.pane(pid, plot.name), "opening the results again")
    recording.call(session, "put the results in front", "bring_to_front", path=str(plot))
    recording.fact(
        "the precision of the results the pane held, and of those it holds once opened again",
        [read_first[1], held()[1]],
    )

    add_trace_by_hand(
        _until(lambda: frame.pane(pid, plot.name), "finding the results pane"),
        lambda: desktop.windows(pid),
        build.exe,
        "V(out)",
        functools.partial(_until, doing="adding a trace by hand"),
    )
    recording.fact(
        "with a trace added by hand it closes the same, and saves no plot settings",
        [*closes(), plot.with_suffix(".plt").exists()],
    )


def _results_alone_facts(
    recording: _Recording,
    session: BridgeSession,
    desktop: HiddenDesktop,
    build: Build,
    pid: int,
    open_sheet: Path,
    results: Path,
) -> None:
    """What results opened on their own do: when they read the plot settings
    beside them, and which of two panes of one title is found first.

    ``results`` are a run's, copied twice under one name into two folders,
    which is how the results of two runs of one deck stand. ``open_sheet`` is
    a sheet the window has open with no plot of its own.
    """
    frame = LtspiceFrame(desktop.windows)
    one, other = (open_sheet.parent / "runs" / run / "run.raw" for run in ("one", "other"))
    for copy in (one, other):
        copy.parent.mkdir(parents=True)
        shutil.copyfile(results, copy)
    settings = plot_settings.plot_settings_path(one)
    transient = plot_settings.SECTION_NAMES["tran"]

    def name(trace: str) -> None:
        """Write plot settings naming ``trace`` beside the results, as the server does."""
        settings.unlink(missing_ok=True)
        if plot_settings.write_beside(one, transient, [[trace]]) is not None:
            raise RecorderError("the plot settings were not written")

    def drawn() -> list[list[str]]:
        pane = _until(lambda: frame.pane(pid, one.name), "finding the results pane")
        return drawn_traces(
            pane, build.exe, one, functools.partial(_until, doing="saving plot settings")
        )

    def in_front() -> str:
        """The folder of the results the window has in front."""
        return Path(session.results_in_front() or "").parent.name

    recording.call(session, "put a sheet in front", "bring_to_front", path=str(open_sheet))
    recording.call(
        session, "put results that are not open in front", "bring_to_front", path=str(one)
    )
    recording.call(session, "the results in front, with a sheet in front", "get_raw_info")
    name("I(R1)")
    recording.call(session, "open results on their own", "get_raw_info", path=str(one))
    recording.call(session, "put them in front", "bring_to_front", path=str(one))
    recording.call(session, "the results in front", "get_raw_info")
    recording.fact(
        "results opened on their own draw the traces the plot settings beside them name", drawn()
    )
    name("V(out)")
    session.show_results(str(one))
    recording.fact("asked for again while open, they draw what they drew", drawn())
    pane = frame.pane(pid, one.name)
    assert pane is not None
    _close_and_wait(frame, pane, "closing results opened on their own")
    recording.fact("closed, they are not in front any more", in_front() != one.parent.name)
    session.show_results(str(one))
    recording.fact("closed and opened again, they draw what the settings name now", drawn())

    session.call("get_raw_info", path=str(other))
    _until(lambda: len(frame.panes_titled(pid, one.name)) == 2, "opening a second run's results")
    found: list[tuple[str, int | None]] = []
    for run in (one, other, one):
        session.bring_to_front(str(run))
        _until(lambda: in_front() == run.parent.name, "putting results in front")  # noqa: B023
        found.append((in_front(), frame.pane(pid, one.name)))
    recording.fact(
        "the results put in front, of two of one name", [folder for folder, _pane in found]
    )
    recording.fact(
        "of two results panes of one title, the first found is the one in front",
        found[0][1] == found[2][1] != found[1][1],
    )
    in_front_now = frame.pane(pid, one.name)
    assert in_front_now is not None
    _close_and_wait(frame, in_front_now, "closing the results in front")
    session.bring_to_front(str(one))
    recording.fact("with that one closed, the results left open are the other run's", in_front())
    _close_all(frame, pid, one.name)


def _box_words(desktop: HiddenDesktop, pid: int) -> list[list[str]]:
    """What each box ``pid`` has up says, and the buttons it offers."""
    return [
        [window_text(part) for part in child_windows(box) if window_text(part)]
        for box in _dialogs(desktop, pid)
    ]


def _sheet_close_facts(
    recording: _Recording,
    session: BridgeSession,
    desktop: HiddenDesktop,
    scrub: _Scrub,
    pid: int,
    work: Path,
    results: Path,
) -> None:
    """What a sheet told to close does: one nobody changed, and one the window was handed.

    The sheet is opened with no results beside it, and ``results``, a run's,
    are put there afterwards, which is how a sheet comes to know of none.
    """
    frame = LtspiceFrame(desktop.windows)
    sheet = work / "closing.asc"
    shutil.copyfile(INPUTS / f"{EDITED}.asc", sheet)
    on_disk = sheet.read_bytes()
    plot = sheet.with_suffix(".raw")

    def in_front() -> int:
        session.bring_to_front(str(sheet))
        return _until(lambda: frame.pane(pid, sheet.name), "finding the sheet's pane")

    def told_to_close() -> int:
        """Tell the sheet to close, and wait for it to have gone or LTspice to ask."""
        pane = in_front()
        frame.close(pane)
        _until(lambda: not frame.is_open(pane) or frame.save_question(pid), "closing or asking")
        return pane

    def asked() -> SaveQuestion:
        question = frame.save_question(pid)
        if question is None:
            raise RecorderError(f"LTspice closed {sheet.name} and asked nothing")
        return question

    def opened_from_the_sheet() -> bool:
        in_front()
        frame.send(pid, VISIBLE_TRACES)
        _until(lambda: frame.pane(pid, plot.name), "opening the results from the sheet")
        return True

    recording.call(
        session, "open a sheet with no results beside it", "open_design", path=str(sheet)
    )
    shutil.copyfile(results, plot)
    recording.fact("the frame takes input while LTspice asks nothing", frame.takes_input(pid))
    pane = told_to_close()
    recording.fact(
        "a sheet nobody changed, told to close, is closed, and LTspice asks nothing",
        [not frame.is_open(pane), frame.save_question(pid) is None],
    )
    session.open_design(str(sheet))
    recording.fact(
        "opened again, it opens the results put beside it while it was open",
        opened_from_the_sheet(),
    )
    pane = told_to_close()
    _until(lambda: frame.pane(pid, plot.name) is None, "closing the plot with its sheet")
    recording.fact(
        "the plot opened from a sheet closes with it",
        [not frame.is_open(pane), frame.pane(pid, plot.name) is None],
    )

    session.open_design(str(sheet))
    held = session.design_text(str(sheet))
    changed = held.replace("SYMATTR Value 1k", "SYMATTR Value 2k")
    recording.call(
        session,
        "hand the window a changed sheet",
        "set_design_content",
        path=str(sheet),
        text=changed,
    )
    recording.call(
        session, "hand it back what it held", "set_design_content", path=str(sheet), text=held
    )
    recording.fact(
        "it then reads as its file does",
        file_difference(on_disk, session.design_text(str(sheet))) is None,
    )
    pane = told_to_close()
    question = asked()
    recording.fact(
        "a sheet the window was handed is asked about before it closes, in these words",
        [[scrub.text(words) for words in box] for box in _box_words(desktop, pid)],
    )
    recording.fact("the document the question names is the sheet", Path(question.path) == sheet)
    recording.fact("the frame takes input while LTspice asks that", frame.takes_input(pid))
    recording.call(
        session, "the window's copy, while LTspice asks", "get_design_content", path=str(sheet)
    )
    frame.answer(question, discard=False)
    _until(lambda: not frame.is_open(question.box), "taking Cancel for an answer")
    recording.fact(
        "answered Cancel, the sheet stays open, holds what it held, and the frame takes input",
        [frame.is_open(pane), session.design_text(str(sheet)) == held, frame.takes_input(pid)],
    )
    pane = told_to_close()
    question = asked()
    frame.answer(question, discard=True)
    _until(lambda: not frame.is_open(pane), "taking No for an answer")
    recording.fact(
        "answered No, the sheet closes and its file is as it was",
        [not frame.is_open(question.box), sheet.read_bytes() == on_disk],
    )

    elsewhere = work / "elsewhere" / sheet.name
    elsewhere.parent.mkdir()
    shutil.copyfile(sheet, elsewhere)
    session.open_design(str(sheet))
    session.open_design(str(elsewhere))
    found: list[int | None] = []
    for each in (sheet, elsewhere, sheet):
        session.bring_to_front(str(each))
        _until(
            lambda: Path(session.active_design() or "") == each,  # noqa: B023
            "putting a sheet in front",
        )
        found.append(frame.pane(pid, sheet.name))
    recording.fact(
        "of two sheets of one title, the first pane found is the one in front",
        found[0] == found[2] != found[1],
    )
    _close_all(frame, pid, sheet.name)


def _tie_and_menu_facts(
    recording: _Recording,
    session: BridgeSession,
    desktop: HiddenDesktop,
    build: Build,
    pid: int,
    work: Path,
    results: Path,
) -> None:
    """Which plots are tied to their sheet, and what the sheet menu's Visible
    Traces item says of a sheet's results.

    A plot is tied when a click on its sheet plots what it lands on: the
    sheet is clicked all over, and the traces the plot draws read before and
    after. Each sheet is a copy of one input with ``results``, a run of it,
    beside it, named for the state it is in.
    """
    frame = LtspiceFrame(desktop.windows)
    sheets = {name: work / f"tie_{name}.asc" for name in ("none", "late", "known", "alone")}

    def with_results(sheet: Path) -> None:
        plot = sheet.with_suffix(".raw")
        shutil.copyfile(results, plot)
        # One trace the clicks will not have drawn, whatever they land on first.
        plot_settings.plot_settings_path(plot).unlink(missing_ok=True)
        transient = plot_settings.SECTION_NAMES["tran"]
        if plot_settings.write_beside(plot, transient, [["I(R1)"]]) is not None:
            raise RecorderError("the plot settings were not written")

    def enabled(sheet: Path) -> bool | None:
        session.bring_to_front(str(sheet))
        return frame.command_enabled(pid, VISIBLE_TRACES)

    def follows_clicks(sheet: Path) -> bool:
        """Whether the plot of the results beside ``sheet`` changes when the sheet is clicked."""
        plot = sheet.with_suffix(".raw")
        until = functools.partial(_until, doing="saving plot settings")

        def drawn() -> list[list[str]]:
            session.bring_to_front(str(plot))
            pane = _until(lambda: frame.pane(pid, plot.name), "finding the plot")
            return drawn_traces(pane, build.exe, plot, until)

        before = drawn()
        session.bring_to_front(str(sheet))
        pane = _until(lambda: frame.pane(pid, sheet.name), "finding the sheet")
        (view,) = child_windows(pane)[:1]
        click_all_over(view)
        return drawn() != before

    for sheet in sheets.values():
        shutil.copyfile(INPUTS / f"{EDITED}.asc", sheet)
    with_results(sheets["known"])
    for sheet in sheets.values():
        session.open_design(str(sheet))
    with_results(sheets["late"])
    with_results(sheets["alone"])
    states = [enabled(sheets[name]) for name in ("none", "late", "known")]
    session.bring_to_front(str(sheets["known"]))
    frame.send(pid, VISIBLE_TRACES)
    _until(lambda: frame.pane(pid, sheets["known"].with_suffix(".raw").name), "opening the plot")
    recording.fact(
        "Visible Traces is enabled for a sheet with no results, one with results put "
        "beside it after it was opened, one with results beside it when it was opened, "
        "and that one with its plot open",
        [*states, enabled(sheets["known"])],
    )
    recording.fact(
        "a plot opened by its sheet's Visible Traces plots what a click on the sheet lands on",
        follows_clicks(sheets["known"]),
    )

    session.show_results(str(sheets["alone"].with_suffix(".raw")))
    recording.fact(
        "a plot of results opened by their own path does, for a sheet that knows of none",
        follows_clicks(sheets["alone"]),
    )
    for each in ("late", "known"):
        for pane in frame.panes_titled(pid, sheets[each].with_suffix(".raw").name):
            frame.close(pane)
    _until(
        lambda: not frame.panes_titled(pid, sheets["known"].with_suffix(".raw").name),
        "closing the plot opened by the command",
    )
    asked_about = [
        Path(str(session.call("get_raw_info", path=str(sheets[each])).get("path"))).name
        for each in ("late", "known")
    ]
    recording.fact(
        "asked about a sheet by its path, the bridge opens the results beside it", asked_about
    )
    recording.fact(
        "that plot does, for a sheet that knows of none and for one that knows of its results",
        [follows_clicks(sheets["late"]), follows_clicks(sheets["known"])],
    )


def reader_subjects() -> dict[str, list[str]]:
    """The results files LTspice's reader is asked about, by the build that wrote them.

    Every file with samples in a main recording's ``raw`` group: one of each
    layout the server reads.
    """
    subjects: dict[str, list[str]] = {}
    for label in ltspice_recorder.recorded_builds(ltspice_recorder.FIXTURES):
        directory = ltspice_recorder.FIXTURES / label / "raw"
        names = sorted(
            path.name.removesuffix(".raw")
            for path in directory.glob("*.raw")
            if path.stat().st_size
        )
        if names:
            subjects[label] = names
    return subjects


def reader_reply(build: str, written_by: str, name: str) -> dict[str, Any]:
    """What the reader of ``build`` said of the results file ``name`` that ``written_by`` wrote."""
    path = FIXTURES / build / READER / written_by / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _reader_replies(
    session: BridgeSession, work: Path, scrub: _Scrub
) -> tuple[dict[str, bytes], dict[str, dict[str, str]]]:
    """What LTspice's own reader says of each results file in the main recordings.

    Each is copied with its log beside it, which is where the reader finds a
    stepped run's parameter values, and read whole. Returns the files to
    record and the digest of each results file read.
    """
    files: dict[str, bytes] = {}
    digests: dict[str, dict[str, str]] = {}
    for label, names in reader_subjects().items():
        recorded = ltspice_recorder.FIXTURES / label / "raw"
        copies = work / "results" / label
        copies.mkdir(parents=True)
        for name in names:
            copy = copies / f"{name}.raw"
            shutil.copyfile(recorded / copy.name, copy)
            log = recorded / f"{name}.log"
            if log.is_file():
                shutil.copyfile(log, copies / log.name)
            try:
                info = session.call("get_raw_info", path=str(copy))
                said: dict[str, Any] = {
                    "info": info,
                    "read": session.call(
                        "read_raw_waveforms",
                        path=str(copy),
                        signals=info["signals"],
                        max_values=_READER_MAX_VALUES,
                    ),
                }
            except BridgeError as error:
                said = {"refused": scrub.text(str(error))}
            reply = json.dumps(scrub(said), separators=(",", ":"), ensure_ascii=False)
            files[f"{READER}/{label}/{name}.json"] = reply.encode("utf-8") + b"\n"
            digests.setdefault(label, {})[name] = sha256_bytes(copy.read_bytes())
    return files, digests


def _resistor_value(netlist: Path) -> str | None:
    """The value of R1 in the netlist a run in the window wrote beside its sheet.

    When the run is made the window holds ``2k`` and the file ``3k``, so this
    says which of the two LTspice simulated.
    """
    if not netlist.is_file():
        return None
    for line in decode_spice_bytes(netlist.read_bytes()).splitlines():
        words = line.split()
        if words and words[0] == "R1":
            return words[-1]
    return None


def _record(build: Build, out: Path, desktop: HiddenDesktop) -> None:
    """The recording itself, with LTspice's window on ``desktop``."""
    command = bridge_command(build.exe)
    if command is None:
        raise RecorderError(f"{build.exe} has no bridge beside it")
    settings = build.settings_file
    assert settings is not None
    with tempfile.TemporaryDirectory(prefix="ltspice-bridge-rec-") as scratch:
        work = Path(scratch).resolve() / "sheets"
        work.mkdir()
        names = input_names()
        for name in names:
            shutil.copyfile(INPUTS / f"{name}.asc", work / f"{name}.asc")
        ini = work.parent / settings.name
        ini.write_bytes(neutral_settings(settings.read_bytes(), {}))
        scrub = _Scrub(work)
        recording = _Recording(scrub)
        sheets: dict[str, bytes] = {}
        first = work / f"{names[0]}.asc"

        with BridgeSession(command) as session:
            recording.call(session, "no LTspice is running", "status")
            recording.call(
                session, "attach to a process that is not LTspice", "attach", pid=NOT_A_PROCESS
            )
            recording.call(
                session, "a request that needs LTspice, with none running", "list_open_designs"
            )

        _started_with_no_document(recording, command, desktop, build, ini)
        window = desktop.start([str(build.exe), str(first), "-ini", str(ini)])
        try:
            scrub.pid = window.pid
            scrub.port = wait_for_window(command, window.pid, first)
            with BridgeSession(command, timeout=_RECORDING_S) as session:
                recording.call(session, "one window is open", "status")
                recording.call(session, "attach to the window", "attach", pid=window.pid)
                recording.call(session, "attached", "status")
                for name in names[1:]:
                    recording.call(
                        session,
                        "open another sheet",
                        "open_design",
                        path=str(work / f"{name}.asc"),
                    )
                recording.call(session, "every sheet is open", "list_open_designs")
                recording.call(session, "the document in front", "get_active_design_path")
                recording.call(
                    session, "open a sheet that is already open", "open_design", path=str(first)
                )
                recording.call(session, "put it in front", "bring_to_front", path=str(first))
                recording.call(session, "the document in front now", "get_active_design_path")
                recording.call(
                    session,
                    "open a sheet that is not there",
                    "open_design",
                    path=str(work / "absent.asc"),
                )
                for name in names:
                    reply = recording.call(
                        session,
                        "the window's copy",
                        "get_design_content",
                        path=str(work / f"{name}.asc"),
                    )
                    sheets[f"sheets/{name}.asc"] = str(reply.get("text", "")).encode("utf-8")

                edited = work / f"{EDITED}.asc"
                on_disk = edited.read_bytes()
                held = sheets[f"sheets/{EDITED}.asc"].decode("utf-8")
                recording.call(
                    session,
                    "replace a sheet with the text it already holds",
                    "set_design_content",
                    path=str(edited),
                    text=held,
                )
                changed = held.replace("SYMATTR Value 1k", "SYMATTR Value 2k")
                recording.call(
                    session,
                    "replace a sheet with a changed one",
                    "set_design_content",
                    path=str(edited),
                    text=changed,
                )
                recording.fact(
                    "the window then reads back exactly what it was given",
                    session.design_text(str(edited)) == changed,
                )
                recording.fact("the file is as it was", edited.read_bytes() == on_disk)
                rewritten = on_disk.replace(b"SYMATTR Value 1k", b"SYMATTR Value 3k")
                edited.write_bytes(rewritten)
                recording.fact(
                    "after the file is rewritten the window still holds its own copy",
                    session.design_text(str(edited)) == changed,
                )
                recording.call(
                    session,
                    "a file no window has open",
                    "get_design_content",
                    path=str(work / "not_open.asc"),
                )
                recording.call(
                    session,
                    "replace a file no window has open",
                    "set_design_content",
                    path=str(work / "not_open.asc"),
                    text=held,
                )

                results = _run_in_window(session, recording, edited)
                recording.fact(
                    "a run in the window is of the window's copy and not of the file",
                    _resistor_value(edited.with_suffix(".net")),
                )
                recording.fact("the run did not write the sheet", edited.read_bytes() == rewritten)
                recording.fact(
                    "what the run left beside the sheet",
                    sorted(path.name[len(EDITED) :] for path in work.glob(f"{EDITED}.*")),
                )
                recording.call(session, "open a results file", "get_raw_info", path=str(results))
                recording.call(session, "put it in front", "bring_to_front", path=str(results))
                recording.fact(
                    "the results file is then the one LTspice has in front",
                    session.call("get_raw_info").get("path") == str(results),
                )
                recording.call(
                    session,
                    "a results file that is not there",
                    "get_raw_info",
                    path=str(work / "absent.raw"),
                )
                placed = _frame_facts(recording, session, desktop, window.pid, first, results)
                _results_pane_facts(
                    recording,
                    session,
                    desktop,
                    build,
                    window.pid,
                    placed,
                    newer=ltspice_recorder.FIXTURES / build.label / "raw" / "tran_double.raw",
                )
                _results_alone_facts(
                    recording, session, desktop, build, window.pid, first, results
                )
                _sheet_close_facts(recording, session, desktop, scrub, window.pid, work, results)
                _tie_and_menu_facts(recording, session, desktop, build, window.pid, work, results)
                reader_files, results_read = _reader_replies(session, work, scrub)

                window.kill()
                window.wait(timeout=30)
                recording.call(session, "the window has closed", "list_open_designs")
                recording.call(session, "the window has closed", "status")
                # This bridge has no LTspice of its own: one it had started
                # would be the one it is bound to.
                recording.fact(
                    "no LTspice was started in its place",
                    not session.call("status").get("current", {}).get("backendPid"),
                )
        finally:
            window.close()

    conversation = (
        json.dumps(recording.steps, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    files = {CONVERSATION: conversation, **sheets, **reader_files}
    assert_private(files, private_strings())
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "build": {
            "label": build.label,
            "file_version": build.file_version,
            "generation": build.generation,
        },
        "bridge": {
            "size": Path(command[0]).stat().st_size,
            "sha256": sha256_bytes(Path(command[0]).read_bytes()),
        },
        "inputs": {name: sha256_bytes((INPUTS / f"{name}.asc").read_bytes()) for name in names},
        "reference": reference_record(build),
        "sheet_commands": {VISIBLE_TRACES: menu_command(build.exe, VISIBLE_TRACES)},
        # The results files the reader was handed, as the main recordings held them then.
        "reader": results_read,
        "files": {name: sha256_bytes(data) for name, data in sorted(files.items())},
    }
    if out.exists():
        shutil.rmtree(out)
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def reference_record(build: Build) -> dict[str, Any] | None:
    """The reference documents ``build`` installs: their names and the keys of
    their front matter. The documents are the vendor's and are not recorded."""
    library = build.library_root
    directory = library.parent / "reference" if library is not None else None
    if directory is None or not directory.is_dir():
        return None
    keys: set[str] = set()
    names: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda entry: entry.name.casefold()):
        if path.is_file():
            names.append(path.name)
            head = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            keys |= set(split_front_matter(head)[0])
    return {"directory": directory.name, "files": names, "front_matter": sorted(keys)}


def bridge_builds() -> list[Build]:
    """The installed builds that ship a bridge."""
    return [build for build in discover_builds() if bridge_command(build.exe) is not None]


def record(build: Build, out: Path) -> None:
    """Record ``build`` into ``out``, its window on a desktop of its own."""
    reason = unavailable_reason(build)
    if reason:
        raise RecorderError(reason)
    with HiddenDesktop(f"ltspice-bridge-recorder-{os.getpid()}") as desktop:
        if not desktop.available:
            raise RecorderError(
                "Windows gave no desktop to record on; not starting LTspice on this one"
            )
        _record(build, out, desktop)


def main() -> int:
    builds = bridge_builds()
    if not builds:
        print("No installed LTspice has ltspice-mcp-bridge.exe beside it (26.1 or later).")
        return 1
    for build in builds:
        record(build, FIXTURES / build.label)
        print(f"recorded {build.label} ({build.file_version}) into {FIXTURES / build.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
