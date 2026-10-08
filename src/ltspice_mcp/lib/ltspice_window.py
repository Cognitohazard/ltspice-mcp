"""A sheet someone has open in LTspice, kept in step with its file.

LTspice reads a sheet once. After that the window holds its own copy: a write
to the file changes nothing on screen, and the next save from the window puts
the old sheet back over the new file. An edit made by this server to a sheet
that is open would be invisible until the sheet was reopened and could be lost
without a word. ``OpenWindows`` is what closes that: it finds the windows that
have a file open (``holding``), and replaces what one shows with the committed
sheet (``show``), which LTspice records as one step of that window's undo
history. It reaches the windows through the bridge LTspice ships
(``BridgeSession``), attaching to instances that are already running. The
bridge is never let start one. ``open_sheet`` opens a sheet in a window and ``show_results``
a finished run's results file, each for a caller who was asked to show it
there; with a plot settings file beside it, the results open with its traces
drawn. A results file opened that way stands alone. ``results_from_sheet``
opens the results beside a sheet from the sheet, as a person does with its
Visible Traces command, and LTspice then ties the plot to the sheet: a click
on a net plots it. The bridge has no call for that, so the window's frame is
asked (``LtspiceFrame``).

Those three are for a caller who was asked to show something in LTspice, and
for that caller alone LTspice is started when no window is open
(``ensure_window``, ``start_in_view``): where a person can see it, which is the
one start of LTspice anywhere here that is meant to be seen. Keeping an open
sheet in step, and saying what is open, never start anything.

The file stays the record. Before an edit is committed, the window's copy is
compared with the file, and one that differs holds work nobody saved: the
edit is refused there, because committing it would leave two sheets, each
missing the other's changes. That comparison cannot be of text. LTspice hands
back its own writing of a sheet, and on 26.1.1 that differs from the file it
read in ways that change nothing (each recorded from a window; see
``docs/TESTING.md``, "An open window and the bridge"):

- the first line of a sheet written by an older build becomes ``Version 4.1``,
  and the ``SHEET`` line's extent is worked out again;
- wires come back sorted, and a symbol's attributes in LTspice's own order;
- an attribute with no value is dropped;
- a ``u`` multiplier is written as a micro sign;
- text placed off the grid of 8 is moved onto it.

``sheet_content`` reads a sheet past all of those, and two sheets are the same
when it reads them the same.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.encoding import decode_spice_bytes, decode_windows_1252
from ltspice_mcp.lib.ltspice_bridge import (
    DEFAULT_TIMEOUT_S,
    BridgeError,
    BridgeSession,
    Instance,
    bridge_command,
)
from ltspice_mcp.lib.ltspice_frame import VISIBLE_TRACES, FrameError, LtspiceFrame

# The lines that say how a sheet is stored and how far it extends, which
# LTspice writes for itself whatever the file said.
_STORAGE_LINES = frozenset({"Version", "SHEET"})
# Lines that belong to the symbol above them.
_SYMBOL_LINES = frozenset({"SYMATTR", "WINDOW"})
_MICRO_SIGNS = str.maketrans({"µ": "u", "μ": "u"})
_TEXT_GRID = 8
_WINDOW = "gui"
_NO_WINDOW = "no LTspice window is open, and none is started for this"
_STARTED_S = 20.0
"""How long a started LTspice is given to offer its window to the bridge. It
takes a second or two; this is a cap on one that never does."""
# Apart from the server's own console and process group, and outside its job
# where Windows allows: the window is the person's and must not end with a
# session.
_APART = 0x00000008 | 0x00000200
_OUTSIDE_THE_JOB = 0x01000000
_RESULTS_SUFFIX = ".raw"
_OPENED_S = 5.0
"""How long LTspice is given to show the results it was asked to open. Opening
a results file is quick; this is a cap on a window that did nothing."""


class WindowsUnavailable(BridgeError):
    """There is no bridge to ask here: off Windows, turned off, or none installed."""


class ResultsAlreadyOpen(BridgeError):
    """The window already has a sheet's results open, and goes on showing those."""


class ResultsNotLookedFor(BridgeError):
    """The window opened the sheet before it had results, and knows of none."""


def _on_text_grid(coordinate: str) -> str:
    try:
        value = int(coordinate)
    except ValueError:
        return coordinate
    # To the nearest grid point, and a coordinate half way between two goes to
    # the one further from zero: 20 becomes 24 and -20 becomes -24.
    return str(int(math.copysign(_TEXT_GRID * math.floor(abs(value) / _TEXT_GRID + 0.5), value)))


def _has_no_value(attribute_line: str) -> bool:
    words = attribute_line.split(None, 2)
    return len(words) < 3 or words[2].strip() in ("", '""')


def sheet_content(text: str) -> list[str]:
    """What a sheet holds, read past everything LTspice rewrites on loading it.

    One entry per symbol (with its attributes, in a fixed order) and one per
    other line, sorted: two sheets with equal content draw the same picture
    and netlist to the same circuit, whichever of them LTspice wrote.
    """
    records: list[list[str]] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        words = line.split()
        if not words or words[0] in _STORAGE_LINES:
            continue
        if words[0] in _SYMBOL_LINES and records and records[-1][0].startswith("SYMBOL "):
            if not (words[0] == "SYMATTR" and _has_no_value(line)):
                records[-1].append(line)
            continue
        if words[0] == "TEXT" and len(words) > 2:
            head = line.split(None, 3)
            head[1], head[2] = _on_text_grid(head[1]), _on_text_grid(head[2])
            line = " ".join(head)
        records.append([line])
    return sorted(
        "\n".join([record[0], *sorted(record[1:])]).translate(_MICRO_SIGNS) for record in records
    )


def content_difference(file_text: str, window_text: str, *, limit: int = 3) -> str | None:
    """How a window's sheet differs from its file's, or None when it does not.

    Names up to ``limit`` entries from each side by their first line, which is
    enough to recognise the change and short enough to put in a refusal.
    """
    in_file = Counter(sheet_content(file_text))
    in_window = Counter(sheet_content(window_text))
    if in_file == in_window:
        return None

    def named(side: Counter[str]) -> str:
        entries = sorted(side.elements())
        shown = "; ".join(entry.split("\n", 1)[0] for entry in entries[:limit])
        more = len(entries) - limit
        return shown + (f"; and {more} more" if more > 0 else "")

    parts = []
    only_window = in_window - in_file
    only_file = in_file - in_window
    if only_window:
        parts.append(f"only in the window: {named(only_window)}")
    if only_file:
        parts.append(f"only in the file: {named(only_file)}")
    return ". ".join(parts)


def file_difference(on_disk: bytes, window_text: str) -> str | None:
    """How a window's copy of a sheet differs from the file's bytes, or None.

    The file is read both as this server reads it and as LTspice does, which
    differ for a sheet stored as UTF-8: a window showing either reading of the
    file holds nothing of its own. A window that differs has unsaved changes
    or was opened before the file last changed, and nothing tells which.
    """
    as_read_here = decode_spice_bytes(on_disk)
    difference = content_difference(as_read_here, window_text)
    if difference is None:
        return None
    as_ltspice_reads = decode_windows_1252(on_disk)
    if as_ltspice_reads != as_read_here and (
        content_difference(as_ltspice_reads, window_text) is None
    ):
        return None
    return difference


@dataclass(frozen=True)
class OpenDesign:
    """A document one LTspice window has open.

    ``path`` is as LTspice spells it. ``active`` says it is the document in
    front in that window. ``text`` is the window's copy, read only when the
    caller asked for this one, and None otherwise.
    """

    pid: int
    version: str
    path: str
    active: bool
    text: str | None = None


@dataclass(frozen=True)
class OpenSheet:
    """A file as one LTspice window holds it.

    ``path`` is the path as LTspice spells it, which is what the bridge takes
    back; ``text`` is the window's copy, saved or not.
    """

    pid: int
    version: str
    path: str
    text: str


def start_in_view(executable: Path) -> int:
    """Start LTspice where the person can see it, and return its process id.

    The one start of LTspice that is meant to be seen: every other is kept
    off the person's desktop. It is for a caller who was asked to show
    something in LTspice when no window is open, and it takes the keyboard
    focus, as starting any program does. No document and no settings file are
    named, so it opens as it does from the Start menu. The process is the
    person's from then on: nothing here waits for it or ends it. Raises
    ``OSError`` when it cannot be started.
    """
    if sys.platform != "win32":
        raise OSError("an LTspice window is started through Windows")

    def started(flags: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [str(executable)],
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        process = started(_APART | _OUTSIDE_THE_JOB)
    except PermissionError:
        # A job that lets no process leave it. Started inside it, the window
        # ends when the job does, which is still the window that was asked for.
        process = started(_APART)
    pid = process.pid
    # Not this process's to wait for: without this, dropping the handle of a
    # program still running is reported as a leak.
    process.returncode = 0
    return pid


def _same_file(ours: Path, theirs: str) -> bool:
    return os.path.normcase(os.path.abspath(theirs)) == os.path.normcase(os.path.abspath(ours))


class OpenWindows:
    """The LTspice windows on this machine, as far as the bridge shows them.

    ``available`` is False where there is no bridge to ask, and every method
    then raises ``WindowsUnavailable``. ``unavailable`` says why, for
    ``inspect(kind="capabilities")``. ``start`` starts LTspice where a person
    can see it; without one, none is ever started.
    """

    def __init__(
        self,
        command: Sequence[str] | None,
        *,
        unavailable: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        frame: LtspiceFrame | None = None,
        start: Callable[[], object] | None = None,
    ) -> None:
        self._command = list(command) if command else None
        self.unavailable = None if self._command else (unavailable or "no bridge was found")
        self._timeout = timeout
        self._frame = frame or LtspiceFrame()
        self._start = start if self._command else None

    @property
    def starts_ltspice(self) -> bool:
        """Whether LTspice is started for a caller asked to show something in it."""
        return self._start is not None

    @property
    def available(self) -> bool:
        return self._command is not None

    @classmethod
    def detect(
        cls, executables: Iterable[str], *, enabled: bool = True, start_ltspice: bool = False
    ) -> OpenWindows:
        """The windows reachable through the bridge beside one of ``executables``.

        Any 26.1 bridge finds every running instance, so the first one found
        is used, and with ``start_ltspice`` the LTspice it stands beside is the
        one started when no window is open. Native Windows only: under WSL the
        bridge is a Windows program that names Windows paths, and nothing here
        translates them.
        """
        if not enabled:
            return cls(None, unavailable="turned off by [schematic] sync_open_window")
        if sys.platform != "win32":
            return cls(None, unavailable="needs the server to run on Windows itself")
        for executable in executables:
            command = bridge_command(executable)
            if command is not None:
                start = functools.partial(start_in_view, Path(executable))
                return cls(command, start=start if start_ltspice else None)
        return cls(
            None,
            unavailable="no detected LTspice has ltspice-mcp-bridge.exe beside it (LTspice 26.1 or later)",
        )

    def check(self) -> None:
        """Raise ``WindowsUnavailable``, saying why, where there is no bridge to ask."""
        if self._command is None:
            raise WindowsUnavailable(f"LTspice windows cannot be reached here: {self.unavailable}")

    def _session(self) -> BridgeSession:
        self.check()
        assert self._command is not None
        return BridgeSession(self._command, timeout=self._timeout)

    @staticmethod
    def _windows(session: BridgeSession) -> list[Instance]:
        """The instances a person has open: never one a bridge started for itself."""
        return [instance for instance in session.instances() if instance.mode == _WINDOW]

    @staticmethod
    def _having(
        session: BridgeSession, windows: Sequence[Instance], path: Path
    ) -> Iterator[tuple[Instance, str]]:
        """Each of ``windows`` that has ``path`` open, with the path as it spells it.

        The session is attached to that window while the caller holds the pair.
        """
        for window in windows:
            session.attach(window.pid)
            for spelled in session.open_designs():
                if _same_file(path, spelled):
                    yield window, spelled

    def ensure_window(self) -> bool:
        """Have a window to show something in; True when LTspice was started for it.

        For a caller who was asked to show something in LTspice, before it
        asks a window to. With a window open, or where starting LTspice is not
        allowed, nothing is done and this is False: the call that follows
        finds the window, or says there is none. Blocks until the started
        LTspice offers its window to the bridge. Raises ``BridgeError`` when
        there is no bridge, when LTspice cannot be started, and when it opens
        no window in time.
        """
        with self._session() as session:
            if self._windows(session):
                return False
        if self._start is None:
            return False
        try:
            self._start()
        except OSError as error:
            raise BridgeError(f"LTspice could not be started: {error}") from error
        deadline = time.monotonic() + _STARTED_S
        while True:
            # A bridge asked while LTspice is still coming up may not answer.
            with contextlib.suppress(BridgeError), self._session() as session:
                if self._windows(session):
                    return True
            if time.monotonic() >= deadline:
                raise BridgeError(
                    f"LTspice was started and did not open a window within {_STARTED_S:g} s"
                )
            time.sleep(0.25)

    def holding(self, path: Path) -> list[OpenSheet]:
        """Every window that has ``path`` open, with the sheet as it holds it.

        Blocks for the bridge's answer. Raises ``BridgeError`` when the bridge
        cannot say, which is not the same answer as "no window has it".
        """
        with self._session() as session:
            return [
                OpenSheet(window.pid, window.version, spelled, session.design_text(spelled))
                for window, spelled in self._having(session, self._windows(session), path)
            ]

    def designs(
        self, wants_text: Callable[[str], bool] = lambda _path: False
    ) -> tuple[int, list[OpenDesign]]:
        """How many windows there are, and every document open in them.

        The window's copy of a document is read where ``wants_text`` says so
        of its path, which is how a caller keeps to the files it may read.
        Blocks; raises ``BridgeError`` when there is no bridge to ask or it
        cannot say.
        """
        found: list[OpenDesign] = []
        with self._session() as session:
            windows = self._windows(session)
            for window in windows:
                session.attach(window.pid)
                in_front = session.active_design()
                found += [
                    OpenDesign(
                        pid=window.pid,
                        version=window.version,
                        path=spelled,
                        active=in_front is not None and _same_file(Path(in_front), spelled),
                        text=session.design_text(spelled) if wants_text(spelled) else None,
                    )
                    for spelled in session.open_designs()
                ]
        return len(windows), found

    def open_sheet(self, path: Path) -> tuple[Instance, str | None]:
        """Open a sheet or netlist in an LTspice window and put it in front.

        Returns the window it is in, and that window's own copy when it
        already had the file open: LTspice does not read the file again, so
        the caller can then say whether what the person is shown is the file.
        Otherwise the first window opens it from the file, and there is no
        copy to hand back. For a caller who was asked to show it. Blocks;
        raises ``BridgeError`` when there is no bridge, when no window is
        running (none is started), or when LTspice refuses the file.
        """
        with self._session() as session:
            windows = self._windows(session)
            if not windows:
                raise BridgeError(_NO_WINDOW)
            for window, spelled in self._having(session, windows, path):
                held = session.design_text(spelled)
                session.bring_to_front(spelled)
                return window, held
            session.attach(windows[0].pid)
            session.open_design(str(path))
            session.bring_to_front(str(path))
            return windows[0], None

    def show_results(self, results: Path) -> Instance:
        """Open a results file in an LTspice window, in front, and return the window.

        For a caller who was asked to show it. Blocks; raises ``BridgeError``
        when there is no bridge, when no window is running (none is started),
        or when LTspice refuses the file.
        """
        with self._session() as session:
            windows = self._windows(session)
            if not windows:
                raise BridgeError(_NO_WINDOW)
            session.attach(windows[0].pid)
            session.show_results(str(results))
            return windows[0]

    def results_pane_open(self, window: Instance, sheet: Path) -> bool:
        """Whether ``window`` has the results beside ``sheet`` open in a pane.

        The pane is known by its title, the results file's name. Raises
        ``BridgeError`` when the window's frame cannot be asked.
        """
        wanted = sheet.with_suffix(_RESULTS_SUFFIX).name.casefold()
        try:
            return any(title.casefold() == wanted for title in self._frame.panes(window.pid))
        except FrameError as error:
            raise BridgeError(str(error)) from error

    def results_from_sheet(
        self, sheet: Path, place: Callable[[], None]
    ) -> tuple[Instance, str | None]:
        """Open the results beside ``sheet`` from the sheet, tied to it. Blocks.

        ``place`` is called to put the results beside the sheet, and the
        sheet's Visible Traces command then opens them. Returns the window
        used, and its own copy of the sheet when it already had the sheet
        open, as ``open_sheet`` does. For a caller who was asked to show them.

        LTspice looks for a sheet's results as it opens the sheet, and
        afterwards knows of none but those of a run it made there. So a sheet
        no window has open is opened only once ``place`` has been called. One
        that is already open keeps what LTspice found then: where that was
        nothing, the command does nothing, and this raises
        ``ResultsNotLookedFor`` once the wait for them runs out, with the
        results in place for the next time the sheet is opened.

        Raises ``ResultsAlreadyOpen``, before ``place`` is called, when the
        window already has those results open: LTspice does not read a
        results file again, and its command would then ask the person which
        traces to show. Raises ``BridgeError`` when there is no bridge or no
        window (none is started), when the sheet cannot be put in front, and
        when this build has no such command; what ``place`` raises is the
        caller's.
        """
        results = sheet.with_suffix(_RESULTS_SUFFIX).name
        with self._session() as session:
            windows = self._windows(session)
            if not windows:
                raise BridgeError(_NO_WINDOW)
            window, held = windows[0], None
            for holder, spelled in self._having(session, windows, sheet):
                window, held = holder, session.design_text(spelled)
                break
        if self.results_pane_open(window, sheet):
            raise ResultsAlreadyOpen(
                f"LTspice already has {results} open, and goes on showing the results it read"
            )
        place()
        with self._session() as session:
            session.attach(window.pid)
            if held is None:
                session.open_design(str(sheet))
            # The command goes to whatever is in front, and time has passed.
            for spelled in session.open_designs():
                if _same_file(sheet, spelled):
                    session.bring_to_front(spelled)
            in_front = session.active_design()
            if in_front is None or not _same_file(sheet, in_front):
                raise BridgeError(f"{sheet.name} could not be put in front in LTspice")
        try:
            self._frame.send(window.pid, VISIBLE_TRACES)
        except FrameError as error:
            raise BridgeError(str(error)) from error
        deadline = time.monotonic() + _OPENED_S
        while not self.results_pane_open(window, sheet):
            if time.monotonic() >= deadline:
                if held is None:
                    raise BridgeError(f"LTspice did not open {results} within {_OPENED_S:g} s")
                raise ResultsNotLookedFor(
                    f"LTspice had {sheet.name} open before these results were beside it, and "
                    "looks for a sheet's results only as it opens the sheet"
                )
            time.sleep(0.05)
        return window, held

    def show(self, sheet: OpenSheet, text: str) -> None:
        """Replace what ``sheet``'s window shows with ``text``. Blocks; raises ``BridgeError``."""
        with self._session() as session:
            session.attach(sheet.pid)
            session.replace_design_text(sheet.path, text)

    def show_each(self, sheets: Sequence[OpenSheet], text: str) -> list[str | None]:
        """``show`` for every window: None where it took the sheet, else why not.

        A window that could not be reached is an answer saying so, never an
        exception: the file is already committed when this runs, and a window
        left behind is a fact about that window.
        """
        refusals: list[str | None] = []
        for sheet in sheets:
            try:
                self.show(sheet, text)
            except BridgeError as error:
                refusals.append(str(error))
            else:
                refusals.append(None)
        return refusals
