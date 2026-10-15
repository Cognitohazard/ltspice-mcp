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

LTspice reads a results file once and looks for a sheet's results once, as it
opens each. So showing a newer run means closing what is open and opening it
again, and both showings do that themselves instead of leaving it to a person:
a results pane closes without a question, and a sheet is closed only where the
window holds what its file does, so that nothing anyone did there can be lost.

Those three are for a caller who was asked to show something in LTspice, and
each of them, and nothing else, starts LTspice when no window is open
(``start_in_view``, kept with the other starts of LTspice on Windows): where a
person can see it, which is the one start of LTspice anywhere here that is
meant to be seen. What they hand back (``Shown``) says so. Keeping an open
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
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ltspice_mcp.lib.encoding import decode_spice_bytes, decode_windows_1252
from ltspice_mcp.lib.ltspice_bridge import (
    DEFAULT_TIMEOUT_S,
    BridgeError,
    BridgeSession,
    Instance,
    bridge_command,
)
from ltspice_mcp.lib.ltspice_frame import VISIBLE_TRACES, LtspiceFrame, SaveQuestion
from ltspice_mcp.lib.ltspice_windows import start_in_view

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
_OPENED_S = 5.0
"""How long LTspice is given to show the results it was asked to open. Opening
a results file is quick; this is a cap on a window that did nothing."""
_LOOKED_S = 2.0
"""How long a sheet that was already open is given to open its results. One
that knows of them does it in a fraction of a second however large they are,
and one that knows of none never does: this is how long it takes to tell."""
_CLOSED_S = 2.0
"""How long LTspice is given to close a pane it was told to close, or to ask
about it. Either takes a moment; this is a cap on a window that did neither."""


class WindowsUnavailable(BridgeError):
    """There is no bridge to ask here: off Windows, turned off, or none installed."""


class NoWindow(BridgeError):
    """No LTspice window is open to show something in."""


def _within(seconds: float, every: float, happened: Callable[[], bool]) -> bool:
    """Whether ``happened`` comes true within ``seconds``, asked ``every`` so often.

    Blocks. For what LTspice does in its own time and tells no one of.
    """
    deadline = time.monotonic() + seconds
    while not happened():
        if time.monotonic() >= deadline:
            return False
        time.sleep(every)
    return True


def results_beside(sheet: Path) -> Path:
    """Where LTspice keeps a sheet's own results and looks for them: beside it, under its name."""
    return sheet.with_suffix(".raw")


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


@dataclass(frozen=True)
class Difference:
    """How a window's copy of a sheet differs from its file's.

    Each side holds the entries of ``sheet_content`` it has and the other
    lacks, sorted: a symbol with its attributes, or one other line. A part
    whose value changed is on both sides, once with each value.
    """

    only_in_window: tuple[str, ...]
    only_in_file: tuple[str, ...]

    def summary(self, limit: int = 3) -> str:
        """The difference in a sentence, naming up to ``limit`` entries a side.

        An entry is named by its first line, which is enough to recognise the
        change and short enough to put in a refusal.
        """

        def named(entries: tuple[str, ...]) -> str:
            shown = "; ".join(entry.split("\n", 1)[0] for entry in entries[:limit])
            more = len(entries) - limit
            return shown + (f"; and {more} more" if more > 0 else "")

        parts = []
        if self.only_in_window:
            parts.append(f"only in the window: {named(self.only_in_window)}")
        if self.only_in_file:
            parts.append(f"only in the file: {named(self.only_in_file)}")
        return ". ".join(parts)


def content_difference(file_text: str, window_text: str) -> Difference | None:
    """How a window's sheet differs from its file's, or None when it does not."""
    in_file = Counter(sheet_content(file_text))
    in_window = Counter(sheet_content(window_text))
    if in_file == in_window:
        return None
    return Difference(
        only_in_window=tuple(sorted((in_window - in_file).elements())),
        only_in_file=tuple(sorted((in_file - in_window).elements())),
    )


def file_difference(on_disk: bytes, window_text: str) -> Difference | None:
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


@dataclass(frozen=True)
class Shown:
    """Where something was shown, for a caller who was asked to show it there.

    ``window`` is the LTspice window it is in. ``held`` is that window's own
    copy of a sheet it already had open, and None where it opened the file:
    LTspice does not read a file again, so with the copy the caller can say
    whether what the person is shown is the file. ``started`` says LTspice
    was not running and was started for this. ``reopened`` says a plot of the
    same results was open and was closed to show these. ``sheet_reopened``
    says the sheet was open, knew of no results, and was closed and opened
    again so that it would. ``tied`` says results were opened from their
    sheet, so the plot is tied to it.
    """

    window: Instance
    held: str | None = None
    started: bool = False
    reopened: bool = False
    sheet_reopened: bool = False
    tied: bool = False


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
        start: Callable[[], None] | None = None,
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
    def _spelling(session: BridgeSession, path: Path) -> str | None:
        """How the attached window spells ``path``, or None when it does not have it open."""
        return next((each for each in session.open_designs() if _same_file(path, each)), None)

    @classmethod
    def _having(
        cls, session: BridgeSession, windows: Sequence[Instance], path: Path
    ) -> Iterator[tuple[Instance, str]]:
        """Each of ``windows`` that has ``path`` open, with the path as it spells it.

        The session is attached to that window while the caller holds the pair.
        """
        for window in windows:
            session.attach(window.pid)
            spelled = cls._spelling(session, path)
            if spelled is not None:
                yield window, spelled

    def _to_show_in(self, session: BridgeSession) -> list[Instance]:
        """``_windows``, for a caller with something to show: raises ``NoWindow`` for none."""
        windows = self._windows(session)
        if not windows:
            raise NoWindow(_NO_WINDOW)
        return windows

    def _holder(self, session: BridgeSession, path: Path) -> tuple[Instance, str | None]:
        """The window to show ``path`` in, and how it spells the path if it has it open.

        The first window that has the file open, else the first window and
        None; the session is left attached to it. Raises ``NoWindow`` for none.
        """
        windows = self._to_show_in(session)
        for window, spelled in self._having(session, windows, path):
            return window, spelled
        if len(windows) > 1:
            session.attach(windows[0].pid)
        return windows[0], None

    def _starting(self, show: Callable[[], Shown]) -> Shown:
        """``show``, with LTspice started for it when no window is open.

        With a window open, or where starting LTspice is not allowed, this is
        ``show`` alone. Otherwise LTspice is started, its window is waited
        for, and ``show`` is asked again, so ``show`` must have done nothing
        when it raises ``NoWindow``. Raises ``BridgeError`` when LTspice
        cannot be started and when it opens no window in time.
        """
        try:
            return show()
        except NoWindow:
            if self._start is None:
                raise
        try:
            self._start()
        except OSError as error:
            raise BridgeError(f"LTspice could not be started: {error}") from error

        def offered() -> bool:
            # A bridge asked while LTspice is still coming up may not answer.
            with contextlib.suppress(BridgeError), self._session() as session:
                return bool(self._windows(session))
            return False

        if not _within(_STARTED_S, 0.5, offered):
            raise BridgeError(
                f"LTspice was started and did not open a window within {_STARTED_S:g} s"
            )
        return replace(show(), started=True)

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

    def open_sheet(self, path: Path) -> Shown:
        """Open a sheet or netlist in an LTspice window and put it in front.

        A window that already has the file open shows the copy it holds, which
        is handed back; otherwise the first window opens it from the file.
        For a caller who was asked to show it, so LTspice is started when no
        window is open. Blocks; raises ``BridgeError`` when there is no
        bridge, when no window is open and none may be started, or when
        LTspice refuses the file.
        """

        def show() -> Shown:
            with self._session() as session:
                window, spelled = self._holder(session, path)
                held = None
                if spelled is None:
                    session.open_design(str(path))
                else:
                    held = session.design_text(spelled)
                session.bring_to_front(spelled or str(path))
                return Shown(window, held)

        return self._starting(show)

    def show_results(self, results: Path) -> Shown:
        """Open a results file in an LTspice window, in front, on its own.

        A window that already has the file open goes on showing what it read
        of it and of the plot settings beside it, so that pane is closed
        first and ``reopened`` says so; traces someone added to it by hand go
        with it. For a caller who was asked to show it, so LTspice is started
        when no window is open. Blocks; raises ``BridgeError`` when there is
        no bridge, when no window is open and none may be started, when the
        open pane does not close, or when LTspice refuses the file.
        """

        def show() -> Shown:
            with self._session() as session:
                window = self._to_show_in(session)[0]
                session.attach(window.pid)
                reopened = self._close_results(session, window.pid, results)
                session.show_results(str(results))
                return Shown(window, reopened=reopened)

        return self._starting(show)

    def _free(self, pid: int) -> None:
        """Raise ``BridgeError`` while LTspice process ``pid`` waits on a box of its own."""
        if not self._frame.takes_input(pid):
            raise BridgeError(
                "LTspice is waiting for an answer to a box of its own, so nothing was "
                "closed in it; answer that box and ask again"
            )

    def _pane_in_front(
        self,
        session: BridgeSession,
        pid: int,
        path: Path,
        spelled: str,
        in_front: Callable[[], str | None],
    ) -> int | None:
        """The pane of ``path`` in the attached window ``pid``, put in front; None
        when the window does not have it open.

        A pane is titled with its file's name, which the results of two runs
        of one deck share, and two sheets can, so the file is found by path:
        the bridge puts ``spelled`` in front, which does nothing with a file
        that is not open, and ``in_front`` asks it what is in front then.
        Where that is ``path``, the first pane of its title is the one
        (recorded). The frame is asked only then, so a file that is not open
        is shown where the frame cannot be reached.
        """
        session.bring_to_front(spelled)
        shown = in_front()
        if shown is None or not _same_file(path, shown):
            return None
        return self._frame.pane(pid, path.name)

    def _close_results(self, session: BridgeSession, pid: int, results: Path) -> bool:
        """Close the pane showing ``results`` in the attached window ``pid``.

        False when the window does not have that file open. Raises
        ``BridgeError`` when it is open and does not close.
        """
        pane = self._pane_in_front(session, pid, results, str(results), session.results_in_front)
        if pane is None:
            return False
        self._free(pid)
        self._frame.close(pane)
        # A pane closes in a moment, so this looks often.
        if not _within(_CLOSED_S, 0.02, lambda: not self._frame.is_open(pane)):
            raise BridgeError(
                f"LTspice has {results.name} open and did not close it within "
                f"{_CLOSED_S:g} s, so it goes on showing the results it read; close "
                "that plot there and ask again"
            )
        return True

    def results_from_sheet(self, sheet: Path, place: Callable[[], None]) -> Shown:
        """Open the results beside ``sheet`` from the sheet, tied to it. Blocks.

        ``place`` is called to put the results beside the sheet, and the
        sheet's Visible Traces command then opens them. The window's own copy
        of the sheet is handed back when it already had the sheet open, as
        ``open_sheet`` does. For a caller who was asked to show them, so
        LTspice is started when no window is open.

        A window that already has those results open goes on showing what
        it read, and with them open its command asks the person which traces
        to show. So that pane is closed first, before ``place`` is called,
        which LTspice does without a question and without saving anything;
        traces someone added to it by hand go with it, and ``reopened`` says
        that it was done.

        LTspice looks for a sheet's results as it opens the sheet, and
        afterwards knows of none but those of a run it made there. So a sheet
        no window has open is opened only once ``place`` has been called. One
        that was open before it had any does nothing with the command, and is
        closed and opened again, which ``sheet_reopened`` says: its place in
        the window and its undo history go with it, and nothing else, because
        it is closed only where the window holds what the file does
        (``_close_sheet``). Where it holds anything else, the results are
        opened on their own instead: ``tied`` is False, and ``held`` is what
        the window holds.

        Raises ``BridgeError`` when a pane does not close, when there is no
        bridge, when no window is open and none may be started, when the
        sheet cannot be put in front, and when this build has no such
        command; what ``place`` raises is the caller's.
        """
        return self._starting(lambda: self._results_from_sheet(sheet, place))

    def _results_from_sheet(self, sheet: Path, place: Callable[[], None]) -> Shown:
        results = results_beside(sheet)
        with self._session() as session:
            window, spelled = self._holder(session, sheet)
            held = None if spelled is None else session.design_text(spelled)
            reopened = self._close_results(session, window.pid, results)
        # A sheet open with no results beside it knows of none: LTspice looks
        # for them only as it opens a sheet, and a run made there leaves them.
        may_know = results.exists()
        place()
        if held is None or may_know:
            if self._open_from_sheet(window.pid, sheet, _OPENED_S if held is None else _LOOKED_S):
                return Shown(window, held, reopened=reopened, tied=True)
            if held is None:
                raise BridgeError(f"LTspice did not open {results.name} within {_OPENED_S:g} s")
        with self._session() as session:
            session.attach(window.pid)
            unsaved = self._close_sheet(session, window.pid, sheet, held)
            if unsaved is not None:
                session.show_results(str(results))
                return Shown(window, unsaved, reopened=reopened)
            self._sheet_in_front(session, sheet)
        if self._visible_traces(window.pid, sheet, _OPENED_S):
            return Shown(window, reopened=reopened, sheet_reopened=True, tied=True)
        raise BridgeError(
            f"LTspice opened {sheet.name} again and did not open {results.name} within "
            f"{_OPENED_S:g} s"
        )

    def _sheet_in_front(self, session: BridgeSession, sheet: Path) -> None:
        """Open ``sheet`` in the attached window where it is not open, and put it in front.

        Raises ``BridgeError`` when it is not in front then: the command that
        opens its results goes to whatever is.
        """
        spelled = self._spelling(session, sheet)
        if spelled is None:
            session.open_design(str(sheet))
        session.bring_to_front(spelled or str(sheet))
        in_front = session.active_design()
        if in_front is None or not _same_file(sheet, in_front):
            raise BridgeError(f"{sheet.name} could not be put in front in LTspice")

    def _visible_traces(self, pid: int, sheet: Path, seconds: float) -> bool:
        """Send the sheet in front in window ``pid`` its Visible Traces command:
        whether a pane of its results opened within ``seconds``.

        What is waited for is a pane of the results' name that was not there
        before, since the results of another run of that name may be open on
        their own.
        """
        results = results_beside(sheet).name
        before = set(self._frame.panes_titled(pid, results))
        self._frame.send(pid, VISIBLE_TRACES)
        return _within(
            seconds,
            0.05,
            lambda: any(pane not in before for pane in self._frame.panes_titled(pid, results)),
        )

    def _open_from_sheet(self, pid: int, sheet: Path, seconds: float) -> bool:
        """``_sheet_in_front`` and then ``_visible_traces``."""
        with self._session() as session:
            session.attach(pid)
            self._sheet_in_front(session, sheet)
        return self._visible_traces(pid, sheet, seconds)

    def _close_sheet(self, session: BridgeSession, pid: int, sheet: Path, held: str) -> str | None:
        """Close ``sheet`` in the attached window ``pid``, so that opened again it
        looks for its results. Returns None once it is closed.

        Only where the window holds what the file does: ``held`` is what it
        held when the caller last read it, and where that differs the sheet is
        left open and what it holds is returned. LTspice counts a sheet it was
        handed through the bridge as changed even when it reads as its file
        does, which every sheet this server edited while it was open is, and
        asks whether to save it before closing (recorded). While it asks
        nobody can change the sheet and the bridge still reads it, so it is
        read again then: the answer is No where it reads as the file, which
        loses nothing, and otherwise Cancel, which leaves it as it was.
        """
        on_disk = sheet.read_bytes()
        if file_difference(on_disk, held) is not None:
            return held
        spelled = self._spelling(session, sheet)
        if spelled is None:
            return None  # closed by someone since
        pane = self._pane_in_front(session, pid, sheet, spelled, session.active_design)
        if pane is None:
            raise BridgeError(f"{sheet.name} could not be put in front in LTspice")
        self._free(pid)
        self._frame.close(pane)
        question: SaveQuestion | None = None

        def closed_or_asked() -> bool:
            nonlocal question
            question = self._frame.save_question(pid)
            return question is not None or not self._frame.is_open(pane)

        if not _within(_CLOSED_S, 0.02, closed_or_asked):
            raise BridgeError(f"LTspice did not close {sheet.name} within {_CLOSED_S:g} s")
        if question is None:
            return None
        now = session.design_text(spelled)
        keep = not _same_file(sheet, question.path) or file_difference(on_disk, now) is not None
        self._frame.answer(question, discard=not keep)
        # Cancelled, the box goes; answered No, the sheet goes after it.
        gone = question.box if keep else pane
        if not _within(_CLOSED_S, 0.02, lambda: not self._frame.is_open(gone)):
            raise BridgeError(f"LTspice did not take its answer about {sheet.name}")
        return now if keep else None

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
