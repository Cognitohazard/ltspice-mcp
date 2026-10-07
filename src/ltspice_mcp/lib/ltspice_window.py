"""A sheet someone has open in LTspice, kept in step with its file.

LTspice reads a sheet once. After that the window holds its own copy: a write
to the file changes nothing on screen, and the next save from the window puts
the old sheet back over the new file. An edit made by this server to a sheet
that is open would be invisible until the sheet was reopened and could be lost
without a word. ``OpenWindows`` is what closes that: it finds the windows that
have a file open (``holding``), and replaces what one shows with the committed
sheet (``show``), which LTspice records as one step of that window's undo
history. It reaches the windows through the bridge LTspice ships
(``BridgeSession``), attaching to instances that are already running and never
starting one. ``open_sheet`` opens a sheet in a window and ``show_results``
a finished run's results file, each for a caller who was asked to show it there; with the plot settings
file ``lib/plot_settings.py`` writes beside it, it opens with its traces drawn.

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

import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.lib.encoding import decode_spice_bytes, decode_windows_1252
from ltspice_mcp.lib.ltspice_bridge import (
    DEFAULT_TIMEOUT_S,
    BridgeError,
    BridgeSession,
    bridge_command,
)

# The lines that say how a sheet is stored and how far it extends, which
# LTspice writes for itself whatever the file said.
_STORAGE_LINES = frozenset({"Version", "SHEET"})
# Lines that belong to the symbol above them.
_SYMBOL_LINES = frozenset({"SYMATTR", "WINDOW"})
_MICRO_SIGNS = str.maketrans({"µ": "u", "μ": "u"})
_TEXT_GRID = 8
_WINDOW = "gui"


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
    readings = (decode_spice_bytes(on_disk), decode_windows_1252(on_disk))
    differences = [content_difference(reading, window_text) for reading in readings]
    return differences[0] if all(differences) else None


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


def _same_file(ours: Path, theirs: str) -> bool:
    return os.path.normcase(os.path.abspath(theirs)) == os.path.normcase(os.path.abspath(ours))


class OpenWindows:
    """The LTspice windows on this machine, as far as the bridge shows them.

    ``available`` is False where there is no bridge to ask, and every method
    then answers as if no window had anything open. ``unavailable`` says why,
    for ``inspect(kind="capabilities")``.
    """

    def __init__(
        self,
        command: Sequence[str] | None,
        *,
        unavailable: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._command = list(command) if command else None
        self.unavailable = None if self._command else (unavailable or "no bridge was found")
        self._timeout = timeout

    @property
    def available(self) -> bool:
        return self._command is not None

    @classmethod
    def detect(cls, executables: Iterable[str], *, enabled: bool = True) -> OpenWindows:
        """The windows reachable through the bridge beside one of ``executables``.

        Any 26.1 bridge finds every running instance, so the first one found
        is used. Native Windows only: under WSL the bridge is a Windows
        program that names Windows paths, and nothing here translates them.
        """
        if not enabled:
            return cls(None, unavailable="turned off by [schematic] sync_open_window")
        if sys.platform != "win32":
            return cls(None, unavailable="needs the server to run on Windows itself")
        for executable in executables:
            command = bridge_command(executable)
            if command is not None:
                return cls(command)
        return cls(
            None,
            unavailable="no detected LTspice has ltspice-mcp-bridge.exe beside it (LTspice 26.1 or later)",
        )

    def holding(self, path: Path) -> list[OpenSheet]:
        """Every window that has ``path`` open, with the sheet as it holds it.

        Blocks for the bridge's answer. Raises ``BridgeError`` when the bridge
        cannot say, which is not the same answer as "no window has it".
        """
        if self._command is None:
            return []
        sheets: list[OpenSheet] = []
        with BridgeSession(self._command, timeout=self._timeout) as session:
            for instance in session.instances():
                if instance.mode != _WINDOW:
                    continue
                session.attach(instance.pid)
                for spelled in session.open_designs():
                    if _same_file(path, spelled):
                        sheets.append(
                            OpenSheet(
                                pid=instance.pid,
                                version=instance.version,
                                path=spelled,
                                text=session.design_text(spelled),
                            )
                        )
        return sheets

    def designs(
        self, wants_text: Callable[[str], bool] = lambda _path: False
    ) -> tuple[int, list[OpenDesign]]:
        """How many windows there are, and every document open in them.

        The window's copy of a document is read where ``wants_text`` says so
        of its path, which is how a caller keeps to the files it may read.
        Blocks; raises ``BridgeError`` when there is no bridge to ask or it
        cannot say.
        """
        if self._command is None:
            raise BridgeError(self.unavailable or "no bridge")
        found: list[OpenDesign] = []
        windows = 0
        with BridgeSession(self._command, timeout=self._timeout) as session:
            for instance in session.instances():
                if instance.mode != _WINDOW:
                    continue
                windows += 1
                session.attach(instance.pid)
                in_front = session.active_design()
                for spelled in session.open_designs():
                    found.append(
                        OpenDesign(
                            pid=instance.pid,
                            version=instance.version,
                            path=spelled,
                            active=in_front is not None and _same_file(Path(in_front), spelled),
                            text=session.design_text(spelled) if wants_text(spelled) else None,
                        )
                    )
        return windows, found

    def open_sheet(self, path: Path) -> OpenSheet | tuple[int, str]:
        """Open a sheet or netlist in an LTspice window and put it in front.

        A window that already has it open is the one used, and what comes
        back is that window's copy (an ``OpenSheet``), because LTspice does
        not read the file again: the caller can then say whether what the
        person is shown is the file. Otherwise the first window opens it from
        the file, and its process and version come back. For a caller who was
        asked to show it. Blocks; raises ``BridgeError`` when there is no
        bridge, when no window is running (none is started), or when LTspice
        refuses the file.
        """
        if self._command is None:
            raise BridgeError(self.unavailable or "no bridge")
        with BridgeSession(self._command, timeout=self._timeout) as session:
            running = [instance for instance in session.instances() if instance.mode == _WINDOW]
            if not running:
                raise BridgeError("no LTspice window is open, and none is started for this")
            for window in running:
                session.attach(window.pid)
                for spelled in session.open_designs():
                    if _same_file(path, spelled):
                        held = OpenSheet(
                            pid=window.pid,
                            version=window.version,
                            path=spelled,
                            text=session.design_text(spelled),
                        )
                        session.bring_to_front(spelled)
                        return held
            window = running[0]
            session.attach(window.pid)
            session.open_design(str(path))
            session.bring_to_front(str(path))
        return window.pid, window.version

    def show_results(self, results: Path) -> tuple[int, str]:
        """Open a results file in an LTspice window and put it in front.

        Returns the process and version of the window it went to. This is
        the one thing here that opens something in a window, so it is for a
        caller who was asked to. Blocks; raises ``BridgeError`` when there is
        no bridge, when no window is running (none is started), or when
        LTspice refuses the file.
        """
        if self._command is None:
            raise BridgeError(self.unavailable or "no bridge")
        with BridgeSession(self._command, timeout=self._timeout) as session:
            running = [instance for instance in session.instances() if instance.mode == _WINDOW]
            if not running:
                raise BridgeError("no LTspice window is open, and none is started for this")
            window = running[0]
            session.attach(window.pid)
            session.show_results(str(results))
        return window.pid, window.version

    def show(self, sheet: OpenSheet, text: str) -> None:
        """Replace what ``sheet``'s window shows with ``text``. Blocks; raises ``BridgeError``."""
        if self._command is None:
            raise BridgeError(self.unavailable or "no bridge")
        with BridgeSession(self._command, timeout=self._timeout) as session:
            session.attach(sheet.pid)
            session.replace_design_text(sheet.path, text)

    def show_each(self, sheets: Sequence[OpenSheet], text: str) -> list[dict[str, object]]:
        """``show`` for every window, as one row each of what happened there.

        A window that could not be reached is a row saying so, never an
        exception: the file is already committed when this runs, and a window
        left behind is a fact about that window.
        """
        rows: list[dict[str, object]] = []
        for sheet in sheets:
            row: dict[str, object] = {"pid": sheet.pid, "version": sheet.version, "shown": True}
            try:
                self.show(sheet, text)
            except BridgeError as error:
                row["shown"] = False
                row["reason"] = str(error)
            rows.append(row)
        return rows
