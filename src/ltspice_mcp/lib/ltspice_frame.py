"""The frame window of a running LTspice, for what the bridge has no call for.

The bridge lists the sheets and netlists a window has open and never its
results files, and it opens a results file only on its own, where LTspice does
not tie the plot to a sheet: a click on a net then plots nothing. A person gets
a tied plot by choosing Visible Traces on the sheet, which opens the results
beside it. So the frame is asked two things directly:

- which documents have a pane in it, results files included (``panes``);
- to carry out a command of its own menu, as choosing it would (``send``).

A command is named by the label the menu shows. Its number is the build's own,
so it is read from the menus of the executable the process runs
(``menu_command``), and a build whose menu has no such label has no such
command here. Both work on a window of another desktop.

What a command does depends on what the window has in front, which the frame
cannot be asked for: Visible Traces on a sheet whose results are already open
shows a dialog to pick traces from, and so does the same command with a plot in
front. The caller makes sure of the state first (``OpenWindows``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import psutil

from ltspice_mcp.lib import hidden_desktop, pe_menu
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.ltspice_bridge import BridgeError

#: The schematic editor's command that opens the results beside a sheet.
VISIBLE_TRACES = "Visible Traces"
# A top-level menu only the schematic editor's menu bar has, which tells its
# menu from the waveform viewer's and the symbol editor's.
_SHEET_MENU = "Hierarchy"


class FrameError(BridgeError):
    """The frame of an LTspice process could not be found, or not told what was asked.

    A ``BridgeError``, as every failure to reach an LTspice window is: a
    caller that reports one reports this.
    """


def is_frame(window: int) -> bool:
    """Whether a top-level window is an LTspice frame, by its class and title."""
    return hidden_desktop.window_class(window).startswith("Afx:") and hidden_desktop.window_text(
        window
    ).startswith("LTspice")


#: The schematic editor's commands by executable, read again when the file changes.
_SHEET_COMMANDS: FileCache[dict[str, int]] = FileCache(maxsize=8)


def menu_command(executable: Path, label: str) -> int | None:
    """The command the schematic editor's menu gives ``label`` in the build at
    ``executable``, or None when its menu has no such item.

    Raises ``OSError`` when the file cannot be read and ``ValueError`` when it
    is not a Windows program.
    """
    sheet_menu = _SHEET_COMMANDS.get(executable, lambda path: pe_menu.commands(path, _SHEET_MENU))
    return sheet_menu.get(label)


class LtspiceFrame:
    """The frames of the LTspice processes on one desktop.

    ``top_level`` lists a process's top-level windows; left out, it is the
    desktop this process runs on, which is where a person's LTspice is.
    """

    def __init__(self, top_level: Callable[[int], list[int]] | None = None) -> None:
        self._top_level = top_level or hidden_desktop.windows_here

    def _frame(self, pid: int) -> int:
        if sys.platform != "win32":
            raise FrameError("an LTspice window is reached through Windows")
        for window in self._top_level(pid):
            if is_frame(window):
                return window
        raise FrameError(f"LTspice process {pid} has no window on this desktop")

    def panes(self, pid: int) -> list[str]:
        """The title of each window inside ``pid``'s frame that has one.

        A document's pane is titled with its file name, so a results file the
        window has open is among these, as ``<name>.raw``.
        """
        titles = (
            hidden_desktop.window_text(child)
            for child in hidden_desktop.child_windows(self._frame(pid))
        )
        return sorted({title for title in titles if title})

    def send(self, pid: int, label: str) -> None:
        """Have ``pid``'s frame carry out the sheet command its menu labels ``label``.

        The command goes to whatever document is in front, and returns before
        it is carried out. Raises ``FrameError`` when the build has no such
        command or the window does not take it.
        """
        frame = self._frame(pid)
        try:
            executable = Path(psutil.Process(pid).exe())
            command = menu_command(executable, label)
        except (psutil.Error, OSError, ValueError) as error:
            raise FrameError(
                f"the menu of LTspice process {pid} could not be read: {error}"
            ) from error
        if command is None:
            raise FrameError(f"this LTspice build's sheet menu has no {label!r} command")
        try:
            hidden_desktop.post_command(frame, command)
        except OSError as error:
            raise FrameError(f"LTspice's window did not take {label!r}: {error}") from error
