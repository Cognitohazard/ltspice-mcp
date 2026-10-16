"""The frame window of a running LTspice, for what the bridge has no call for.

The bridge lists the sheets and netlists a window has open and never its
results files, and it opens a results file only on its own, where LTspice does
not tie the plot to a sheet: a click on a net then plots nothing. A person gets
a tied plot by choosing Visible Traces on the sheet, which opens the results
beside it. LTspice reads a results file once: a pane that is open goes on
showing what it read, whatever the file holds since. And it looks for a
sheet's results once, as it opens the sheet. So the frame is asked directly:

- which documents have a pane in it, results files included (``panes``);
- to carry out a command of its own menu, as choosing it would (``send``);
- to close one of its panes, as that pane's close box would (``close``), and
  whether it has (``is_open``);
- what LTspice asks before it closes a sheet with changes nobody saved, and
  to answer that without saving (``save_question``, ``answer``);
- whether it takes a person's input, which it does not while LTspice has a
  box up that must be answered first (``takes_input``);
- whether a command of its menu is enabled for the document in front, which
  for Visible Traces says whether a sheet knows of results
  (``command_enabled``).

A pane is known by its title, its file's name, which two files in different
folders share. Windows lists the panes from the front, so the first pane of a
title is the one in front of the others of that title (``pane``); the caller
puts the file it means in front first, which the bridge does by path.

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
from dataclasses import dataclass
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
# How the window class of the frame, and of each document's pane in it, begins.
_DOCUMENT_CLASS = "Afx:"
# The window class of the words in a box a program puts up.
_WORDS_CLASS = "Static"
# What LTspice asks before it closes a document holding changes nobody saved,
# on either side of that document's path.
_SAVE_QUESTION = ("Save changes to ", "?")
# The two answers to it that leave the file as it is: close without saving, and
# do not close.
_NO = 7
_CANCEL = 2


class FrameError(BridgeError):
    """The frame of an LTspice process could not be found, or not told what was asked.

    A ``BridgeError``, as every failure to reach an LTspice window is: a
    caller that reports one reports this.
    """


@dataclass(frozen=True)
class SaveQuestion:
    """LTspice asking whether to save a document it was told to close.

    ``box`` is the window it asks in and ``path`` the document it names, as
    LTspice spells it. Nobody can change a document while LTspice asks.
    """

    box: int
    path: str


def is_frame(window: int) -> bool:
    """Whether a top-level window is an LTspice frame, by its class and title."""
    return hidden_desktop.window_class(window).startswith(
        _DOCUMENT_CLASS
    ) and hidden_desktop.window_text(window).startswith("LTspice")


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
        command = self._command(pid, label)
        try:
            hidden_desktop.post_command(frame, command)
        except OSError as error:
            raise FrameError(f"LTspice's window did not take {label!r}: {error}") from error

    @staticmethod
    def _command(pid: int, label: str) -> int:
        """The number of the sheet command ``label`` in the build process ``pid`` runs."""
        try:
            executable = Path(psutil.Process(pid).exe())
            command = menu_command(executable, label)
        except (psutil.Error, OSError, ValueError) as error:
            raise FrameError(
                f"the menu of LTspice process {pid} could not be read: {error}"
            ) from error
        if command is None:
            raise FrameError(f"this LTspice build's sheet menu has no {label!r} command")
        return command

    def command_enabled(self, pid: int, label: str) -> bool | None:
        """Whether the sheet command ``label`` is enabled for the document in
        front of ``pid``'s frame, as its menu would show it.

        None where the menu cannot be read. Visible Traces is turned off for a
        sheet that knows of no results, which is one LTspice opened with none
        beside it, whatever was put there since (recorded).
        """
        try:
            return hidden_desktop.menu_item_enabled(self._frame(pid), self._command(pid, label))
        except OSError:
            return None

    def panes_titled(self, pid: int, title: str) -> list[int]:
        """The window of each pane titled ``title`` in ``pid``'s frame, the one in front first.

        A pane's title is its file's name, whatever its case, and is on more
        than one window inside the frame: the pane is the one whose class
        begins as the frame's does.
        """
        wanted = title.casefold()
        return [
            child
            for child in hidden_desktop.child_windows(self._frame(pid))
            if hidden_desktop.window_text(child).casefold() == wanted
            and hidden_desktop.window_class(child).startswith(_DOCUMENT_CLASS)
        ]

    def pane(self, pid: int, title: str) -> int | None:
        """The pane titled ``title`` that is in front of the others so titled, or None."""
        return next(iter(self.panes_titled(pid, title)), None)

    def close(self, pane: int) -> None:
        """Ask a pane to close; returns before it has.

        A results pane closes without a question and saves nothing. A sheet
        closes the same unless LTspice counts it changed, and then LTspice
        asks first (``save_question``); all three are recorded. Raises
        ``FrameError`` when the window does not take the request.
        """
        try:
            hidden_desktop.post_close(pane)
        except OSError as error:
            raise FrameError(
                f"LTspice's window did not take the request to close: {error}"
            ) from error

    def is_open(self, window: int) -> bool:
        """Whether a pane, or a box LTspice asked in, is still there."""
        return hidden_desktop.window_exists(window)

    def takes_input(self, pid: int) -> bool:
        """Whether a person could use ``pid``'s frame now.

        False while LTspice has a box up that must be answered first. A pane
        closed from outside then would go from under whatever the box is about.
        """
        return hidden_desktop.takes_input(self._frame(pid))

    def save_question(self, pid: int) -> SaveQuestion | None:
        """The question LTspice process ``pid`` is asking about saving a document, or None."""
        before, after = _SAVE_QUESTION
        for window in self._top_level(pid):
            if hidden_desktop.window_class(window) != hidden_desktop.DIALOG_CLASS:
                continue
            for child in hidden_desktop.child_windows(window):
                words = hidden_desktop.window_text(child)
                if (
                    hidden_desktop.window_class(child) == _WORDS_CLASS
                    and words.startswith(before)
                    and words.endswith(after)
                ):
                    return SaveQuestion(window, words[len(before) : -len(after)])
        return None

    def answer(self, question: SaveQuestion, *, discard: bool) -> None:
        """Answer ``question`` without saving: No with ``discard``, which closes
        the document as its file has it, and otherwise Cancel, which leaves it
        open as it is. Returns before LTspice has acted on it. Raises
        ``FrameError`` when the box does not take the answer.
        """
        try:
            hidden_desktop.post_command(question.box, _NO if discard else _CANCEL)
        except OSError as error:
            raise FrameError(f"LTspice's question did not take an answer: {error}") from error
