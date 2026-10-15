"""A stand-in LTspice window, put in front of a handler.

``tests/fake_ltspice_bridge.py`` answers as LTspice's bridge was recorded
answering, about the windows a "world" file describes. This writes and reads
that file and points a session at the stand-in: the part every test with a
window in it shares.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zlib
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.encoding import decode_windows_1252
from ltspice_mcp.lib.ltspice_frame import (
    VISIBLE_TRACES,
    FrameError,
    LtspiceFrame,
    SaveQuestion,
)
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from tests.conftest import LIVENESS_S
from tests.fake_ltspice_bridge import front_design, in_front

FAKE = Path(__file__).with_name("fake_ltspice_bridge.py")
PID = 4242
#: The process a stand-in start of LTspice makes.
STARTED_PID = 5151
VERSION = "26.1.1"


def fake_command(world: Path) -> list[str]:
    return [sys.executable, str(FAKE), str(world)]


def write_world(world: Path, windows: list[dict[str, Any]], **extra: Any) -> None:
    world.write_text(json.dumps({"windows": windows, **extra}), encoding="utf-8")


def read_world(world: Path) -> dict[str, Any]:
    return json.loads(world.read_text(encoding="utf-8"))


def a_window(designs: dict[str, str] | None = None, **entry: Any) -> dict[str, Any]:
    """One window's entry in a world: the text it holds for each path it has open."""
    return {"pid": PID, "version": VERSION, "designs": designs or {}, **entry}


def put_windows(
    state: SessionState,
    world: Path,
    windows: list[dict[str, Any]],
    *,
    timeout: float = LIVENESS_S,
    start: FakeStart | None = None,
    **extra: Any,
) -> None:
    """Write ``windows`` into ``world`` and have ``state`` reach them through the stand-in.

    Without a ``start``, LTspice is never started, as with the setting off.
    """
    write_world(world, windows, **extra)
    state.open_windows = OpenWindows(
        fake_command(world), timeout=timeout, frame=FakeFrame(world), start=start
    )


class FakeStart:
    """Starting LTspice, for the stand-in: a window with nothing open appears in
    the world, as an LTspice started with no document was recorded appearing.

    ``calls`` counts the starts. With ``opens_window`` false the program starts
    and never offers a window; with an ``error`` it cannot be started.
    """

    def __init__(
        self, world: Path, *, opens_window: bool = True, error: OSError | None = None
    ) -> None:
        self._world = world
        self._opens_window = opens_window
        self._error = error
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._opens_window:
            world = read_world(self._world)
            world["windows"].append(a_window(pid=STARTED_PID))
            self._world.write_text(json.dumps(world), encoding="utf-8")


def _name(path: str) -> str:
    """A file's name from a path in either spelling, on any platform."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def results_of(sheet: str) -> str:
    """The results beside a sheet, from its path in either spelling."""
    return sheet.rsplit(".", 1)[0] + ".raw"


class FakeFrame(LtspiceFrame):
    """The frame of a stand-in window, kept in the same world file.

    A window's entry lists the results files it has open under ``panes``, by
    path and the one in front first, and the commands it was sent under
    ``commands``. Visible Traces opens the results of the sheet in front
    where the window knows of any, as LTspice was recorded doing: the sheets
    under ``with_results``, which the stand-in bridge adds a sheet to when it
    opens one that has results beside it. A world with ``frame_ignores`` is a
    window that does nothing with a command, and one with
    ``frame_has_no_command`` a build whose menu lacks it.

    Whatever is told to close is noted under ``closed``, and all of it stays
    in a world with ``frame_keeps_panes``. Otherwise a results pane goes, and
    so does a sheet, with the plot opened from it, unless the window counts
    it changed (``unsaved``): then the window asks (``asking``) and takes no
    input until it is answered, which is noted under ``answers``; No closes
    the sheet and Cancel leaves it. All as LTspice was recorded doing. A
    world with ``box_up`` is a window waiting on a box of its own.

    A pane's window is a number made from its path and how often that path
    was closed before, so one opened again is another window.
    """

    def __init__(self, world: Path) -> None:
        super().__init__(lambda _pid: [])
        self._world = world

    def _entry(self, world: dict[str, Any], pid: int) -> dict[str, Any]:
        for entry in world["windows"]:
            if entry["pid"] == pid:
                return entry
        raise FrameError(f"LTspice process {pid} has no window on this desktop")

    def _keep(self, world: dict[str, Any]) -> None:
        write_world(self._world, **world)

    @staticmethod
    def _window_of(entry: dict[str, Any], path: str, kind: str = "pane") -> int:
        times_closed = entry.get("closings", {}).get(path, 0)
        return zlib.crc32(f"{kind}|{entry['pid']}|{path}|{times_closed}".encode())

    @staticmethod
    def _in_order(entry: dict[str, Any]) -> list[str]:
        """Every path with a pane, results first and the one in front of each kind first."""
        front = front_design(entry)
        sheets = sorted(entry["designs"], key=lambda path: path != front)
        return [*entry.get("panes", []), *sheets]

    def _open_windows(self, world: dict[str, Any]) -> dict[int, tuple[dict[str, Any], str]]:
        """Each pane that is open, by its window: the entry it is in and its path."""
        return {
            self._window_of(entry, path): (entry, path)
            for entry in world["windows"]
            for path in self._in_order(entry)
        }

    def panes(self, pid: int) -> list[str]:
        entry = self._entry(read_world(self._world), pid)
        return sorted({_name(path) for path in self._in_order(entry)})

    def panes_titled(self, pid: int, title: str) -> list[int]:
        entry = self._entry(read_world(self._world), pid)
        return [
            self._window_of(entry, path)
            for path in self._in_order(entry)
            if _name(path).casefold() == title.casefold()
        ]

    def send(self, pid: int, label: str) -> None:
        world = read_world(self._world)
        entry = self._entry(world, pid)
        if world.get("frame_has_no_command"):
            raise FrameError(f"this LTspice build's sheet menu has no {label!r} command")
        entry.setdefault("commands", []).append(label)
        front = in_front(entry) or ""
        known = front in entry.get("with_results", [])
        results = results_of(front)
        opened = results in entry.get("panes", [])
        if label == VISIBLE_TRACES and known and not opened and not world.get("frame_ignores"):
            entry.setdefault("panes", []).insert(0, results)
            entry.setdefault("tied", []).append(results)
        self._keep(world)

    @staticmethod
    def _closed(entry: dict[str, Any], path: str) -> None:
        """Take ``path``'s pane out of a window: a sheet's goes with the plot opened from it."""
        closings = entry.setdefault("closings", {})
        closings[path] = closings.get(path, 0) + 1
        if path in entry.get("panes", []):
            if entry["panes"][0] == path:
                entry["plot_in_front"] = False
            entry["panes"].remove(path)
            if path in entry.get("tied", []):
                entry["tied"].remove(path)
            return
        del entry["designs"][path]
        for listed in ("with_results", "unsaved"):
            if path in entry.get(listed, []):
                entry[listed].remove(path)
        if entry.get("active") == path:
            entry["active"] = next(iter(entry["designs"]), None)
        if results_of(path) in entry.get("tied", []):
            FakeFrame._closed(entry, results_of(path))

    def close(self, pane: int) -> None:
        world = read_world(self._world)
        found = self._open_windows(world).get(pane)
        if found is None:
            return  # gone already: Windows takes a message for a window that has closed
        entry, path = found
        entry.setdefault("closed", []).append(path)
        if world.get("frame_keeps_panes"):
            pass  # a window that does not close it
        elif path in entry.get("unsaved", []):
            entry["asking"] = path
        else:
            self._closed(entry, path)
        self._keep(world)

    def is_open(self, window: int) -> bool:
        world = read_world(self._world)
        boxes = {
            self._window_of(entry, entry["asking"], "box")
            for entry in world["windows"]
            if entry.get("asking")
        }
        return window in boxes or window in self._open_windows(world)

    def takes_input(self, pid: int) -> bool:
        world = read_world(self._world)
        return not (world.get("box_up") or self._entry(world, pid).get("asking"))

    def save_question(self, pid: int) -> SaveQuestion | None:
        entry = self._entry(read_world(self._world), pid)
        asked_of = entry.get("asking")
        if asked_of is None:
            return None
        return SaveQuestion(self._window_of(entry, asked_of, "box"), asked_of)

    def answer(self, question: SaveQuestion, *, discard: bool) -> None:
        world = read_world(self._world)
        for entry in world["windows"]:
            if entry.get("asking") != question.path:
                continue
            entry.setdefault("answers", []).append("no" if discard else "cancel")
            del entry["asking"]
            if discard:
                self._closed(entry, question.path)
        self._keep(world)


def as_ltspice_reads(sheet: Path) -> str:
    return decode_windows_1252(sheet.read_bytes())


def digest(sheet: Path) -> str:
    return hashlib.sha256(sheet.read_bytes()).hexdigest()
