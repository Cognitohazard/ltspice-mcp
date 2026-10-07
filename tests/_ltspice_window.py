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
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.encoding import decode_windows_1252
from ltspice_mcp.lib.ltspice_frame import VISIBLE_TRACES, FrameError, LtspiceFrame
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from tests.conftest import LIVENESS_S

FAKE = Path(__file__).with_name("fake_ltspice_bridge.py")
PID = 4242
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
    **extra: Any,
) -> None:
    """Write ``windows`` into ``world`` and have ``state`` reach them through the stand-in."""
    write_world(world, windows, **extra)
    state.open_windows = OpenWindows(fake_command(world), timeout=timeout, frame=FakeFrame(world))


def _name(path: str) -> str:
    """A file's name from a path in either spelling, on any platform."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


class FakeFrame(LtspiceFrame):
    """The frame of a stand-in window, kept in the same world file.

    A window's entry lists the results files it has open under ``panes`` and
    the commands it was sent under ``commands``. Visible Traces opens the
    results of the sheet in front where the window knows of any, as LTspice
    was recorded doing: the sheets under ``with_results``, which the stand-in
    bridge adds a sheet to when it opens one that has results beside it. A
    world with ``frame_ignores`` is a window that does nothing with a command,
    and one with ``frame_has_no_command`` a build whose menu lacks it.
    """

    def __init__(self, world: Path) -> None:
        super().__init__(lambda _pid: [])
        self._world = world

    def _entry(self, world: dict[str, Any], pid: int) -> dict[str, Any]:
        for entry in world["windows"]:
            if entry["pid"] == pid:
                return entry
        raise FrameError(f"LTspice process {pid} has no window on this desktop")

    def panes(self, pid: int) -> list[str]:
        entry = self._entry(read_world(self._world), pid)
        return sorted({_name(path) for path in entry["designs"]} | set(entry.get("panes", [])))

    def send(self, pid: int, label: str) -> None:
        world = read_world(self._world)
        entry = self._entry(world, pid)
        if world.get("frame_has_no_command"):
            raise FrameError(f"this LTspice build's sheet menu has no {label!r} command")
        entry.setdefault("commands", []).append(label)
        in_front = str(entry.get("active") or "")
        known = in_front in entry.get("with_results", [])
        if label == VISIBLE_TRACES and known and not world.get("frame_ignores"):
            results = _name(in_front).rsplit(".", 1)[0] + ".raw"
            entry.setdefault("panes", []).append(results)
        self._world.write_text(json.dumps(world), encoding="utf-8")


def as_ltspice_reads(sheet: Path) -> str:
    return decode_windows_1252(sheet.read_bytes())


def digest(sheet: Path) -> str:
    return hashlib.sha256(sheet.read_bytes()).hexdigest()
