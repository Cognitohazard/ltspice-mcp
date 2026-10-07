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
    state.open_windows = OpenWindows(fake_command(world), timeout=timeout)


def as_ltspice_reads(sheet: Path) -> str:
    return decode_windows_1252(sheet.read_bytes())


def digest(sheet: Path) -> str:
    return hashlib.sha256(sheet.read_bytes()).hexdigest()
