"""A stand-in for ``ltspice-mcp-bridge.exe`` that runs anywhere.

It speaks the bridge's stdio protocol for the calls the server makes, and
answers each the way LTspice 26.1.1 was recorded answering it
(``tests/fixtures/ltspice_bridge_recorded/ltspice26/conversation.json``;
``test_ltspice_bridge.py`` replays that recording against this program, so the
two cannot drift apart). The LTspice it stands in front of is a JSON file, the
world, named on its command line and read again for every request, so a test
changes what is running between two calls by rewriting it:

    {"windows": [{"pid": 1000, "version": "26.1.1", "designs": {"<path>": "<text>"},
                  "active": "<path>"}],
     "silent_on": "set_design_content"}

``active`` is the document in front in that window; left out, it is the last
one listed, which is the one LTspice had in front after opening them in turn.
A window opens a sheet that exists on disk, or one the world lists under
``files`` with the text it would then hold, and puts it in front. It opens a
results file that exists on disk, or one the world lists under ``results``;
each one opened is added to the window's ``shown``.

``silent_on`` names a tool this program stops answering at, for the tests of a
bridge that hangs. A replaced design is written back to the world. One tool is
not the bridge's: ``where_am_i`` answers with the Windows desktop this program
runs on, for the tests of where the server starts it.

    python tests/fake_ltspice_bridge.py <world.json>
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

PORT = 50000
NO_LTSPICE = (
    "Could not start or attach an LTspice instance. "
    "Open an LTspice window, or use start_headless/attach."
)


def own_desktop() -> str:
    """The name of the desktop this program's windows would open on."""
    assert sys.platform == "win32"
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32")
    kernel = ctypes.WinDLL("kernel32")
    user.GetThreadDesktop.restype = wintypes.HANDLE
    user.GetThreadDesktop.argtypes = [wintypes.DWORD]
    user.GetUserObjectInformationW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    name = ctypes.create_unicode_buffer(256)
    desktop = user.GetThreadDesktop(kernel.GetCurrentThreadId())
    user.GetUserObjectInformationW(desktop, 2, name, ctypes.sizeof(name), None)
    return name.value


def _kind(path: str) -> str:
    return "schematic" if path.lower().endswith(".asc") else "netlist"


class _Refused(Exception):
    """A tool result marked as an error, carrying its text."""


class _NoBackend(Exception):
    """A JSON-RPC error: the request needed an LTspice and there is none."""


class Bridge:
    def __init__(self, world: Path) -> None:
        self._world = world
        self._attached: int | None = None
        self._was_attached = False
        self._committed = False

    def _load(self) -> dict[str, Any]:
        return json.loads(self._world.read_text(encoding="utf-8"))

    def _windows(self) -> list[dict[str, Any]]:
        return self._load().get("windows", [])

    def _window(self) -> dict[str, Any]:
        """The window this session is attached to, or the newest if it has none yet."""
        windows = self._windows()
        if self._attached is None and not self._was_attached and windows:
            self._attached = windows[-1]["pid"]
            self._was_attached = True
        for window in windows:
            if window["pid"] == self._attached:
                return window
        self._attached = None
        raise _NoBackend

    def _status(self) -> dict[str, Any]:
        try:
            self._window()
        except _NoBackend:
            state = "detached" if self._was_attached else "background"
        else:
            state = "gui-attached"
        return {
            "current": {
                "backendPid": self._attached or 0,
                "committed": self._committed,
                "state": state,
            },
            "instances": [
                {"mode": "gui", "pid": w["pid"], "port": PORT, "version": w["version"]}
                for w in self._windows()
            ],
        }

    def _attach(self, pid: int | None) -> dict[str, Any]:
        if not any(window["pid"] == pid for window in self._windows()):
            raise _Refused(
                json.dumps({"error": "no live LTspice instance with that pid", "ok": False})
            )
        self._attached = pid
        self._was_attached = True
        return {"attachedPid": pid, "ok": True, "port": PORT}

    def _design(self, path: str) -> tuple[dict[str, Any], str]:
        window = self._window()
        if path not in window["designs"]:
            raise _Refused("document not found")
        return window, window["designs"][path]

    def _in_window(self, change: Any) -> None:
        """Apply ``change`` to this session's window in the world, and keep it."""
        window = self._window()
        world = self._load()
        for entry in world["windows"]:
            if entry["pid"] == window["pid"]:
                change(entry, world)
        self._world.write_text(json.dumps(world), encoding="utf-8")

    def _open(self, path: str) -> dict[str, Any]:
        """Open a document: one the world lists under ``files``, or one on disk."""
        already = path in self._window()["designs"]

        def load(entry: dict[str, Any], world: dict[str, Any]) -> None:
            if path in world.get("files", {}):
                entry["designs"][path] = world["files"][path]
            elif Path(path).is_file():
                entry["designs"][path] = Path(path).read_bytes().decode("cp1252", "replace")
            else:
                raise _Refused("file not found")
            entry["active"] = path
            if Path(path).with_suffix(".raw").is_file():
                entry.setdefault("with_results", []).append(path)

        if not already:
            self._in_window(load)
        return {
            "already_open": "true" if already else "false",
            "path": path,
            "status": "ok",
            "type": _kind(path),
        }

    def _bring_to_front(self, path: str | None) -> None:
        def front(entry: dict[str, Any], _world: dict[str, Any]) -> None:
            if path in entry["designs"]:
                entry["active"] = path

        self._in_window(front)

    def _show_results(self, path: str) -> str:
        def show(entry: dict[str, Any], world: dict[str, Any]) -> None:
            if path not in world.get("results", []) and not Path(path).is_file():
                raise _Refused("file not found")
            entry.setdefault("shown", []).append(path)

        self._in_window(show)
        return path

    def _replace(self, path: str, text: str) -> dict[str, Any]:
        _window, held = self._design(path)
        if held == text:
            return {"status": "ok", "unchanged": "true"}

        def replace(entry: dict[str, Any], _world: dict[str, Any]) -> None:
            entry["designs"][path] = text

        self._in_window(replace)
        self._committed = True
        return {"message": "", "status": "ok", "unchanged": "false"}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._load().get("silent_on") == name:
            # Never answers, as a bridge that has stopped: the caller's deadline ends it.
            threading.Event().wait()
        if name == "where_am_i":
            return {"desktop": own_desktop()}
        if name == "status":
            return self._status()
        if name == "attach":
            return self._attach(arguments.get("pid"))
        if name == "list_open_designs":
            return {"paths": "\n".join(self._window()["designs"])}
        if name == "get_active_design_path":
            window = self._window()
            if not window["designs"]:
                # A window with no document has none in front: the recording
                # has that, and not the bridge's wording, which the client
                # does not read (a refusal and an empty path are alike to it).
                raise _Refused("document not found")
            in_front = window.get("active") or list(window["designs"])[-1]
            return {"path": in_front, "type": _kind(in_front)}
        if name == "get_raw_info":
            return {"path": self._show_results(arguments["path"])}
        if name == "open_design":
            return self._open(arguments["path"])
        if name == "bring_to_front":
            self._bring_to_front(arguments.get("path"))
            return {"status": "ok"}
        if name == "get_design_content":
            return {"path": arguments["path"], "text": self._design(arguments["path"])[1]}
        if name == "set_design_content":
            return self._replace(arguments["path"], arguments["text"])
        raise _Refused(f"unknown tool {name}")


def main() -> int:
    bridge = Bridge(Path(sys.argv[1]))
    for line in sys.stdin.buffer:
        request = json.loads(line)
        if "id" not in request:
            continue
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"]}
        if request["method"] == "initialize":
            reply["result"] = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ltspice-bridge", "version": "1.0"},
            }
        else:
            params = request.get("params", {})
            try:
                value = bridge.call(params.get("name", ""), params.get("arguments", {}))
                reply["result"] = {"content": [{"type": "text", "text": json.dumps(value)}]}
            except _Refused as refusal:
                reply["result"] = {
                    "content": [{"type": "text", "text": str(refusal)}],
                    "isError": True,
                }
            except _NoBackend:
                reply["error"] = {"code": -32001, "message": NO_LTSPICE}
        sys.stdout.buffer.write(json.dumps(reply).encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
