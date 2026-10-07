"""Reaching an LTspice someone has open, through the bridge LTspice ships.

From 26.1 LTspice carries an MCP server of its own, and every LTspice window
listens for it on the loopback interface. The supported way in is a small
program installed beside ``LTspice.exe``, ``ltspice-mcp-bridge.exe``: a stdio
MCP server that finds the running instances and forwards to one. This module
is a client for that program and nothing more. It is the only route to what a
window holds in memory, which is not what the file holds: LTspice never reads
a sheet again once it is open, and a sheet replaced through the bridge is not
written to disk.

Three things about the bridge shape this client, each observed on 26.1.1 and
kept with the suite's recordings of the bridge (``docs/TESTING.md``, "An open
window and the bridge"):

- **It starts an LTspice of its own when it has none to talk to**, on the
  first request that needs one, even a read, and even when it was told which
  instance to use and that instance has gone. Two things stand between that
  and whoever is at the machine. The bridge is started on the server's hidden
  desktop, in a job that ends with the session: whatever it launches has its
  windows there, where they cannot take the keyboard focus, and does not
  outlive the session. That holds whatever the bridge does. It is also started
  with ``--ltspice-path`` naming a file that does not exist, which makes the
  launch itself fail, so that as a rule nothing is started at all; that is a
  failure the bridge reports and not a mode it offers (it still describes
  itself as able to launch), and it is all there is where there is no desktop
  to start it on.
- **Its session tools answer without an instance.** ``status`` lists the
  running instances and ``attach`` binds to one by process id, so a session
  here asks which windows exist and names the one it means, and never relies
  on the bridge's choice.
- **An LTspice-backed tool answers in JSON text whose values are strings**
  (``"unchanged": "true"``), and a failure is either a result marked as an
  error or a JSON-RPC error. Both arrive here as ``BridgeError``.

The bridge is third-party code on a path that can hold an edit lock, so a
session is bounded as a whole: past its deadline the process is ended, which
is also what unblocks a write it stopped reading. The calls block; a caller on
the event loop makes them through ``asyncio.to_thread``.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol, Self

from ltspice_mcp.lib import hidden_desktop

BRIDGE_NAME = "ltspice-mcp-bridge.exe"

DEFAULT_TIMEOUT_S = 5.0
"""How long one session may take in all. A session that finds a window is a
handful of loopback calls, each a few milliseconds; this is a cap on a bridge
that has stopped answering, never an estimate."""

_PROTOCOL_VERSION = "2025-03-26"
_CREATE_NO_WINDOW = 0x08000000
# What the bridge reads when the matching flag is absent. An instance is named
# here by attaching, and where LTspice is said to be by --ltspice-path, so none
# of these may reach it from this process's environment.
_BRIDGE_VARIABLES = frozenset({"LTSPICE_MCP_PID", "LTSPICE_MCP_PORT", "LTSPICE_INSTALL_DIR"})
_ATTACHED_TO_A_WINDOW = "gui-attached"


class BridgeError(RuntimeError):
    """The bridge could not be started, stopped answering, or refused a request."""


@dataclass(frozen=True)
class Instance:
    """A running LTspice the bridge found.

    ``mode`` is ``"gui"`` for a window someone has open and ``"headless"`` for
    an instance some bridge started for itself.
    """

    pid: int
    mode: str
    version: str


def bridge_command(executable: str | os.PathLike[str]) -> list[str] | None:
    """The command that starts the bridge installed beside ``executable``.

    None when that build has none (before 26.1, or an install without it). The
    command names an LTspice that does not exist, so a bridge started with it
    can attach to a running instance and fails when it tries to launch one.
    """
    bridge = Path(executable).with_name(BRIDGE_NAME)
    if not bridge.is_file():
        return None
    nowhere = bridge.with_name("launch-disabled-by-ltspice-mcp") / "LTspice.exe"
    return [str(bridge), "--ltspice-path", str(nowhere)]


class _Bridge(Protocol):
    """What a session needs of the process it started."""

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


def _nothing() -> None:
    return None


def _start_hidden(
    desktop: hidden_desktop.HiddenDesktop, command: Sequence[str], environment: Mapping[str, str]
) -> tuple[_Bridge, IO[bytes], IO[bytes], Callable[[], None]]:
    """Start the bridge where an LTspice it launches cannot be seen or left behind.

    A program started on a desktop takes whatever it starts there with it, and
    the launch puts it in a job that ends when the returned release is called.
    """
    its_input, ours_to_write = os.pipe()
    ours_to_read, its_output = os.pipe()
    try:
        with open(os.devnull, "wb") as discarded:
            process = desktop.start(
                command, env=environment, stdin=its_input, stdout=its_output, stderr=discarded
            )
    except BaseException:
        os.close(ours_to_write)
        os.close(ours_to_read)
        raise
    finally:
        # The program holds its own copies; these would keep its output open
        # past its end, and the reader waiting.
        os.close(its_input)
        os.close(its_output)
    return (
        process,
        os.fdopen(ours_to_write, "wb", buffering=0),
        os.fdopen(ours_to_read, "rb"),
        process.close,
    )


def _start_plain(
    command: Sequence[str], environment: Mapping[str, str]
) -> tuple[_Bridge, IO[bytes], IO[bytes], Callable[[], None]]:
    """Start the bridge on the caller's own desktop: off Windows, and where
    Windows gave no other or the setting turned it off."""
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=dict(environment),
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    release = _nothing
    if sys.platform == "win32":
        from ltspice_mcp.lib.windows_job import WindowsJob

        # Without a job the bridge is still ended by close and by the
        # deadline; only what it started could then outlive the session.
        with contextlib.suppress(OSError):
            release = WindowsJob(process.pid, allow_breakaway=False).close
    return process, process.stdin, process.stdout, release


class BridgeSession:
    """One conversation with the bridge, ended by ``close`` or by its deadline.

    On Windows the bridge runs on the server's hidden desktop
    (``hidden_desktop.shared``), in a job this session ends: see the module
    docstring for why that matters for a program that can start LTspice.
    """

    def __init__(self, command: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._name = Path(command[0]).name
        self._deadline = time.monotonic() + timeout
        self._timeout = timeout
        self._expired = False
        self._requests = 0
        self._ended = False
        self._ending = threading.Lock()
        self._replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.upper() not in _BRIDGE_VARIABLES
        }
        desktop = hidden_desktop.shared()
        try:
            if desktop is not None:
                started = _start_hidden(desktop, command, environment)
            else:
                started = _start_plain(command, environment)
        except OSError as error:
            raise BridgeError(f"{self._name} could not be started: {error}") from error
        self._process, self._stdin, self._stdout, self._release = started
        self._watchdog = threading.Timer(timeout, self._expire)
        self._watchdog.daemon = True
        self._watchdog.start()
        threading.Thread(target=self._read, name="ltspice-bridge-replies", daemon=True).start()
        try:
            self._request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ltspice-mcp", "version": "1"},
                },
            )
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # The wire
    # ------------------------------------------------------------------

    def _expire(self) -> None:
        self._expired = True
        self._end()

    def _end(self) -> None:
        """End the bridge and whatever it started. Once, from either thread."""
        with self._ending:
            if self._ended:
                return
            self._ended = True
            with contextlib.suppress(OSError):
                self._process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                self._process.wait(timeout=2.0)
            with contextlib.suppress(OSError):
                self._release()

    def _read(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            for line in self._stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._replies.put(message)
        self._replies.put(None)

    def _gave_up(self) -> BridgeError:
        return BridgeError(f"{self._name} did not answer within {self._timeout:g} s and was ended")

    def _send(self, message: dict[str, Any]) -> None:
        try:
            self._stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            self._stdin.flush()
        except (OSError, ValueError) as error:
            if self._expired:
                raise self._gave_up() from error
            raise BridgeError(f"{self._name} stopped reading: {error}") from error

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._requests += 1
        wanted = self._requests
        self._send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params})
        while True:
            # The watchdog ends the bridge at the deadline, which closes its
            # output and so ends this wait; the second is for an end that
            # Windows refused.
            remaining = self._deadline - time.monotonic() + 1.0
            try:
                reply = self._replies.get(timeout=max(remaining, 0.0))
            except queue.Empty:
                raise self._gave_up() from None
            if reply is None:
                # Left for whoever asks next: the output does not close twice.
                self._replies.put(None)
                if self._expired:
                    raise self._gave_up()
                raise BridgeError(f"{self._name} closed before answering {method}")
            if reply.get("id") != wanted or "method" in reply:
                continue
            error = reply.get("error")
            if error is not None:
                message = error.get("message") if isinstance(error, dict) else error
                raise BridgeError(str(message))
            result = reply.get("result")
            return result if isinstance(result, dict) else {}

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """One of the bridge's tools by name; its reply as the object it sent."""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content")
        text = "\n".join(
            str(part.get("text", ""))
            for part in (content if isinstance(content, list) else [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if result.get("isError"):
            said = value.get("error") if isinstance(value, dict) else value
            raise BridgeError(str(said if said else text or f"{name} failed"))
        if not isinstance(value, dict):
            raise BridgeError(f"{name} answered with something other than a JSON object")
        return value

    # ------------------------------------------------------------------
    # What a caller asks
    # ------------------------------------------------------------------

    def instances(self) -> list[Instance]:
        """Every running LTspice the bridge can see. Starts nothing."""
        found = self.call("status").get("instances")
        return [
            Instance(
                pid=int(row["pid"]), mode=str(row.get("mode")), version=str(row.get("version"))
            )
            for row in (found if isinstance(found, list) else [])
            if isinstance(row, dict) and "pid" in row
        ]

    def attach(self, pid: int) -> None:
        """Bind this session to the LTspice window with process id ``pid``.

        Raises unless the bridge then reports itself attached to that window:
        a session bound to anything else must not be read from or written to.
        """
        self.call("attach", pid=pid)
        current = self.call("status").get("current")
        if (
            not isinstance(current, dict)
            or current.get("backendPid") != pid
            or current.get("state") != _ATTACHED_TO_A_WINDOW
        ):
            raise BridgeError(f"the bridge did not attach to LTspice process {pid}")

    def open_designs(self) -> list[str]:
        """The path of every document open in the attached window, as LTspice spells it."""
        paths = self.call("list_open_designs").get("paths")
        return [line for line in str(paths or "").split("\n") if line.strip()]

    def active_design(self) -> str | None:
        """The path of the document in front in the attached window, or None.

        None when the window has no document, which the bridge reports as a
        refusal like any other.
        """
        try:
            path = self.call("get_active_design_path").get("path")
        except BridgeError:
            return None
        return path if isinstance(path, str) and path.strip() else None

    def design_text(self, path: str) -> str:
        """The open document at ``path`` as the window holds it, saved or not."""
        text = self.call("get_design_content", path=path).get("text")
        if not isinstance(text, str):
            raise BridgeError(f"LTspice returned no text for {path}")
        return text

    def replace_design_text(self, path: str, text: str) -> bool:
        """Replace the open document at ``path``; False when it already read so.

        The window changes and the file does not. LTspice records the change
        as one step of that document's undo history.
        """
        reply = self.call("set_design_content", path=path, text=text)
        if reply.get("status") != "ok":
            raise BridgeError(str(reply.get("message") or f"LTspice did not take {path}"))
        return str(reply.get("unchanged")).lower() != "true"

    def close(self) -> None:
        """End the bridge. It leaves the LTspice window it was attached to running."""
        self._watchdog.cancel()
        with contextlib.suppress(OSError, ValueError):
            self._stdin.close()
        # Its input closed, the bridge takes its leave of LTspice and exits.
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self._process.wait(timeout=2.0)
        self._end()
        with contextlib.suppress(OSError, ValueError):
            self._stdout.close()
