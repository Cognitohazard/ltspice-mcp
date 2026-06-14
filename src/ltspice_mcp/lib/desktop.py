"""Plot delivery: client classification + non-blocking local-desktop opening.

``plot_waveform`` renders an interactive HTML chart and delivers it one of two
ways: as an in-chat ``ui://`` widget for hosts that advertise MCP Apps support
(SEP-1865), or by opening the HTML on the user's desktop for everyone else. This
module owns the pure client classifier and the detached opener so the chart core
stays ignorant of how the result is delivered.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.wsl import is_wsl, to_windows_path

DeliveryChannel = Literal["terminal", "ui"]

# SEP-1865 (MCP Apps) capability identifier. A client advertises it under
# ``capabilities.extensions[...]`` at initialize to signal it can render ui://
# HTML widgets in-chat. (Final/Stable since 2026-01-26.)
_UI_EXTENSION = "io.modelcontextprotocol/ui"


def client_supports_ui(caps: Any | None) -> bool:
    """Whether the client advertised MCP Apps (``ui://`` widget) support.

    The capability lives at ``capabilities.extensions["io.modelcontextprotocol/ui"]``.
    The mcp SDK's ``ClientCapabilities`` has no typed ``extensions`` field, so it
    arrives as an extra attribute (the model is ``extra="allow"``) — read it
    defensively and default to unsupported.
    """
    if caps is None:
        return False
    extensions = getattr(caps, "extensions", None)
    return isinstance(extensions, dict) and _UI_EXTENSION in extensions


def resolve_delivery_channel(caps: Any | None) -> DeliveryChannel:
    """Pick the plot delivery channel for the connected client.

    ``"ui"`` (embed a ``ui://`` widget the host renders in-chat) when the client
    advertises MCP Apps support, else ``"terminal"`` (write the HTML and open it
    locally). The safe default is ``"terminal"``: a local server can always open a
    window, and a non-UI host gets the universally-working path.
    """
    return "ui" if client_supports_ui(caps) else "terminal"


def open_in_desktop(path: Path, *, spawn: Any = subprocess.Popen) -> tuple[bool, str | None]:
    """Open ``path`` with the OS default app — non-blocking and best-effort.

    Spawns a detached child and returns immediately, never waiting on the opener
    (a browser may outlive the request). Returns ``(opened, method)``; on any
    failure (headless box, missing opener) returns ``(False, None)`` rather than
    raising, so the caller still has the written file path to report.

    ``spawn`` is injectable for tests (defaults to :class:`subprocess.Popen`) so
    the platform branch and argv can be asserted without launching anything.
    """
    quiet: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    try:
        if is_wsl():
            # explorer.exe opens a wslpath-converted path (incl. a \\wsl.localhost
            # UNC) for reading; it returns exit 1 even on success, so the result is
            # never inspected — fire and forget. (wslview is not installed.)
            spawn(["explorer.exe", to_windows_path(path)], **quiet)
            return True, "explorer.exe"
        if sys.platform == "darwin":
            spawn(["open", str(path)], **quiet)
            return True, "open"
        if sys.platform == "win32":
            startfile = getattr(os, "startfile", None)
            if startfile is None:  # pragma: no cover - platform guard
                return False, None
            startfile(str(path))
            return True, "startfile"
        spawn(["xdg-open", str(path)], **quiet)
        return True, "xdg-open"
    except (OSError, subprocess.SubprocessError):
        return False, None
