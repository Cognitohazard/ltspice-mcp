"""Plot delivery: client classification + non-blocking local-desktop opening.

``plot_waveform`` renders an interactive HTML chart and delivers it one of two
ways: as an in-chat ``ui://`` widget for hosts that advertise MCP Apps support
(SEP-1865), or by opening the HTML on the user's desktop for everyone else. This
module owns the pure client classifier and the detached opener so the chart core
stays ignorant of how the result is delivered.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from ltspice_mcp.lib.wsl import is_wsl, to_windows_path

DeliveryChannel = Literal["terminal", "ui"]

# SEP-1865 (MCP Apps) capability identifier. A client advertises it under
# ``capabilities.extensions[...]`` at initialize to signal it can render ui://
# HTML widgets in-chat. (Final/Stable since 2026-01-26.)
_UI_EXTENSION = "io.modelcontextprotocol/ui"

# Chromium-family executables for an "app mode" (chromeless pop-up) window.
# On WSL we address the Windows-side exes by their /mnt path; elsewhere we look
# them up on PATH. Edge first (always present on Windows), then Chrome.
_WSL_CHROMIUM = (
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
)
_CHROMIUM_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
    "chrome",
    "msedge",
)
_APP_WINDOW_SIZE = "--window-size=1100,860"


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


def _chromium_exes() -> list[str]:
    """Chromium-family executables available for an app-mode window, best first.

    Factored out so tests can force it empty (exercise the default-opener
    fallback) or fake (exercise the app-window argv) without a real browser.
    """
    if is_wsl():
        return [e for e in _WSL_CHROMIUM if Path(e).exists()]
    return [e for n in _CHROMIUM_NAMES if (e := shutil.which(n))]


def _file_url(path: Path) -> str:
    """A browser-openable ``file://`` URL for ``path``.

    On WSL, Windows browsers reach the file by its ``wslpath -w`` Windows path,
    which is either a drive path (``C:\\...`` for a /mnt/c-backed workspace) or a
    ``\\wsl.localhost`` UNC (Linux FS). The two need different URI shapes — a
    drive path is ``file:///C:/...`` (three slashes), a UNC is
    ``file:////wsl.localhost/...`` (the leading ``//`` of the UNC supplies two of
    them). Spaces/special chars are percent-encoded so a bad URI can't be handed
    to the browser (and silently reported as opened). Off WSL, ``Path.as_uri()``.
    """
    if not is_wsl():
        return path.as_uri()
    win = to_windows_path(path).replace("\\", "/")
    if win.startswith("//"):  # UNC: //wsl.localhost/... -> file:////wsl.localhost/...
        return "file://" + quote(win, safe="/:")
    return "file:///" + quote(win, safe="/:")  # drive: C:/... -> file:///C:/...


def _open_app_window(path: Path, spawn: Any, quiet: dict[str, Any]) -> str | None:
    """Open ``path`` in a chromeless Chromium app window; ``None`` if unavailable.

    Tries each discovered Chromium exe in turn; returns the exe name on the first
    spawn that doesn't raise, else ``None`` so the caller falls back to the OS
    default opener (a normal browser tab) — never worse than before.
    """
    url = _file_url(path)
    for exe in _chromium_exes():
        try:
            spawn([exe, f"--app={url}", _APP_WINDOW_SIZE], **quiet)
            return Path(exe).name
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def open_in_desktop(path: Path, *, spawn: Any = subprocess.Popen) -> tuple[bool, str | None]:
    """Open ``path`` on the desktop — non-blocking and best-effort.

    Prefers a chromeless Chromium "app" window (a real plot pop-up, no browser
    chrome); on any failure falls back to the OS default opener (a normal tab),
    so this is never worse than a plain open. Spawns a detached child and returns
    immediately, never waiting (a browser may outlive the request). Returns
    ``(opened, method)``; on total failure (headless box, no opener) returns
    ``(False, None)`` rather than raising, so the caller still has the file path.

    ``spawn`` is injectable for tests (defaults to :class:`subprocess.Popen`) so
    the platform branch and argv can be asserted without launching anything.
    """
    quiet: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    app_method = _open_app_window(path, spawn, quiet)
    if app_method is not None:
        return True, app_method
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
