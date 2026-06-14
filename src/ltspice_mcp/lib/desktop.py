"""Non-blocking, best-effort local-desktop file opening.

``plot_waveform`` writes an interactive HTML chart and opens it on the user's
desktop. This module owns the detached opener so the chart core stays ignorant
of how the result is delivered.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.wsl import is_wsl, to_windows_path


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
