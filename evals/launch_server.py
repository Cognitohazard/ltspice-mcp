"""Start one variant's ltspice-mcp server with the environment the harness sets.

Both client profiles start the server through this script, because in the
Claude Code profile the client, not the harness, decides what a stdio server
inherits. The script keeps only the variables the MCP SDK passes a server by
default, plus the temp, locale and WSL ones, and sets the trial's private
``LTSPICE_MCP_HOME``. Every other ``LTSPICE_MCP_*`` variable is dropped,
because each one overrides the TOML the variant is defined by; so are
``VIRTUAL_ENV`` and ``PYTHONPATH``, which would let the caller's environment
choose the code that runs.

Run it with ``python -I``. It uses the standard library only, and it runs the
server with the variant checkout's own Python, so nothing in the caller's
interpreter reaches the server::

    python -I launch_server.py --python <venv python> --cwd <workspace> --home <dir>

On POSIX the script replaces itself with the server, so the client's process
is the server's. Windows has no such replacement, so there it runs the server
as a child that shares its stdin and stdout, and exits with its exit code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping

#: What the MCP SDK's ``get_default_environment`` passes a stdio server
#: (``mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS``), copied so this script
#: imports nothing. ``tests/test_eval_harness.py`` checks the copy.
WINDOWS_INHERITED = (
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "USERNAME",
    "USERPROFILE",
)
POSIX_INHERITED = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")

#: Kept besides those: the temp directory (``TMP`` on Windows, ``TMPDIR``
#: elsewhere), the locale, and the variable the server reads to tell it runs
#: under WSL.
EXTRA = ("TMP", "TMPDIR", "LANG", "LC_ALL", "WSL_DISTRO_NAME")

HOME_VARIABLE = "LTSPICE_MCP_HOME"


def server_environment(source: Mapping[str, str], home: str, *, windows: bool) -> dict[str, str]:
    """The environment the server runs with, built from ``source``.

    A value that starts with ``()`` is a shell function exported into the
    environment, which the MCP SDK also refuses to pass on.
    """
    names = (*(WINDOWS_INHERITED if windows else POSIX_INHERITED), *EXTRA)
    env = {
        name: source[name]
        for name in names
        if name in source and not source[name].startswith("()")
    }
    env[HOME_VARIABLE] = home
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start one variant's ltspice-mcp server.")
    parser.add_argument("--python", required=True, help="the variant checkout's venv Python")
    parser.add_argument("--cwd", required=True, help="the trial workspace")
    parser.add_argument("--home", required=True, help="the trial's private LTSPICE_MCP_HOME")
    args = parser.parse_args(argv)

    windows = sys.platform == "win32"
    env = server_environment(os.environ, args.home, windows=windows)
    command = [args.python, "-m", "ltspice_mcp"]
    if windows:
        return subprocess.run(command, env=env, cwd=args.cwd, check=False).returncode
    os.chdir(args.cwd)
    os.execve(args.python, command, env)
    return 0  # not reached: execve replaces this process


if __name__ == "__main__":
    sys.exit(main())
