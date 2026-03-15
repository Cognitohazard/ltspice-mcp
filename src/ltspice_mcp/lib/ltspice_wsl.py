"""WSL-aware LTspice simulator subclass.

On WSL, spicelib's LTspice.run() prepends 'Z:' to paths (Wine convention),
but WSL runs the Windows LTspice.exe directly via interop — not through Wine.
This subclass overrides run() to use wslpath for proper path conversion.
"""

import subprocess
from pathlib import Path
from typing import Optional, Union

from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.lib.wsl import is_wsl, to_windows_path


class LTspiceWSL(LTspice):
    """LTspice simulator class that handles WSL path conversion correctly.

    On WSL, instead of using Wine's Z: drive mapping, converts paths using
    wslpath to get proper Windows paths (e.g., C:\\Users\\... or
    \\\\wsl.localhost\\...) that the Windows LTspice binary understands.
    """

    @classmethod
    def run(
        cls,
        netlist_file: Union[str, Path],
        cmd_line_switches: Optional[list] = None,
        timeout: Optional[float] = None,
        stdout=None,
        stderr=None,
        cwd: Union[str, Path, None] = None,
        exe_log: bool = False,
    ) -> int:
        if not is_wsl():
            # Not WSL — delegate to parent (handles native Windows and Wine)
            return super().run(
                netlist_file, cmd_line_switches, timeout, stdout, stderr, cwd, exe_log
            )

        # WSL path: convert netlist to Windows path for the LTspice process
        netlist_file = Path(netlist_file)
        win_path = to_windows_path(netlist_file)

        # Build command: run Windows LTspice.exe directly (no wine)
        if cmd_line_switches is None:
            cmd_line_switches = []

        # Use the exe path as-is (already set via create_from with Linux path)
        cmd_run = cls.spice_exe + ["-Run", "-b", win_path] + cmd_line_switches

        from spicelib.sim.simulator import run_function

        if exe_log:
            log_exe_file = netlist_file.with_suffix(".exe.log")
            with open(log_exe_file, "w") as outfile:
                error = run_function(
                    cmd_run,
                    timeout=timeout,
                    stdout=outfile,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                )
        else:
            error = run_function(
                cmd_run, timeout=timeout, stdout=stdout, stderr=stderr, cwd=cwd
            )
        return error
