"""LTspice on native Windows, started where it cannot take the keyboard focus.

spicelib starts LTspice with a plain ``subprocess.run``, and LTspice opens a
window for every batch run and every export and holds the foreground for most
of it (``lib/hidden_desktop.py`` has the measurements). This subclass starts
the same command lines on the process's hidden desktop instead.

Only the launch differs. The commands are spicelib's, the exit code and the
timeout behave as ``subprocess.run`` does, and where there is no hidden
desktop (the ``hidden_desktop`` setting is off, or Windows refused one) each
method is spicelib's own.

The class keeps the name ``LTspice``: a job records its simulator by class
name, and the raw dialect and the linter key on it.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any

from spicelib.simulators.ltspice_simulator import LTspice as _SpicelibLTspice

from ltspice_mcp.lib import hidden_desktop

SEE_THE_BOX = (
    "To see LTspice's window and answer the box yourself, set [simulator] "
    "hidden_desktop = false (or LTSPICE_MCP_HIDDEN_DESKTOP=0) and restart"
)


def _switches(cmd_line_switches: list | str | None) -> list:
    if cmd_line_switches is None:
        return []
    return [cmd_line_switches] if isinstance(cmd_line_switches, str) else list(cmd_line_switches)


class LTspice(_SpicelibLTspice):
    """spicelib's LTspice with its windows kept off the user's desktop."""

    @classmethod
    def _off_desktop(cls, stdout: Any, stderr: Any) -> hidden_desktop.HiddenDesktop | None:
        """The desktop to launch on, or None when spicelib's own launch applies.

        A caller that hands over its own streams gets spicelib's launch, which
        takes every form ``subprocess`` does; no caller in this package does.
        A missing executable is left to spicelib too, for its error.
        """
        if stdout is not None or stderr is not None or not cls.is_available():
            return None
        return hidden_desktop.shared()

    @classmethod
    def _launch(
        cls,
        desktop: hidden_desktop.HiddenDesktop,
        mode: list[str],
        subject: Path,
        cmd_line_switches: list | str | None,
        *,
        timeout: float | None,
        cwd: str | Path | None,
        exe_log: bool,
    ) -> int:
        """Start LTspice on ``subject`` in ``mode`` and return its exit code."""
        command = [*cls.spice_exe, *mode, subject.as_posix(), *_switches(cmd_line_switches)]
        with contextlib.ExitStack() as opened:
            # The console log spicelib keeps beside the input. LTspice writes
            # nothing to it; it is connected all the same, as spicelib connects it.
            console = (
                opened.enter_context(open(subject.with_suffix(".exe.log"), "wb"))
                if exe_log
                else None
            )
            return desktop.run(
                command,
                timeout=timeout,
                cwd=cwd,
                stdout=console,
                stderr=subprocess.STDOUT,
                remedy=SEE_THE_BOX,
            )

    @classmethod
    def run(
        cls,
        netlist_file: str | Path,
        cmd_line_switches: list | None = None,
        timeout: float | None = None,
        stdout=None,
        stderr=None,
        cwd: str | Path | None = None,
        exe_log: bool = False,
    ) -> int:
        """Run ``netlist_file`` in batch mode and return LTspice's exit code.

        Raises ``subprocess.TimeoutExpired`` past ``timeout``, having ended
        LTspice, and ``hidden_desktop.DialogError`` when LTspice stopped on a
        message box, with what the box said.
        """
        desktop = cls._off_desktop(stdout, stderr)
        if desktop is None:
            return super().run(
                netlist_file, cmd_line_switches, timeout, stdout, stderr, cwd, exe_log
            )
        return cls._launch(
            desktop,
            ["-Run", "-b"],
            Path(netlist_file),
            cmd_line_switches,
            timeout=timeout,
            cwd=cwd,
            exe_log=exe_log,
        )

    @classmethod
    def create_netlist(
        cls,
        circuit_file: str | Path,
        cmd_line_switches: list | None = None,
        timeout: float | None = None,
        stdout=None,
        stderr=None,
        cwd: str | Path | None = None,
        exe_log: bool = False,
    ) -> Path:
        """Export ``circuit_file`` to the ``.net`` beside it and return that path.

        Raises ``RuntimeError`` when LTspice exits without writing it, as
        spicelib does, and what ``run`` raises for a timeout or a message box.
        """
        desktop = cls._off_desktop(stdout, stderr)
        if desktop is None:
            return super().create_netlist(
                circuit_file, cmd_line_switches, timeout, stdout, stderr, cwd, exe_log
            )
        sheet = Path(circuit_file)
        code = cls._launch(
            desktop,
            ["-netlist"],
            sheet,
            cmd_line_switches,
            timeout=timeout,
            cwd=cwd,
            exe_log=exe_log,
        )
        netlist = sheet.with_suffix(".net")
        if code == 0 and netlist.exists():
            return netlist
        raise RuntimeError("Failed to create netlist")
