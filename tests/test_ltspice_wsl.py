"""Tests for LTspiceWSL subclass."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from ltspice_mcp.lib.ltspice_wsl import LTspiceWSL


class TestLTspiceWSLRun:
    def test_not_wsl_delegates_to_parent(self, tmp_path: Path):
        netlist = tmp_path / "x.cir"
        netlist.write_text("R1 1 0 1k\n.END\n")
        with (
            patch("ltspice_mcp.lib.ltspice_wsl.is_wsl", return_value=False),
            patch(
                "spicelib.simulators.ltspice_simulator.LTspice.run",
                return_value=0,
            ) as mock_parent,
        ):
            result = LTspiceWSL.run(netlist)
            assert result == 0
            mock_parent.assert_called_once()

    def test_wsl_path_converts_and_runs(self, tmp_path: Path):
        netlist = tmp_path / "x.cir"
        netlist.write_text("R1 1 0 1k\n.END\n")
        with (
            patch("ltspice_mcp.lib.ltspice_wsl.is_wsl", return_value=True),
            patch(
                "ltspice_mcp.lib.ltspice_wsl.to_windows_path",
                return_value="C:\\tmp\\x.cir",
            ),
            patch("spicelib.sim.simulator.run_function", return_value=0) as mock_run,
            patch.object(LTspiceWSL, "spice_exe", ["/fake/exe"]),
        ):
            result = LTspiceWSL.run(netlist)
            assert result == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args.args[0]
            # Exe first, batch-mode switches, then the wslpath-converted
            # Windows path — never the raw Linux path.
            assert cmd == ["/fake/exe", "-Run", "-b", "C:\\tmp\\x.cir"]
            assert str(netlist) not in cmd

    def test_wsl_appends_extra_switches_after_netlist(self, tmp_path: Path):
        netlist = tmp_path / "x.cir"
        netlist.write_text("R1 1 0 1k\n.END\n")
        with (
            patch("ltspice_mcp.lib.ltspice_wsl.is_wsl", return_value=True),
            patch(
                "ltspice_mcp.lib.ltspice_wsl.to_windows_path",
                return_value="C:\\tmp\\x.cir",
            ),
            patch("spicelib.sim.simulator.run_function", return_value=0) as mock_run,
            patch.object(LTspiceWSL, "spice_exe", ["/fake/exe"]),
        ):
            result = LTspiceWSL.run(netlist, cmd_line_switches=["-ascii"], timeout=42.0)
            assert result == 0
            cmd = mock_run.call_args.args[0]
            assert cmd == ["/fake/exe", "-Run", "-b", "C:\\tmp\\x.cir", "-ascii"]
            assert mock_run.call_args.kwargs["timeout"] == 42.0

    def test_wsl_with_exe_log(self, tmp_path: Path):
        netlist = tmp_path / "x.cir"
        netlist.write_text("R1 1 0 1k\n.END\n")
        with (
            patch("ltspice_mcp.lib.ltspice_wsl.is_wsl", return_value=True),
            patch(
                "ltspice_mcp.lib.ltspice_wsl.to_windows_path",
                return_value="C:\\tmp\\x.cir",
            ),
            patch("spicelib.sim.simulator.run_function", return_value=0) as mock_run,
            patch.object(LTspiceWSL, "spice_exe", ["/fake/exe"]),
        ):
            result = LTspiceWSL.run(netlist, exe_log=True)
            assert result == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args.args[0]
            assert cmd == ["/fake/exe", "-Run", "-b", "C:\\tmp\\x.cir"]
            # Simulator stdout must be redirected into <netlist>.exe.log,
            # with stderr merged into the same stream.
            kwargs = mock_run.call_args.kwargs
            assert Path(kwargs["stdout"].name) == netlist.with_suffix(".exe.log")
            assert kwargs["stderr"] == subprocess.STDOUT
            assert netlist.with_suffix(".exe.log").exists()
