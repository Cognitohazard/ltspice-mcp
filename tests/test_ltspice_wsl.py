"""Tests for LTspiceWSL subclass."""

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
            assert mock_run.called
