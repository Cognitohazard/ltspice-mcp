"""LTspice started off the user's desktop (``lib/ltspice_windows.py``).

A stand-in program takes LTspice's two command lines and writes what LTspice
writes, saying which desktop it ran on. It is started through the class a
Windows server launches LTspice with, so the launch itself is the real one;
only the simulator is not. The tests against a real LTspice are in
``test_ltspice_integration.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from spicelib.simulators import ltspice_simulator
from spicelib.simulators.ltspice_simulator import LTspice as SpicelibLTspice

from ltspice_mcp import engine
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import hidden_desktop
from ltspice_mcp.lib.hidden_desktop import DialogError
from ltspice_mcp.lib.ltspice_windows import LTspice
from ltspice_mcp.lib.simulator import (
    SIMULATORS,
    bind_named_executable,
    detect_simulators,
    simulator_dialect,
    simulator_family,
)
from ltspice_mcp.lib.windows_job import python_launch
from tests.conftest import LIVENESS_S
from tests.test_hidden_desktop import OWN_DESKTOP_SOURCE, own_desktop, windows_only

# LTspice as far as its files go: ``-Run -b <deck>`` leaves a log and a raw,
# ``-netlist <sheet>`` a netlist. A deck or sheet whose name says so makes it
# exit 1 without them, put up a message box, or never finish.
STAND_IN = OWN_DESKTOP_SOURCE + textwrap.dedent(
    """
    import ctypes, json, os, threading
    from pathlib import Path

    arguments = sys.argv[1:]
    export = arguments[0] == "-netlist"
    subject = Path(arguments[1] if export else arguments[2])
    facts = {"desktop": own_desktop(), "arguments": arguments, "cwd": os.getcwd()}
    subject.with_suffix(".ran.json").write_text(json.dumps(facts), encoding="utf-8")
    print("console line")
    if "refused" in subject.stem:
        sys.exit(1)
    if "asks" in subject.stem:
        ctypes.WinDLL("user32").MessageBoxW(
            None, "Aborting: Unknown schematic syntax", "LTspice", 0
        )
    if "hangs" in subject.stem:
        threading.Event().wait()
    if export:
        subject.with_suffix(".net").write_text("* exported\\n.end\\n", encoding="utf-8")
    else:
        subject.with_suffix(".log").write_text("Circuit: * t\\n", encoding="utf-8")
        subject.with_suffix(".raw").write_bytes(b"Title: * t\\n")
    """
)


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    script = tmp_path / "stand_in_ltspice.py"
    script.write_text(STAND_IN, encoding="utf-8")
    return script


@pytest.fixture(autouse=True)
def _interpreter_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """What the base interpreter needs to act as this environment's Python.

    The stand-in is started as that interpreter, not through the environment's
    redirecting launcher, so that the process started is the one that runs.
    """
    _, launch = python_launch()
    for key, value in (launch or {}).items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)


def bound(base: type, script: Path) -> type:
    """``base`` launching the stand-in."""
    python, _ = python_launch()
    return type(
        base.__name__, (base,), {"spice_exe": [python, str(script)], "process_name": "python.exe"}
    )


def ran(subject: Path) -> dict:
    return json.loads(subject.with_suffix(".ran.json").read_text(encoding="utf-8"))


def spicelib_command(tmp_path: Path, method: str, subject: Path, switches: list | None) -> list:
    """The command spicelib's own LTspice builds for the same call."""
    seen: list[list[str]] = []

    def record(command, timeout=None, stdout=None, stderr=None, cwd=None):
        seen.append(list(command))
        subject.with_suffix(".net").write_text("* exported\n", encoding="utf-8")
        return 0

    original = ltspice_simulator.run_function
    ltspice_simulator.run_function = record
    try:
        cls = type("LTspice", (SpicelibLTspice,), {"spice_exe": [sys.executable, "stand-in"]})
        getattr(cls, method)(subject, switches)
    finally:
        ltspice_simulator.run_function = original
    return seen[0][2:]


class TestTheClassAServerLaunches:
    def test_it_keeps_the_name_records_and_dialects_key_on(self):
        assert LTspice.__name__ == "LTspice"
        assert issubclass(LTspice, SpicelibLTspice)
        assert simulator_family(LTspice) == "ltspice"
        assert simulator_family(LTspice.__name__) == "ltspice"
        assert simulator_dialect(LTspice) == "ltspice"

    @windows_only
    def test_on_windows_it_is_the_ltspice_family_class(self):
        assert SIMULATORS["ltspice"] is LTspice

    @windows_only
    def test_a_named_executable_is_launched_the_same_way(self, tmp_path: Path, stand_in: Path):
        """A further build is bound as a subclass of the family's class, so it
        inherits the launch; binding it does not touch the family's program."""
        family_program = list(LTspice.spice_exe)
        python, _ = python_launch()
        named = bind_named_executable(SIMULATORS["ltspice"], "ltspice:xvii", Path(python))
        assert named.run.__func__ is LTspice.run.__func__
        assert named.create_netlist.__func__ is LTspice.create_netlist.__func__
        assert LTspice.spice_exe == family_program


@windows_only
class TestTheSetting:
    """``[simulator] hidden_desktop`` reaches the launch through detection,
    which every way of starting the engine runs."""

    @pytest.mark.parametrize("wanted", [True, False])
    def test_detection_applies_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wanted: bool
    ):
        monkeypatch.delenv("LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION", raising=False)
        detect_simulators(ServerConfig(working_dir=tmp_path, hidden_desktop=wanted))
        assert (hidden_desktop.shared() is not None) == wanted

    def test_the_python_api_takes_it(self, tmp_path: Path):
        config = engine._library_config(tmp_path, None, {"hidden_desktop": False})
        assert config.hidden_desktop is False


@windows_only
class TestRun:
    def test_a_batch_run_is_off_the_callers_desktop(self, tmp_path: Path, stand_in: Path):
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        code = bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S)
        assert code == 0
        assert deck.with_suffix(".raw").is_file()
        assert ran(deck)["desktop"] == f"ltspice-mcp-{os.getpid()}"
        assert ran(deck)["desktop"] != own_desktop()

    @pytest.mark.parametrize("switches", [None, [], ["-alt"], ["-ini", "settings.ini"]])
    def test_the_command_is_the_one_spicelib_builds(
        self, tmp_path: Path, stand_in: Path, switches: list | None
    ):
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        bound(LTspice, stand_in).run(deck, switches, timeout=LIVENESS_S)
        assert ran(deck)["arguments"] == spicelib_command(tmp_path, "run", deck, switches)

    def test_the_console_log_is_kept_when_asked_for(self, tmp_path: Path, stand_in: Path):
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        cls = bound(LTspice, stand_in)
        cls.run(deck, timeout=LIVENESS_S)
        assert not deck.with_suffix(".exe.log").exists()
        cls.run(deck, timeout=LIVENESS_S, exe_log=True)
        assert deck.with_suffix(".exe.log").read_bytes().strip() == b"console line"

    def test_the_working_directory_is_the_one_given(self, tmp_path: Path, stand_in: Path):
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        folder = tmp_path / "elsewhere"
        folder.mkdir()
        bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S, cwd=folder)
        assert Path(ran(deck)["cwd"]) == folder

    def test_the_exit_code_is_ltspices(self, tmp_path: Path, stand_in: Path):
        deck = tmp_path / "refused.cir"
        deck.write_text("* t\n.end\n")
        assert bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S) == 1

    def test_past_its_timeout_it_raises_as_subprocess_does(self, tmp_path: Path, stand_in: Path):
        deck = tmp_path / "hangs.cir"
        deck.write_text("* t\n.end\n")
        with pytest.raises(subprocess.TimeoutExpired):
            # timing: the bound is the behaviour under test; the stand-in never exits
            bound(LTspice, stand_in).run(deck, timeout=0.2)

    @pytest.mark.usefixtures("quick_looks")
    def test_a_message_box_ends_the_run_with_what_it_said(self, tmp_path: Path, stand_in: Path):
        """On a desktop nobody sees, a box LTspice waits on would hold the run
        to its timeout and say nothing. It is ended and reported instead."""
        deck = tmp_path / "asks.cir"
        deck.write_text("* t\n.end\n")
        with pytest.raises(DialogError) as stopped:
            bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S)
        assert "LTspice; Aborting: Unknown schematic syntax" in str(stopped.value)
        # It says how to get to see the box.
        assert "hidden_desktop = false" in str(stopped.value)

    def test_turned_off_the_launch_is_spicelibs(self, tmp_path: Path, stand_in: Path):
        hidden_desktop.configure(enabled=False)
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        assert bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S) == 0
        assert ran(deck)["desktop"] == own_desktop()

    def test_a_caller_with_its_own_streams_gets_spicelibs_launch(
        self, tmp_path: Path, stand_in: Path
    ):
        deck = tmp_path / "deck.cir"
        deck.write_text("* t\n.end\n")
        with open(tmp_path / "mine.log", "wb") as mine:
            bound(LTspice, stand_in).run(deck, timeout=LIVENESS_S, stdout=mine)
        assert (tmp_path / "mine.log").read_bytes().strip() == b"console line"
        assert ran(deck)["desktop"] == own_desktop()


@windows_only
class TestCreateNetlist:
    def test_an_export_is_off_the_callers_desktop(self, tmp_path: Path, stand_in: Path):
        sheet = tmp_path / "sheet.asc"
        sheet.write_text("Version 4\n")
        net = bound(LTspice, stand_in).create_netlist(sheet, timeout=LIVENESS_S)
        assert net == sheet.with_suffix(".net")
        assert net.is_file()
        assert ran(sheet)["desktop"] == f"ltspice-mcp-{os.getpid()}"

    @pytest.mark.parametrize("switches", [None, ["-alt"]])
    def test_the_command_is_the_one_spicelib_builds(
        self, tmp_path: Path, stand_in: Path, switches: list | None
    ):
        sheet = tmp_path / "sheet.asc"
        sheet.write_text("Version 4\n")
        bound(LTspice, stand_in).create_netlist(sheet, switches, timeout=LIVENESS_S)
        expected = spicelib_command(tmp_path, "create_netlist", sheet, switches)
        assert ran(sheet)["arguments"] == expected

    def test_no_netlist_is_the_error_spicelib_raises(self, tmp_path: Path, stand_in: Path):
        sheet = tmp_path / "refused.asc"
        sheet.write_text("Version 4\n")
        with pytest.raises(RuntimeError, match="Failed to create netlist"):
            bound(LTspice, stand_in).create_netlist(sheet, timeout=LIVENESS_S)

    @pytest.mark.usefixtures("quick_looks")
    def test_a_message_box_ends_the_export_with_what_it_said(self, tmp_path: Path, stand_in: Path):
        """What LTspice XVII does with a sheet that starts with a byte order
        mark: it says so in a box and waits."""
        sheet = tmp_path / "asks.asc"
        sheet.write_text("Version 4\n")
        with pytest.raises(DialogError) as stopped:
            bound(LTspice, stand_in).create_netlist(sheet, timeout=LIVENESS_S)
        assert stopped.value.text == "LTspice\nAborting: Unknown schematic syntax"
