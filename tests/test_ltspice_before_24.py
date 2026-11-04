"""Whether an LTspice is one before 24: one answer, from its name and its runs.

LTspice XVII and IV read a deck as cp1252, refuse ``.options logopinfo`` with
"unrecognized option" (recorded: ``ltspice17/log/device_op.log``), and name a
subcircuit's instances without their X. An XVII is recognised by its stock file
name, or, under any other name, by the build an earlier run on it reported. The
run's deck, the hierarchy's spelling and the encoding warning all take that one
answer, so an XVII under another name is not handed a deck it refuses.
"""

from __future__ import annotations

import ast
import asyncio
import shutil
from pathlib import Path
from typing import ClassVar

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import services
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.state import SessionState
from tests.conftest import stand_in_program, terminal_experiment
from tests.test_inspect_tools import _run

_XVII = Path(__file__).parent / "fixtures" / "ltspice_recorded" / "ltspice17" / "raw"
_DECK = (
    "* divider\n.subckt leaf p\nR1 p internal 1k\nR2 internal 0 1k\n.ends leaf\n"
    "V1 a 0 1\nXA a leaf\n.op\n.end\n"
)


class _RecordedXVII(LTspice):
    """An LTspice whose every run leaves what LTspice XVII wrote for an ``.op``
    deck, and which keeps each deck it was handed."""

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated
    decks: ClassVar[list[bytes]]

    @classmethod
    def run(
        cls,
        netlist_file,
        cmd_line_switches=None,
        timeout=None,
        stdout=None,
        stderr=None,
        cwd=None,
        exe_log=False,
    ) -> int:
        netlist = Path(netlist_file)
        cls.decks.append(netlist.read_bytes())
        shutil.copy(_XVII / "op.log", netlist.with_suffix(".log"))
        shutil.copy(_XVII / "op.raw", netlist.with_suffix(".raw"))
        return 0

    @classmethod
    def valid_switch(cls, switch: str, path: str = "") -> list:
        return []


def _session(work_dir: Path, name: str) -> tuple[SessionState, type[_RecordedXVII]]:
    program = stand_in_program(work_dir / "sim" / name, b"an LTspice XVII build")
    simulator = type("LTspice", (_RecordedXVII,), {"spice_exe": [str(program)], "decks": []})
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice")
    return SessionState.create(config, available={"ltspice": simulator}), simulator


async def _answers(state: SessionState, simulator: type, deck: Path) -> dict[str, bool]:
    """Whether the encoding check and the hierarchy each take this LTspice to be
    one before 24."""
    identity = await asyncio.to_thread(executable_identity, simulator)
    (hierarchy,) = await _run(
        state, [{"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}]
    )
    assert hierarchy["ok"], hierarchy
    [resistor] = [
        row for row in hierarchy["data"]["instances"] if row["instance"] == ["XA", "R1"]
    ]
    internal = resistor["nodes"][1]["voltage_trace"]
    assert internal in {"V(a:internal)", "V(xa:internal)"}, internal
    return {
        "cp1252_ltspice": services.cp1252_ltspice(state, identity) is not None,
        "hierarchy": internal == "V(a:internal)",
    }


def _payload(deck: Path, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 30},
        "lint": "off",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "known_by_name"),
    [("XVIIx64.exe", True), ("LTspiceXVII.exe", False), ("LTspice.exe", False)],
)
async def test_an_xvii_is_known_by_its_name_or_by_what_its_runs_report(
    name: str, known_by_name: bool, work_dir: Path
):
    state, simulator = _session(work_dir, name)
    deck = work_dir / "divider.cir"
    deck.write_text(_DECK, encoding="utf-8")

    before = await _answers(state, simulator, deck)
    first = await terminal_experiment(state, _payload(deck, "first-run"))
    after = await _answers(state, simulator, deck)
    second = await terminal_experiment(state, _payload(deck, "second-run"))

    assert first["status"] == second["status"] == "completed", (first, second)
    injected = [b".options logopinfo" in deck_bytes for deck_bytes in simulator.decks]
    # Before any run only the stock name tells: every route agrees on it,
    # including the deck the first run is handed.
    assert before == dict.fromkeys(before, known_by_name)
    assert injected[0] is not known_by_name
    # Its run named the build, so from then on every route knows it.
    assert after == dict.fromkeys(after, True)
    assert injected[1] is False


# ---------------------------------------------------------------------------
# Ownership of the question
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"


def _calls(name: str) -> list[tuple[str, ast.Call]]:
    found: list[tuple[str, ast.Call]] = []
    for path in _SRC.rglob("*.py"):
        module = path.relative_to(_SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == name)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            ):
                found.append((module, node))
    return found


def test_only_the_owner_reads_an_executable_s_name_for_its_age() -> None:
    stray = sorted(
        f"{module}:{call.lineno}"
        for module, call in _calls("is_cp1252_ltspice_executable")
        if module != "lib/simulator_build.py"
    )
    assert not stray, f"the executable's name read for its age around the owner: {stray}"


def test_every_age_decision_passes_the_reported_build() -> None:
    """``ltspice_before_24`` takes the executable and the build its runs
    reported; a call with the name alone would answer from half the evidence."""
    calls = _calls("ltspice_before_24")
    assert calls
    short = sorted(f"{module}:{call.lineno}" for module, call in calls if len(call.args) < 2)
    assert not short, f"ltspice_before_24 called without the reported build: {short}"
