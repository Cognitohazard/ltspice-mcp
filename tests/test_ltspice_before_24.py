"""Which LTspice an executable is: one answer, from evidence about it.

LTspice XVII and IV read a deck as cp1252, refuse ``.options logopinfo`` with
"unrecognized option" (recorded: ``ltspice17/log/device_op.log``), name a
subcircuit's instances without their X, word their logs otherwise and keep
their libraries elsewhere. An XVII is recognised by the build an earlier run on
it reported, by the file version in its own version resource (recorded:
``17.0.37.0``), or by its stock file name; ``LTspice.exe`` is a name an XVII can
carry too. The run's deck, the hierarchy's spelling, the log's wording, the
encoding warning and the library all take that one answer
(``simulator_build.which_ltspice``), so an XVII under another name is not
handed a deck it refuses, and an executable nothing speaks for is not taken for
LTspice 26.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import ClassVar

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import services, wsl
from ltspice_mcp.lib.device_names import naming_for
from ltspice_mcp.lib.log_conditions import rules_for
from ltspice_mcp.lib.simulator import generation_of
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._stand_in_pe import stand_in_windows_program
from tests.conftest import stand_in_program, terminal_experiment
from tests.test_inspect_tools import _run

_XVII = rec.recorded("ltspice17", "raw")
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


def _session(
    work_dir: Path, name: str, file_version: str | None
) -> tuple[SessionState, type[_RecordedXVII]]:
    program = (
        stand_in_windows_program(work_dir / "sim" / name, file_version)
        if name.endswith(".exe")
        else stand_in_program(work_dir / "sim" / name, b"#!/bin/sh\nexec wine XVIIx64.exe")
    )
    simulator = type("LTspice", (_RecordedXVII,), {"spice_exe": [str(program)], "decks": []})
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice")
    return SessionState.create(config, available={"ltspice": simulator}), simulator


async def _answers(state: SessionState, simulator: type, deck: Path) -> dict[str, bool]:
    """Whether each place that turns on it takes this LTspice to be XVII: the
    encoding check, the hierarchy's spelling, the names and log wording of a
    deck run on it, and the library it stages against."""
    identity = await asyncio.to_thread(executable_identity, simulator)
    producer = await asyncio.to_thread(services.deck_producer, state, "ltspice", simulator)
    (hierarchy,) = await _run(
        state, [{"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}]
    )
    assert hierarchy["ok"], hierarchy
    [resistor] = [row for row in hierarchy["data"]["instances"] if row["instance"] == ["XA", "R1"]]
    internal = resistor["nodes"][1]["voltage_trace"]
    assert internal in {"V(a:internal)", "V(xa:internal)"}, internal
    return {
        "cp1252_ltspice": services.cp1252_ltspice(state, identity) is not None,
        "hierarchy": internal == "V(a:internal)",
        "naming": naming_for(producer).xvii,
        "log_wording": rules_for(producer)[0] == "ltspice17",
        "library": generation_of(simulator) == "xvii",
    }


def _payload(deck: Path, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 30},
        "lint": "off",
    }


_XVII_VERSION = rec.manifest("ltspice17")["executable"]["file_version"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "file_version", "known_before_a_run"),
    [
        # Its stock name, with and without the version resource.
        ("XVIIx64.exe", _XVII_VERSION, True),
        ("XVIIx64.exe", None, True),
        # Other names: the version resource shows it.
        ("LTspice.exe", _XVII_VERSION, True),
        ("XVIIx64_2019.exe", _XVII_VERSION, True),
        ("LTspiceXVII.exe", _XVII_VERSION, True),
        ("spice17.exe", _XVII_VERSION, True),
        # Nothing about the program shows it: a copy without its version
        # resource, a wrapper script.
        ("LTspice.exe", None, False),
        ("ltspice-wrapper", None, False),
    ],
)
async def test_an_xvii_is_known_by_what_it_is_or_what_its_runs_report(
    name: str,
    file_version: str | None,
    known_before_a_run: bool,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # WSL interop is off whatever the host: the recorded XVII is an LTspice,
    # which under WSL would have its runs moved to the Windows temp directory,
    # and a Linux container on a WSL kernel has none.
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    state, simulator = _session(work_dir, name, file_version)
    deck = work_dir / "divider.cir"
    deck.write_text(_DECK, encoding="utf-8")

    before = await _answers(state, simulator, deck)
    first = await terminal_experiment(state, _payload(deck, "first-run"))
    after = await _answers(state, simulator, deck)
    second = await terminal_experiment(state, _payload(deck, "second-run"))

    assert first["status"] == second["status"] == "completed", (first, second)
    # Before any run every route gives the one answer the executable gives.
    assert before == dict.fromkeys(before, known_before_a_run)
    # An XVII is never handed '.options logopinfo', which ends its run: when
    # nothing shows which LTspice it is, the first run goes without it and
    # says why.
    assert not any(b"logopinfo" in deck_bytes for deck_bytes in simulator.decks)
    assert _withheld(first) == (not known_before_a_run), first
    # Its run named the build, so from then on every route that has the
    # session's runs knows it.
    library_after = after.pop("library")
    assert after == dict.fromkeys(after, True)
    # The library is chosen without the session's runs (the symbol library is
    # built at startup, and a library search has no session), so it follows
    # the executable alone: where that is silent, every build's library is
    # kept, XVII's among them, rather than LTspice 26's alone.
    assert library_after is known_before_a_run
    assert not _withheld(second)


def _withheld(receipt: dict) -> bool:
    return any(item["code"] == "logopinfo_withheld" for item in receipt.get("observations", []))


@pytest.mark.asyncio
async def test_an_established_ltspice_26_gets_the_option_on_its_first_run(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The same evidence shows a current build, under any name: its first
    .op run carries the option that makes its log name each device's
    operating point."""
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    version = rec.manifest("ltspice26")["executable"]["file_version"]
    state, simulator = _session(work_dir, "renamed.exe", version)
    deck = work_dir / "divider.cir"
    deck.write_text(_DECK, encoding="utf-8")
    assert generation_of(simulator) == "current"
    await terminal_experiment(state, _payload(deck, "first-run"))
    assert b".options logopinfo" in simulator.decks[0]
