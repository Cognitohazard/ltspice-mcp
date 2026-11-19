"""Why a read of a log's device table holds nothing, on every route that reads it.

LTspice 24 and later print the operating point of each semiconductor device
(gm, vth, vdsat) only under ``.options logopinfo``; LTspice XVII prints it
always (recorded: ``log/device_op_numdgt_off``, whose LTspice 26 log has no
table and whose XVII log has one). ``run_experiments`` adds the option to an
``.op`` run, except when which LTspice the executable is is not established,
since XVII ends a run on it: the run then carries ``logopinfo_withheld``.

An empty table read later is "no devices" only if the run asked for it. So
every route that reads the table gives the reason with the empty table:
the case's ``logopinfo_withheld`` by job, and, by path, where no job record
says whether the table was asked for, that this is not known. Every route
is listed once, in :func:`_every_route`.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import wsl
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi, dispatched, stand_in_program, terminal_experiment

#: An LTspice 26 .op run of a deck with a MOSFET, a BJT and a diode, made
#: without '.options logopinfo': its log holds no device table.
_NO_TABLE = rec.recorded("ltspice26", "log/device_op_numdgt_off")
_RAW = rec.recorded("ltspice26", "log/device_op_numdgt_off.raw")
_DECK = rec.INPUTS / "log/device_op_numdgt_off.cir"


class _RecordedLTspice26(LTspice):
    """An LTspice whose every run leaves what LTspice 26 wrote for the deck
    run without the option."""

    spice_exe = []  # noqa: RUF012 - spicelib declares it unannotated

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
        shutil.copy(_NO_TABLE.with_suffix(".log"), netlist.with_suffix(".log"))
        shutil.copy(_RAW, netlist.with_suffix(".raw"))
        return 0

    @classmethod
    def valid_switch(cls, switch: str, path: str = "") -> list:
        return []


def _session(work_dir: Path) -> SessionState:
    """A session whose LTspice is a wrapper script: nothing shows which
    LTspice it is, so its first .op run is made without the option."""
    program = stand_in_program(work_dir / "sim" / "ltspice-wrapper", b"#!/bin/sh\nexec x.exe")
    simulator = type("LTspice", (_RecordedLTspice26,), {"spice_exe": [str(program)]})
    config = ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], simulator="ltspice")
    return SessionState.create(config, available={"ltspice": simulator})


def _said(reply: dict[str, Any]) -> list[str]:
    """What an analyze_results reply says beside its values: each
    observation's code and detail, and each refusal's message."""
    said = [f"{item['code']}: {item['detail']}" for item in reply.get("observations", [])]
    for result in reply.get("results", {}).values():
        said.extend(failure["message"] for failure in result.get("failures", []))
    return said


async def _every_route(state: SessionState, source: dict[str, str]) -> dict[str, list[str]]:
    """What each route that reads the device table says with it: the
    operating_point recipe and a device parameter's value recipe through the
    tool and the Python API's analyze_results, and the Python API's
    device_op_points; ``source`` is ``{"job_id": ...}`` or ``{"path": raw}``."""
    recipes = {
        "operating_point": [{"key": "op", "metric": "operating_point"}],
        "value of m1.gm": [{"key": "gm", "metric": "value", "expr": "m1.gm"}],
    }
    api = SyncApi(state)
    routes: dict[str, list[str]] = {}
    for name, recipe in recipes.items():
        arguments = {"sources": [{**source, "label": "run"}], "recipes": recipe}
        routes[f"tool {name}"] = _said(await dispatched(state, "analyze_results", **arguments))
        reply = await asyncio.to_thread(
            lambda arguments=arguments: api.analyze_results(**arguments)
        )
        routes[f"api {name}"] = _said(reply)
    where = (
        {"job_id": source["job_id"]}
        if "job_id" in source
        else {"path": Path(source["path"]).with_suffix(".log")}
    )
    ops = await asyncio.to_thread(lambda: api.device_op_points(**where))
    assert dict(ops) == {}, dict(ops)
    routes["api device_op_points"] = [f"{o['code']}: {o['detail']}" for o in ops.observations]
    return routes


@pytest.mark.asyncio
async def test_a_table_the_server_did_not_ask_for_says_so_on_every_route_by_job(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The run withheld the option (``logopinfo_withheld`` on its receipt),
    and its log holds no table. Every route that reads the table by job gives
    that reason with the empty table, rather than an empty table that reads
    as "no devices"."""
    monkeypatch.setattr(wsl, "is_wsl", lambda: False)
    state = _session(work_dir)
    deck = work_dir / "devices.cir"
    shutil.copyfile(_DECK, deck)
    receipt = await terminal_experiment(
        state,
        {
            "request_id": "withheld",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 30},
            "lint": "off",
        },
    )
    assert receipt["status"] == "completed", receipt
    assert any(o["code"] == "logopinfo_withheld" for o in receipt.get("observations", []))

    routes = await _every_route(state, {"job_id": receipt["job_id"]})
    for route, said in routes.items():
        assert any(line.startswith("logopinfo_withheld: ") for line in said) or any(
            "'.options logopinfo'" in line and "not added" in line for line in said
        ), (route, said)


@pytest.mark.asyncio
async def test_a_table_read_by_path_says_it_is_not_known_whether_it_was_asked_for(
    state_no_sim: SessionState, work_dir: Path
):
    """By path no job record says whether the run asked for the table: every
    route says that it is not known, with the empty table."""
    raw = work_dir / "devices.raw"
    shutil.copyfile(_RAW, raw)
    shutil.copyfile(_NO_TABLE.with_suffix(".log"), raw.with_suffix(".log"))
    routes = await _every_route(state_no_sim, {"path": str(raw)})
    for route, said in routes.items():
        assert any("not known whether" in line for line in said), (route, said)
