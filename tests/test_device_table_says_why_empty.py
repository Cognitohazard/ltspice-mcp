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
    said.extend(failure["message"] for failure in reply.get("failures", []))
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
    ops = await asyncio.to_thread(
        lambda: api.device_op_points(job_id=source["job_id"])
        if "job_id" in source
        else api.device_op_points(Path(source["path"]).with_suffix(".log"))
    )
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


#: Scopes that name the device table's section and read no table to a caller.
_DEFINES_THE_SECTION = {
    "lib/log_decode.py:decode_logs": "decodes the section from the log",
    "lib/decoded_log.py:_numeric_nulls": "checks the decoded section's shape",
    "lib/decoded_log.py:_check_sections": "checks the decoded section's shape",
    "lib/log_types.py:<module>": "lists the section names",
}


def test_every_reader_of_the_device_table_asks_why_it_may_be_empty():
    """The guard: a scope that reads the decoded device table (the
    ``device_op`` section) also asks ``services.device_table_unread``, so a
    new route cannot give an empty table without its reason."""
    import ast

    from tests._source_scan import modules, scopes

    def called(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        return (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )

    silent = []
    readers = []
    naming = set()
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            if not any(isinstance(n, ast.Constant) and n.value == "device_op" for n in nodes):
                continue
            key = f"{path}:{scope}"
            naming.add(key)
            if key in _DEFINES_THE_SECTION:
                continue
            readers.append(key)
            if not any(called(node) == "device_table_unread" for node in nodes):
                silent.append(key)
    assert readers, "the scan found no reader of the device table"
    assert set(_DEFINES_THE_SECTION) <= naming, set(_DEFINES_THE_SECTION) - naming
    assert silent == [], (
        "These scopes read the log's device table without asking "
        f"services.device_table_unread why an empty one is empty: {silent}"
    )


@pytest.mark.asyncio
async def test_an_empty_table_that_means_no_devices_gets_no_reason(
    state_no_sim: SessionState, work_dir: Path
):
    """Where an empty table does mean "no such device", no route adds a
    reason: an ngspice log, which never holds the table; an XVII log, which
    prints it unasked; and, through the operating-point recipe, an LTspice
    26 run of a divider, whose raw shows no semiconductor."""
    api = SyncApi(state_no_sim)
    for log in (
        Path("tests/fixtures/ngspice_ac_no_meas.log"),
        rec.recorded("ltspice17", "raw/op.log"),
    ):
        copy = work_dir / f"{log.parent.name}_{log.name}"
        shutil.copyfile(log, copy)
        ops = await asyncio.to_thread(lambda copy=copy: api.device_op_points(copy))
        assert dict(ops) == {}, dict(ops)
        assert not [o for o in ops.observations if o["code"] == "device_table_request_unknown"]
    raw = work_dir / "divider.raw"
    shutil.copyfile(rec.recorded("ltspice26", "raw/op.raw"), raw)
    shutil.copyfile(rec.recorded("ltspice26", "raw/op.log"), raw.with_suffix(".log"))
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[{"key": "op", "metric": "operating_point"}],
    )
    assert not [
        line for line in _said(reply) if line.startswith("device_table_request_unknown")
    ], reply


def _dc_sweep_raw(directory: Path) -> Path:
    """A .dc sweep's raw holding a MOSFET's terminal current and no device
    parameter, with the LTspice 26 log of a run made without the option
    beside it. No recording holds a .dc sweep of a semiconductor."""
    raw = directory / "sweep.raw"
    raw.write_text(
        "Title: * sweep\n"
        "Date: Thu Jan 15 00:00:00 2026\n"
        "Plotname: DC transfer characteristic\n"
        "Flags: real forward\n"
        "No. Variables: 3\n"
        "No. Points: 2\n"
        "Offset: 0.0000000000000000e+00\n"
        "Command: Linear Technology Corporation LTspice\n"
        "Variables:\n"
        "\t0\tv1\tvoltage\n"
        "\t1\tV(d)\tvoltage\n"
        "\t2\tId(M1)\tdevice_current\n"
        "Values:\n"
        "0\t0.0\n\t0.0\n\t0.0\n"
        "1\t1.0\n\t0.9\n\t1.5e-05\n"
    )
    shutil.copyfile(_NO_TABLE.with_suffix(".log"), raw.with_suffix(".log"))
    return raw


@pytest.mark.asyncio
async def test_a_sweep_read_by_path_promises_no_rerun_of_itself(
    state_no_sim: SessionState, work_dir: Path
):
    """LTspice prints the table for an .op run under the option; with it, a
    .tran run prints none (recorded: log/device_op_tran), and no recording
    shows a .dc sweep. So the reason given with a sweep's empty table says
    that, and points to an .op run for the table, not to running the sweep
    again; an .op's reason names the option for a rerun."""
    raw = _dc_sweep_raw(work_dir)
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(raw), "label": "run"}],
        recipes=[{"key": "op", "metric": "operating_point"}],
    )
    (said,) = [line for line in _said(reply) if line.startswith("device_table_request_unknown")]
    assert "a .dc sweep prints one" in said, said
    assert "rerun" not in said, said
    assert "run the circuit's bias point as an .op" in said, said
    op = work_dir / "devices.raw"
    shutil.copyfile(_RAW, op)
    shutil.copyfile(_NO_TABLE.with_suffix(".log"), op.with_suffix(".log"))
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(op), "label": "run"}],
        recipes=[{"key": "op", "metric": "operating_point"}],
    )
    (said,) = [line for line in _said(reply) if line.startswith("device_table_request_unknown")]
    assert "rerun it with the option" in said, said
