"""Which instance a trace's name stands for, and which traces of two results are one.

The question is which instance path a trace's name names on each build
(``device_names.trace_keys``), and from it which trace of one result is which
trace of another (``device_names.pair_traces``, behind the Python API's
``compare``). LTspice XVII writes every instance without the X of its
netlist card; a schematic's export writes a part given a Prefix of X as ``X``
and its reference, so XVII writes that part by its reference, as LTspice 26
does, and a card a directive on the sheet writes without its X (recorded:
``export/hier_plain_top_run``, ``export/sheet_part_names``). The name alone
does not say which an instance of a schematic's XVII run is, so it has a key
for each reading, and a trace that meets more than one trace of the other
result is not paired.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.device_names import pair_traces, trace_keys
from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.lib.trace_names import from_schematic
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._ngspice_decks import run_ngspice
from tests.conftest import SyncApi

pytestmark = pytest.mark.skipif(
    set(rec.BUILDS) != {"ltspice26", "ltspice17"}, reason="needs both recorded builds"
)


def _signals(build: str, case_id: str) -> list[str]:
    (plot,) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
    axis = 1 if plot.variables[0].declared_type in ("time", "frequency") else 0
    return [variable.name for variable in plot.variables[axis:]]


async def _compare(state: SessionState, here: Path, there: Path) -> dict[str, Any]:
    api = SyncApi(state)

    def compared() -> dict[str, Any]:
        return api.load_raw(here).compare(api.load_raw(there))

    return await asyncio.to_thread(compared)


#: Recorded runs of a schematic: every trace of one build's run is a trace
#: of the other's.
_SHEET_RUNS = ["export/sheet_part_names", "export/hier_plain_top_run"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", _SHEET_RUNS)
@pytest.mark.parametrize(
    ("here", "there"),
    [("ltspice17", "ltspice17"), ("ltspice17", "ltspice26"), ("ltspice26", "ltspice17")],
)
async def test_two_runs_of_one_sheet_pair_every_trace(
    case_id: str, here: str, there: str, state_no_sim: SessionState, work_dir: Path
):
    """Two recorded runs of one sheet, of XVII and of either build, pair
    every trace: a part the sheet names (``mb1``, ``x1``) and a card a
    directive writes (``XMB2``) alike."""
    ours = rec.stage_run(here, case_id, work_dir / "here")
    theirs = rec.stage_run(there, case_id, work_dir / "there")
    result = await _compare(state_no_sim, ours, theirs)
    assert not result.get("only_here"), result
    assert not result.get("only_there"), result
    assert not result.get("ambiguous"), result
    assert sorted(result["signals"]) == sorted(_signals(here, case_id))
    # Paired with the trace that holds the same value: an operating point of
    # one circuit, solved by either build.
    for name, compared in result["signals"].items():
        assert compared["max_abs"] <= 1e-6 + 1e-4 * abs(compared["value"]), (name, compared)


@pytest.mark.asyncio
async def test_two_runs_of_one_deck_pair_every_trace_both_builds_hold(
    state_no_sim: SessionState, work_dir: Path
):
    """A deck's run on XVII has one reading of each instance (its card, X
    dropped), which pairs it with LTspice 26's spelling: every trace the
    builds both hold is paired."""
    case_id = "raw/subckt_names_tran_saved"
    ours = rec.stage_run("ltspice17", case_id, work_dir / "here")
    theirs = rec.stage_run("ltspice26", case_id, work_dir / "there")
    result = await _compare(state_no_sim, ours, theirs)
    assert not result.get("ambiguous"), result
    assert not result.get("only_here"), result
    assert len(result["signals"]) == len(_signals("ltspice17", case_id))


@pytest.mark.parametrize("build", ["ltspice26", "ltspice17"])
@pytest.mark.parametrize(
    ("case_id", "user"),
    [
        ("export/sheet_part_names", "Id(mb1:M1)"),
        ("export/sheet_part_names", "Id(xmb2:M1)"),
        ("export/sheet_part_names", "Ix(mb1:d)"),
        ("export/hier_plain_top_run", "Ix(x1:in)"),
    ],
)
def test_the_name_a_user_writes_is_a_key_of_the_trace_on_each_build(
    build: str, case_id: str, user: str
):
    """The user's name (LTspice 26's spelling) is one of the keys of the
    trace each build wrote for it, and the trace's keys name no instance the
    sheet does not hold under either reading."""
    (plot,) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
    assert from_schematic(plot)
    xvii = build == "ltspice17"
    (asked,) = trace_keys(user, xvii=False, schematic=False)
    found = [
        variable.name
        for variable in plot.variables
        if asked in trace_keys(variable.name, xvii=xvii, schematic=True)
    ]
    assert len(found) == 1, (user, found)


def test_a_deck_check_reads_an_xvii_schematic_trace_as_the_card_its_deck_writes():
    """The checks of a deck's ``.save`` and of its subcircuit nodes compare a
    trace with the deck's own cards, so they read each instance as its card
    (``schematic=False``), where pairing two results reads both ways: XVII
    exports part MB1 as the card ``XMB1``, and writes its device ``Id(mb1:M1)``
    (``export/sheet_part_names``)."""
    cards = rec.export_instances("ltspice17", "export/sheet_part_names")
    assert "XMB1" in {name.upper() for name in cards}
    assert trace_keys("Id(mb1:M1)", xvii=True, schematic=False) == (("id", ("xmb1", "m1")),)
    assert trace_keys("Id(mb1:M1)", xvii=True, schematic=True) == (
        ("id", ("mb1", "m1")),
        ("id", ("xmb1", "m1")),
    )


def test_a_trace_that_meets_two_traces_is_not_paired():
    """LTspice 26 can hold both ``mb1`` (a part MB1, exported ``X§MB1``) and
    ``xmb1`` (a card ``XMB1``); XVII's ``mb1`` of a schematic's run could be
    either, so it is paired with neither, and the pairing names both."""
    here = [trace_keys("Ix(mb1:D)", xvii=True, schematic=True)]
    there = [
        trace_keys("Ix(mb1:d)", xvii=False, schematic=True),
        trace_keys("Ix(xmb1:d)", xvii=False, schematic=True),
        trace_keys("V(out)", xvii=False, schematic=True),
    ]
    pairing = pair_traces(here, there)
    assert pairing.pairs == ()
    assert pairing.ambiguous == {0: (0, 1)}
    assert pairing.ambiguous_there == {0: (0,), 1: (0,)}


def test_two_ngspice_runs_pair_every_trace(state_no_sim: SessionState, tmp_path: Path):
    """ngspice names a path with dots and keeps every X, one reading of
    each instance: two runs of the grid's deck pair every trace, and a name a
    user writes for LTspice is a key of the trace it finds."""
    first = run_ngspice(rec.INPUTS / "raw/trace_grid_tran.cir", tmp_path / "a", text=False)
    second = run_ngspice(rec.INPUTS / "raw/trace_grid_tran.cir", tmp_path / "b", text=True)
    api = SyncApi(state_no_sim)
    ours = api.load_raw(first, dialect="ngspice")
    theirs = api.load_raw(second, dialect="ngspice")
    result = ours.compare(theirs)
    assert not result.get("only_here") and not result.get("only_there"), result
    assert not result.get("ambiguous"), result
    assert len(result["signals"]) == len(ours.signals) - 1
    (asked,) = trace_keys("V(xa:xin:mid)", xvii=False, schematic=False)
    assert asked in trace_keys("v(xa.xin.mid)", xvii=False, schematic=False)


def _xvii_sheet_raw(path: Path, title: str, command: str, traces: list[str]) -> Path:
    lines = [
        f"Title: {title}",
        "Date: Thu Jan 15 00:00:00 2026",
        "Plotname: Operating Point",
        "Flags: real",
        f"No. Variables: {len(traces)}",
        "No. Points: 1",
        f"Command: {command}",
        "Variables:",
        *(f"\t{i}\t{name}\tsubckt_current" for i, name in enumerate(traces)),
        "Values:",
        "0" + "".join(f"\t{i + 1}e-3\n" for i in range(len(traces))),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_compare_lists_a_trace_that_could_be_two_as_ambiguous(
    state_no_sim: SessionState, tmp_path: Path
):
    """An XVII run of a schematic's ``Ix(mb1:D)`` could be LTspice 26's
    ``Ix(mb1:d)`` (part MB1) or ``Ix(xmb1:d)`` (a card XMB1): compare pairs
    neither, lists it under ambiguous with both, and calls none of the three
    only one result's."""
    here = _xvii_sheet_raw(
        tmp_path / "xvii.raw",
        "* C:\\work\\sheet.asc",
        "Linear Technology Corporation LTspice XVII",
        ["Ix(mb1:D)"],
    )
    there = _xvii_sheet_raw(
        tmp_path / "26.raw",
        "C:\\work\\sheet.net",
        "Linear Technology Corporation LTspice",
        ["Ix(mb1:d)", "Ix(xmb1:d)"],
    )
    result = await _compare(state_no_sim, here, there)
    assert result["signals"] == {}
    assert result["ambiguous"] == {"Ix(mb1:D)": ["Ix(mb1:d)", "Ix(xmb1:d)"]}
    assert not result.get("only_here") and not result.get("only_there"), result


@pytest.mark.asyncio
async def test_a_device_of_an_xvii_sheet_run_is_selected_by_the_sheets_name(
    state_no_sim: SessionState, work_dir: Path
):
    """The operating point of part MB1 is selected as ``mb1``, the name the
    sheet and LTspice 26 give it, on either build, and as its card ``xmb1``
    on XVII, whose export writes ``XMB1`` (``export/sheet_part_names``)."""
    from tests.conftest import dispatched

    for build, devices in (("ltspice26", ["mb1"]), ("ltspice17", ["mb1", "xmb1"])):
        staged = rec.stage_run(build, "export/sheet_part_names", work_dir / build)
        for device in devices:
            reply = await dispatched(
                state_no_sim,
                "analyze_results",
                sources=[{"path": str(staged), "label": "dut"}],
                recipes=[{"key": "k", "metric": "operating_point", "device": device}],
            )
            assert not reply.get("failures"), (build, device, reply)
            (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
            held = {name.lower() for name in value["currents"]}
            assert "id(mb1:m1)" in held and "id(mb2:m1)" not in held, (build, device, held)
            assert not any("xmb2" in name for name in held), held
