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

import ast
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
    here = [("Ix(mb1:D)", trace_keys("Ix(mb1:D)", xvii=True, schematic=True))]
    there = [
        (name, trace_keys(name, xvii=False, schematic=False))
        for name in ("Ix(mb1:d)", "Ix(xmb1:d)", "V(out)")
    ]
    pairing = pair_traces(here, there, same_spelling=False)
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
            if build == "ltspice17":
                # XVII prints the device table without logopinfo (LTspice 26
                # does not, and says so): MB1's values are its own too, and
                # nothing says they are missing.
                params = {name.lower() for name in value["device_op_points"]}
                assert params, reply
                assert all("mb1" in n and "mb2" not in n for n in params), params
                assert "logopinfo" not in str(reply), reply


@pytest.mark.asyncio
async def test_compare_of_chosen_signals_still_names_a_trace_that_could_be_two(
    state_no_sim: SessionState, tmp_path: Path
):
    """With ``signals`` given, a trace that could be either of two traces of
    the other result is still listed under ambiguous: every trace is in one
    of compare's lists or named with the reason."""
    here = _xvii_sheet_raw(
        tmp_path / "xvii.raw",
        "* C:\\\\work\\\\sheet.asc",
        "Linear Technology Corporation LTspice XVII",
        ["Ix(mb1:D)", "Ix(r1:A)"],
    )
    there = _xvii_sheet_raw(
        tmp_path / "26.raw",
        "C:\\\\work\\\\sheet.net",
        "Linear Technology Corporation LTspice",
        ["Ix(mb1:d)", "Ix(xmb1:d)", "Ix(r1:a)"],
    )
    api = SyncApi(state_no_sim)

    def compared() -> dict[str, Any]:
        return api.load_raw(here).compare(api.load_raw(there), signals=["Ix(r1:A)"])

    result = await asyncio.to_thread(compared)
    assert list(result["signals"]) == ["Ix(r1:A)"]
    assert result.get("ambiguous") == {"Ix(mb1:D)": ["Ix(mb1:d)", "Ix(xmb1:d)"]}, result


#: The scopes that read a trace's keys, each with why: a deck's names, read
#: with no schematic, have one key, which a scope may take as the key.
_KEY_READERS = {
    ("lib/device_names.py", "_deck_key"): "a deck's own names: one reading",
    ("lib/result_observations.py", "saves_not_stored"): "a deck's .save names: one reading",
    ("lib/result_observations.py", "subcircuit_nodes_not_saved"): "a deck's cards: one reading",
    ("lib/trace_names.py", "_respelled"): "names read literally, then told apart by an X",
    ("api/_primitives.py", "RawResult._keyed_traces"): "every key, for pair_traces",
}
#: The scopes that pair two results' traces.
_PAIRERS = {("api/_primitives.py", "RawResult.compare"): "the Python API's compare"}
#: The scopes that read a name's instances (``instance_readings``).
_READERS = {
    ("lib/device_names.py", "_param_owner"): "a device-table name",
    ("lib/device_names.py", "_trace_paths"): "a trace's or a current's name",
}


def test_one_place_reads_a_names_instances_and_one_pairs_two_results():
    """Which instances a name stands for is read only by ``instance_readings``
    and its two readers; ``trace_keys`` is called only by the scopes named,
    and only with no schematic where a scope takes one key; two results'
    traces are paired only by ``pair_traces``. A caller that took the first
    of several keys, or paired traces by name itself, is how a trace comes to
    name another instance in silence."""
    from tests._source_scan import modules, scopes

    calls: dict[str, set[tuple[str, str]]] = {
        "trace_keys": set(),
        "pair_traces": set(),
        "instance_readings": set(),
    }
    single: set[tuple[str, str]] = set()
    for path, tree in modules().items():
        for scope, nodes in scopes(tree):
            for node in nodes:
                if not isinstance(node, ast.Call):
                    continue
                # A bare name or an attribute (``device_names.trace_keys``) alike.
                called = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if called in calls:
                    calls[called].add((path, scope))
                if called == "trace_keys":
                    schematic = next(k.value for k in node.keywords if k.arg == "schematic")
                    if not (isinstance(schematic, ast.Constant) and schematic.value is False):
                        single.add((path, scope))
    assert calls["trace_keys"] <= set(_KEY_READERS), sorted(
        calls["trace_keys"] - set(_KEY_READERS)
    )
    assert calls["pair_traces"] <= set(_PAIRERS), sorted(calls["pair_traces"])
    assert calls["instance_readings"] <= set(_READERS), sorted(calls["instance_readings"])
    # Only the pairing's own reader reads a result's names with their schematic
    # flag; every other caller reads a deck's names, which have one key.
    assert single <= {("api/_primitives.py", "RawResult._keyed_traces")}, sorted(single)
    for path, scope in _KEY_READERS:
        tree = modules()[path]
        (nodes,) = [n for s, n in scopes(tree) if s == scope]
        for node in nodes:
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Call)
                and (
                    (isinstance(node.value.func, ast.Name) and node.value.func.id == "trace_keys")
                    or (
                        isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "trace_keys"
                    )
                )
            ):
                pytest.fail(f"{path}:{node.lineno} takes one of trace_keys' keys")


# One instance a name, never two ----------------------------------------------

#: An XVII schematic run's raw with a block X1 (written x1) beside a card X1
#: a directive writes (written 1), a pair no recording holds: built by hand.
_COLLIDING = ["V(n001)", "Id(x1:M1)", "Id(1:M1)", "Ix(x1:D)", "Ix(1:D)"]


def _xvii_schematic_op(
    path: Path, traces: list[str], *, title: str = "* C:\\x\\sheet.asc"
) -> Path:
    lines = [
        f"Title: {title}",
        "Date: Thu Jan 15 00:00:00 2026",
        "Plotname: Operating Point",
        "Flags: real",
        f"No. Variables: {len(traces)}",
        "No. Points: 1",
        "Command: Linear Technology Corporation LTspice XVII",
        "Variables:",
        *(
            f"\t{i}\t{name}\t"
            + (
                "voltage"
                if name.startswith("V(")
                else "subckt_current"
                if name.startswith("Ix(")
                else "device_current"
            )
            for i, name in enumerate(traces)
        ),
        "Values:",
        "0" + "".join(f"\t{i + 1}e-3\n" for i in range(len(traces))),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def _bias(state: SessionState, raw: Path, **recipe: Any) -> dict[str, Any]:
    from tests.conftest import dispatched

    return await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(raw), "label": "dut"}],
        recipes=[{"key": "k", "metric": "operating_point", **recipe}],
    )


@pytest.mark.asyncio
async def test_a_device_named_as_the_result_writes_it_selects_that_instance_alone(
    state_no_sim: SessionState, tmp_path: Path
):
    """``x1`` is the block the result writes ``x1``, not also the card it
    writes ``1``: a name the result writes stands for that instance first."""
    raw = _xvii_schematic_op(tmp_path / "sheet.raw", _COLLIDING)
    for device, held in (
        ("x1", {"Id(x1:M1)", "Ix(x1:D)"}),
        ("x1:m1", {"Id(x1:M1)"}),
        ("1", {"Id(1:M1)", "Ix(1:D)"}),
    ):
        reply = await _bias(state_no_sim, raw, device=device)
        assert not reply.get("failures"), (device, reply)
        (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
        assert set(value["currents"]) == held, (device, value["currents"])


@pytest.mark.asyncio
async def test_a_device_whose_readings_reach_two_instances_is_refused_with_both(
    state_no_sim: SessionState, tmp_path: Path
):
    """``xa:xb:m1`` names nothing the result writes. Read with the first X
    dropped it is the device in ``a:xb``, with the second dropped the one in
    ``xa:b``: two readings that reach different devices, refused with both
    named, never merged."""
    raw = _xvii_schematic_op(tmp_path / "sheet.raw", ["V(n001)", "Id(a:xb:M1)", "Id(xa:b:M1)"])
    reply = await _bias(state_no_sim, raw, device="xa:xb:m1")
    (failure,) = reply["failures"]
    assert "a:xb:m1, xa:b:m1" in failure["message"], failure


@pytest.mark.asyncio
async def test_one_reading_of_a_name_is_one_selection_under_every_parent(
    state_no_sim: SessionState, tmp_path: Path
):
    """One reading of a name, matched as a part of a path, selects it under
    every parent that has it: ``xin`` (read ``in`` on XVII) is the instance
    inside both ``a`` and ``b``, as the name the result writes is."""
    raw = _xvii_schematic_op(
        tmp_path / "deck.raw",
        ["V(n001)", "Id(a:in:Mtail)", "Id(b:in:Mtail)", "Id(a:Mdrv)"],
        title="* a deck",
    )
    held = {"Id(a:in:Mtail)", "Id(b:in:Mtail)"}
    for device in ("xin", "in", "xin:mtail", "xin.mtail", "in:mtail"):
        reply = await _bias(state_no_sim, raw, device=device)
        assert not reply.get("failures"), (device, reply)
        (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
        assert set(value["currents"]) == held, (device, value["currents"])


@pytest.mark.asyncio
async def test_a_device_and_an_instance_of_one_written_name_are_two_things(
    state_no_sim: SessionState, tmp_path: Path
):
    """An XVII deck with a top-level ``M1`` and a card ``XM1`` (written
    ``m1``) holding ``M2``: ``m1`` is the device, as the result writes it,
    on the selector, ``find_trace`` and the shorthand alike; the instance is
    reached by its card, ``xm1``."""
    from ltspice_mcp.lib.trace_names import read_in_plot

    names = ["V(n001)", "Id(M1)", "Id(m1:M2)", "Ix(m1:D)"]
    raw = _xvii_schematic_op(tmp_path / "deck.raw", names, title="* a deck")
    for device, held in (("m1", {"Id(M1)"}), ("xm1", {"Id(m1:M2)", "Ix(m1:D)"})):
        reply = await _bias(state_no_sim, raw, device=device)
        assert not reply.get("failures"), (device, reply)
        (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
        assert set(value["currents"]) == held, (device, value["currents"])
    (plot,) = preflight_raw(raw, limits=rec.RAW_LIMITS).plots
    assert read_in_plot(plot, "Id(m1)").traces == ("Id(M1)",)
    assert read_in_plot(plot, "Id(xm1:m2)").traces == ("Id(m1:M2)",)


#: Recorded runs of decks whose instances XVII writes without their X, and
#: what a selector names in them under both spellings (``subckt-device-names``,
#: ``deck/subckt_multiplier_nested``).
_DECK_SELECTIONS = [
    # LTspice 26 prints its device table only under logopinfo, which
    # log/subckt_names carries and XVII refuses.
    (
        "log/subckt_names_off",
        ("xin", "xin:mtail", "xin.mtail"),
        ("in", "in:tail"),
        "log/subckt_names",
    ),
    # LTspice 26's run holds no current inside X0 and prints no table: it has
    # nothing of X0 to select.
    ("deck/subckt_multiplier_nested", ("x0",), ("0",), None),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "card", "written", "on_26"), _DECK_SELECTIONS)
async def test_the_card_and_the_written_spelling_select_alike_on_a_recorded_xvii_deck(
    case_id: str,
    card: tuple[str, ...],
    written: tuple[str, ...],
    on_26: str | None,
    state_no_sim: SessionState,
    work_dir: Path,
):
    """On XVII the card's spelling (``xin``, ``x0``) and the result's own
    (``in``, ``0``) select the same instances, under every parent; LTspice 26
    writes the card's spelling and selects them by it."""
    selections: dict[str, frozenset[str]] = {}
    for build in rec.BUILDS:
        run = case_id if build == "ltspice17" else on_26
        if run is None or not rec.has(build, f"{run}.raw"):
            continue
        raw = rec.stage_run(build, run, work_dir / build)
        spellings = card + (written if build == "ltspice17" else ())
        for device in spellings:
            reply = await _bias(state_no_sim, raw, device=device)
            assert not reply.get("failures"), (build, device, reply)
            (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
            held = frozenset(
                name for bucket in ("currents", "device_op_points") for name in value[bucket]
            )
            assert held, (build, device, value)
            selections[f"{build}:{device}"] = held
        mine = {
            key.split(":", 1)[1]: held
            for key, held in selections.items()
            if key.startswith(f"{build}:")
        }
        # Each card spelling selects what its written spelling selects.
        for spelled, as_written in zip(card, written, strict=False):
            if as_written in mine:
                assert mine[spelled] == mine[as_written], (build, spelled, mine)
        for spelled in card[1:]:
            assert mine[spelled] == mine[card[1]], (build, spelled, mine)


@pytest.mark.asyncio
async def test_compare_of_one_schematic_run_with_itself_pairs_every_trace_by_its_name(
    state_no_sim: SessionState, tmp_path: Path
):
    """Two results of one build and one kind of run name an instance alike:
    each trace pairs with the trace of its own name, the block's and the
    card's apart."""
    here = _xvii_schematic_op(tmp_path / "a.raw", _COLLIDING)
    there = _xvii_schematic_op(tmp_path / "b.raw", _COLLIDING)
    result = await _compare(state_no_sim, here, there)
    assert sorted(result["signals"]) == sorted(_COLLIDING)
    assert all(entry["max_abs"] == 0 for entry in result["signals"].values()), result
    assert not result.get("ambiguous") and not result.get("only_here"), result


_SHORTHAND_CASES = [
    ("ltspice17", "export/sheet_part_names", ["mb1.m1.gm", "xmb1.m1.gm"], "@m:mb1:1[gm]"),
    ("ltspice17", "export/hier_device_plain", ["u1.d1.id", "xu1.d1.id"], "@d:u1:1[id]"),
    ("ltspice26", "export/hier_device_logopinfo", ["u1.d1.id", "xu1.d1.id"], "@u1:d1[id]"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id", "spellings", "table"), _SHORTHAND_CASES)
async def test_a_devices_parameter_is_read_by_either_reading_of_its_block_on_a_sheet_run(
    build: str,
    case_id: str,
    spellings: list[str],
    table: str,
    state_no_sim: SessionState,
    work_dir: Path,
):
    """The ``dev.param`` shorthand reads a part's device-table value by the
    sheet's name and by its card, as the device selector does: one reading
    of the instances for both (``device_names.instance_readings``)."""
    from tests.conftest import dispatched

    raw = rec.stage_run(build, case_id, work_dir / "run")
    for expr in spellings:
        reply = await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(raw), "label": "dut"}],
            recipes=[{"key": "k", "metric": "value", "expr": expr}],
        )
        assert not reply.get("failures"), (expr, reply)
        (value,) = reply["results"]["k"]["values"]
        assert value["value"]["signal"].lower() == table, (expr, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["u1", "xu1"])
async def test_a_block_on_an_ltspice_26_sheet_selects_its_table_values_and_its_currents(
    device: str, state_no_sim: SessionState, work_dir: Path
):
    """LTspice 26 writes block U1 (exported ``X§U1``) ``u1`` in its device
    table and its raw alike (``export/hier_device_logopinfo``): a name that
    selects D1's table values selects U1's port currents too, by the sheet's
    name or the card's."""
    raw = rec.stage_run("ltspice26", "export/hier_device_logopinfo", work_dir / "run")
    reply = await _bias(state_no_sim, raw, device=device)
    assert not reply.get("failures"), reply
    (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
    assert set(value["currents"]) == {"Ix(u1:in)", "Ix(u1:out)"}, value
    assert {name.lower() for name in value["device_op_points"]} >= {"@u1:d1[id]"}, value


def _op_runs() -> list[Any]:
    """Every recorded run of each build whose raw is an operating point."""
    runs = []
    for build in rec.BUILDS:
        for case_id, entry in sorted(rec.manifest(build)["cases"].items()):
            if f"{case_id}.raw" not in entry["outputs"]:
                continue
            try:
                (plot, *_) = preflight_raw(
                    rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS
                ).plots
            except Exception:
                continue
            if plot.plot_name.lower().startswith("operating point"):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
    return runs


@pytest.mark.parametrize(("build", "case_id"), _op_runs())
def test_a_names_spelling_does_not_change_what_the_selector_selects(build: str, case_id: str):
    """For every instance and device of every recorded operating point, the
    card's spelling (``xa:xin``) and the result's own (``a:in`` on XVII)
    select the same, for the whole path and every part of it, and a device's
    own name selects it under every parent that has it
    (``device_names.device_query``). Every name that selects any of a
    device's values, its currents or its table's, selects all of them. A
    refusal is a difference too."""
    from ltspice_mcp.lib.device_names import _owner, device_query, writes_xvii_command
    from ltspice_mcp.lib.log_parser import read_device_op_points
    from ltspice_mcp.lib.trace_names import plot_names_sheet_parts

    (plot, *_) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
    xvii = writes_xvii_command(dict(plot.fields))
    schematic = plot_names_sheet_parts(plot)
    names = [variable.name for variable in plot.variables]
    if rec.has(build, f"{case_id}.log"):
        names += list(read_device_op_points(rec.recorded(build, f"{case_id}.log")))
    owners = [
        (name, owner)
        for name in names
        if (owner := _owner(name, xvii=xvii, schematic=schematic)) is not None
    ]

    def selected(device: str) -> frozenset[str]:
        try:
            chosen = device_query(device, xvii=xvii, schematic=schematic, among=names)
        except ValueError as refused:
            return frozenset({f"refused: {refused}"})
        return frozenset(name for name in names if chosen(name))

    differ = []
    for name, owner in owners:
        # Every part of the path, from any instance to any below it: ``xin``
        # as well as ``xa:xin`` and ``xa:xin:mtail``.
        for start in range(len(owner.path)):
            for end in range(start + 1, len(owner.path) + 1):
                card = ":".join(owner.path[start:end])
                written = ":".join(owner.written[start:end])
                if selected(card) != selected(written) or name not in selected(card):
                    differ.append(
                        (name, card, written, sorted(selected(card)), sorted(selected(written)))
                    )
        if not owner.port:
            leaf = owner.path[-1]
            under = {n for n, o in owners if not o.port and o.path[-1] == leaf}
            if not under <= selected(leaf):
                differ.append((name, leaf, sorted(under - selected(leaf))))
    # A device is one identity, whether its table's name or its currents'
    # names carry it: every name that selects any of a device's values selects
    # all of them, the table's split spelling (``in:tail`` for ``Mtail``)
    # among the names asked.
    devices: dict[tuple[str, ...], set[str]] = {}
    for name, owner in owners:
        if not owner.port:
            devices.setdefault(owner.path, set()).add(name)
    asked: set[str] = set()
    for _, owner in owners:
        forms = [owner.path, owner.written]
        if not owner.port and len(owner.written[-1]) > 1:
            forms.append((*owner.written[:-1], owner.written[-1][1:]))
        for form in forms:
            for start in range(len(form)):
                for end in range(start + 1, len(form) + 1):
                    asked.add(":".join(form[start:end]))
    for spelled in sorted(asked):
        chosen = selected(spelled)
        for path, values in devices.items():
            if chosen & values and not values <= chosen:
                differ.append((spelled, ":".join(path), sorted(values - chosen)))
    assert not differ, differ


#: Devices of recorded XVII runs that hold both a device's currents and its
#: table, named as the table splits the device's name and as the result
#: writes it: each pair selects the same (``device_names._fragment``).
_SPLIT_NAMES = [
    (
        "log/subckt_names_off_saved",
        [("tail", "mtail"), ("in:tail", "in:mtail"), ("a:in:tail", "a:in:mtail"), ("drv", "mdrv")],
    ),
    ("export/sheet_part_names", [("1", "m1"), ("mb1:1", "mb1:m1")]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "pairs"), _SPLIT_NAMES)
async def test_a_device_named_as_xviis_table_splits_it_selects_its_currents_and_its_table(
    case_id: str, pairs: list[tuple[str, str]], state_no_sim: SessionState, work_dir: Path
):
    """XVII's device table names Mtail ``m:a:in:tail``. ``tail`` and
    ``in:tail`` name the device as ``mtail`` and ``in:mtail`` do: its
    currents (``Id(a:in:Mtail)``) and its table values together."""
    raw = rec.stage_run("ltspice17", case_id, work_dir / "run")
    for split, own in pairs:
        held = {}
        for device in (split, own):
            reply = await _bias(state_no_sim, raw, device=device)
            assert not reply.get("failures"), (device, reply)
            (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
            held[device] = (frozenset(value["currents"]), frozenset(value["device_op_points"]))
        assert held[split] == held[own], (split, own, held)
        currents, table = held[own]
        assert currents and table, (own, held[own])


@pytest.mark.asyncio
async def test_an_instance_the_result_writes_outranks_a_tables_split_spelling(
    state_no_sim: SessionState, work_dir: Path
):
    """An XVII deck with a card ``Xtail`` (written ``tail``) beside the
    device Mtail the recorded table names ``m:a:in:tail``: ``tail`` is the
    instance, as the result writes it, and ``mtail`` the device."""
    folder = work_dir / "run"
    folder.mkdir()
    raw = _xvii_schematic_op(
        folder / "deck.raw", ["V(n001)", "Ix(tail:A)", "I(tail:R1)"], title="* a deck"
    )
    (folder / "deck.log").write_bytes(
        rec.recorded("ltspice17", "log/subckt_names_off_saved.log").read_bytes()
    )
    for device, wanted in (("tail", {"Ix(tail:A)", "I(tail:R1)"}),):
        reply = await _bias(state_no_sim, raw, device=device)
        assert not reply.get("failures"), (device, reply)
        (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
        assert set(value["currents"]) == wanted, (device, value)
        assert not value["device_op_points"], (device, value)
    reply = await _bias(state_no_sim, raw, device="mtail")
    assert not reply.get("failures"), reply
    (value,) = [entry["value"] for entry in reply["results"]["k"]["values"]]
    assert not value["currents"] and value["device_op_points"], value
    assert all("tail" in name for name in value["device_op_points"]), value


@pytest.mark.asyncio
@pytest.mark.parametrize("expr", ["xin.mtail.gm", "in.mtail.gm", "in.tail.gm", "mtail.gm"])
async def test_one_value_of_two_devices_is_refused_naming_both(
    expr: str, state_no_sim: SessionState, work_dir: Path
):
    """``log/subckt_names_off`` on XVII holds Mtail inside ``a:in`` and
    inside ``b:in``: a ``dev.param`` that reaches both is refused, and the
    refusal names each device."""
    from tests.conftest import dispatched

    raw = rec.stage_run("ltspice17", "log/subckt_names_off", work_dir / "run")
    reply = await dispatched(
        state_no_sim,
        "analyze_results",
        sources=[{"path": str(raw), "label": "dut"}],
        recipes=[{"key": "k", "metric": "value", "expr": expr}],
    )
    (failure,) = reply["failures"]
    assert "a:in:mtail, b:in:mtail" in failure["message"], failure
