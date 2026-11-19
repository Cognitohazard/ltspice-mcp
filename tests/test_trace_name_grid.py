"""Which trace a signal name reads, and what that trace holds, on every route that answers it.

The question is the trace-name grid's (``[grid.trace-name]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). A build's spelling of a
name inside subcircuit instances is read off who wrote the result
(``device_names.naming_for``, and from a raw's header ``writes_xvii_command``);
two spellings of one trace share a key (``device_names.trace_keys``); a name
finds its trace (``trace_names.find_trace``, through ``read_in_plot`` for
every reader of a plot); and what the trace holds is read off its name and
declared type (``trace_names.trace_quantity``), its plot
(``raw_parser.node_voltages``) and its unit (``decoded_raw.declared_unit``).

A cell is one name a user writes (LTspice 26's spelling, each instance with
its X), with the trace each build wrote for it, from the recording. What the
trace holds is derived here from the cell's ``what``, never from the reader
under test. Every recorded cell, on each build, is read through the naming
facts of its build, ``find_trace``, ``trace_keys``, ``read_in_plot``, the
quantity of the trace, ``inspect``'s signals, ``analyze_results``,
``plot_waveform`` and the Python API's ``load_raw``. Every trace of every
recorded raw of the grid's cases is also found under the name a user would
write, and a trace no rule names is reported.

ngspice names the same nodes with dots; its answer is checked live where
ngspice is on PATH, and nothing of it is recorded.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.decoded_raw import declared_unit
from ltspice_mcp.lib.device_names import naming_for, trace_keys, writes_xvii_command
from ltspice_mcp.lib.raw_header import RawPlotHeader, preflight_raw
from ltspice_mcp.lib.raw_parser import node_voltages
from ltspice_mcp.lib.result_kinds import header_analysis, sampled_axis, step_variables
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.spice_lex import directive_head
from ltspice_mcp.lib.trace_names import (
    find_trace,
    from_schematic,
    read_in_plot,
    trace_quantity,
)
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests._ngspice_decks import run_ngspice
from tests._recorded_grids import Cell, grid
from tests.conftest import SyncApi, dispatched

GRID = grid("trace-name")

#: How many routes answer otherwise than a cell, per build, each declared on
#: its cell (``differs``, with the route and the reason). A count may only
#: fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 0, "ltspice17": 0}

_DIRECTIVES = {".tran": "tran", ".ac": "ac", ".dc": "dc", ".noise": "noise", ".op": "op"}


def _trace(cell: Cell, build: str) -> str:
    trace = cell.extra["trace"]
    return trace[build] if isinstance(trace, dict) else trace


def _source(cell: Cell) -> str:
    return rec.CASES.case(cell.case).source


# The index, held to the decks ------------------------------------------------


def _analysis(source: str) -> str:
    """The analysis a deck runs."""
    heads = [directive_head(card) for card in rec.deck_cards(source) if card.kind == "directive"]
    (analysis,) = {_DIRECTIVES[head] for head in heads if head in _DIRECTIVES}
    return analysis


#: The function a name of each kind is written with.
_FUNCTION = {
    "node_voltage": "v",
    "current": "i",
    "port_current": "ix",
    "noise": "v",
}


def test_a_cell_is_the_form_its_name_and_its_case_say():
    """Each cell's analysis is its deck's; its depth is the number of
    instances its name passes through; what it holds is the function its
    name is written with; a section-sign name carries the sign. An axis is
    the deck's swept variable, and a gain is named ``gain``."""
    for cell in GRID.cells:
        at = cell.coordinates
        if cell.case.startswith("export/"):
            assert at["analysis"] == _analysis_of_export(cell.case), cell.id
        else:
            assert at["analysis"] == _analysis(_source(cell)), cell.id
        match = re.fullmatch(r"(\w+)\((.*)\)", cell.item)
        if at["what"] == "axis":
            assert match is None and at["depth"] == "top", cell.id
            continue
        if at["what"] == "gain":
            assert cell.item == "gain", cell.id
            continue
        if match is None:
            assert at["what"] == "noise" and cell.item.lower() in ("onoise", "inoise"), cell.id
            continue
        function, path = match.group(1).lower(), match.group(2)
        depth = path.count(":")
        assert at["depth"] == ("top", "block", "nested")[depth], cell.id
        if at["what"] == "terminal_current":
            assert re.fullmatch(r"i[a-z]", function) and function != "ix", cell.id
        else:
            assert function == _FUNCTION[at["what"]], cell.id
        assert (at["form"] == "section_sign") == ("§" in cell.item), cell.id


def _analysis_of_export(case_id: str) -> str:
    """The analysis a run of a sheet ran: the one its recorded netlist holds."""
    text = rec.export_text("ltspice26", case_id)
    heads = [line.split()[0].lower() for line in text.splitlines() if line.startswith(".")]
    (analysis,) = {_DIRECTIVES[head] for head in heads if head in _DIRECTIVES}
    return analysis


def test_every_case_of_the_grid_is_waiting_or_has_no_prediction_left():
    """A recorded cell declares what the recording shows: a prediction
    (``not_established``) is settled against its recording, and the key
    removed, once the case is recorded."""
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if cell.case in rec.manifest(build)["cases"]]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle its trace against the "
                "recording and remove not_established"
            )


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(
            1
            for cell in GRID.cells
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        )
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# What a cell holds, by its form -----------------------------------------------

#: What each kind of trace holds, as each route says it: ``trace_quantity``,
#: membership of ``node_voltages`` and the unit.
_HOLDS: dict[str, tuple[str | None, bool, str | None]] = {
    "node_voltage": ("node_voltage", True, "V"),
    "current": ("current", False, "A"),
    "terminal_current": ("current", False, "A"),
    "port_current": ("current", False, "A"),
    # A noise density: LTspice types it a voltage, which trace_quantity reads
    # without its plot, so its plot's readers ask node_voltages instead.
    "noise": (None, False, "V/√Hz"),
    "gain": ("other", False, None),
    "axis": ("other", False, None),
}

_AXIS_UNITS = {"time": "s", "frequency": "Hz"}


def _expected_unit(cell: Cell) -> str | None:
    at = cell.coordinates
    if at["what"] == "axis":
        # A DC sweep's axis is its swept source: a voltage source's in volts,
        # a current source's in amperes.
        return _AXIS_UNITS.get(cell.item, "V" if cell.item[0].lower() == "v" else "A")
    if at["what"] == "noise" and cell.item.lower() in ("v(inoise)", "inoise"):
        # The input-referred noise has the unit of the input source, which
        # the raw does not name.
        return None
    return _HOLDS[at["what"]][2]


def _expected(cell: Cell, build: str) -> dict[str, Any]:
    trace = _trace(cell, build)
    quantity, voltage, _ = _HOLDS[cell.coordinates["what"]]
    unit = _expected_unit(cell)
    want: dict[str, Any] = {
        "naming_for xvii": build == "ltspice17",
        "raw header xvii": build == "ltspice17",
        "find_trace": trace,
        "trace_keys": True,
        "read_in_plot": (trace,),
        "node_voltages": voltage,
        "unit": unit,
        "inspect signals": True,
        "analyze_results": "read",
        "api load_raw": "read",
        # A chart of an AC sweep draws each trace's magnitude in decibels.
        "plot_waveform unit": "dB" if cell.coordinates["analysis"] == "ac" else unit,
    }
    if quantity is not None:
        want["trace_quantity"] = quantity
    return want


# Every route --------------------------------------------------------------------


def _read_or_refusal(reply: dict[str, Any]) -> str:
    if reply.get("failures"):
        (message,) = {failure["message"] for failure in reply["failures"]}
        return message
    return "read"


async def _every_route(
    build: str,
    cell: Cell,
    plot: RawPlotHeader,
    state: SessionState,
    staged: Path,
    scratch: Path,
    not_run: dict[str, str],
) -> dict[str, Any]:
    user = cell.item
    xvii = naming_for(rec.producer(build)).xvii
    names = [variable.name for variable in plot.variables]
    routes: dict[str, Any] = {
        "naming_for xvii": xvii,
        "raw header xvii": writes_xvii_command(dict(plot.fields)),
    }
    found = find_trace(names, user, xvii=xvii)
    routes["find_trace"] = found
    trace = _trace(cell, build)
    if cell.coordinates["form"] == "section_sign" or _NAME.fullmatch(user) is None:
        # A section sign and a bare noise name are find_trace's own aliases;
        # a key is one name's function and path, compared between results.
        not_run["trace_keys"] = "the name is an alias find_trace resolves, not a spelling of a path"
    else:
        # The user's name is LTspice 26's spelling, of one reading; the
        # trace's may stand for more than one (an XVII run of a schematic),
        # and one of them is the user's.
        (asked,) = trace_keys(user, xvii=False, schematic=False)
        routes["trace_keys"] = asked in trace_keys(
            trace, xvii=xvii, schematic=from_schematic(plot)
        )
    routes["read_in_plot"] = read_in_plot(plot, user).traces
    decoded = rec.decode_file(staged, scratch, log=None).raw
    descriptor = decoded.descriptor
    axis = descriptor.axis.name if descriptor.axis is not None else None
    (described,) = [t for t in descriptor.traces if t.name == trace]
    if cell.coordinates["what"] == "noise":
        not_run["trace_quantity"] = "it reads no plot; a noise plot's readers ask node_voltages"
    else:
        routes["trace_quantity"] = trace_quantity(
            trace, described.declared_type, axis=trace == axis
        )
    routes["node_voltages"] = trace in node_voltages(decoded)
    routes["unit"] = described.unit
    listed = await inspect_one(state, {"kind": "results", "view": "signals", "path": str(staged)})
    routes["inspect signals"] = trace in {row["name"] for row in listed["signals"]} or (
        trace == axis
    )
    if cell.coordinates["what"] == "axis":
        for route in ("analyze_results", "api load_raw", "plot_waveform unit"):
            not_run[route] = "the axis is read as a plot's axis, not as a signal"
        return routes
    swept = axis is not None and plot.point_count > 1
    recipe = (
        {"metric": "signal_stats", "signal": user} if swept else {"metric": "value", "expr": user}
    )
    reply = await dispatched(
        state,
        "analyze_results",
        sources=[{"path": str(staged), "label": "dut"}],
        recipes=[{"key": "k", **recipe}],
    )
    routes["analyze_results"] = _read_or_refusal(reply)
    api = SyncApi(state)

    def load() -> str:
        loaded = api.load_raw(staged)
        try:
            loaded.trace(user)
        except Exception as refused:
            return str(refused)
        return "read"

    routes["api load_raw"] = await asyncio.to_thread(load)
    if swept:
        plotted = await dispatched(
            state, "plot_waveform", path=str(staged), signals=[user], open=False
        )
        routes["plot_waveform unit"] = (
            plotted["traces"][0].get("unit") if plotted.get("traces") else plotted.get("error")
        )
    else:
        not_run["plot_waveform unit"] = "a plot of one point has nothing to chart"
    return routes


def _runs() -> list[Any]:
    runs = []
    for case_id in sorted({cell.case for cell in GRID.cells}):
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if case_id in rec.manifest(build)["cases"]:
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


@pytest.mark.asyncio
@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_finds_its_trace_on_every_route(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """Each name a user writes reads the trace the build wrote for it, on
    every route, and the trace holds what the name's form says. A refused
    item is not read; a route that answers otherwise is declared on its cell
    (``differs``)."""
    staged = rec.stage_run(build, case_id, work_dir / "run")
    if not rec.has(build, f"{case_id}.raw"):
        assert all(build in cell.refused for cell in GRID.cells_of(case_id)), case_id
        return
    (plot,) = preflight_raw(staged, limits=rec.RAW_LIMITS).plots
    mismatched: list[Any] = []
    for index, cell in enumerate(GRID.cells_of(case_id)):
        if build in cell.refused:
            names = [variable.name for variable in plot.variables]
            assert cell.extra["trace"] not in names, (cell.id, "refused, yet written")
            continue
        not_run: dict[str, str] = {}
        routes = await _every_route(
            build, cell, plot, state_no_sim, staged, work_dir / f"cell{index}", not_run
        )
        print(f"{build}/{cell.id}: routes not run: {not_run}")
        want = _expected(cell, build)
        declared = {
            entry["route"]: entry["reason"]
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        }
        differ = {route: (got, want[route]) for route, got in routes.items() if got != want[route]}
        if set(differ) != set(declared):
            mismatched.append((cell.id, build, differ, sorted(declared)))
    assert not mismatched, mismatched


# Every recorded trace of the grid's cases -------------------------------------

_NAME = re.compile(r"(\w+)\((.*)\)")


def what_by_rule(plot: RawPlotHeader, index: int) -> str | None:
    """What a recorded trace holds, read off its name and declared type
    without the reader under test: the sampled axis first, a stepped value, a
    noise run's gain and densities, then the function its name is written
    with. None for a trace no rule names."""
    variable = plot.variables[index]
    declared = variable.declared_type.lower()
    if index == 0 and sampled_axis(header_analysis(plot), plot.first_type) is not None:
        return "axis"
    if variable.name in step_variables(plot):
        return "step_value"
    if declared == "gain":
        return "gain"
    if header_analysis(plot) == "noise":
        return "noise"
    match = _NAME.fullmatch(variable.name)
    if match is None:
        return None
    function = match.group(1).lower()
    if function == "v" and declared == "voltage":
        return "node_voltage"
    if function == "i" and declared == "device_current":
        return "current"
    if function == "ix" and declared == "subckt_current":
        return "port_current"
    if re.fullmatch(r"i[a-z]", function) and declared == "device_current":
        return "terminal_current"
    return None


def user_spelling(trace: str, build: str) -> str:
    """The name a user writes for a recorded trace: LTspice 26's own, and
    XVII's with each instance's X put back (every instance of a deck is
    named with an X; ``subckt-device-names``)."""
    if build != "ltspice17":
        return trace
    match = _NAME.fullmatch(trace)
    if match is None or ":" not in match.group(2):
        return trace
    *instances, last = match.group(2).split(":")
    return f"{match.group(1)}({':'.join(f'x{segment}' for segment in instances)}:{last})"


#: Recorded traces of the grid's cases that no rule names, by build: each is
#: reported. None today.
UNNAMED: dict[str, frozenset[str]] = {"ltspice26": frozenset(), "ltspice17": frozenset()}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_recorded_trace_is_found_under_the_name_a_user_writes(build: str):
    """Every trace of every recorded raw of the grid's cases is named by a
    rule, and found by ``find_trace`` under the name a user writes: a cell's
    own name where a cell names the trace, else the build's spelling with
    each instance's X."""
    xvii = naming_for(rec.producer(build)).xvii
    unnamed: set[str] = set()
    missed: list[str] = []
    for case_id in sorted({cell.case for cell in GRID.cells}):
        if not rec.has(build, f"{case_id}.raw"):
            continue
        (plot,) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
        named = {
            _trace(cell, build): cell.item
            for cell in GRID.cells_of(case_id)
            if build not in cell.refused and cell.coordinates["form"] != "mixed_case"
        }
        names = [variable.name for variable in plot.variables]
        for index, variable in enumerate(plot.variables):
            what = what_by_rule(plot, index)
            if what is None:
                unnamed.add(f"{case_id}:{variable.name}")
                continue
            if what == "step_value":
                continue
            user = named.get(variable.name, user_spelling(variable.name, build))
            if find_trace(names, user, xvii=xvii) != variable.name:
                missed.append(
                    f"{case_id}: {user!r} -> {find_trace(names, user, xvii=xvii)!r}, not {variable.name!r}"
                )
    assert not missed, missed
    assert unnamed == UNNAMED[build], sorted(unnamed)


# ngspice ----------------------------------------------------------------------

#: The names of the grid's node voltages at each depth, which ngspice saves
#: by default (it saves no device current inside a block unless asked).
_NGSPICE_NAMES = ("V(top)", "V(xa:n1)", "V(xa:xin:mid)", "v(TOP)")


@pytest.mark.parametrize("analysis", ["tran", "ac", "dc", "op"])
def test_ngspice_names_a_node_at_every_depth_as_a_user_writes_it(analysis: str, tmp_path: Path):
    """An ngspice run of the grid's deck names each node voltage with dots
    between instances (``v(xa.xin.mid)``): the name a user writes for LTspice
    finds it, its key is the trace's, and it holds a node voltage, in volts.
    The rest of what ngspice saves differs by version and option, and is not
    asserted."""
    raw = run_ngspice(rec.INPUTS / f"raw/trace_grid_{analysis}.cir", tmp_path / "run", text=False)
    header = preflight_raw(raw, limits=rec.RAW_LIMITS, producing_dialect="ngspice")
    plot = header.plots[-1]
    naming = naming_for(Producer("ngspice", None, None))
    assert naming.family == "ngspice" and not naming.xvii
    names = [variable.name for variable in plot.variables]
    decoded = rec.decode_file(raw, tmp_path, producing="ngspice").raw
    for user in _NGSPICE_NAMES:
        found = find_trace(names, user, xvii=False)
        assert found is not None, (user, names)
        assert trace_keys(user, xvii=False, schematic=False) == trace_keys(
            found, xvii=False, schematic=False
        ), (user, found)
        (described,) = [t for t in decoded.descriptor.traces if t.name == found]
        assert trace_quantity(found, described.declared_type) == "node_voltage", found
        assert found in node_voltages(decoded), found
        assert declared_unit(described.declared_type) == "V", found
