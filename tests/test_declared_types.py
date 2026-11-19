"""What a trace of each declared type holds and its unit, on every route.

The question is what a trace holds and its unit, which the server reads off
the type the raw's header declares it with (``trace_names.DECLARED_TYPES``,
read by ``trace_names.trace_quantity`` and ``decoded_raw.declared_unit``).
Every type a recorded LTspice raw declares, on each build, is one the server
knows; a type it does not know is said to be unknown in the reply, never
passed as "other" in silence. A block's port current, which LTspice declares
``subckt_current``, is a current in amperes on every route that reads one.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import pytest

from ltspice_mcp.lib.decoded_raw import UNKNOWN_DECLARED_TYPE, declared_unit
from ltspice_mcp.lib.raw_parser import extract_operating_point, node_voltages
from ltspice_mcp.lib.trace_names import DECLARED_TYPES, trace_quantity
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests._ngspice_decks import run_ngspice
from tests.conftest import SyncApi, dispatched

#: A variable line of a raw header: index, name, declared type.
_VARIABLE = re.compile(r"^\s*\d+\s+(\S+)\s+(\S+)")


def _text(data: bytes) -> str:
    """A whole raw as text, every plot's header readable: LTspice writes its
    headers in UTF-16, an even number of bytes each, and its payloads hold
    an even number too, so the headers after a payload stay aligned."""
    if data[1:2] == b"\x00" or data[:2] == b"\xff\xfe":
        return data[: len(data) - len(data) % 2].decode("utf-16-le", errors="replace")
    return data.decode("latin-1")


def _declared_types(raw: Path) -> dict[str, str]:
    """Every type the plots of ``raw`` declare, each to a trace declaring it,
    read off the header's text without the reader under test."""
    found: dict[str, str] = {}
    text = _text(raw.read_bytes())
    for block in re.findall(r"Variables:\s*\n(.*?)(?=\n(?:Binary|Values):|\Z)", text, re.S):
        for line in block.splitlines():
            match = _VARIABLE.match(line)
            if match is not None:
                found.setdefault(match.group(2).lower(), match.group(1))
    return found


#: The types the recorded raws declare, per build: the table of the change
#: that made every one of them known. A type a new recording declares is
#: added here once ``DECLARED_TYPES`` knows it, or to ``UNKNOWN``.
RECORDED_TYPES = frozenset(
    {
        "voltage",
        "device_current",
        "subckt_current",
        "time",
        "frequency",
        "gain",
        "impedance",
        "transfer",
        "param",
        "temperature",
    }
)

#: Recorded types the server does not know, per build, each said to be
#: unknown in the reply (``decoded_raw.UNKNOWN_DECLARED_TYPE``). None today.
UNKNOWN: dict[str, frozenset[str]] = {"ltspice26": frozenset(), "ltspice17": frozenset()}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_type_a_recorded_raw_declares_has_a_quantity_and_a_unit_or_is_said_unknown(
    build: str,
):
    seen: dict[str, str] = {}
    for raw in sorted((rec.FIXTURES / build).rglob("*.raw")):
        for kind, trace in _declared_types(raw).items():
            seen.setdefault(kind, f"{raw.relative_to(rec.FIXTURES / build)}:{trace}")
    assert set(seen) == RECORDED_TYPES, seen
    unknown = {kind for kind in seen if kind not in DECLARED_TYPES}
    assert unknown == UNKNOWN[build], {kind: seen[kind] for kind in unknown}


def _ascii_raw(path: Path, declared: str) -> Path:
    path.write_text(
        "Title: * a trace of a type no simulator is recorded declaring\n"
        "Date: Thu Jan 15 00:00:00 2026\nPlotname: Transient Analysis\n"
        "Flags: real forward\nNo. Variables: 3\nNo. Points: 2\n"
        "Command: Linear Technology Corporation LTspice\nVariables:\n"
        "\t0\ttime\ttime\n\t1\tV(out)\tvoltage\n"
        f"\t2\tFlux(L1)\t{declared}\n"
        "Values:\n0\t0\n\t1\n\t0.5\n1\t1e-3\n\t2\n\t0.25\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_a_type_the_server_does_not_know_is_said_to_be_unknown(
    state_no_sim: SessionState, tmp_path: Path
):
    """A trace of a type the server does not know holds "other" and has no
    unit, and every reply that describes it says the type is not known."""
    assert "flux" not in DECLARED_TYPES
    raw = _ascii_raw(tmp_path / "flux.raw", "flux")
    assert trace_quantity("Flux(L1)", "flux") == "other"
    assert declared_unit("flux") is None
    listed = await inspect_one(
        state_no_sim, {"kind": "results", "view": "signals", "path": str(raw)}
    )
    (row,) = [row for row in listed["signals"] if row["name"] == "Flux(L1)"]
    assert row.get("unit") is None
    assert row["unit_evidence"] == UNKNOWN_DECLARED_TYPE
    loaded = await asyncio.to_thread(SyncApi(state_no_sim).load_raw, raw)
    (described,) = [t for t in loaded.descriptor["traces"] if t["name"] == "Flux(L1)"]
    assert described["unit_evidence"] == UNKNOWN_DECLARED_TYPE
    # A known type with no unit is not said to be unknown.
    known = _ascii_raw(tmp_path / "gain.raw", "gain")
    listed = await inspect_one(
        state_no_sim, {"kind": "results", "view": "signals", "path": str(known)}
    )
    (row,) = [row for row in listed["signals"] if row["name"] == "Flux(L1)"]
    assert row.get("unit_evidence") is None


#: Recorded runs that hold a block's port current, with one on each build.
_PORT_CURRENTS = [
    ("raw/subckt_names_tran_saved", "Ix(xa:out)", {"ltspice17": "Ix(a:OUT)"}),
    ("log/subckt_names_off_saved", "Ix(xa:xin:n)", {"ltspice17": "Ix(a:in:N)"}),
    ("export/sheet_part_names", "Ix(mb1:d)", {"ltspice17": "Ix(mb1:D)"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(("case_id", "user", "spelled"), _PORT_CURRENTS)
async def test_a_port_current_is_a_current_in_amperes_on_every_route(
    build: str,
    case_id: str,
    user: str,
    spelled: dict[str, str],
    state_no_sim: SessionState,
    work_dir: Path,
):
    """LTspice declares a block's port current ``subckt_current``: what it
    holds and its unit, read through every route that answers it."""
    trace = spelled.get(build, user)
    staged = rec.stage_run(build, case_id, work_dir / "run")
    decoded = rec.decode_file(staged, work_dir / "decode", log=None).raw
    (described,) = [t for t in decoded.descriptor.traces if t.name == trace]
    assert described.declared_type == "subckt_current"
    assert trace_quantity(trace, described.declared_type) == "current"
    assert declared_unit(described.declared_type) == "A"
    assert described.unit == "A"
    assert described.unit_evidence == "declared_type"
    assert trace not in node_voltages(decoded)
    listed = await inspect_one(
        state_no_sim, {"kind": "results", "view": "signals", "path": str(staged)}
    )
    (row,) = [row for row in listed["signals"] if row["name"] == trace]
    assert row["unit"] == "A"
    if decoded.descriptor.axis is None:
        # An operating point files it among the currents, in amperes.
        assert trace in extract_operating_point(decoded)["currents"]
        reply = await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(staged), "label": "dut"}],
            recipes=[{"key": "bias", "metric": "operating_point"}],
        )
        (value,) = [entry["value"] for entry in reply["results"]["bias"]["values"]]
        assert trace in value["currents"] and trace not in value["other"]
        assert value["units"][trace] == "A"
    else:
        plotted = await dispatched(
            state_no_sim, "plot_waveform", path=str(staged), signals=[user], open=False
        )
        assert [t.get("unit") for t in plotted["traces"]] == ["A"]


def test_an_ngspice_transient_holds_no_port_current_and_only_known_types(tmp_path: Path):
    """ngspice writes no block's port current: the grid's transient deck
    holds no ``Ix`` trace, and every type it declares is one the server
    knows. Other ngspice analyses declare types it does not (a ``.dc`` of a
    temperature declares ``temp-sweep``), which their descriptors say."""
    raw = run_ngspice(rec.INPUTS / "raw/trace_grid_tran.cir", tmp_path / "run", text=True)
    found = _declared_types(raw)
    names = [
        match.group(1)
        for line in _text(raw.read_bytes()).splitlines()
        if (match := _VARIABLE.match(line)) is not None
    ]
    assert not [name for name in names if name.lower().startswith("ix(")], names
    unknown = {kind: trace for kind, trace in found.items() if kind not in DECLARED_TYPES}
    assert not unknown, unknown


#: The scopes outside the table that name two or more declared types in one
#: collection, each with the other question it answers.
_NOT_A_TABLE = {
    ("lib/raw_header.py", "leading_parameters"): "which variables hold a step's stepped values",
    ("lib/result_kinds.py", "plot_analysis"): "which analysis a plot's axis says it is",
    ("lib/result_kinds.py", "sampled_axis"): "whether a plot has a sampled axis",
    ("lib/spice_validator.py", "<module>"): "the words a deck's analysis card sweeps",
}

#: The scopes outside the table that test a trace's declared type against a
#: type's name, each with the other question it answers.
_NOT_A_QUANTITY = {
    (
        "lib/decoded_raw.py",
        "_trace_descriptor",
    ): "a .tf plot's impedance, which the analysis overrides",
    ("lib/raw_header.py", "leading_parameters"): "which variables hold a step's stepped values",
    ("lib/raw_header.py", "_resolve_dialect"): "which simulator wrote a header",
    ("lib/raw_header.py", "step_parameter_name"): "how a stepped temperature is named",
    ("lib/raw_parser.py", "_has_sampled_axis"): "whether a plot has a sampled axis",
    ("lib/result_kinds.py", "sampled_axis"): "whether a plot has a sampled axis",
}


def test_only_the_table_of_declared_types_says_what_a_type_holds():
    """No scope but ``trace_names`` maps declared types to what a trace holds
    or its unit: neither a collection of two or more of them nor a test of a
    trace's declared type against one, but where it answers another question,
    named here. A second table is how one reader comes to know a type the
    other does not."""
    from tests._source_scan import modules, scopes

    def declared_word(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lower() in DECLARED_TYPES
        )

    collections: set[tuple[str, str]] = set()
    tests: set[tuple[str, str]] = set()
    for path, tree in modules().items():
        if path == "lib/trace_names.py":
            continue
        for scope, nodes in scopes(tree):
            where = (path, scope)
            for node in nodes:
                if isinstance(node, (ast.Dict, ast.Set, ast.Tuple, ast.List)):
                    elements = node.keys if isinstance(node, ast.Dict) else node.elts
                    if sum(1 for e in elements if e is not None and declared_word(e)) >= 2:
                        collections.add(where)
                if (
                    isinstance(node, ast.Compare)
                    and "declared" in ast.unparse(node.left).lower()
                    and any(declared_word(c) for c in ast.walk(node))
                ):
                    tests.add(where)
    assert collections <= set(_NOT_A_TABLE), sorted(collections - set(_NOT_A_TABLE))
    assert tests <= set(_NOT_A_QUANTITY), sorted(tests - set(_NOT_A_QUANTITY))
