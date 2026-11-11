"""Whether a trace holds a node voltage is answered in one place.

``trace_names.trace_quantity`` reads a trace by its name and declared type,
and by whether it is its plot's sampled axis. Every reader that asks the
question goes through it: the operating point's buckets, the
stepping-failure gate of a run's summary, the value scan's comparison with
the deck's sources and an AC summary's default signal. The source is held to
having no second answer that tests a name for a bare ``V(`` prefix, which
reads ngspice's ``v(@m1[vth])`` (a device's parameter) and ``v(v-sweep)`` (a
``.dc`` sweep's own variable) as node voltages.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ltspice_mcp.lib import raw_parser
from ltspice_mcp.lib.raw_parser import build_simulation_summary, extract_operating_point
from ltspice_mcp.lib.result_observations import parse_source_amplitudes
from tests import _ltspice_recorded as rec
from tests._source_scan import PACKAGE, parsed
from tests.conftest import FIXTURES_DIR
from tests.test_summary_log_facts import captured_log_facts

#: ngspice-42 runs of the decks beside them, as the server runs ngspice
#: (``-D ngbehavior=kiltpsa``).
NGSPICE_RUNS = FIXTURES_DIR / "ngspice_runs"

#: The module that owns the rule.
OWNER = "lib/trace_names.py"


def _ngspice_run(name: str, tmp_path: Path):
    return rec.decode_file(NGSPICE_RUNS / f"{name}.raw", tmp_path, producing="ngspice").raw


def test_a_device_parameter_named_v_is_no_node_voltage_to_any_reader(tmp_path: Path):
    """``op_device_params``: a floating gate, whose bias point every stepping
    method fails to find before ngspice takes its transient op's last point,
    saying "Transient op finished successfully", as it says of a circuit with
    no DC solution too: no recovery the log parser reads. The deck saves
    three device parameters and no node, and ngspice writes them
    ``v(@m1[vth])`` (typed voltage), ``@m1[gm]`` and ``i(@m1[id])`` (typed
    current). The operating point files them with the device, the
    stepping-failure gate finds no node voltage to vouch for the bias point,
    and the value scan does not compare the 1.8 V threshold with the 20 mV
    supplies."""
    raw = _ngspice_run("op_device_params", tmp_path)
    log = captured_log_facts(tmp_path, NGSPICE_RUNS / "op_device_params.log")
    deck = (NGSPICE_RUNS / "op_device_params.cir").read_text(encoding="utf-8")
    assert raw.get_trace_names() == ["v(@m1[vth])", "@m1[gm]", "i(@m1[id])"]
    assert raw.get_trace("v(@m1[vth])").whattype == "voltage"

    op = extract_operating_point(raw)
    assert set(op["device_op_points"]) == {"v(@m1[vth])", "@m1[gm]", "i(@m1[id])"}
    assert op["voltages"] == op["currents"] == op["other"] == {}

    summary = build_simulation_summary(
        raw, log, value_scan=True, source_amplitudes=parse_source_amplitudes(deck)
    )
    assert summary["errors"] == [
        "Warning: Dynamic gmin stepping failed",
        "Warning: True gmin stepping failed",
        "Warning: source stepping failed",
    ]
    assert not any("finite node data" in w for w in summary.get("warnings", []))
    assert not any(o["code"] == "extreme_value" for o in summary["observations"])
    assert raw_parser.node_voltages(raw) == []


def test_a_bias_point_ngspices_source_stepping_finds_is_no_failure(tmp_path: Path):
    """``op_source_stepping``: gmin stepping fails and source stepping finds
    the bias point, which ngspice ends with "Source stepping completed". The
    failed rungs are no error of the run, though the deck saves no node
    voltage that could vouch for it."""
    raw = _ngspice_run("op_source_stepping", tmp_path)
    written = (NGSPICE_RUNS / "op_source_stepping.log").read_text(encoding="utf-8")
    assert "Warning: Dynamic gmin stepping failed" in written
    assert "Note: Source stepping completed" in written
    assert raw_parser.node_voltages(raw) == []
    summary = build_simulation_summary(
        raw, captured_log_facts(tmp_path, NGSPICE_RUNS / "op_source_stepping.log")
    )
    assert not summary.get("errors"), summary


def test_a_dc_sweeps_own_variable_is_no_node_voltage_to_any_reader(tmp_path: Path):
    """``dc_current_only``: ngspice writes a ``.dc V1`` sweep's variable as
    ``v(v-sweep)``, typed voltage, beside the one current the deck saved. The
    swept values are finite at every point whether a bias point was found or
    not, so they vouch for none, and the operating point at a sweep point
    files them apart from the node voltages."""
    raw = _ngspice_run("dc_current_only", tmp_path)
    assert raw.get_trace_names() == ["v(v-sweep)", "i(v1)"]

    op = extract_operating_point(raw, point_index=2)
    assert op["voltages"] == {}
    assert op["other"] == {"v(v-sweep)": 10.0}
    assert op["currents"] == {"i(v1)": pytest.approx(-5e-3)}

    log = captured_log_facts(tmp_path, text="Circuit: test\ngmin stepping failed\n")
    summary = build_simulation_summary(raw, log)
    assert summary["errors"] == ["gmin stepping failed"]
    assert raw_parser.node_voltages(raw) == []


@pytest.mark.parametrize("build", rec.BUILDS)
def test_ltspices_dc_sweep_variable_is_no_node_voltage(build: str, tmp_path: Path):
    """``raw/dc``: both builds write a ``.dc V1`` sweep's variable as a trace
    named for the source (``V1`` on LTspice 26, ``v1`` on XVII), typed
    voltage, ahead of the node voltages."""
    raw = rec.decode(build, "raw/dc", tmp_path).raw
    axis = raw.get_trace_names()[0]
    assert axis.lower() == "v1"
    assert raw.get_trace(axis).whattype == "voltage"

    op = extract_operating_point(raw, point_index=3)
    assert op["voltages"] == {"V(in)": 3.0, "V(out)": 1.5}
    assert op["other"] == {axis: 3.0}
    assert raw_parser.node_voltages(raw) == ["V(in)", "V(out)"]


def _is_prefix(node: ast.AST) -> bool:
    """Whether ``node`` is the literal ``V(`` or ``I(``, in either case."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.lower() in ("v(", "i(")
    )


def _prefix_tests_in(tree: ast.AST) -> list[int]:
    """The lines that test a name for a bare ``V(`` or ``I(`` prefix: a
    ``startswith`` of it, alone or among others, or a comparison with it."""
    found: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
            and node.args
        ):
            first = node.args[0]
            candidates = first.elts if isinstance(first, ast.Tuple) else [first]
            if any(_is_prefix(candidate) for candidate in candidates):
                found.append(node.lineno)
        elif isinstance(node, ast.Compare) and any(
            _is_prefix(operand) for operand in (node.left, *node.comparators)
        ):
            found.append(node.lineno)
    return sorted(found)


def test_no_module_but_the_owner_reads_a_trace_by_its_v_prefix():
    found = [
        f"{path.relative_to(PACKAGE).as_posix()}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).as_posix() != OWNER
        for line in _prefix_tests_in(parsed(path))
    ]
    assert found == [], "ask trace_names.trace_quantity whether a trace is a node voltage"
    assert _prefix_tests_in(parsed(PACKAGE / OWNER)), f"{OWNER} no longer holds the rule"


def test_the_prefix_rule_catches_what_it_names():
    tree = ast.parse(
        "def a(name):\n"
        "    return name.lower().startswith('v(')\n"  # 2
        "def b(name):\n"
        "    return name.upper().startswith(('X', 'V('))\n"  # 4
        "def c(name):\n"
        "    return name[:2].lower() == 'v('\n"  # 6
        "def d(name):\n"
        "    return 'I(' == name[:2]\n"  # 8
        "def unrelated(name):\n"
        "    return name.startswith('v(output_impedance_at_'), 'V(' + name + ')'\n"
    )
    assert _prefix_tests_in(tree) == [2, 4, 6, 8]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_noise_plots_densities_are_no_node_voltages(build: str, tmp_path: Path):
    """LTspice types a noise plot's traces as voltages (``V(onoise)``, each
    source's contribution); they are spectral densities, so no reader of
    node voltages takes one."""
    raw = rec.decode(build, "raw/noise", tmp_path).raw
    assert raw.descriptor.analysis == "noise"
    typed = [t.name for t in raw.descriptor.signals() if t.declared_type == "voltage"]
    assert typed, raw.descriptor.signals()
    assert raw_parser.node_voltages(raw) == []
