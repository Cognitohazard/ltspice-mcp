"""Ops-at-scale battery: whole-circuit construction through edit_schematic.

The retired CircuitPlan compiler was measured at whole-circuit-construction
scale — one call builds a complete circuit, zero rejections. This battery proves
the ops language reaches the same capability through ``edit_schematic{base:
"blank"}``: for every device-class archetype in the shipped build battery
(``tests/test_circuit_asc.py::BUILD_ARCHETYPES``), a COMPLETE, fully-connected
circuit is authored as ONE transactional op batch (components + wires + net
labels + directives), and one LARGE synthetic archetype (a 25-bit R-2R ladder,
51 components / 129 ops) hits the ~50-component scale the design's risk flag
names.

Per archetype the assertions are:

* ``outcome == "complete"`` and ``commit_state == "committed"`` — the whole
  batch committed atomically with no aborting op.
* zero op rejections — ``failures == []`` and no error envelope (an op-failure
  would flip the outcome to ``failed`` and write nothing).
* component census — every ``add_component`` reference is present in the
  committed ``.asc`` (read back through spicelib's ``AscEditor``, which is the
  schematic's component enumeration). LTspice's ``.asc``->netlist exporter is
  unavailable in the test environment, so the census is taken on the committed
  sheet directly rather than through the tool's reference stage; it is the same
  set of parts a netlist census would list.
* every directive landed in the committed text.
* wiring — ``pins_wired + pins_label_only == pins_total`` (zero unwired /
  floating pins), ``pins_total`` equals the sum of the placed symbols' terminal
  counts (no component silently dropped from the geometry layer), and
  ``pins_label_only`` equals the archetype's intentional label-only pins (the
  shared supply / ground / IO / control nets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests.test_circuit_asc import BUILD_ARCHETYPES

# Fixture-symbol terminal counts (mirrors the .asy PIN records in
# tests/fixtures/symbols) — the expected pins_total denominator per placed part.
_PIN_COUNTS = {
    "res": 2,
    "cap": 2,
    "ind": 2,
    "diode": 2,
    "voltage": 2,
    "current": 2,
    "bv": 2,
    "nmos": 3,
    "e": 4,
    "g": 4,
}


def _census(path: Path) -> set[str]:
    """Component references present in the committed .asc (sync — no async I/O)."""
    return set(AscEditor(str(path)).get_components())


def _expected_refs(ops: list[dict]) -> set[str]:
    return {op["reference"] for op in ops if op["op"] == "add_component"}


def _expected_pins_total(ops: list[dict]) -> int:
    return sum(_PIN_COUNTS[op["symbol"]] for op in ops if op["op"] == "add_component")


def _directives(ops: list[dict]) -> list[str]:
    return [op["instruction"] for op in ops if op["op"] == "add_directive"]


async def _build(state: SessionState, name: str, ops: list[dict]) -> dict:
    """Author ``ops`` onto a blank sheet in one call; return the envelope dict."""
    result = await handle_edit_schematic(
        EditSchematicInput.model_validate(
            {"target": f"{name}.asc", "base": "blank", "ops": ops, "view_limit": 500}
        ),
        state,
    )
    data = result.structuredContent
    assert data is not None
    return data


def _assert_one_call_build(
    data: dict,
    work_dir: Path,
    name: str,
    ops: list[dict],
    *,
    expected_label_only: int,
) -> None:
    """The shared per-archetype contract: complete, zero-rejection, fully wired."""
    # 1. One atomic commit, no aborting op.
    assert data["outcome"] == "complete", data
    assert data["commit_state"] == "committed", data
    # 2. Zero op rejections (an op-failure aborts -> outcome "failed", nothing written).
    assert data["failures"] == [], data["failures"]
    assert data.get("error") is None, data.get("error")

    committed = work_dir / f"{name}.asc"
    assert committed.is_file()

    # 3. Component census: every placed part is in the committed sheet.
    assert _census(committed) == _expected_refs(ops)

    # 4. Every directive landed.
    text = committed.read_text()
    for instruction in _directives(ops):
        assert instruction in text, instruction

    # 5. Wiring: zero unwired pins, full pin denominator, intentional labels only.
    w = data["wiring"]
    assert w["pins_total"] == _expected_pins_total(ops), w
    assert w["pins_wired"] + w["pins_label_only"] == w["pins_total"], w  # zero unwired
    assert w["pins_label_only"] == expected_label_only, w


# ---------------------------------------------------------------------------
# Device-class archetypes — each a complete, fully-connected circuit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Archetype:
    symbol: str  # the device class under test, keyed to BUILD_ARCHETYPES
    name: str
    ops: list[dict]
    label_only: int  # intentional label-only (shared supply/ground/IO) pins


# diode — half-wave rectifier: V1 -> D1 -> R1 -> gnd. Fixture pins: voltage
# +/-=(0,-32)/(0,32); diode A/K=(0,-48)/(0,48); res 1/2=(0,-48)/(0,48).
_DIODE = _Archetype(
    symbol="diode",
    name="rectifier",
    label_only=4,  # in x2 (V1.+, D1.A), 0 x2 (V1.-, R1.2)
    ops=[
        {
            "op": "add_component",
            "reference": "V1",
            "symbol": "voltage",
            "x": 100,
            "y": 300,
            "value": "SINE(0 5 60)",
        },
        {
            "op": "add_component",
            "reference": "D1",
            "symbol": "diode",
            "x": 300,
            "y": 150,
            "value": "Dmod",
        },
        {
            "op": "add_component",
            "reference": "R1",
            "symbol": "res",
            "x": 300,
            "y": 320,
            "value": "1k",
        },
        # D1.K (300,198) -> R1.1 (300,272): collinear vertical.
        {"op": "wire_pins", "from_pin": "D1.K", "to_pin": "R1.1"},
        {"op": "add_net_label", "net": "in", "pin": "V1.+"},
        {"op": "add_net_label", "net": "in", "pin": "D1.A"},
        {"op": "add_net_label", "net": "0", "pin": "V1.-"},
        {"op": "add_net_label", "net": "0", "pin": "R1.2"},
        {"op": "add_directive", "instruction": ".model Dmod D(Is=1e-14)"},
        {"op": "add_directive", "instruction": ".tran 50m"},
    ],
)

# nmos — common-source amplifier with source degeneration. Fixture nmos pins:
# D=(0,-96), G=(-48,0), S=(0,96).
_NMOS = _Archetype(
    symbol="nmos",
    name="common_source",
    label_only=7,  # vdd x2, 0 x3 (Vdd.-, RS.2, Vin.-), in x2 (M1.G, Vin.+)
    ops=[
        {
            "op": "add_component",
            "reference": "M1",
            "symbol": "nmos",
            "x": 400,
            "y": 300,
            "value": "NMOS_MOD",
        },
        {
            "op": "add_component",
            "reference": "RD",
            "symbol": "res",
            "x": 400,
            "y": 120,
            "value": "5k",
        },
        {
            "op": "add_component",
            "reference": "RS",
            "symbol": "res",
            "x": 400,
            "y": 500,
            "value": "500",
        },
        {
            "op": "add_component",
            "reference": "Vdd",
            "symbol": "voltage",
            "x": 200,
            "y": 120,
            "value": "5",
        },
        {
            "op": "add_component",
            "reference": "Vin",
            "symbol": "voltage",
            "x": 200,
            "y": 300,
            "value": "SINE(1.5 0.1 1k)",
        },
        # M1.D (400,204) -> RD.2 (400,168); M1.S (400,396) -> RS.1 (400,452): vertical.
        {"op": "wire_pins", "from_pin": "M1.D", "to_pin": "RD.2"},
        {"op": "wire_pins", "from_pin": "M1.S", "to_pin": "RS.1"},
        {"op": "add_net_label", "net": "vdd", "pin": "RD.1"},
        {"op": "add_net_label", "net": "vdd", "pin": "Vdd.+"},
        {"op": "add_net_label", "net": "0", "pin": "Vdd.-"},
        {"op": "add_net_label", "net": "0", "pin": "RS.2"},
        {"op": "add_net_label", "net": "in", "pin": "M1.G"},
        {"op": "add_net_label", "net": "in", "pin": "Vin.+"},
        {"op": "add_net_label", "net": "0", "pin": "Vin.-"},
        {"op": "add_directive", "instruction": ".model NMOS_MOD NMOS(Kp=200u Vto=1)"},
        {"op": "add_directive", "instruction": ".op"},
    ],
)

# e (VCVS) — a controlled voltage source driving a load, driven by Vin. Fixture
# e pins: +=(0,16), -=(0,96), P=(-48,32), N=(-48,80).
_VCVS = _Archetype(
    symbol="e",
    name="vcvs_stage",
    label_only=6,  # out (RL.1), 0 x3 (E1.-, E1.N, Vin.-), in x2 (E1.P, Vin.+)
    ops=[
        {
            "op": "add_component",
            "reference": "E1",
            "symbol": "e",
            "x": 400,
            "y": 300,
            "value": "2",
        },
        {
            "op": "add_component",
            "reference": "RL",
            "symbol": "res",
            "x": 400,
            "y": 220,
            "value": "10k",
        },
        {
            "op": "add_component",
            "reference": "Vin",
            "symbol": "voltage",
            "x": 200,
            "y": 340,
            "value": "AC 1",
        },
        # E1.+ (400,316) -> RL.2 (400,268): vertical, upward (clear of E1.- at 396).
        {"op": "wire_pins", "from_pin": "E1.+", "to_pin": "RL.2"},
        {"op": "add_net_label", "net": "out", "pin": "RL.1"},
        {"op": "add_net_label", "net": "0", "pin": "E1.-"},
        {"op": "add_net_label", "net": "in", "pin": "E1.P"},
        {"op": "add_net_label", "net": "0", "pin": "E1.N"},
        {"op": "add_net_label", "net": "in", "pin": "Vin.+"},
        {"op": "add_net_label", "net": "0", "pin": "Vin.-"},
        {"op": "add_directive", "instruction": ".ac dec 10 1 1meg"},
    ],
)

# g (VCCS) — a controlled current source into a load, driven by Vin. Fixture g
# pins: +=(0,96), -=(0,16), NC+=(-48,32), NC-=(-48,80).
_VCCS = _Archetype(
    symbol="g",
    name="vccs_stage",
    label_only=6,  # vdd (G1.-), 0 x3 (RL.2, G1.NC-, Vin.-), in x2 (G1.NC+, Vin.+)
    ops=[
        {
            "op": "add_component",
            "reference": "G1",
            "symbol": "g",
            "x": 400,
            "y": 200,
            "value": "1m",
        },
        {
            "op": "add_component",
            "reference": "RL",
            "symbol": "res",
            "x": 400,
            "y": 380,
            "value": "1k",
        },
        {
            "op": "add_component",
            "reference": "Vin",
            "symbol": "voltage",
            "x": 200,
            "y": 240,
            "value": "AC 1",
        },
        # G1.+ (400,296) -> RL.1 (400,332): vertical, downward (clear of G1.- at 216).
        {"op": "wire_pins", "from_pin": "G1.+", "to_pin": "RL.1"},
        {"op": "add_net_label", "net": "vdd", "pin": "G1.-"},
        {"op": "add_net_label", "net": "0", "pin": "RL.2"},
        {"op": "add_net_label", "net": "in", "pin": "G1.NC+"},
        {"op": "add_net_label", "net": "0", "pin": "G1.NC-"},
        {"op": "add_net_label", "net": "in", "pin": "Vin.+"},
        {"op": "add_net_label", "net": "0", "pin": "Vin.-"},
        {"op": "add_directive", "instruction": ".op"},
    ],
)

_ARCHETYPES = [_DIODE, _NMOS, _VCVS, _VCCS]


def test_coverage_matches_shipped_build_battery():
    """Every device class in the shipped build battery has an ops-at-scale build.

    Structural link so a class added to BUILD_ARCHETYPES that this battery does
    not build fails here instead of leaving a silent coverage gap.
    """
    assert {a.symbol for a in _ARCHETYPES} == {sym for sym, _pins in BUILD_ARCHETYPES}


@pytest.mark.parametrize("arch", _ARCHETYPES, ids=[a.name for a in _ARCHETYPES])
async def test_archetype_builds_in_one_call(arch: _Archetype, asc_state, work_dir):
    data = await _build(asc_state, arch.name, arch.ops)
    _assert_one_call_build(
        data, work_dir, arch.name, arch.ops, expected_label_only=arch.label_only
    )


# ---------------------------------------------------------------------------
# LARGE synthetic archetype — a 25-node R-2R ladder built programmatically.
# ---------------------------------------------------------------------------


def _r2r_ladder_ops(k: int) -> tuple[list[dict], int]:
    """A k-node R-2R ladder: a vertical series spine (RS0..RS{k-1}) with a 2R
    shunt (RT0..RT{k-1}) dropped off each node, plus a reference source.

    Returns (ops, expected_label_only). Every junction is a real wire; only the
    output node, each shunt's ground end, and the source terminals carry labels.
    """
    x_spine, x_shunt = 400, 600
    ops: list[dict] = []
    for i in range(k):
        cy = 300 + i * 200
        ops.append(
            {
                "op": "add_component",
                "reference": f"RS{i}",
                "symbol": "res",
                "x": x_spine,
                "y": cy,
                "value": "1k",
            }
        )
        # Shunt placed so RT{i}.1 = (x_shunt, cy+48) is collinear with RS{i}.2.
        ops.append(
            {
                "op": "add_component",
                "reference": f"RT{i}",
                "symbol": "res",
                "x": x_shunt,
                "y": cy + 96,
                "value": "2k",
            }
        )
    ops.append(
        {
            "op": "add_component",
            "reference": "V1",
            "symbol": "voltage",
            "x": 200,
            "y": 252,
            "value": "5",
        }
    )
    # Series spine: RS{i}.2 (bottom) -> RS{i+1}.1 (top), collinear vertical.
    for i in range(k - 1):
        ops.append({"op": "wire_pins", "from_pin": f"RS{i}.2", "to_pin": f"RS{i + 1}.1"})
    # Shunt drops: RT{i}.1 -> RS{i}.2, collinear horizontal onto the node.
    for i in range(k):
        ops.append({"op": "wire_pins", "from_pin": f"RT{i}.1", "to_pin": f"RS{i}.2"})
    # Labels: output node, each shunt ground end, and the source rails.
    ops.append({"op": "add_net_label", "net": "out", "pin": "RS0.1"})
    for i in range(k):
        ops.append({"op": "add_net_label", "net": "0", "pin": f"RT{i}.2"})
    ops.append({"op": "add_net_label", "net": "out", "pin": "V1.+"})
    ops.append({"op": "add_net_label", "net": "0", "pin": "V1.-"})
    ops.append({"op": "add_directive", "instruction": ".op"})
    # label-only pins = RS0.1 (out) + k shunt grounds + V1.+ + V1.- = k + 3.
    return ops, k + 3


async def test_large_r2r_ladder_builds_in_one_call(asc_state, work_dir):
    """~50-component whole-circuit build in one op batch — the design's scale flag.

    51 components (25 series + 25 shunt + 1 source), 129 ops, 49 wires. Proves the
    ops language builds at the scale the retired CircuitPlan compiler was measured
    at, with zero rejections and every pin connected.
    """
    k = 25
    ops, expected_label_only = _r2r_ladder_ops(k)
    assert len(_expected_refs(ops)) == 2 * k + 1  # 51 components
    data = await _build(asc_state, "r2r_ladder", ops)
    _assert_one_call_build(
        data, work_dir, "r2r_ladder", ops, expected_label_only=expected_label_only
    )
