"""The builds of the schematic build benchmark: how each reference circuit is drawn.

Each circuit has two builds (`tests/_build_benchmark.py` says what a build is):

- ``labelled``: parts placed with no layout in mind, pins joined by labels
  except where two pins already line up. The five circuits that
  `test_ops_at_scale.py` used to build keep its ops unchanged; the others
  are generated from the netlist, a label on every pin.
- ``drawn``: the drawing a person would make with the present ops. Signal
  left to right, supplies above and ground below, matched parts mirrored,
  wires between parts, labels for supplies, ground and ports.

A drawn build works out where each part goes by the pin it should put at a
point (`Drawing.part`), which is the arithmetic a caller does today before
it can write a part's origin.

A non-test helper module (leading underscore), so pytest does not collect it.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ltspice_mcp.lib.symbol_file import read_symbol
from ltspice_mcp.lib.symbol_geometry import place_point
from tests._build_benchmark import Build, Card, Circuit, circuit, circuits
from tests._schematic_fixtures import TESTS

_SYMBOLS = TESTS / "fixtures" / "symbols"

Point = tuple[int, int]


@functools.cache
def _pins(symbol: str) -> dict[str, tuple[int, int, int]]:
    """Each pin of a suite symbol, by name: its SpiceOrder and where it is on the symbol."""
    file = read_symbol((_SYMBOLS / f"{symbol}.asy").read_text(encoding="ascii"))
    return {pin.name: (pin.order, pin.x, pin.y) for pin in file.pins}


def _pin_named(symbol: str, order: int) -> str:
    return next(name for name, (o, _x, _y) in _pins(symbol).items() if o == order)


def symbol_of(card: Card, reference: Circuit) -> str:
    """The suite symbol a card is drawn with."""
    if card.letter == "M":
        if len(card.nodes) == 3:
            return "nmos"
        model = card.value.split()[0].lower()
        kinds = {
            words[1].lower(): words[2].split("(")[0].lower()
            for words in (d.split() for d in reference.directives)
            if words[0].lower() == ".model" and len(words) > 2
        }
        return "pmos4" if kinds.get(model) == "pmos" else "nmos4"
    return {
        "R": "res",
        "C": "cap",
        "L": "ind",
        "V": "voltage",
        "I": "current",
        "D": "diode",
        "E": "e",
        "G": "g",
        "B": "bv",
        "Q": "npn",
    }[card.letter]


# ---------------------------------------------------------------------------
# Labelled builds
# ---------------------------------------------------------------------------


def _generated_labelled(name: str) -> Build:
    """Each part on a grid in netlist order, a label with its node's name on every pin."""
    reference = circuit(name)
    ops: list[dict[str, Any]] = []
    for number, card in enumerate(reference.cards):
        symbol = symbol_of(card, reference)
        ops.append(
            {
                "op": "add_component",
                "reference": card.ref,
                "symbol": symbol,
                "x": 160 + 160 * (number % 8),
                "y": 160 + 240 * (number // 8),
                "value": card.value,
            }
        )
        for order, node in enumerate(card.nodes, start=1):
            pin = _pin_named(symbol, order)
            ops.append({"op": "add_net_label", "net": node, "pin": f"{card.ref}.{pin}"})
    ops += [{"op": "add_directive", "instruction": d} for d in reference.directives]
    return Build(name, "labelled", (tuple(ops),))


# The builds `test_ops_at_scale.py` made, unchanged. Fixture pins: voltage
# +/- (0,-32)/(0,32); diode A/K (0,-48)/(0,48); res 1/2 (0,-48)/(0,48); nmos D
# (0,-96), G (-48,0), S (0,96); e + (0,16), - (0,96), P (-48,32), N (-48,80); g
# + (0,96), - (0,16), NC+ (-48,32), NC- (-48,80).
_RECTIFIER = (
    {"op": "add_component", "reference": "V1", "symbol": "voltage", "x": 100, "y": 300, "value": "SINE(0 5 60)"},
    {"op": "add_component", "reference": "D1", "symbol": "diode", "x": 300, "y": 150, "value": "Dmod"},
    {"op": "add_component", "reference": "R1", "symbol": "res", "x": 300, "y": 320, "value": "1k"},
    # D1.K (300,198) -> R1.1 (300,272): collinear vertical.
    {"op": "wire_pins", "from_pin": "D1.K", "to_pin": "R1.1"},
    {"op": "add_net_label", "net": "in", "pin": "V1.+"},
    {"op": "add_net_label", "net": "in", "pin": "D1.A"},
    {"op": "add_net_label", "net": "0", "pin": "V1.-"},
    {"op": "add_net_label", "net": "0", "pin": "R1.2"},
    {"op": "add_directive", "instruction": ".model Dmod D(Is=1e-14)"},
    {"op": "add_directive", "instruction": ".tran 50m"},
)  # fmt: skip

_COMMON_SOURCE = (
    {"op": "add_component", "reference": "M1", "symbol": "nmos", "x": 400, "y": 300, "value": "NMOS_MOD"},
    {"op": "add_component", "reference": "RD", "symbol": "res", "x": 400, "y": 120, "value": "5k"},
    {"op": "add_component", "reference": "RS", "symbol": "res", "x": 400, "y": 500, "value": "500"},
    {"op": "add_component", "reference": "Vdd", "symbol": "voltage", "x": 200, "y": 120, "value": "5"},
    {"op": "add_component", "reference": "Vin", "symbol": "voltage", "x": 200, "y": 300, "value": "SINE(1.5 0.1 1k)"},
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
)  # fmt: skip

_VCVS_STAGE = (
    {"op": "add_component", "reference": "E1", "symbol": "e", "x": 400, "y": 300, "value": "2"},
    {"op": "add_component", "reference": "RL", "symbol": "res", "x": 400, "y": 220, "value": "10k"},
    {"op": "add_component", "reference": "Vin", "symbol": "voltage", "x": 200, "y": 340, "value": "AC 1"},
    # E1.+ (400,316) -> RL.2 (400,268): vertical, upward (clear of E1.- at 396).
    {"op": "wire_pins", "from_pin": "E1.+", "to_pin": "RL.2"},
    {"op": "add_net_label", "net": "out", "pin": "RL.1"},
    {"op": "add_net_label", "net": "0", "pin": "E1.-"},
    {"op": "add_net_label", "net": "in", "pin": "E1.P"},
    {"op": "add_net_label", "net": "0", "pin": "E1.N"},
    {"op": "add_net_label", "net": "in", "pin": "Vin.+"},
    {"op": "add_net_label", "net": "0", "pin": "Vin.-"},
    {"op": "add_directive", "instruction": ".ac dec 10 1 1meg"},
)  # fmt: skip

_VCCS_STAGE = (
    {"op": "add_component", "reference": "G1", "symbol": "g", "x": 400, "y": 200, "value": "1m"},
    {"op": "add_component", "reference": "RL", "symbol": "res", "x": 400, "y": 380, "value": "1k"},
    {"op": "add_component", "reference": "Vin", "symbol": "voltage", "x": 200, "y": 240, "value": "AC 1"},
    # G1.+ (400,296) -> RL.1 (400,332): vertical, downward (clear of G1.- at 216).
    {"op": "wire_pins", "from_pin": "G1.+", "to_pin": "RL.1"},
    {"op": "add_net_label", "net": "vdd", "pin": "G1.-"},
    {"op": "add_net_label", "net": "0", "pin": "RL.2"},
    {"op": "add_net_label", "net": "in", "pin": "G1.NC+"},
    {"op": "add_net_label", "net": "0", "pin": "G1.NC-"},
    {"op": "add_net_label", "net": "in", "pin": "Vin.+"},
    {"op": "add_net_label", "net": "0", "pin": "Vin.-"},
    {"op": "add_directive", "instruction": ".op"},
)  # fmt: skip


def _r2r_ladder_labelled(k: int = 25) -> tuple[dict[str, Any], ...]:
    """A k-node R-2R ladder: a vertical series spine (RS0..RS{k-1}) with a 2R
    shunt (RT0..RT{k-1}) dropped off each node, plus a reference source.

    Every junction is a real wire; only the output node, each shunt's ground
    end, and the source terminals carry labels.
    """
    x_spine, x_shunt = 400, 600
    ops: list[dict[str, Any]] = []
    for i in range(k):
        cy = 300 + i * 200
        ops.append(
            {"op": "add_component", "reference": f"RS{i}", "symbol": "res", "x": x_spine, "y": cy, "value": "1k"}
        )  # fmt: skip
        # Shunt placed so RT{i}.1 = (x_shunt, cy+48) is collinear with RS{i}.2.
        ops.append(
            {"op": "add_component", "reference": f"RT{i}", "symbol": "res", "x": x_shunt, "y": cy + 96, "value": "2k"}
        )  # fmt: skip
    ops.append(
        {"op": "add_component", "reference": "V1", "symbol": "voltage", "x": 200, "y": 252, "value": "5"}
    )  # fmt: skip
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
    return tuple(ops)


# ---------------------------------------------------------------------------
# Drawn builds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placed:
    """A part placed by a drawing, with where each of its pins went."""

    ref: str
    pins: Mapping[str, Point]

    def __getitem__(self, name: str) -> Point:
        return self.pins[name]


class Drawing:
    """The ops of one drawn build, written as a caller works them out today."""

    def __init__(self, name: str) -> None:
        self.reference = circuit(name)
        self.ops: list[dict[str, Any]] = []
        self.origins_from_pins = 0

    def part(
        self, ref: str, rotation: str, pin: str, at: Point, symbol: str | None = None
    ) -> Placed:
        """Place ``ref`` turned by ``rotation`` so that its pin ``pin`` lands on ``at``,
        drawn with ``symbol`` or the one `symbol_of` gives."""
        card = self.reference.card(ref)
        assert card is not None, ref
        symbol = symbol or symbol_of(card, self.reference)
        _order, px, py = _pins(symbol)[pin]
        dx, dy = place_point(px, py, 0, 0, rotation)
        x, y = at[0] - dx, at[1] - dy
        self.origins_from_pins += 1
        op: dict[str, Any] = {
            "op": "add_component",
            "reference": ref,
            "symbol": symbol,
            "x": x,
            "y": y,
            "value": card.value,
        }
        if rotation != "R0":
            op["rotation"] = rotation
        self.ops.append(op)
        placed = {
            name: place_point(lx, ly, x, y, rotation)
            for name, (_o, lx, ly) in _pins(symbol).items()
        }
        return Placed(ref, placed)

    def wire(self, start: str | Point, end: str | Point, *via: Point) -> None:
        """A wire from ``start`` to ``end`` (a ``REF.PIN`` or a point on wiring) through ``via``."""
        op: dict[str, Any] = {"op": "wire_pins", "from_pin": _end(start), "to_pin": _end(end)}
        if via:
            op["waypoints"] = [{"x": x, "y": y} for x, y in via]
        self.ops.append(op)

    def label(self, net: str, *pins: str) -> None:
        for pin in pins:
            self.ops.append({"op": "add_net_label", "net": net, "pin": pin})

    def ground(self, *pins: str) -> None:
        self.label("0", *pins)

    def directives(self, at: Point) -> None:
        """The reference's directives, one under another from ``at``."""
        x, y = at
        for number, text in enumerate(self.reference.directives):
            self.ops.append(
                {"op": "add_directive", "instruction": text, "x": x, "y": y + 32 * number}
            )

    def build(self, expected_findings: Mapping[str, int] | None = None) -> Build:
        return Build(
            self.reference.name,
            "drawn",
            (tuple(self.ops),),
            dict(expected_findings or {}),
            self.origins_from_pins,
        )


def _end(end: str | Point) -> str | dict[str, int]:
    return end if isinstance(end, str) else {"x": end[0], "y": end[1]}


def _rc_lowpass() -> Build:
    d = Drawing("rc_lowpass")
    v1 = d.part("V1", "R0", "+", (96, 96))
    r1 = d.part("R1", "R90", "2", (176, 96))
    c1 = d.part("C1", "R0", "1", (r1["1"][0] + 80, 96))
    d.wire("V1.+", "R1.2")
    d.wire("R1.1", "C1.1")
    d.label("in", "V1.+")
    d.label("out", "C1.1")
    d.ground("V1.-", "C1.2")
    d.directives((v1["-"][0], c1["2"][1] + 80))
    return d.build()


def _rectifier() -> Build:
    d = Drawing("rectifier")
    d.part("V1", "R0", "+", (96, 96))
    d1 = d.part("D1", "R270", "A", (176, 96))
    r1 = d.part("R1", "R0", "1", (d1["K"][0] + 80, 96))
    d.wire("V1.+", "D1.A")
    d.wire("D1.K", "R1.1")
    d.label("in", "V1.+")
    d.label("out", "R1.1")
    d.ground("V1.-", "R1.2")
    d.directives((96, r1["2"][1] + 80))
    return d.build()


def _common_source() -> Build:
    d = Drawing("common_source")
    m1 = d.part("M1", "R0", "G", (272, 320), symbol="nmos")
    d.part("RD", "R0", "2", (m1["D"][0], m1["D"][1] - 48))
    rs = d.part("RS", "R0", "1", (m1["S"][0], m1["S"][1] + 48))
    d.part("Vin", "R0", "+", (176, 320))
    d.part("Vdd", "R0", "+", (96, 80))
    d.wire("RD.2", "M1.D")
    d.wire("M1.S", "RS.1")
    d.wire("Vin.+", "M1.G")
    d.label("vdd", "Vdd.+", "RD.1")
    d.label("in", "Vin.+")
    d.label("out", "RD.2")
    d.ground("Vdd.-", "Vin.-", "RS.2")
    d.directives((96, rs["2"][1] + 80))
    return d.build({"label_island": 1})


def _vcvs_stage() -> Build:
    d = Drawing("vcvs_stage")
    e1 = d.part("E1", "R0", "P", (272, 192))
    d.part("Vin", "R0", "+", (176, 192))
    d.part("RL", "R90", "2", (e1["+"][0] + 80, e1["+"][1]))
    d.wire("Vin.+", "E1.P")
    d.wire("E1.+", "RL.2")
    d.label("in", "Vin.+")
    d.label("out", "RL.1")
    d.ground("Vin.-", "E1.-", "E1.N")
    d.directives((176, e1["-"][1] + 80))
    return d.build()


def _vccs_stage() -> Build:
    d = Drawing("vccs_stage")
    g1 = d.part("G1", "R0", "NC+", (272, 192))
    d.part("Vin", "R0", "+", (176, 192))
    rl = d.part("RL", "R0", "1", (g1["+"][0], g1["+"][1] + 64))
    d.wire("Vin.+", "G1.NC+")
    d.wire("G1.+", "RL.1")
    d.label("in", "Vin.+")
    d.label("vdd", "G1.-")
    d.ground("Vin.-", "G1.NC-", "RL.2")
    d.directives((176, rl["2"][1] + 80))
    return d.build()


def _nmos_mirror() -> Build:
    """The diode-connected side on the left, mirrored so the gates face each other."""
    d = Drawing("nmos_mirror")
    m1 = d.part("M1", "M0", "G", (256, 400))
    m2 = d.part("M2", "R0", "G", (352, 400))
    d.part("Iref", "R0", "Out", (m1["D"][0], m1["D"][1] - 80))
    d.part("RL", "R0", "2", (m2["D"][0], m2["D"][1] - 48))
    d.part("Vdd", "R0", "+", (96, 176))
    d.wire("M1.G", "M2.G")
    d.wire("Iref.Out", "M1.D")
    # The diode connection: from the drain wire to the gate wire, between the two.
    tap = (m1["D"][0], m1["D"][1] - 32)
    between = (m1["G"][0] + 48, m1["G"][1])
    d.wire(tap, between, (between[0], tap[1]))
    d.wire("RL.2", "M2.D")
    d.label("vdd", "Vdd.+", "Iref.In", "RL.1")
    d.label("ref", "Iref.Out")
    d.label("out", "RL.2")
    d.ground("Vdd.-", "M1.S", "M1.B", "M2.S", "M2.B")
    d.directives((96, m1["S"][1] + 96))
    return d.build({"label_island": 1})


def _diff_pair() -> Build:
    """The pair mirrored about the tail, inputs from either side."""
    d = Drawing("diff_pair")
    m1 = d.part("M1", "R0", "G", (256, 400))
    m2 = d.part("M2", "M0", "G", (448, 400))
    middle = (m1["S"][0] + m2["S"][0]) // 2
    tail = d.part("Itail", "R0", "In", (middle, m1["S"][1] + 64))
    d.part("R1", "R0", "2", (m1["D"][0], m1["D"][1] - 48))
    d.part("R2", "R0", "2", (m2["D"][0], m2["D"][1] - 48))
    d.part("Vinp", "R0", "+", (m1["G"][0] - 96, m1["G"][1]))
    d.part("Vinn", "R0", "+", (m2["G"][0] + 96, m2["G"][1]))
    d.part("Vdd", "R0", "+", (64, 176))
    d.wire("M1.S", "Itail.In", (m1["S"][0], tail["In"][1]))
    d.wire("M2.S", "Itail.In", (m2["S"][0], tail["In"][1]))
    d.wire("R1.2", "M1.D")
    d.wire("R2.2", "M2.D")
    d.wire("Vinp.+", "M1.G")
    d.wire("Vinn.+", "M2.G")
    d.label("vdd", "Vdd.+", "R1.1", "R2.1")
    d.label("inp", "Vinp.+")
    d.label("inn", "Vinn.+")
    d.label("outn", "R1.2")
    d.label("outp", "R2.2")
    d.label("tail", "Itail.In")
    d.ground("Vdd.-", "Vinp.-", "Vinn.-", "M1.B", "M2.B", "Itail.Out")
    d.directives((64, tail["Out"][1] + 64))
    return d.build({"label_island": 1})


def _r2r_ladder(k: int = 25) -> Build:
    """The spine left to right, each 2R shunt dropping to ground below its node."""
    d = Drawing("r2r_ladder")
    d.part("V1", "R0", "+", (96, 96))
    for i in range(k):
        rs = d.part(f"RS{i}", "R270", "1", (176 + 160 * i, 96))
        d.part(f"RT{i}", "R0", "1", (rs["2"][0], 176))
        d.wire("V1.+" if i == 0 else f"RS{i - 1}.2", f"RS{i}.1")
        d.wire(f"RT{i}.1", f"RS{i}.2")
    d.label("out", "V1.+")
    d.ground("V1.-", *(f"RT{i}.2" for i in range(k)))
    d.directives((96, 352))
    return d.build()


def _rlc_line(sections: int = 50, per_row: int = 10) -> Build:
    """Rows of sections left to right, each row carried on to the next by a label.

    In a section the inductor runs along the row and the capacitor and the
    resistor drop to ground from its far end.
    """
    d = Drawing("rlc_line")
    d.part("V1", "R0", "+", (96, 96))
    d.part("Rs", "R270", "1", (176, 96))
    d.wire("V1.+", "Rs.1")
    d.label("in", "V1.+")
    d.ground("V1.-")
    start = 352
    for i in range(sections):
        row, column = divmod(i, per_row)
        y = 96 + 352 * row
        x = start + 224 * column
        d.part(f"L{i}", "R270", "1", (x, y))
        if i == 0:
            d.wire("Rs.2", "L0.1")
        elif column == 0:
            # The row starts from the label that ended the one above. A wire
            # cannot end at a free point, so the label is on the pin.
            d.label(f"n{i}", f"L{i}.1")
        else:
            d.wire(f"L{i - 1}.2", f"L{i}.1")
    for i in range(sections):
        row, column = divmod(i, per_row)
        y = 96 + 352 * row
        x = start + 224 * column + 96
        d.part(f"C{i}", "R0", "1", (x, y + 64))
        d.part(f"R{i}", "R0", "1", (x + 64, y + 64))
        d.wire(f"C{i}.1", f"L{i}.2")
        if i == sections - 1:
            d.part("RL", "R0", "1", (x + 192, y + 64))
            d.wire(f"L{i}.2", "RL.1", (x + 192, y))
            d.ops.append({"op": "add_net_label", "net": "out", "x": x + 192, "y": y})
            d.ground("RL.2")
            d.wire(f"R{i}.1", (x + 64, y))
        elif column == per_row - 1:
            d.wire(f"R{i}.1", f"L{i}.2", (x + 64, y))
            d.label(f"n{i + 1}", f"L{i}.2")
        else:
            d.wire(f"R{i}.1", (x + 64, y))
        d.ground(f"C{i}.2", f"R{i}.2")
    d.directives((96, 96 + 352 * ((sections - 1) // per_row) + 256))
    return d.build()


def _sallen_key() -> Build:
    """The filter along one line into the amplifier, its feedback capacitor over the top."""
    d = Drawing("sallen_key")
    d.part("Vin", "R0", "+", (96, 256))
    r1 = d.part("R1", "R270", "1", (176, 256))
    r2 = d.part("R2", "R270", "1", (r1["2"][0] + 80, 256))
    e1 = d.part("E1", "R0", "P", (r2["2"][0] + 80, 256))
    node_a = (r1["2"][0] + 32, 256)
    node_b = (r2["2"][0] + 32, 256)
    c1 = d.part("C1", "R0", "2", (node_a[0], node_a[1] - 64))
    d.part("C2", "R0", "1", (node_b[0], node_b[1] + 64))
    out = (e1["+"][0] + 112, e1["+"][1])
    d.wire("Vin.+", "R1.1")
    d.wire("R1.2", "R2.1")
    d.wire("R2.2", "E1.P")
    d.wire("C1.2", node_a)
    d.wire("C2.1", node_b)
    # The feedback capacitor over the top and down to the output, whose last
    # leg is the output: a wire cannot end at a free point.
    top = c1["1"][1] - 32
    d.wire("C1.1", "E1.+", (c1["1"][0], top), (out[0], top), out)
    d.ops.append({"op": "add_net_label", "net": "out", "x": out[0], "y": out[1]})
    below = e1["-"][1] + 96
    d.wire("E1.N", (out[0] - 48, out[1]), (e1["N"][0], below), (out[0] - 48, below))
    d.label("in", "Vin.+")
    d.ground("Vin.-", "C2.2", "E1.-")
    d.directives((96, below + 80))
    return d.build()


def _two_stage_opamp() -> Build:
    """The usual drawing: the mirror load over the input pair over the tail, the
    bias diode at the left, the second stage at the right, vdd as a rail along
    the top, inputs as ports."""
    d = Drawing("two_stage_opamp")
    # The mirror load, gates facing each other.
    m3 = d.part("M3", "R180", "S", (320, 96))
    m4 = d.part("M4", "M180", "S", (448, 96))
    m7 = d.part("M7", "M180", "S", (640, 96))
    # The input pair under it, mirrored the same way.
    m1 = d.part("M1", "R0", "D", (320, 288))
    m2 = d.part("M2", "M0", "D", (448, 288))
    # The tail and the bias diode.
    m6 = d.part("M6", "R0", "D", (384, 448))
    m5 = d.part("M5", "M0", "G", (m6["G"][0] - 112, m6["G"][1]))
    d.part("Ibias", "R0", "Out", (m5["D"][0], m5["D"][1] - 64))
    d.part("M8", "R0", "D", (m7["D"][0], m6["D"][1]))
    # Compensation and load.
    cc = d.part("Cc", "R270", "1", (560, 336))
    cl = d.part("CL", "R0", "1", (736, 320))
    # The sources, apart at the left.
    d.part("Vdd", "R0", "+", (64, 96))
    d.part("Vinp", "R0", "+", (64, 256))
    d.part("Vinn", "R0", "+", (64, 416))

    rail = m3["S"][1] - 32
    d.wire("M3.S", "M4.S", (m3["S"][0], rail), (m4["S"][0], rail))
    d.wire("M7.S", (m4["S"][0], rail), (m7["S"][0], rail))
    d.ops.append({"op": "add_net_label", "net": "vdd", "x": m3["S"][0], "y": rail})
    d.wire("M3.G", "M4.G")
    d.wire("M3.D", "M1.D")
    d.wire("M4.D", "M2.D")
    # M3 as a diode: from its drain wire up to the gate wire, between the two.
    gap = (m3["G"][0] + m4["G"][0]) // 2
    tap = (m3["D"][0], m3["D"][1] + 48)
    d.wire(tap, (gap, m3["G"][1]), (gap, tap[1]))
    d.wire("M1.S", "M6.D", (m1["S"][0], m6["D"][1]))
    d.wire("M2.S", "M6.D", (m2["S"][0], m6["D"][1]))
    d.wire("M5.G", "M6.G")
    d.wire("Ibias.Out", "M5.D")
    knee = (m5["G"][0] + 48, m5["D"][1] - 32)
    d.wire((m5["D"][0], knee[1]), (knee[0], m5["G"][1]), knee)
    # The second stage, driven from the first stage's output.
    first_out = (m4["D"][0], m4["D"][1] + 48)
    turn = (cc["1"][0] - 32, first_out[1])
    d.wire(first_out, "M7.G", turn, (turn[0], m7["G"][1]))
    d.wire("M7.D", "M8.D")
    d.wire("Cc.1", turn, (turn[0], cc["1"][1]))
    d.wire("Cc.2", (m7["D"][0], cc["2"][1]))
    out = (cl["1"][0], m7["D"][1] + 80)
    d.wire((m7["D"][0], out[1]), "CL.1", out)
    d.ops.append({"op": "add_net_label", "net": "out", "x": out[0], "y": out[1]})
    d.label("bias", "M5.G", "M8.G")
    d.label("inn", "M1.G", "Vinn.+")
    d.label("inp", "M2.G", "Vinp.+")
    d.label("vdd", "Vdd.+", "Ibias.In", "M3.B", "M4.B", "M7.B")
    d.ground("Vdd.-", "Vinp.-", "Vinn.-", "CL.2")
    d.ground("M1.B", "M2.B", "M5.S", "M5.B", "M6.S", "M6.B", "M8.S", "M8.B")
    d.directives((64, m6["S"][1] + 96))
    # The inputs are ports, each joined to its source by a label alone.
    return d.build({"label_island": 2})


_DRAWN = {
    "r2r_ladder": _r2r_ladder,
    "rlc_line": _rlc_line,
    "sallen_key": _sallen_key,
    "two_stage_opamp": _two_stage_opamp,
    "rc_lowpass": _rc_lowpass,
    "rectifier": _rectifier,
    "common_source": _common_source,
    "vcvs_stage": _vcvs_stage,
    "vccs_stage": _vccs_stage,
    "nmos_mirror": _nmos_mirror,
    "diff_pair": _diff_pair,
}

_LABELLED_KEPT = {
    "rectifier": _RECTIFIER,
    "common_source": _COMMON_SOURCE,
    "vcvs_stage": _VCVS_STAGE,
    "vccs_stage": _VCCS_STAGE,
}


def builds() -> list[Build]:
    """Every build of the benchmark, by circuit and then style."""

    made: list[Build] = []
    for name in circuits():
        if name in _LABELLED_KEPT:
            made.append(Build(name, "labelled", (_LABELLED_KEPT[name],)))
        elif name == "r2r_ladder":
            made.append(Build(name, "labelled", (_r2r_ladder_labelled(),)))
        else:
            made.append(_generated_labelled(name))
        if name in _DRAWN:
            made.append(_DRAWN[name]())
    return made
