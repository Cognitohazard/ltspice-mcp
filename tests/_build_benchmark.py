"""The schematic build benchmark: reference circuits, builds of them, and their scores.

A circuit is a reference netlist under ``fixtures/build_benchmark``. A build is
a list of `edit_schematic` calls that draws one on a blank sheet. A build is
scored on three things, all read from the sheet it wrote and the replies it
was given:

- whether the sheet is the reference circuit (`equivalence`);
- what it cost the caller (`cost`);
- what the drawing is like (`drawing`): each sheet rule's count of findings,
  and measures the rules do not have. Those measures are this benchmark's own
  definitions, stated on `drawing`; they are not the readability facts the
  design names as a server feature.

A non-test helper module (leading underscore), so pytest does not collect it.
`scripts/build_benchmark.py` scores any sheet with the same functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.connectivity import GROUND, build_on_wire_predicate, signature
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.netlist_graph import canon_ref, parse_netlist_graph
from ltspice_mcp.lib.schematic_scene import scene_of_sheet, sheet_view
from ltspice_mcp.lib.sheet import Sheet, SheetPart
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests._schematic_fixtures import TESTS

CORPUS = TESTS / "fixtures" / "build_benchmark"

#: The grid LTspice snaps a drawing to, in sheet units.
GRID = 16

#: Both ends of a resistor, capacitor or inductor are one role: such a part
#: drawn either way round is the same circuit. `netlist_graph`'s comparison
#: reads them the same way (`_pin_role`).
_EITHER_WAY_ROUND = frozenset("RCL")

#: How many results each call asks to be listed, fixed so that a reply's size
#: does not depend on a default.
VIEW_LIMIT = 500


# ---------------------------------------------------------------------------
# The circuits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """One element of a reference circuit."""

    ref: str
    letter: str
    #: Node names, with ground's spellings folded to ``0`` and in lower case.
    nodes: tuple[str, ...]
    #: The card's text after its nodes, as written.
    value: str


@dataclass(frozen=True)
class Circuit:
    name: str
    title: str
    #: The nodes a drawing of it must name, from the ``* names:`` line.
    names: tuple[str, ...]
    cards: tuple[Card, ...]
    directives: tuple[str, ...]

    def card(self, ref: str) -> Card | None:
        key = canon_ref(ref)
        return next((c for c in self.cards if canon_ref(c.ref) == key), None)


def circuit(name: str) -> Circuit:
    """The reference circuit ``name``, read from ``fixtures/build_benchmark/<name>.cir``."""
    return read_circuit(CORPUS / f"{name}.cir")


def read_circuit(path: Path) -> Circuit:
    """The reference circuit in the netlist at ``path``."""
    lines = decode_spice_bytes(path.read_bytes()).splitlines()
    names: tuple[str, ...] = ()
    for line in lines:
        if line.startswith("* names:"):
            names = tuple(line.removeprefix("* names:").split())
    cards = []
    for component in parse_netlist_graph(path).components:
        tokens = lines[component.line - 1].split()
        assert canon_ref(tokens[0]) == canon_ref(component.ref), (path.name, component)
        cards.append(
            Card(
                component.ref,
                component.type_letter,
                component.nodes,
                " ".join(tokens[1 + len(component.nodes) :]),
            )
        )
    directives = tuple(
        line.strip()
        for line in lines
        if line.startswith(".") and line.split()[0].lower() not in (".end", ".title")
    )
    return Circuit(path.stem, lines[0].removeprefix("*").strip(), names, tuple(cards), directives)


def circuits() -> list[str]:
    return sorted(path.stem for path in CORPUS.glob("*.cir"))


# ---------------------------------------------------------------------------
# Running a build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Build:
    """One way of drawing a circuit: the calls, in order, each a batch of ops.

    ``expected_findings`` is every finding the drawing is meant to have, by
    rule; any other finding fails the build. ``None`` leaves the findings to
    the record alone.

    ``origins_from_pins`` is how many part origins the build worked out from
    where one of the part's pins should go: the pin's offset at the part's
    rotation looked up (`inspect(kind="symbol")` gives every rotation, for up
    to 64 symbols in one call) and subtracted from the point. The ops do not
    show it, since an origin is two numbers however it was found.
    """

    circuit: str
    style: str
    calls: tuple[tuple[Mapping[str, Any], ...], ...]
    expected_findings: Mapping[str, int] | None = None
    origins_from_pins: int = 0

    @property
    def id(self) -> str:
        return f"{self.circuit}_{self.style}"


@dataclass
class Built:
    """What running a build left: the sheet, each reply and the sheet's digest after each call."""

    build: Build
    path: Path
    replies: list[dict[str, Any]] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)


async def run_build(state: SessionState, build: Build, folder: Path) -> Built:
    """Draw ``build`` on a blank sheet in ``folder``, one `edit_schematic` call at a time."""
    built = Built(build, folder / f"{build.id}.asc")
    for number, ops in enumerate(build.calls):
        args: dict[str, Any] = {
            "target": str(built.path),
            "ops": [dict(op) for op in ops],
            "view_limit": VIEW_LIMIT,
        }
        if number == 0:
            args["base"] = "blank"
        else:
            args["expected_sha256"] = built.digests[-1]
        reply = await handle_edit_schematic(EditSchematicInput.model_validate(args), state)
        assert reply.structured_content is not None
        built.replies.append(reply.structured_content)
        data = built.path.read_bytes() if built.path.is_file() else b""
        built.digests.append(hashlib.sha256(data).hexdigest())
    return built


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------

_Pin = tuple[str, int]


def _pin(letter: str, ref: str, order: int) -> _Pin:
    return (canon_ref(ref), 0 if letter in _EITHER_WAY_ROUND else order)


def _pin_text(pin: _Pin) -> str:
    return f"{pin[0]}.{pin[1]}" if pin[1] else f"{pin[0]}.*"


def _net_text(pins: Iterable[_Pin]) -> list[str]:
    return sorted(_pin_text(pin) for pin in pins)


def _reference_nets(reference: Circuit) -> dict[str, tuple[_Pin, ...]]:
    nets: dict[str, list[_Pin]] = {}
    for card in reference.cards:
        for order, node in enumerate(card.nodes, start=1):
            nets.setdefault(node, []).append(_pin(card.letter, card.ref, order))
    return {node: tuple(sorted(pins)) for node, pins in nets.items()}


def netlisted(sheet: Sheet) -> list[tuple[frozenset[tuple[str, int]], frozenset[str]]]:
    """Each net the sheet draws, as LTspice would netlist it: its pins and its names.

    A pin is its part's reference, as `netlist_graph.canon_ref` spells it, and
    its SpiceOrder, the order a netlist lists a part's nodes in (the
    ``symbol-orientation`` recording). Names are in lower case, ground as ``0``
    (``connectivity.name_key``).

    A transistor drawn with three pins is exported with four nodes. A
    MOSFET's fourth is its source's net when the source is connected, and a
    net of its own when it is not; a bipolar transistor's is ground (the
    ``build-benchmark-export`` and ``export-boilerplate`` recordings, on the
    suite's ``nmos`` and LTspice's own symbols).
    """
    pins = []
    implied: list[tuple[str, str]] = []
    for part in sheet.parts:
        for pin in sorted(part.pins, key=lambda p: p.order):
            assert pin.order > 0, (part.reference, pin)
            pins.append(((pin.x, pin.y), (part.reference, str(pin.order))))
        letter = _prefix(part)[:1]
        if letter and letter in "MQ" and len(part.pins) == 3:
            implied.append((canon_ref(part.reference), letter))
    nets = [
        (
            {(canon_ref(ref), int(order)) for ref, order in owners},
            names,
        )
        for owners, names in signature(
            pins,
            [((x, y), name) for x, y, name in sheet.labels],
            [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in sheet.wires],
        )
    ]
    for ref, letter in implied:
        if letter == "Q":
            ground = next((net for net in nets if GROUND in net[1]), None)
            if ground is None:
                nets.append(({(ref, 4)}, frozenset({GROUND})))
            else:
                ground[0].add((ref, 4))
            continue
        source = next(net for net in nets if (ref, 3) in net[0])
        if source[0] == {(ref, 3)} and not source[1]:
            nets.append(({(ref, 4)}, frozenset()))
        else:
            source[0].add((ref, 4))
    return [(frozenset(pins), names) for pins, names in nets]


def _prefix(part: SheetPart) -> str:
    return (part.symbol.file.attr("Prefix") if part.symbol else "").upper()


def _drawn_nets(sheet: Sheet, reference: Circuit) -> list[tuple[tuple[_Pin, ...], frozenset[str]]]:
    """`netlisted`, with the two ends of a resistor, capacitor or inductor one
    role. A part the reference does not have keeps its symbol's prefix as its
    letter. A transistor's fourth node, which the export adds to one drawn with
    three pins, is left out where the reference's card writes three."""
    letters = {}
    implied = set()
    for part in sheet.parts:
        ref = canon_ref(part.reference)
        card = reference.card(part.reference)
        letters[ref] = card.letter if card else _prefix(part)[:1]
        if card is not None and len(card.nodes) == len(part.pins) == 3:
            implied.add((ref, 4))
    drawn = []
    for pins, names in netlisted(sheet):
        kept = [pin for pin in pins if pin not in implied]
        if kept:
            drawn.append((tuple(sorted(_pin(letters[r], r, o) for r, o in kept)), names))
    return drawn


def equivalence(sheet: Sheet, reference: Circuit) -> dict[str, Any]:
    """How the circuit ``sheet`` draws differs from ``reference``.

    Nets are compared by the pins on them, not by their names, except that
    the reference's ground is the sheet's net labelled ``0``. Separately: the
    nodes the reference says a drawing must name and the sheet does not, the
    parts whose value differs from the card's text after its nodes, and the
    directives the sheet lacks. ``equivalent`` is about parts and nets only.
    """
    expected = _reference_nets(reference)
    drawn = _drawn_nets(sheet, reference)
    on_sheet = {canon_ref(part.reference) for part in sheet.parts}
    in_reference = {canon_ref(card.ref) for card in reference.cards}
    nets_expected = Counter(expected.values())
    nets_drawn = Counter(pins for pins, _names in drawn)
    ground_drawn = next((pins for pins, names in drawn if GROUND in names), ())
    named = {name: pins for pins, names in drawn for name in names}
    values = []
    for part in sheet.parts:
        card = reference.card(part.reference)
        value = " ".join((part.attribute("Value") or "").split())
        if card is not None and value.casefold() != card.value.casefold():
            values.append(f"{part.reference}: {value!r} drawn, {card.value!r} in the reference")
    texts = {" ".join(t.body.split()).casefold() for t in sheet.document.texts if t.is_directive}
    said = {
        "missing_parts": sorted(in_reference - on_sheet),
        "extra_parts": sorted(on_sheet - in_reference),
        "nets_only_in_reference": sorted(
            _net_text(pins) for pins in (nets_expected - nets_drawn).elements()
        ),
        "nets_only_drawn": sorted(
            _net_text(pins) for pins in (nets_drawn - nets_expected).elements()
        ),
        "ground_differs": ground_drawn != expected.get(GROUND, ()),
        "names_missing": sorted(
            name for name in reference.names if named.get(name) != expected.get(name)
        ),
        "values_differ": sorted(values),
        "directives_missing": sorted(
            d for d in reference.directives if " ".join(d.split()).casefold() not in texts
        ),
    }
    said["equivalent"] = not any(
        said[key]
        for key in (
            "missing_parts",
            "extra_parts",
            "nets_only_in_reference",
            "nets_only_drawn",
            "ground_differs",
        )
    )
    return said


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def coordinates_written(op: Mapping[str, Any]) -> int:
    """How many numbers locating something on the sheet the caller wrote in ``op``.

    Two for each part origin, label or directive placed at a point, wire
    waypoint, wire end given as a point, point a removal names, and a move's
    offset; a wire's two segment ends and a region's two corners count four. A
    pin or a part given by name counts nothing.
    """
    count = 2 * len(op.get("waypoints", ()))
    count += sum(2 for key in ("from_pin", "to_pin") if isinstance(op.get(key), Mapping))
    count += sum(
        1 for key in ("x", "y", "x1", "y1", "x2", "y2", "dx", "dy") if op.get(key) is not None
    )
    count += 2 * len(op.get("labels", ())) + 4 * len(op.get("wires", ()))
    count += 4 if op.get("region") is not None else 0
    return count


def cost(built: Built, working_dir: Path) -> dict[str, Any]:
    """What the build cost the caller: calls, ops, refusals, coordinates, part
    origins worked out from a pin, reply size.

    ``reply_bytes`` is the size of every reply as JSON with the working
    directory replaced by one fixed name, so it does not depend on where the
    build ran.
    """
    ops = [op for call in built.build.calls for op in call]
    folder = json.dumps(str(working_dir))[1:-1]
    replies = [
        json.dumps(reply, sort_keys=True).replace(folder, "WORK") for reply in built.replies
    ]
    return {
        "calls": len(built.build.calls),
        "ops": len(ops),
        "refused": sum(
            len(reply.get("failures") or []) + (1 if reply.get("error") else 0)
            for reply in built.replies
        ),
        "coordinates": sum(coordinates_written(op) for op in ops),
        "origins_from_pins": built.build.origins_from_pins,
        "reply_bytes": sum(len(text.encode("utf-8")) for text in replies),
    }


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------

_Point = tuple[int, int]
_Segment = tuple[int, int, int, int]


def _through(point: _Point, segment: _Segment) -> int:
    """How many wire arms ``segment`` has at ``point``: one at an end, two inside it."""
    x, y = point
    x1, y1, x2, y2 = segment
    if (x, y) in ((x1, y1), (x2, y2)):
        return 1
    if (x2 - x1) * (y - y1) != (y2 - y1) * (x - x1):
        return 0
    inside = min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
    return 2 if inside else 0


def _crossing_points(segments: Sequence[_Segment]) -> set[_Point]:
    """Every point where a level wire and an upright one pass through each other."""
    level = [s for s in segments if s[1] == s[3] and s[0] != s[2]]
    upright = [s for s in segments if s[0] == s[2] and s[1] != s[3]]
    points = set()
    for x1, y, x2, _ in level:
        for x, y1, _, y2 in upright:
            if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2):
                points.add((x, y))
    return points


def drawing(sheet: Sheet) -> dict[str, Any]:
    """Measures of the drawing the sheet rules do not have. Each is a definition:

    - ``crossings``: points where two wires pass through each other, neither
      ending there, with no pin or label on the point. LTspice leaves such a
      point unjoined (the ``wire-connectivity`` recording).
    - ``four_way_junctions``: joined points with four or more wire arms. A
      wire counts one arm at its end and two through a point inside it.
    - ``bends``: points where exactly two wire ends meet out of line, with no
      pin or label there.
    - ``wire_length``: the wires' total length in grid steps of 16.
    - ``wires``: wire segments.
    - ``off_grid_points``: distinct part origins, wire ends and label points
      off the 16-unit grid.
    - ``label_only_pins`` and ``ground_only_pins``: pins on no wire with a
      label on them, as `edit_schematic`'s ``wiring`` counts them, apart from
      those whose only label is ground.
    - ``extent``: width and height, in grid steps, of the box around every
      part, wire and label.
    """
    segments = list(sheet.wires)
    pins = [(pin.x, pin.y) for part in sheet.parts for pin in part.pins]
    labels: dict[_Point, set[str]] = {}
    for x, y, name in sheet.labels:
        labels.setdefault((x, y), set()).add(name)
    occupied = set(pins) | set(labels)
    crossings = {p for p in _crossing_points(segments) if p not in occupied}
    ends = Counter(p for x1, y1, x2, y2 in segments for p in ((x1, y1), (x2, y2)))
    four_way = 0
    for point in set(ends) | _crossing_points(segments):
        arms = sum(_through(point, s) for s in segments)
        if arms >= 4 and point not in crossings:
            four_way += 1
    bends = 0
    for point, count in ends.items():
        if count != 2 or point in occupied:
            continue
        meeting = [s for s in segments if _through(point, s)]
        if len(meeting) != 2:
            continue
        (a1, b1, a2, b2), (c1, d1, c2, d2) = meeting
        if (a2 - a1) * (d2 - d1) != (b2 - b1) * (c2 - c1):
            bends += 1
    length = sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segments)
    points = {part.at for part in sheet.parts} | set(ends) | set(labels)
    on_wire = build_on_wire_predicate([((x1, y1), (x2, y2)) for x1, y1, x2, y2 in segments])
    label_only = [p for p in pins if not on_wire(p) and p in labels]
    xs: list[int] = []
    ys: list[int] = []
    for part in sheet.parts:
        if part.box is not None:
            xs += [part.box.x1, part.box.x2]
            ys += [part.box.y1, part.box.y2]
    for x, y in set(ends) | set(labels):
        xs.append(x)
        ys.append(y)
    return {
        "crossings": len(crossings),
        "four_way_junctions": four_way,
        "bends": bends,
        "wire_length": round(length / GRID, 1),
        "wires": len(segments),
        "off_grid_points": sum(1 for x, y in points if x % GRID or y % GRID),
        "label_only_pins": sum(1 for p in label_only if labels[p] != {GROUND}),
        "ground_only_pins": sum(1 for p in label_only if labels[p] == {GROUND}),
        "extent": [
            round((max(xs) - min(xs)) / GRID, 1) if xs else 0,
            round((max(ys) - min(ys)) / GRID, 1) if ys else 0,
        ],
    }


def finding_counts(sheet: Sheet, path: Path) -> dict[str, int]:
    """How many findings each sheet rule has of ``sheet``, the rules with none left out."""
    view = sheet_view(scene_of_sheet(sheet, path))
    return dict(sorted(Counter(f.rule for f in findings(view)).items()))
