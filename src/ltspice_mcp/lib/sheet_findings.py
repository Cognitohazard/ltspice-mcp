"""What can be said of a sheet: the rules, and one type for what they find.

A check on a sheet is a rule. ``RULES`` is the registry: for each rule its id,
what kind of thing it says, what it is about, where the claim comes from, what
a finding of it is called on a whole sheet, and which of ``verify_circuit``'s
checks reports it. ``findings`` runs them and returns :class:`Finding` values,
the same ones whichever tool asked: ``edit_schematic`` reads them off the sheet
it is about to write and ``verify_circuit`` off the file it was given, and each
words them into the rows its own reply carries.

The rules are in three families, kept apart because they are different kinds
of statement:

- *electrical*: the sheet will netlist differently from how it is drawn. The
  provenance of such a rule is the LTspice recording that shows the behaviour
  (``docs/TESTING.md``, "Recorded LTspice behaviour").
- *structural*: something is left undone, such as a pin or a label on nothing.
- *drawing*: how the sheet reads, such as one part's box over another's.

None of them is a verdict on the sheet. A finding says what is there and where.

A finding's sentence stands alone, because an edit's reply shows the sentence
and nothing else; and everything the sentence says is also in the finding's
parts, points and facts, so those three with the rule are what makes two
findings the same finding (:attr:`Finding.identity`).

The checks read a :class:`SheetView`, a plain picture of a sheet. A part whose
symbol was not found is in it, and is reported as that; nothing is said about
its pins or its extent, which are not known.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ltspice_mcp.lib.connectivity import (
    NetPartition,
    build_on_wire_predicate,
    build_wire_counter,
    crossing_points,
    is_ground,
    label_folded_nets,
    name_key,
    partition,
    point_on_segment,
    same_instance_dropped_segments,
)
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.spice_lex import copies_by_name, key_value_tokens

Point = tuple[int, int]
Family = Literal["electrical", "structural", "drawing"]
Scope = Literal["point", "wire", "part", "net", "sheet"]
Check = Literal["symbols", "layout", "quality", "export"]


@dataclass(frozen=True)
class Rule:
    """One thing that can be said of a sheet.

    ``provenance`` is the recorded behaviour an electrical rule rests on (a key
    of the recording inventory), or ``"definition"`` for a rule that says what
    is drawn and claims nothing about LTspice. ``severity`` is what a finding
    of the rule is called on a whole sheet, and ``check`` the ``verify_circuit``
    check that reports it. ``edit_schematic`` reports every rule.
    """

    rule_id: str
    family: Family
    scope: Scope
    summary: str
    check: Check
    provenance: str = "definition"
    severity: Literal["observation", "warning", "error"] = "observation"


_RULES = (
    Rule(
        "unresolved_symbol",
        "structural",
        "part",
        "A part whose symbol is not found, or has a pin line that does not read, "
        "so that it has no pins.",
        check="symbols",
        severity="error",
    ),
    Rule(
        "dropped_wire",
        "electrical",
        "wire",
        "A wire straight between two pins of one part, which LTspice leaves out of the netlist.",
        check="export",
        provenance="same-instance-wire",
        severity="warning",
    ),
    Rule(
        "duplicate_parameter",
        "electrical",
        "part",
        "A part that gives one parameter more than once across Value, Value2, "
        "SpiceLine and SpiceLine2, which LTspice writes onto one netlist line.",
        check="quality",
        provenance="export-attributes",
    ),
    Rule(
        "subcircuit_value_slot",
        "electrical",
        "part",
        "A Value on a subcircuit call that LTspice writes last on its line, where the "
        "subcircuit is named: an opamp's complexity level (Level.2), or any Value on a "
        "part whose model its SpiceModel attribute selects.",
        check="quality",
        provenance="export-attributes",
        severity="warning",
    ),
    Rule(
        "net_with_two_names",
        "electrical",
        "net",
        "A net carrying labels of two or more names; LTspice gives its node one of them.",
        check="export",
        provenance="net-naming",
    ),
    Rule(
        "joined_at_crossing",
        "electrical",
        "point",
        "A pin or label where wires cross, which joins nets that would be apart without it.",
        check="layout",
        provenance="wire-connectivity",
    ),
    Rule(
        "unjoined_crossing",
        "electrical",
        "point",
        "Two wires crossing where neither ends and nothing sits, which LTspice leaves apart.",
        check="layout",
        provenance="wire-connectivity",
    ),
    Rule(
        "diagonal_wire",
        "electrical",
        "wire",
        "A wire neither horizontal nor vertical, which LTspice joins anywhere along it.",
        check="layout",
        provenance="wire-connectivity",
    ),
    Rule(
        "floating_pin",
        "structural",
        "point",
        "A pin with no wire, label or other pin on it.",
        check="layout",
    ),
    Rule(
        "dangling_wire_end",
        "structural",
        "point",
        "A wire end on no pin, label or other wire.",
        check="layout",
    ),
    Rule(
        "dangling_label",
        "structural",
        "point",
        "A net label on no wire and no pin.",
        check="layout",
    ),
    Rule(
        "duplicate_wire",
        "structural",
        "wire",
        "One wire drawn more than once between the same two points.",
        check="layout",
    ),
    Rule(
        "symbol_overlap",
        "drawing",
        "part",
        "Two parts whose boxes share an area.",
        check="layout",
    ),
    Rule(
        "wire_through_symbol",
        "drawing",
        "part",
        "A wire passing through what a part draws, other than out from a pin of "
        "the part that lies inside it.",
        check="layout",
    ),
    Rule(
        "attribute_text_overlap",
        "drawing",
        "part",
        "A part's attribute text drawn across a wire, another part's body, or other text.",
        check="layout",
    ),
    Rule(
        "label_over_component",
        "drawing",
        "part",
        "A net label placed inside a part's box and on no pin.",
        check="quality",
    ),
    Rule(
        "text_in_symbol_body",
        "drawing",
        "part",
        "Text anchored inside what another part draws.",
        check="quality",
    ),
    Rule(
        "stacked_directive",
        "drawing",
        "point",
        "Two or more directives or comments at one anchor.",
        check="quality",
    ),
    Rule(
        "label_island",
        "drawing",
        "net",
        "A net joined only by labels of one name, with no wire on any of them.",
        check="quality",
    ),
)

#: Every rule made of a whole sheet, in the order ``findings`` lists them: what
#: changes the circuit or leaves it undone first, how the sheet reads after.
RULES: Mapping[str, Rule] = {rule.rule_id: rule for rule in _RULES}


@dataclass(frozen=True)
class Finding:
    """What one rule found at one place.

    ``refs`` are the parts involved and ``points`` the places to look, both in
    the order the rule gives them. ``facts`` are the rule's own named values
    (a pin, a label's text, a count). ``detail`` is one sentence saying all of
    it, for a reader who is shown nothing else.
    """

    rule: str
    detail: str
    refs: tuple[str, ...] = ()
    points: tuple[Point, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    #: What the finding is about in place of its facts, where they hold an
    #: attribute's words: so a part given a new value keeps the finding it
    #: had. Said by the other fields already, so two findings equal without it.
    about: tuple[Any, ...] | None = field(default=None, compare=False)

    @property
    def identity(self) -> tuple[Any, ...]:
        """What makes this the same finding as another, before and after an
        edit: its rule, its parts, its places and its facts. A rule whose
        findings move with the parts they name (``_finds(placeless=True)``)
        leaves the places out, as does a finding with an ``about``."""
        what = self.about if self.about is not None else tuple(sorted(self.facts.items()))
        placeless = self.about is not None or self.rule in _PLACELESS
        return (self.rule, self.refs, () if placeless else self.points, what)


@dataclass(frozen=True)
class Part:
    """A placed part as a check sees it.

    ``box`` is the part's extent with its pins, the box ``inspect`` and
    ``add_component`` report, and ``body`` the extent of what it draws alone;
    either is ``None`` when the part has none. ``at`` is where the part is
    placed. ``pins`` are ``(name, x, y)`` in SpiceOrder, the order a symbol
    gives them in, and ``texts`` the anchors of its drawn attributes as
    ``(x, y, first line)``, and ``text_extents`` where each is drawn, as
    ``(x, y, first line, extent)``. ``strokes`` are the lines it draws, as segments,
    and ``areas`` the rectangles and circles it draws, as boxes. ``missing``
    says its symbol was not found: whatever box it carries is then a
    placeholder's, and no rule reads it.
    """

    ref: str
    symbol: str = ""
    box: BBox | None = None
    pins: tuple[tuple[str, int, int], ...] = ()
    texts: tuple[tuple[int, int, str], ...] = ()
    text_extents: tuple[tuple[int, int, str, BBox], ...] = ()
    strokes: tuple[tuple[Point, Point], ...] = ()
    areas: tuple[BBox, ...] = ()
    missing: bool = False
    body: BBox | None = None
    at: Point | None = None
    #: Why a symbol that was found cannot be used, as words that follow its
    #: name (``FoundSymbol.problem``); the part is then ``missing`` too.
    problem: str = ""
    #: The attributes whose text LTspice writes onto the part's netlist line,
    #: as ``(name, text)`` in that order (``sheet.PARAMETER_ATTRIBUTES``).
    parameters: tuple[tuple[str, str], ...] = ()
    #: The element letter the part netlists as (``sheet.SheetPart.element``),
    #: and the model its ``SpiceModel`` attribute selects, if any.
    element: str = ""
    spice_model: str = ""

    @property
    def name(self) -> str:
        """What to call the part in a finding: its reference, else its symbol."""
        return self.ref or self.symbol or "<unnamed>"

    @property
    def draws_known(self) -> bool:
        """Whether the lines, rectangles or circles it draws are known; a part
        drawn without them counts as drawing its whole body."""
        return bool(self.strokes or self.areas)

    @property
    def drawn(self) -> BBox | None:
        """The extent of what the part draws: its body where known, else its box."""
        return self.body if self.body is not None else self.box


@dataclass(frozen=True)
class SheetView:
    """A sheet as the checks read it, in file order.

    ``wires`` are ``(x1, y1, x2, y2)``, ``labels`` are ``(x, y, name)`` with
    ground among them, and ``texts`` are the sheet's directives and comments as
    ``(x, y, first line)``. ``text_extents`` are the labels', directives' and
    comments' text as drawn, as ``(x, y, first line, extent)``.
    """

    parts: tuple[Part, ...] = ()
    wires: tuple[tuple[int, int, int, int], ...] = ()
    labels: tuple[tuple[int, int, str], ...] = ()
    texts: tuple[tuple[int, int, str], ...] = ()
    text_extents: tuple[tuple[int, int, str, BBox], ...] = ()


class _Index:
    """What several rules ask of one view, worked out once."""

    def __init__(self, view: SheetView) -> None:
        self.view = view
        every = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in view.wires]
        #: The wires that have a length. One of no length has no span to cross
        #: anything, no end to dangle, and connects nothing.
        self.wires = [(a, b) for a, b in every if a != b]
        self.on_a_wire = build_on_wire_predicate(self.wires)
        self.wires_through = build_wire_counter(self.wires)
        #: As ``on_a_wire``, counting the point of a wire of no length too.
        self.on_any_wire = build_on_wire_predicate(every)
        self.pins_at = {(x, y) for part in view.parts for _name, x, y in part.pins}
        self.labelled = {(x, y) for x, y, _text in view.labels}
        #: The parts whose symbols were found, by their place in the view: the
        #: only ones whose extent is known.
        found = [(index, part) for index, part in enumerate(view.parts) if not part.missing]
        #: Each of those with a box, and each with something drawn.
        self.boxed = [(index, part, part.box) for index, part in found if part.box is not None]
        self.drawn = [
            (index, part, drawn) for index, part in found if (drawn := part.drawn) is not None
        ]
        #: Every point a wire ends at.
        self.wire_ends = {end for wire in self.wires for end in wire}


def _wired_nets(view: SheetView, without: Collection[Point] = ()) -> NetPartition:
    """The sheet's nets as its wiring joins them (``connectivity.partition``),
    leaving out the pins and labels on the points in ``without``."""
    return partition(
        (
            ((x, y), (part.name, name))
            for part in view.parts
            for name, x, y in part.pins
            if (x, y) not in without
        ),
        (((x, y), text) for x, y, text in view.labels if (x, y) not in without),
        [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in view.wires],
    )


def _net_name(labels: list[tuple[Point, str]]) -> str:
    """Of labels whose names differ only in case, the spelling LTspice gives
    the node: the label lowest on the sheet, and of those the one furthest
    right (the net-naming recordings)."""
    return max(labels, key=lambda label: (label[0][1], label[0][0], label[1]))[1]


_Finder = Callable[[_Index], list[Finding]]
_FINDERS: dict[str, _Finder] = {}


#: The rules whose findings move with the parts they name.
_PLACELESS: set[str] = set()


def _finds(rule_id: str, *, placeless: bool = False) -> Callable[[_Finder], _Finder]:
    """Register the function that finds ``rule_id``. A ``placeless`` rule's
    findings move with the parts they name, so a part moved keeps the finding
    it had (``Finding.identity``)."""

    def register(finder: _Finder) -> _Finder:
        _FINDERS[rule_id] = finder
        if placeless:
            _PLACELESS.add(rule_id)
        return finder

    return register


def _at(point: Point) -> str:
    return f"({point[0]},{point[1]})"


def _wire(a: Point, b: Point) -> str:
    return f"{_at(a)}->{_at(b)}"


@_finds("unresolved_symbol", placeless=True)
def _parts_without_a_symbol(ix: _Index) -> list[Finding]:
    found: list[Finding] = []
    for part in ix.view.parts:
        if not part.missing:
            continue
        facts: dict[str, Any] = {"symbol": part.symbol}
        why = "was not found"
        if part.problem:
            facts["problem"] = why = part.problem
        found.append(
            Finding(
                "unresolved_symbol",
                f"Symbol '{part.symbol}' of {part.name} {why}: the part has no pins "
                "and no extent here, so nothing about them is checked",
                refs=(part.name,),
                points=(part.at,) if part.at is not None else (),
                facts=facts,
            )
        )
    return found


@_finds("dropped_wire")
def _dropped_wires(ix: _Index) -> list[Finding]:
    # LTspice leaves out a run whose two ends both land on pins of one part,
    # with nothing on it between them: the pins stay on separate nodes, so the
    # sheet shows a tie the netlist does not have. A label or a ground flag on
    # the run keeps it. A part is told from another by its place in the view,
    # since two parts may carry one reference, or none.
    owners: dict[Point, list[tuple[str, str]]] = {}
    for index, part in enumerate(ix.view.parts):
        for _name, x, y in part.pins:
            owners.setdefault((x, y), []).append((str(index), ""))
    found: list[Finding] = []
    for drop in same_instance_dropped_segments(owners, list(ix.view.wires), ix.labelled):
        x1, y1, x2, y2 = drop["segment"]
        name = ix.view.parts[int(drop["ref"])].name
        found.append(
            Finding(
                "dropped_wire",
                f"Wire {_wire((x1, y1), (x2, y2))} joins two pins of the same instance "
                f"{name} and is not exported",
                refs=(name,),
                points=((x1, y1), (x2, y2)),
            )
        )
    return found


@_finds("duplicate_parameter", placeless=True)
def _duplicate_parameters(ix: _Index) -> list[Finding]:
    found: list[Finding] = []
    for part in ix.view.parts:
        copies = copies_by_name(
            (tok.key or "", f"{tok.key}={tok.value} in {attribute}")
            for attribute, text in part.parameters
            for tok in key_value_tokens(text)
        )
        for key, given in copies.items():
            if len(given) < 2:
                continue
            found.append(
                Finding(
                    "duplicate_parameter",
                    f"{part.name} gives {key} {len(given)} times: {'; '.join(given)}. "
                    "LTspice writes every copy onto the part's line, and which one it "
                    "reads is not known here",
                    refs=(part.name,),
                    points=(part.at,) if part.at is not None else (),
                    facts={"parameter": key, "copies": tuple(given)},
                )
            )
    return found


def _floating(ix: _Index) -> list[tuple[Part, tuple[str, int, int]]]:
    # For each point, the last pin each part has there.
    last: dict[Point, dict[int, int]] = {}
    for index, part in enumerate(ix.view.parts):
        for position, (_name, x, y) in enumerate(part.pins):
            last.setdefault((x, y), {})[index] = position
    floating: list[tuple[Part, tuple[str, int, int]]] = []
    for index, part in enumerate(ix.view.parts):
        for position, pin in enumerate(part.pins):
            coord = (pin[1], pin[2])
            if coord in ix.labelled or ix.on_a_wire(coord):
                continue
            here = last[coord]
            if len(here) > 1 and here[index] == position:
                continue
            floating.append((part, pin))
    return floating


def floating_pins(view: SheetView) -> list[tuple[Part, tuple[str, int, int]]]:
    """Each pin connected to nothing, with its part, in the sheet's own order.

    A pin is connected by a label on its point, by a wire that touches its
    point anywhere along the wire's length, or by another part's pin on its
    point. It is not connected by another pin of its own part. And where a part
    has several pins on one point with no wire or label there, only the last of
    them in SpiceOrder touches another part's pin; the rest are on nothing, even
    with that other pin on the same point. Both builds were recorded doing this
    (the ``pins-of-one-part-on-one-point`` cases).

    A wire of no length connects nothing.
    """
    return _floating(_Index(view))


#: An opamp's complexity level as LTspice's symbol dialog names it.
_LEVEL_LABEL = re.compile(r"\s*level\.\d+\s*", re.IGNORECASE)


@_finds("subcircuit_value_slot")
def _subcircuit_value_slots(ix: _Index) -> list[Finding]:
    # A part that netlists as a subcircuit call (element letter X) writes its
    # Value last, after its nodes and any SpiceModel (export-attributes). A
    # library part that names its subcircuit in Value, with no SpiceModel, is
    # the ordinary case and is not said; on a transistor or diode a Value after
    # its SpiceModel is an area factor.
    found: list[Finding] = []
    for part in ix.view.parts:
        value = dict(part.parameters).get("Value", "")
        if part.element != "X" or not value.strip():
            continue
        if _LEVEL_LABEL.fullmatch(value):
            detail = (
                f"{part.name}: Value {value!r} is an opamp's complexity-level label, but "
                "LTspice writes a subcircuit call's Value last on its X line, where the "
                f"subcircuit it calls is named, so it calls one named {value!r}. Select "
                "the level through the SpiceModel attribute (or the symbol's default)"
            )
        elif part.spice_model:
            detail = (
                f"{part.name}: its model is selected by its SpiceModel attribute "
                f"({part.spice_model!r}), and LTspice writes Value {value!r} after it on "
                f"the X line, so {value!r} is named as the subcircuit and "
                f"{part.spice_model!r} as one more node. Clear the Value"
            )
        else:
            continue
        found.append(
            Finding("subcircuit_value_slot", detail, refs=(part.name,), facts={"value": value})
        )
    return found


@_finds("net_with_two_names")
def _nets_with_two_names(ix: _Index) -> list[Finding]:
    # One node per label-folded net, names compared as LTspice compares them,
    # ground no name. Each name is given in the spelling LTspice gives it.
    named = [
        ((x, y), name)
        for x, y, text in ix.view.labels
        if (name := text.strip()) and not is_ground(name)
    ]
    if len({name_key(name) for _point, name in named}) < 2:
        return []
    net_of = label_folded_nets(_wired_nets(ix.view))
    nets: dict[Point, dict[str, list[tuple[Point, str]]]] = {}
    for point, name in named:
        nets.setdefault(net_of(point), {}).setdefault(name_key(name), []).append((point, name))
    found: list[Finding] = []
    for by_key in nets.values():
        if len(by_key) < 2:
            continue
        names = tuple(sorted(_net_name(labels) for labels in by_key.values()))
        points = tuple(sorted(point for labels in by_key.values() for point, _ in labels))
        found.append(
            Finding(
                "net_with_two_names",
                f"One net carries labels {list(names)}; LTspice gives the node one of these "
                "names, so a directive or probe that uses another may not find it",
                points=points,
                facts={"names": names},
            )
        )
    return found


@_finds("joined_at_crossing")
def _joined_at_crossings(ix: _Index) -> list[Finding]:
    # Wires crossing where neither ends are apart; a pin or a label on the
    # crossing joins them (pin_at_crossing, label_at_crossing). Said only where
    # what sits there is what joins them: without the pins and labels on such
    # crossings, the wires through each would be on separate nets, nets that
    # share a name being one.
    on: dict[Point, list[str]] = {}
    refs: dict[Point, dict[str, None]] = {}
    for part in ix.view.parts:
        for name, x, y in part.pins:
            on.setdefault((x, y), []).append(f"{part.name}.{name}")
            refs.setdefault((x, y), {})[part.name] = None
    for x, y, text in ix.view.labels:
        on.setdefault((x, y), []).append(f"label '{text.strip()}'")
    # A point no wire ends at lies away from the ends of every wire through it.
    crossings = {
        point: [(a, b) for a, b in ix.wires if point_on_segment(point, a, b)]
        for point in on
        if point not in ix.wire_ends and ix.wires_through(point) > 1
    }
    if not crossings:
        return []
    net_of = label_folded_nets(_wired_nets(ix.view, without=crossings))
    found: list[Finding] = []
    for point, through in sorted(crossings.items()):
        nets = {net_of(a) for a, _b in through}
        if len(nets) < 2:
            continue
        what = ", ".join(on[point])
        found.append(
            Finding(
                "joined_at_crossing",
                f"{what} at {_at(point)} sits where {len(through)} wires cross, and joins "
                f"{len(nets)} nets there that are apart without it",
                refs=tuple(refs.get(point, ())),
                points=(point,),
                facts={"wires": len(through), "nets": len(nets)},
            )
        )
    return found


@_finds("unjoined_crossing")
def _unjoined_crossings(ix: _Index) -> list[Finding]:
    # Horizontal and vertical wires through each other, at a point that is no
    # wire's end and has no pin or label on it.
    return [
        Finding(
            "unjoined_crossing",
            f"Wires cross at {_at(point)} without joining: neither ends there and "
            "nothing sits on the crossing",
            points=(point,),
        )
        for point in sorted(crossing_points(ix.wires))
        if point not in ix.wire_ends and point not in ix.pins_at and point not in ix.labelled
    ]


@_finds("diagonal_wire")
def _diagonal_wires(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "diagonal_wire",
            f"Wire {_wire(a, b)} is diagonal; LTspice joins anything on it, along its "
            "length as at its ends",
            points=(a, b),
        )
        for a, b in ix.wires
        if a[0] != b[0] and a[1] != b[1]
    ]


@_finds("floating_pin", placeless=True)
def _floating_pins(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "floating_pin",
            f"Floating pin: {f'{part.name}.{name}' if name else part.name} at {_at((x, y))}",
            refs=(part.name,),
            points=((x, y),),
            facts={"pin": name},
        )
        for part, (name, x, y) in _floating(ix)
    ]


@_finds("dangling_wire_end")
def _dangling_wire_ends(ix: _Index) -> list[Finding]:
    # Once per coordinate: two loose ends meeting nothing at one point are one
    # place to look, not two findings.
    seen: set[Point] = set()
    found: list[Finding] = []
    for a, b in ix.wires:
        for end in (a, b):
            if end in seen or end in ix.pins_at or end in ix.labelled:
                continue
            # Its own wire is one; a second is another wire touching it there.
            if ix.wires_through(end) > 1:
                continue
            seen.add(end)
            found.append(
                Finding(
                    "dangling_wire_end",
                    f"Wire end at {_at(end)} meets no pin, net label, or other wire",
                    points=(end,),
                )
            )
    return found


@_finds("dangling_label")
def _dangling_labels(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "dangling_label",
            f"Dangling label '{text}' at {_at((x, y))}",
            points=((x, y),),
            facts={"label": text},
        )
        for x, y, text in ix.view.labels
        if (x, y) not in ix.pins_at and not ix.on_any_wire((x, y))
    ]


@_finds("duplicate_wire")
def _duplicate_wires(ix: _Index) -> list[Finding]:
    drawn: dict[tuple[Point, Point], int] = {}
    for v1, v2 in ix.wires:
        key = (v1, v2) if v1 <= v2 else (v2, v1)
        drawn[key] = drawn.get(key, 0) + 1
    return [
        Finding(
            "duplicate_wire",
            f"Duplicate wire ({count}×): {_wire(a, b)}",
            points=(a, b),
            facts={"count": count},
        )
        for (a, b), count in drawn.items()
        if count > 1
    ]


@_finds("symbol_overlap", placeless=True)
def _symbol_overlaps(ix: _Index) -> list[Finding]:
    found: list[Finding] = []
    for position, (_index, part_a, box_a) in enumerate(ix.boxed):
        for _other, part_b, box_b in ix.boxed[position + 1 :]:
            if not box_a.overlaps(box_b):
                continue
            ox1, oy1 = max(box_a.x1, box_b.x1), max(box_a.y1, box_b.y1)
            ox2, oy2 = min(box_a.x2, box_b.x2), min(box_a.y2, box_b.y2)
            found.append(
                Finding(
                    "symbol_overlap",
                    f"{part_a.name} and {part_b.name}: bounding boxes share a "
                    f"{ox2 - ox1}x{oy2 - oy1} region at {_at((ox1, oy1))}",
                    refs=(part_a.name, part_b.name),
                    points=((ox1, oy1), (ox2, oy2)),
                )
            )
    return found


def _through(a: Point, b: Point, box: BBox) -> bool:
    """True if segment ``a``-``b`` passes through ``box``'s interior.

    Parametric (Liang-Barsky) clip, then a strict containment test on the
    midpoint of the clipped span. Both steps matter: a segment that only
    terminates on the boundary clips to zero length, and one that runs *along*
    an edge clips to a span whose midpoint is on the boundary, not inside. So
    neither the normal way a wire meets a pin nor a wire tracking an edge counts
    as crossing. A degenerate (zero-length) segment never crosses.
    """
    (x1, y1), (x2, y2) = a, b
    # A segment whose own extent misses the interior cannot pass through it.
    if not (
        min(x1, x2) < box.x2
        and max(x1, x2) > box.x1
        and min(y1, y2) < box.y2
        and max(y1, y2) > box.y1
    ):
        return False
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return False
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - box.x1), (dx, box.x2 - x1), (-dy, y1 - box.y1), (dy, box.y2 - y1)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    if t1 <= t0:
        return False
    tm = (t0 + t1) / 2
    return box.strictly_contains(x1 + tm * dx, y1 + tm * dy)


def _way_out_from_own_pin(a: Point, b: Point, part: Part, box: BBox) -> bool:
    """True if wire ``a``-``b`` is the way out of ``box`` from a pin of ``part`` inside it.

    A pin can lie inside the box of what its part draws: a frame drawn a unit
    outside its column of pins, a pin on the slanted side of a triangle, a box
    made larger by something drawn apart. Any wire to such a pin has a stretch
    inside the box, and that stretch is the connection, not a wire through the
    part, when the wire ends on the pin and runs level or upright from it out
    of the box by the nearer of the two edges on its line. Out by the farther
    edge is the wire running back across the body. A wire that ends inside the
    box has not left it, and a diagonal one has no line of its own to judge by;
    both stay reported.

    Which edge counts is from LTspice's own example sheets, where this
    separates every wire drawn out of a body from such a pin, some 1,300 of
    them, from the wires that end inside one.
    """
    own = {(x, y) for _name, x, y in part.pins}
    for (ex, ey), (ox, oy) in ((a, b), (b, a)):
        if (ex, ey) not in own or not box.strictly_contains(ex, ey):
            continue
        if box.strictly_contains(ox, oy):
            continue
        if oy == ey:
            ahead, behind = ex - box.x1, box.x2 - ex
            if ox > ex:
                ahead, behind = behind, ahead
        elif ox == ex:
            ahead, behind = ey - box.y1, box.y2 - ey
            if oy > ey:
                ahead, behind = behind, ahead
        else:
            continue
        if ahead <= behind:
            return True
    return False


@_finds("wire_through_symbol")
def _wires_through_symbols(ix: _Index) -> list[Finding]:
    # A wire attached to one of the part's own pins is NOT exempt for that:
    # leaving a pin and running straight back across the body is the very error
    # this looks for. Landing on a pin at the edge and heading outward clips to
    # no length, so the ordinary connection does not register, and the way out
    # from a pin that lies inside the body is told apart from a wire across it.
    return [
        Finding(
            "wire_through_symbol",
            f"Wire {_wire(a, b)} passes through {part.name}'s body",
            refs=(part.name,),
            points=(a, b),
        )
        for a, b in ix.wires
        for _index, part, box in ix.drawn
        if _through(a, b, box)
        and not _way_out_from_own_pin(a, b, part, box)
        and _meets_drawing(part, a, b)
    ]


def _side(p: Point, q: Point, r: Point) -> int:
    """Which side of the line ``p``-``q`` the point ``r`` lies on: 1, -1, or
    0 on the line."""
    cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (cross > 0) - (cross < 0)


def _meet(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True if segments ``a``-``b`` and ``c``-``d`` share a point, crossing or
    touching."""
    if _side(a, b, c) * _side(a, b, d) < 0 and _side(c, d, a) * _side(c, d, b) < 0:
        return True
    return (
        point_on_segment(c, a, b)
        or point_on_segment(d, a, b)
        or point_on_segment(a, c, d)
        or point_on_segment(b, c, d)
    )


def _meets_drawing(part: Part, a: Point, b: Point) -> bool:
    """True if segment ``a``-``b`` meets what ``part`` draws: crosses or
    touches one of its lines, or passes through one of its rectangles or
    circles. A part drawn without those is met anywhere."""
    if not part.draws_known:
        return True
    return any(_meet(a, b, c, d) for c, d in part.strokes) or any(
        _through(a, b, area) for area in part.areas
    )


def _encloses(part: Part, body: BBox, x: int, y: int) -> bool:
    """True if ``(x, y)`` lies within what ``part`` draws: inside one of its
    rectangles or circles, or closed in by what it draws to the left, right,
    above and below, as a point inside an amplifier's triangle or on one of
    its lines is and one in the empty corner of its body beside the triangle
    is not. A part drawn without lines or areas encloses its whole body."""
    if any(area.strictly_contains(x, y) for area in part.areas):
        return True
    beyond = ((body.x1 - 1, y), (body.x2 + 1, y), (x, body.y1 - 1), (x, body.y2 + 1))
    return all(_meets_drawing(part, (x, y), edge) for edge in beyond)


def _draws_over(part: Part, body: BBox, box: BBox) -> bool:
    """True if what ``part`` draws reaches into ``box``: one of its lines, or
    one of its rectangles or circles, rather than an empty corner of its body
    such as the space beside an amplifier's triangle. A part drawn without
    those reaches wherever its body does."""
    if not body.overlaps(box):
        return False
    if not part.draws_known:
        return True
    return any(_through(a, b, box) for a, b in part.strokes) or any(
        area.overlaps(box) for area in part.areas
    )


def _core(extent: BBox) -> BBox:
    """What a text's glyphs surely cover: its extent, which is estimated from
    its font's metrics rather than its own glyphs, less a quarter of its
    thickness all round. Two lines that only graze, one's descender band on
    the other's cap height, do not overlap."""
    return extent.expanded(-(min(extent.width, extent.height) // 4))


@_finds("attribute_text_overlap")
def _attribute_texts_across_drawing(ix: _Index) -> list[Finding]:
    # Where the text lies is LTspice's own layout of it, so a part whose symbol
    # was not found has none to report. Each text is compared with the texts
    # after it: the other attribute texts, then the sheet's own.
    drawn = [
        (index, part, (x, y), text, _core(extent))
        for index, part in enumerate(ix.view.parts)
        if not part.missing
        for x, y, text, extent in part.text_extents
    ]
    texts: list[tuple[Part | None, Point, str, BBox]] = [
        (part, at, text, core) for _index, part, at, text, core in drawn
    ]
    texts += [(None, (x, y), text, _core(extent)) for x, y, text, extent in ix.view.text_extents]
    found: list[Finding] = []
    for n, (index, part, at, text, core) in enumerate(drawn):
        # What the text is drawn across: words for it, the parts and the
        # points it names, its own facts, and what it is by name, which a move
        # of the parts it names or a new value leaves as it was.
        across: list[
            tuple[str, tuple[str, ...], tuple[Point, ...], dict[str, str], tuple[Any, ...]]
        ] = [
            (f"wire {_wire(a, b)}", (), (a, b), {"across": "wire"}, ("wire", a, b))
            for a, b in ix.wires
            if _through(a, b, core)
        ]
        across += [
            (f"{other.name}'s body", (other.name,), (), {"across": "body"}, ("body",))
            for other_index, other, body in ix.drawn
            if other_index != index and _draws_over(other, body, core)
        ]
        across += [
            (
                (f"{owner.name}'s text" if owner else "the text")
                + f" {other_text!r} at {_at(other_at)}",
                (owner.name,) if owner else (),
                (other_at,),
                {"across": "text", "other_text": other_text},
                ("text", _attribute_of(owner, other_text))
                if owner
                else ("text", other_text, other_at),
            )
            for owner, other_at, other_text, other_core in texts[n + 1 :]
            if other_core.overlaps(core)
        ]
        found += [
            Finding(
                "attribute_text_overlap",
                f"{part.name}'s text {text!r} at {_at(at)} is drawn across {what}",
                refs=(part.name, *others),
                points=(at, *points),
                facts={"text": text, **facts},
                about=(_attribute_of(part, text), *of),
            )
            for what, others, points, facts, of in across
        ]
    return found


def _attribute_of(part: Part, text: str) -> str:
    """Which of a part's two drawn attributes ``text`` is: its reference, which
    the InstName window shows, or else its value."""
    return "InstName" if text == part.ref else "Value"


@_finds("label_over_component")
def _labels_over_components(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "label_over_component",
            f"Label '{text}' at {_at((x, y))} is inside {part.name}'s bounding box",
            refs=(part.name,),
            points=((x, y),),
            facts={"label": text},
        )
        for x, y, text in ix.view.labels
        # A label on any part's pin is the ordinary flag (pins sit on a box's
        # edge), even where it also lies inside another part's box.
        if (x, y) not in ix.pins_at
        # With boxes that overlap a label can be inside more than one, and each
        # is its own fact.
        for _index, part, box in ix.boxed
        if box.strictly_contains(x, y)
    ]


@_finds("text_in_symbol_body")
def _texts_in_symbol_bodies(ix: _Index) -> list[Finding]:
    anchors: list[tuple[int, int, str, int | None]] = [
        (x, y, text, index)
        for index, part in enumerate(ix.view.parts)
        for x, y, text in part.texts
    ]
    anchors += [(x, y, text, None) for x, y, text in ix.view.texts]
    return [
        Finding(
            "text_in_symbol_body",
            f"Text {text!r} at {_at((x, y))} is anchored inside {part.name}'s body",
            refs=(part.name,),
            points=((x, y),),
            facts={"text": text},
        )
        for x, y, text, owner in anchors
        for index, part, box in ix.drawn
        if owner != index and box.strictly_contains(x, y) and _encloses(part, box, x, y)
    ]


@_finds("stacked_directive")
def _stacked_directives(ix: _Index) -> list[Finding]:
    anchored: dict[Point, int] = {}
    for x, y, _text in ix.view.texts:
        anchored[(x, y)] = anchored.get((x, y), 0) + 1
    return [
        Finding(
            "stacked_directive",
            f"{count} directives/comments share anchor {_at(point)} — "
            "they render on top of each other",
            points=(point,),
            facts={"count": count},
        )
        for point, count in anchored.items()
        if count > 1
    ]


@_finds("label_island")
def _label_islands(ix: _Index) -> list[Finding]:
    # Ground, by any of its names, is exempt: joining ground by flag is the
    # ordinary practice. Names that differ only in case are one net, reported
    # under the name LTspice gives it: the label lowest on the sheet, and of
    # those the one furthest right (the net-naming recordings).
    stubs: dict[str, list[tuple[Point, str]]] = {}
    for x, y, text in ix.view.labels:
        name = text.strip()
        if name and not is_ground(name):
            stubs.setdefault(name_key(name), []).append(((x, y), name))
    islands = [
        (_net_name(labels), tuple(point for point, _name in labels))
        for labels in stubs.values()
        # A lone label is not a connection by name standing in for a wire.
        if len(labels) > 1 and not any(ix.on_a_wire(point) for point, _name in labels)
    ]
    return [
        Finding(
            "label_island",
            f"Net '{name}' is connected by {len(points)} net-label stubs and "
            "no drawn wire segment",
            points=points,
            facts={"net": name},
        )
        for name, points in sorted(islands)
    ]


assert list(_FINDERS) == list(RULES), "every rule has its finder, in the registry's order"


def findings(view: SheetView, rules: Collection[str] | None = None) -> list[Finding]:
    """What every rule finds of ``view``, or what the rules named in ``rules`` do.

    Grouped by rule in the registry's order, and within a rule in the sheet's
    own order, so the list is the same for the same sheet.

    - ``unresolved_symbol``: a part whose symbol was not found. No other rule
      says anything of its pins or its extent.
    - ``dropped_wire``: a wire LTspice leaves out of the netlist because its
      two ends land on pins of one part
      (``connectivity.same_instance_dropped_segments``).
    - ``duplicate_parameter``: a part giving one parameter, in any case, more
      than once across the attributes LTspice writes onto its line.
    - ``subcircuit_value_slot``: a Value on a part that netlists as a
      subcircuit call (element letter X), where it is an opamp's complexity
      level or the part's model comes from SpiceModel.
    - ``net_with_two_names``: a net, labels of one name being one node, whose
      labels carry two or more names, compared as LTspice compares them, ground
      no name. LTspice gives the node one of them.
    - ``joined_at_crossing``: a pin or label on a point where two or more wires
      cross, none ending there, which joins nets that are apart without it,
      nets that share a name being one.
    - ``unjoined_crossing``: a horizontal and a vertical wire through each
      other at a point no wire ends at and nothing sits on.
    - ``diagonal_wire``: a wire neither horizontal nor vertical.
    - ``floating_pin``: a pin connected to nothing (:func:`floating_pins`).
    - ``dangling_wire_end``: a wire end on no pin, no label and no other wire.
    - ``dangling_label``: a label that is on no wire and at no pin.
    - ``duplicate_wire``: two wires with the same two ends, in either order.
    - ``symbol_overlap``: two parts whose boxes share an area, each box with
      its pins: the box ``inspect`` and ``add_component`` report.
    - ``wire_through_symbol``: a wire through a part's body that meets what
      the part draws: one of its lines, rectangles or circles, not an empty
      corner such as the space beside an amplifier's triangle. The body is
      without its pins: a pin drawn apart from the body leaves room between
      them that an ordinary wire to a nearer pin crosses.
    - ``attribute_text_overlap``: a part's attribute text, laid out as
      LTspice lays it out with its extent estimated from LTspice's font,
      that a wire or a line another part draws passes through, or that shares
      an area with a rectangle or circle another part draws or with other
      text, by more than the estimate's slack (:func:`_core`).
    - ``label_over_component``: a label strictly inside a part's box and on no
      pin. A label on a pin, any part's, is the ordinary flag.
    - ``text_in_symbol_body``: text anchored inside what another part draws:
      in one of its rectangles or circles, or closed in by its lines on every
      side, not in an empty corner of its body. Only the anchor is tested, so
      the later lines of a directive that runs down into a part are not
      reported.
    - ``stacked_directive``: two or more directives or comments at exactly one
      anchor, with no guess at how far text reaches.
    - ``label_island``: a net joined only by labels of one name, with no wire
      on any of them. It is electrically sound; whether a given rail is
      acceptable drawn that way is the caller's to judge.

    A box or a body also spans leads and empty corners, so sharing an area with
    one is where something is, not proof that ink overlaps.
    """
    ix = _Index(view)
    wanted = RULES if rules is None else rules
    return [
        found for rule_id, finder in _FINDERS.items() if rule_id in wanted for found in finder(ix)
    ]
