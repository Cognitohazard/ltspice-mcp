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

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ltspice_mcp.lib.connectivity import (
    build_on_wire_predicate,
    build_wire_counter,
    same_instance_dropped_segments,
)
from ltspice_mcp.lib.geometry import BBox

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
        "A part whose symbol is not found, so that it has no pins.",
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
        "A wire passing through what a part draws.",
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

    @property
    def identity(self) -> tuple[Any, ...]:
        """What makes this the same finding as another: its rule and what it is
        about. The sentence says no more than these do."""
        return (self.rule, self.refs, self.points, tuple(sorted(self.facts.items())))


@dataclass(frozen=True)
class Part:
    """A placed part as a check sees it.

    ``box`` is the part's extent with its pins, the box ``inspect`` and
    ``add_component`` report, and ``body`` the extent of what it draws alone;
    either is ``None`` when the part has none. ``at`` is where the part is
    placed. ``pins`` are ``(name, x, y)`` in SpiceOrder, the order a symbol
    gives them in, and ``texts`` the anchors of its drawn attributes as
    ``(x, y, first line)``. ``missing`` says its symbol was not found: whatever
    box it carries is then a placeholder's, and no rule reads it.
    """

    ref: str
    symbol: str = ""
    box: BBox | None = None
    pins: tuple[tuple[str, int, int], ...] = ()
    texts: tuple[tuple[int, int, str], ...] = ()
    missing: bool = False
    body: BBox | None = None
    at: Point | None = None

    @property
    def name(self) -> str:
        """What to call the part in a finding: its reference, else its symbol."""
        return self.ref or self.symbol or "<unnamed>"

    @property
    def drawn(self) -> BBox | None:
        """The extent of what the part draws: its body where known, else its box."""
        return self.body if self.body is not None else self.box


@dataclass(frozen=True)
class SheetView:
    """A sheet as the checks read it, in file order.

    ``wires`` are ``(x1, y1, x2, y2)``, ``labels`` are ``(x, y, name)`` with
    ground among them, and ``texts`` are the sheet's directives and comments as
    ``(x, y, first line)``.
    """

    parts: tuple[Part, ...] = ()
    wires: tuple[tuple[int, int, int, int], ...] = ()
    labels: tuple[tuple[int, int, str], ...] = ()
    texts: tuple[tuple[int, int, str], ...] = ()


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


_Finder = Callable[[_Index], list[Finding]]
_FINDERS: dict[str, _Finder] = {}


def _finds(rule_id: str) -> Callable[[_Finder], _Finder]:
    """Register the function that finds ``rule_id``."""

    def register(finder: _Finder) -> _Finder:
        _FINDERS[rule_id] = finder
        return finder

    return register


def _at(point: Point) -> str:
    return f"({point[0]},{point[1]})"


def _wire(a: Point, b: Point) -> str:
    return f"{_at(a)}->{_at(b)}"


@_finds("unresolved_symbol")
def _parts_without_a_symbol(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "unresolved_symbol",
            f"Symbol '{part.symbol}' of {part.name} was not found: the part has no pins "
            "and no extent here, so nothing about them is checked",
            refs=(part.name,),
            points=(part.at,) if part.at is not None else (),
            facts={"symbol": part.symbol},
        )
        for part in ix.view.parts
        if part.missing
    ]


@_finds("dropped_wire")
def _dropped_wires(ix: _Index) -> list[Finding]:
    # LTspice leaves out a run whose two ends both land on pins of one part: the
    # pins stay on separate nodes, so the sheet shows a tie the netlist does not
    # have. A part is told from another by its place in the view, since two
    # parts may carry one reference, or none.
    owners: dict[Point, list[tuple[str, str]]] = {}
    for index, part in enumerate(ix.view.parts):
        for _name, x, y in part.pins:
            owners.setdefault((x, y), []).append((str(index), ""))
    found: list[Finding] = []
    for drop in same_instance_dropped_segments(owners, list(ix.view.wires)):
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


@_finds("floating_pin")
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


@_finds("symbol_overlap")
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


@_finds("wire_through_symbol")
def _wires_through_symbols(ix: _Index) -> list[Finding]:
    # A wire attached to one of the part's own pins is NOT exempt: leaving a pin
    # and running straight back across the body is the very error this looks
    # for. Landing on a pin at the edge and heading outward clips to no length,
    # so the ordinary connection still does not register.
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
    ]


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
        if owner != index and box.strictly_contains(x, y)
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
    # Ground is exempt: joining ground by flag is the ordinary practice.
    stubs: dict[str, list[Point]] = {}
    for x, y, text in ix.view.labels:
        name = text.strip()
        if name and name != "0":
            stubs.setdefault(name, []).append((x, y))
    return [
        Finding(
            "label_island",
            f"Net '{name}' is connected by {len(points)} net-label stubs and "
            "no drawn wire segment",
            points=tuple(points),
            facts={"net": name},
        )
        for name, points in sorted(stubs.items())
        # A lone label is not a connection by name standing in for a wire.
        if len(points) > 1 and not any(ix.on_a_wire(point) for point in points)
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
    - ``floating_pin``: a pin connected to nothing (:func:`floating_pins`).
    - ``dangling_wire_end``: a wire end on no pin, no label and no other wire.
    - ``dangling_label``: a label that is on no wire and at no pin.
    - ``duplicate_wire``: two wires with the same two ends, in either order.
    - ``symbol_overlap``: two parts whose boxes share an area, each box with
      its pins: the box ``inspect`` and ``add_component`` report.
    - ``wire_through_symbol``: a wire through what a part draws, without its
      pins. A pin drawn apart from the body leaves room between them that an
      ordinary wire to a nearer pin crosses.
    - ``label_over_component``: a label strictly inside a part's box and on no
      pin. A label on a pin, any part's, is the ordinary flag.
    - ``text_in_symbol_body``: text anchored inside what another part draws.
      Only the anchor is tested, so the later lines of a directive that runs
      down into a part are not reported.
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
