"""What can be said of a sheet: the rules, and one type for what they find.

A check on a sheet is a rule. ``RULES`` is the registry: for each rule its id,
what kind of thing it says, what it is about, where the claim comes from, what
a finding of it is called on a whole sheet, and which tool reports it. A check
returns :class:`Finding` values, whichever tool asked, and the tool turns them
into the rows its own reply has always carried.

The rules are in three families, kept apart because they are different kinds
of statement:

- *electrical*: the sheet will netlist differently from how it is drawn. The
  provenance of such a rule is the LTspice recording that shows the behaviour
  (``docs/TESTING.md``, "Recorded LTspice behaviour").
- *structural*: something is left undone, such as a pin or a label on nothing.
- *drawing*: how the sheet reads, such as one part's box over another's.

None of them is a verdict on the sheet. A finding says what is there and where.

The checks read a :class:`SheetView`, a plain picture of a sheet that the
schematic editor builds from the sheet it holds and the checker from the file
it read. A rule means one thing whichever tool asks: a floating pin is
:func:`floating_pins` for both, and a part's box is the box with its pins for
both. What still differs is which rules each tool reports and how each words
a finding, which is why there are two lists: ``editor_findings`` and
``checker_findings``. ``docs/design/schematic_engine.md`` (sections 7 and 8)
says where that goes next.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


@dataclass(frozen=True)
class Rule:
    """One thing that can be said of a sheet.

    ``provenance`` is the recorded behaviour an electrical rule rests on (a key
    of the recording inventory), or ``"definition"`` for a rule that says what
    is drawn and claims nothing about LTspice. ``severity`` is what a finding
    of the rule is called on a whole sheet. ``editor`` says the schematic
    editor's reply lists the rule, and ``check`` names the ``verify_circuit``
    check that reports it, if one does.
    """

    rule_id: str
    family: Family
    scope: Scope
    summary: str
    provenance: str = "definition"
    severity: Literal["observation", "warning", "error"] = "observation"
    editor: bool = False
    check: Literal["symbols", "layout", "quality", "export"] | None = None


_RULES = (
    Rule(
        "floating_pin",
        "structural",
        "point",
        "A pin with no wire, label or other pin on it.",
        editor=True,
        check="layout",
    ),
    Rule(
        "duplicate_wire",
        "structural",
        "wire",
        "One wire drawn more than once between the same two points.",
        editor=True,
    ),
    Rule(
        "dangling_label",
        "structural",
        "point",
        "A net label on no wire and no pin.",
        editor=True,
    ),
    Rule(
        "label_over_component",
        "drawing",
        "part",
        "A net label placed inside a part's box and on no pin.",
        editor=True,
    ),
    Rule(
        "stacked_directive",
        "drawing",
        "point",
        "Two or more directives or comments at one anchor.",
        editor=True,
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
        "A wire passing through a part's box.",
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
        "text_in_symbol_body",
        "drawing",
        "part",
        "Text anchored inside another part's box.",
        check="quality",
    ),
    Rule(
        "label_island",
        "drawing",
        "net",
        "A net joined only by labels of one name, with no wire on any of them.",
        check="quality",
    ),
    Rule(
        "dropped_wire",
        "electrical",
        "wire",
        "A wire straight between two pins of one part, which LTspice leaves out of the netlist.",
        provenance="same-instance-wire",
        severity="warning",
        check="export",
    ),
    Rule(
        "unresolved_symbol",
        "structural",
        "part",
        "A part whose symbol is not found, so that it has no pins.",
        severity="error",
        editor=True,
        check="symbols",
    ),
)

#: Every rule either tool reports on a whole sheet. The rules the schematic
#: editor lists come in the order its reply lists them.
RULES: Mapping[str, Rule] = {rule.rule_id: rule for rule in _RULES}


@dataclass(frozen=True)
class Finding:
    """What one rule found at one place.

    ``refs`` are the parts involved and ``points`` the places to look, both in
    the order the rule gives them. ``facts`` are the rule's own named values
    (a pin, a label's text, a count). ``detail`` is one sentence saying it.
    """

    rule: str
    detail: str
    refs: tuple[str, ...] = ()
    points: tuple[Point, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Part:
    """A placed part as a check sees it.

    ``box`` is the part's extent with its pins, the box ``inspect`` and
    ``add_component`` report, and ``None`` when it has none. ``body`` is the
    extent of what it draws alone, where the view knows it. ``at`` is where the
    part is placed. ``pins`` are ``(name, x, y)`` in SpiceOrder, the order a
    symbol gives them in, and ``texts`` the anchors of its drawn attributes as
    ``(x, y, first line)``. ``missing`` says its symbol was not found.
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
        #: Each part that draws something, with what it draws, by its place in the view.
        self.drawn = [
            (index, part, drawn)
            for index, part in enumerate(view.parts)
            if (drawn := part.drawn) is not None
        ]


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


# ---------------------------------------------------------------------------
# The schematic editor's list
# ---------------------------------------------------------------------------


def _floating_pins_by_name(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "floating_pin",
            f"Floating pin: {f'{part.ref}.{name}' if name else part.ref} at ({x},{y})",
            refs=(part.ref,),
            points=((x, y),),
            facts={"pin": name},
        )
        for part, (name, x, y) in _floating(ix)
    ]


def _duplicate_wires(ix: _Index) -> list[Finding]:
    drawn: dict[tuple[Point, Point], int] = {}
    for v1, v2 in ix.wires:
        key = (v1, v2) if v1 <= v2 else (v2, v1)
        drawn[key] = drawn.get(key, 0) + 1
    return [
        Finding(
            "duplicate_wire",
            f"Duplicate wire ({count}×): ({a[0]},{a[1]})->({b[0]},{b[1]})",
            points=(a, b),
            facts={"count": count},
        )
        for (a, b), count in drawn.items()
        if count > 1
    ]


def _dangling_labels(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "dangling_label",
            f"Dangling label '{text}' at ({x},{y})",
            points=((x, y),),
            facts={"label": text},
        )
        for x, y, text in ix.view.labels
        if (x, y) not in ix.pins_at and not ix.on_any_wire((x, y))
    ]


def _labels_over_components(ix: _Index) -> list[Finding]:
    boxes = [(part.ref, part.box) for part in ix.view.parts if part.box is not None]
    return [
        Finding(
            "label_over_component",
            f"Label '{text}' at ({x},{y}) is inside {ref}'s bounding box",
            refs=(ref,),
            points=((x, y),),
            facts={"label": text},
        )
        for x, y, text in ix.view.labels
        # A label on any part's pin is the ordinary flag (pins sit on a box's
        # edge), even where it also lies inside another part's box.
        if (x, y) not in ix.pins_at
        # With boxes that overlap a label can be inside more than one, and each
        # is its own fact.
        for ref, box in boxes
        if box.strictly_contains(x, y)
    ]


def _stacked_directives(ix: _Index) -> list[Finding]:
    anchored: dict[Point, int] = {}
    for x, y, _text in ix.view.texts:
        anchored[(x, y)] = anchored.get((x, y), 0) + 1
    return [
        Finding(
            "stacked_directive",
            f"{count} directives/comments share anchor ({x},{y}) — "
            "they render on top of each other",
            points=((x, y),),
            facts={"count": count},
        )
        for (x, y), count in anchored.items()
        if count > 1
    ]


def _parts_without_a_symbol(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "unresolved_symbol",
            f"Symbol '{part.symbol}' of {part.name} was not found: the part has "
            "no pins here, so nothing at them is checked",
            refs=(part.ref,),
            points=(part.at,) if part.at is not None else (),
            facts={"symbol": part.symbol},
        )
        for part in ix.view.parts
        if part.missing
    ]


_Check = Callable[[_Index], list[Finding]]

_EDITOR: tuple[_Check, ...] = (
    _floating_pins_by_name,
    _duplicate_wires,
    _dangling_labels,
    _labels_over_components,
    _stacked_directives,
    _parts_without_a_symbol,
)


def editor_findings(view: SheetView) -> list[Finding]:
    """The whole-sheet findings the schematic editor reports after an edit.

    - ``floating_pin``: a pin connected to nothing (:func:`floating_pins`).
    - ``duplicate_wire``: two wires with the same two ends, in either order.
    - ``dangling_label``: a label that is on no wire and at no pin.
    - ``label_over_component``: a label strictly inside a part's box and on no
      pin. The box also spans leads and empty corners, so this is where the
      anchor is, not a promise that ink overlaps. A label on a pin, any part's,
      is the ordinary flag and is never reported.
    - ``stacked_directive``: two or more directives or comments at exactly one
      anchor. Only an exact match counts, with no guess at how far text reaches.
    - ``unresolved_symbol``: a part whose symbol was not found. It has no pins
      here, so nothing at them is checked, and the pin counts leave it out.
    """
    ix = _Index(view)
    return [finding for check in _EDITOR for finding in check(ix)]


# ---------------------------------------------------------------------------
# The checker's list
# ---------------------------------------------------------------------------


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


def _symbol_overlaps(ix: _Index) -> list[Finding]:
    whole = [(part, part.box) for part in ix.view.parts if part.box is not None]
    findings: list[Finding] = []
    for position, (part_a, box_a) in enumerate(whole):
        for part_b, box_b in whole[position + 1 :]:
            if not box_a.overlaps(box_b):
                continue
            ox1, oy1 = max(box_a.x1, box_b.x1), max(box_a.y1, box_b.y1)
            ox2, oy2 = min(box_a.x2, box_b.x2), min(box_a.y2, box_b.y2)
            findings.append(
                Finding(
                    "symbol_overlap",
                    f"bounding boxes share a {ox2 - ox1}x{oy2 - oy1} region",
                    refs=(part_a.name, part_b.name),
                    points=((ox1, oy1), (ox2, oy2)),
                )
            )
    return findings


def _wires_through_symbols(ix: _Index) -> list[Finding]:
    # A wire attached to one of the part's own pins is NOT exempt: leaving a pin
    # and running straight back across the body is the very error this looks
    # for. Landing on a pin at the edge and heading outward clips to no length,
    # so the ordinary connection still does not register.
    return [
        Finding(
            "wire_through_symbol",
            "wire segment passes through the symbol's body box",
            refs=(part.name,),
            points=(a, b),
        )
        for a, b in ix.wires
        for _index, part, box in ix.drawn
        if _through(a, b, box)
    ]


def _floating_pins_by_part(ix: _Index) -> list[Finding]:
    return [
        Finding(
            "floating_pin",
            "pin has no wire, net label, or mating pin on it",
            refs=(part.name,),
            points=((x, y),),
        )
        for part, (_name, x, y) in _floating(ix)
    ]


def _dangling_wire_ends(ix: _Index) -> list[Finding]:
    # Once per coordinate: two loose ends meeting nothing at one point are one
    # place to look, not two findings.
    seen: set[Point] = set()
    findings: list[Finding] = []
    for a, b in ix.wires:
        for end in (a, b):
            if end in seen or end in ix.pins_at or end in ix.labelled:
                continue
            # Its own wire is one; a second is another wire touching it there.
            if ix.wires_through(end) > 1:
                continue
            seen.add(end)
            findings.append(
                Finding(
                    "dangling_wire_end",
                    "wire end meets no pin, net label, or other wire",
                    points=(end,),
                )
            )
    return findings


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
            f"text {text!r} is anchored inside the symbol's body box",
            refs=(part.name,),
            points=((x, y),),
        )
        for x, y, text, owner in anchors
        for index, part, box in ix.drawn
        if owner != index and box.strictly_contains(x, y)
    ]


_CHECKER: tuple[_Check, ...] = (
    _symbol_overlaps,
    _wires_through_symbols,
    _floating_pins_by_part,
    _dangling_wire_ends,
    _texts_in_symbol_bodies,
)


def checker_findings(view: SheetView) -> list[Finding]:
    """The layout findings the checker reports, grouped by rule in this order.

    Overlapping boxes, wires through a box, pins connected to nothing, wire
    ends connected to nothing, and text anchored inside another part's box.
    Within a rule the findings are in the sheet's own order, so the list is the
    same for the same sheet.

    Two parts overlap when their boxes do, each with its pins. A wire or a
    text is through or inside a part when it is within what the part draws,
    without its pins: a pin drawn apart from the body leaves room between them
    that an ordinary wire to a nearer pin crosses. Either extent also spans
    leads and empty corners, so sharing an area is not proof that ink does. The
    text check tests a text's anchor only, so the later lines of a directive
    that runs down into a part are not reported.
    """
    ix = _Index(view)
    return [finding for check in _CHECKER for finding in check(ix)]


def label_islands(view: SheetView) -> list[Finding]:
    """Nets joined only by labels of one name, with no wire on any of them.

    Such a net is electrically sound and reads as a netlist wearing symbols.
    Ground is exempt: joining ground by flag is the ordinary practice. Whether
    a given rail is acceptable that way is the caller's to judge.
    """
    stubs: dict[str, list[Point]] = {}
    for x, y, text in view.labels:
        name = text.strip()
        if name and name != "0":
            stubs.setdefault(name, []).append((x, y))
    on_a_wire = build_on_wire_predicate(
        [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in view.wires if (x1, y1) != (x2, y2)]
    )
    return [
        Finding(
            "label_island",
            f"net '{name}' is connected by {len(points)} net-label stubs and "
            "no drawn wire segment",
            points=tuple(points),
            facts={"net": name},
        )
        for name, points in sorted(stubs.items())
        # A lone label is not a connection by name standing in for a wire.
        if len(points) > 1 and not any(on_a_wire(point) for point in points)
    ]


def dropped_wires(view: SheetView) -> list[Finding]:
    """Wires drawn on the sheet and absent from the netlist LTspice exports.

    LTspice leaves out a run whose two ends both land on pins of one part: the
    pins stay on separate nodes, so the sheet shows a tie the netlist does not
    have. The rule is ``connectivity.same_instance_dropped_segments``.
    """
    owners: dict[Point, list[tuple[str, str]]] = {}
    for part in view.parts:
        for _name, x, y in part.pins:
            owners.setdefault((x, y), []).append((part.ref, ""))
    findings: list[Finding] = []
    for drop in same_instance_dropped_segments(owners, list(view.wires)):
        x1, y1, x2, y2 = drop["segment"]
        ref = drop["ref"]
        findings.append(
            Finding(
                "dropped_wire",
                f"wire joins two pins of the same instance {ref} and is not exported",
                refs=(ref,),
                points=((x1, y1), (x2, y2)),
            )
        )
    return findings


def unresolved_symbols(view: SheetView) -> list[Finding]:
    """One finding for each symbol that was not found, naming the parts that use it."""
    users: dict[str, list[str]] = {}
    for part in view.parts:
        if part.missing:
            users.setdefault(part.symbol, []).append(part.ref)
    return [
        Finding(
            "unresolved_symbol",
            "drawn as a placeholder box; searched the schematic directory, "
            "the configured symbol paths, and the stock library",
            refs=tuple(refs),
            facts={"symbol": name},
        )
        for name, refs in sorted(users.items())
    ]
