"""What a record of answers to proposals is made of, for the tests that keep one.

A check made before an edit is written (may this route be drawn, may this label
be placed) is held by putting a broad, fixed set of proposals to it on every
sheet of the suite and keeping what it answered. This is the part those tests
share: a sheet staged where it can be opened, a :class:`Driver` through which
they read it and put each proposal to it, the *shape* of an answer, and an
entry of the record with an example of each shape.

An entry keeps how many proposals there were, a digest of them and of every
answer in order, how many answers had each shape (the answer with its numbers
and quoted names taken out, so that one refusal worded for different pins is
one shape), and each proposal's shape in order, as its place among the
sheet's shapes. So a test held to the record names each proposal whose answer
moved to another shape, and from which (:func:`differences`); a digest that
differs with no shape moved says an answer changed in its particulars only.

A non-test helper module (leading underscore), so pytest does not collect it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from spicelib import AscEditor

from ltspice_mcp.lib.connectivity import NamedNets
from ltspice_mcp.lib.schematic_ops import (
    GridPoint,
    OpAddNetLabel,
    OpMoveComponent,
    _plan_connect_route,  # pyright: ignore[reportPrivateUsage]  # the planner behind wire_pins
    apply_op_inplace,
    collect_component_geometry,
    placed_geometry,
    resolve_pin,
    sheet_nets,
    wire_segments_of,
)
from ltspice_mcp.lib.schematic_scene import scene_of_text, sheet_view
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.lib.symbol_library import SymbolResolver
from ltspice_mcp.tools.schematic_edit import (
    _render_editor_text,  # pyright: ignore[reportPrivateUsage]
)
from tests._schematic_fixtures import open_editor, suite_resolver

Point = tuple[int, int]

#: The sheet rules that say what placing a label or moving a part joined.
_JOIN_RULES = ("net_with_two_names", "joined_at_crossing")

_T = TypeVar("_T")


@dataclass(frozen=True)
class Asked:
    """One proposal, what it was answered, and the shape of that answer."""

    proposal: dict[str, Any]
    answer: dict[str, Any]
    shape: str


def stage(sheet: Path, sandbox: Path) -> Path:
    """``sheet`` copied into ``sandbox`` with the symbol files beside it."""
    for symbol in sheet.parent.rglob("*.asy"):
        target = sandbox / symbol.relative_to(sheet.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(symbol, target)
    return Path(shutil.copyfile(sheet, sandbox / sheet.name))


def stage_afresh(sheet: Path, sandbox: Path) -> Path:
    """``sheet`` staged in ``sandbox``, with every symbol to be read afresh.

    spicelib's symbol cache is keyed by file name alone for the whole process:
    a symbol of one sheet's folder must not answer for another sheet's of the
    same name.
    """
    AscEditor.symbol_cache = {}
    return stage(sheet, sandbox)


def every_nth(items: list[_T], most: int) -> list[_T]:
    """``items``, or every n-th of them when there are more than ``most``."""
    return items[:: -(-len(items) // most)] if len(items) > most else items


def open_staged(sheet: Path, sandbox: Path) -> Driver | None:
    """A :class:`Driver` of ``sheet`` staged afresh in ``sandbox``, or ``None``
    if it cannot be opened."""
    try:
        return driver_of(stage_afresh(sheet, sandbox))
    except Exception:
        return None


def without_particulars(text: str) -> str:
    """``text`` with what varies from one proposal to the next taken out of it."""
    text = re.sub(r"net:[^\s:,]+", "net:NAME", text)
    text = re.sub(r"'[^']*'", "'…'", text)
    text = re.sub(r"\b[\w+\-]+\.[\w+\-]+", "PIN", text)
    return re.sub(r"-?\d+", "#", text)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def entry_of(asked: Sequence[Asked]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """What the record holds of one sheet's answers, and an example of each shape."""
    shapes: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for one in asked:
        shapes[one.shape] = shapes.get(one.shape, 0) + 1
        examples.setdefault(one.shape, {**one.proposal, **one.answer})
    place = {shape: index for index, shape in enumerate(sorted(shapes))}
    entry = {
        "proposals": len(asked),
        "asked_sha256": _digest([one.proposal for one in asked]),
        "sha256": _digest([one.answer for one in asked]),
        "shapes": dict(sorted(shapes.items())),
        "order": " ".join(str(place[one.shape]) for one in asked),
    }
    return entry, examples


def recorded_shapes(recorded: dict[str, Any]) -> list[str]:
    """Each recorded proposal's shape, in order, from an entry :func:`entry_of` made."""
    shapes = sorted(recorded["shapes"])
    return [shapes[int(index)] for index in recorded["order"].split()]


def differences(asked: Sequence[Asked], recorded: dict[str, Any]) -> list[str]:
    """How ``asked`` answers otherwise than the ``recorded`` entry, a line each;
    none where it answers as recorded."""
    entry, _examples = entry_of(asked)
    if entry == recorded:
        return []
    if entry["asked_sha256"] != recorded.get("asked_sha256"):
        return [
            f"the proposals are not those recorded ({recorded['proposals']} then, "
            f"{len(asked)} now), so no answer can be held to its own; record again"
        ]
    moved = [
        (one, was)
        for one, was in zip(asked, recorded_shapes(recorded), strict=True)
        if one.shape != was
    ]
    if not moved:
        return [
            "no answer moved to another shape, but one changed in its particulars "
            "(a name, a number, an order); record again and compare the examples"
        ]
    return moved_lines(moved)


def moved_lines(moved: Sequence[tuple[Asked, str]]) -> list[str]:
    """The lines that say which proposals' answers moved to another shape, and
    from which: each of ``moved`` is a proposal as answered now, with the shape
    it was recorded with."""
    # One line for each move from a shape to a shape, the most proposals first,
    # naming the first few: a change moves many answers the same way.
    by_move: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for one, was in moved:
        by_move.setdefault((was, one.shape), []).append(one.proposal)
    lines = []
    for (was, now), proposals in sorted(by_move.items(), key=lambda item: -len(item[1])):
        shown = "; ".join(json.dumps(proposal, sort_keys=True) for proposal in proposals[:3])
        more = f" and {len(proposals) - 3} more" if len(proposals) > 3 else ""
        lines.append(
            f"{len(proposals)} from {_shortened(was)!r} to {_shortened(now)!r}: {shown}{more}"
        )
    return lines


def _shortened(shape: str, most: int = 160) -> str:
    return shape if len(shape) <= most else shape[: most - 1] + "…"


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class Driver:
    """What the record harnesses read of a sheet and do to it, in one place.

    Each proposal is put to the same sheet: an edit made to answer one is
    taken back when its ``with`` block ends, refused or not. Implemented on
    the schematic editor the ops edit today, so the harnesses do not change
    when the ops stop using it; only this class does.
    """

    def __init__(self, editor: AscEditor, resolver: SymbolResolver) -> None:
        self._editor = editor
        self._sheet = Path(editor.asc_file_path)
        self._resolver = resolver

    def parts(self) -> list[dict[str, Any]]:
        """Each part's ``ref``, place (``x``, ``y``, ``width``, ``height``) and
        ``pins``, each pin's ``name``, ``x`` and ``y``."""
        return collect_component_geometry(self._editor)

    def wires(self) -> list[tuple[int, int, int, int]]:
        return wire_segments_of(self._editor)

    def labels(self) -> list[tuple[Point, str]]:
        """Each label and ground flag, as its point and its text."""
        return [((int(lbl.coord.X), int(lbl.coord.Y)), lbl.text) for lbl in self._editor.labels]

    def position(self, reference: str) -> Point:
        at, _rotation = self._editor.get_component_position(reference)
        return int(at.X), int(at.Y)

    def pin_point(self, spec: str) -> Point:
        """Where the pin ``spec`` (``'R1.A'``) is, as the ops resolve it."""
        return resolve_pin(spec, self._editor)

    def pins_of(self, reference: str) -> list[Point]:
        geometry = placed_geometry(self._editor, reference)
        return [(pin["x"], pin["y"]) for pin in (geometry or {}).get("pins", ())]

    def nets(self) -> NamedNets:
        return sheet_nets(self._editor)

    def propose_route(
        self, from_pin: str | GridPoint, to_pin: str | GridPoint, waypoints: list[GridPoint]
    ) -> Any:
        """The route ``wire_pins`` would draw, its ``segments`` and ``junctions``;
        raises ``NetlistError`` where it refuses it."""
        return _plan_connect_route(self._editor, from_pin, to_pin, waypoints)

    @contextmanager
    def label_placed(self, proposal: dict[str, Any]) -> Iterator[None]:
        """``add_net_label`` with ``proposal``, taken back on leaving; raises
        ``NetlistError`` where the op refuses it."""
        had = len(self._editor.labels)
        try:
            apply_op_inplace(
                self._editor, OpAddNetLabel(op="add_net_label", **proposal), self._sheet
            )
            yield
        finally:
            del self._editor.labels[had:]

    @contextmanager
    def part_moved(self, proposal: dict[str, Any]) -> Iterator[None]:
        """``move_component`` with ``proposal``, taken back on leaving; raises
        ``NetlistError`` where the op refuses it."""
        at, rotation = self._editor.get_component_position(proposal["reference"])
        try:
            apply_op_inplace(
                self._editor, OpMoveComponent(op="move_component", **proposal), self._sheet
            )
            yield
        finally:
            self._editor.set_component_position(proposal["reference"], at, rotation)

    def findings_at(self, points: set[Point]) -> str:
        """The join rules the sheet reports at ``points``: the word an edit's
        reply carries of them. Read as ``edit_schematic`` reads it
        (``sheet_findings_of``), with one resolver for every reading of a sheet."""
        view = sheet_view(
            scene_of_text(_render_editor_text(self._editor), self._sheet, self._resolver)
        )
        said = sorted(
            {found.rule for found in findings(view, _JOIN_RULES) if points & set(found.points)}
        )
        return f" | sheet: {', '.join(said)}" if said else ""


def driver_of(sheet: Path, resolver: SymbolResolver | None = None) -> Driver:
    """A :class:`Driver` of ``sheet``, its symbols found with ``resolver``,
    the suite's own beside the sheet when none is given."""
    resolver = resolver or suite_resolver(sheet)
    return Driver(open_editor(sheet, resolver), resolver)


def what_is_there(nets: NamedNets) -> Any:
    """For a point and a name: whether the name is already on the sheet, how
    many nets meet at the point, and how many of the nets there carry a name,
    as a sentence and as the two counts.

    The nets that meet are those of the wires through the point, a wire LTspice
    drops between two pins of one part counting as one; with no wire there,
    what sits on the point is one place, and a bare point is none. Worked out
    here from the sheet's ``nets``, apart from the op's own reading of them: it
    is what tells apart answers the op words alike.
    """
    on_the_sheet = {label.text for label in nets.labels}

    def there(net: str, at: Point) -> tuple[str, int, int]:
        wires = set(nets.wired_at(at))
        nodes = set().union(*wires) if wires else set(nets.at(at))
        met = len(wires) if wires else min(len(nodes), 1)
        named = [node for node in nodes if nets.names(node)]
        if net == "0":
            name = "ground"
        elif any(net in nets.names(node) for node in nodes):
            name = "a name already here"
        elif net in on_the_sheet:
            name = "a name already elsewhere"
        else:
            name = "a new name"
        return f"{name}; {met} at the point, {len(named)} named", met, len(named)

    return there
