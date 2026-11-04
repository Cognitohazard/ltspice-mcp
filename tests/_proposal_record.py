"""What a record of answers to proposals is made of, for the tests that keep one.

A check made before an edit is written (may this route be drawn, may this label
be placed) is held by putting a broad, fixed set of proposals to it on every
sheet of the suite and keeping what it answered. This is the part those tests
share: a sheet staged where it can be opened, the *shape* of an answer, and an
entry of the record with an example of each shape.

An entry keeps how many proposals there were, a digest of every answer in
order, and how many answers had each shape: the answer with its numbers and
quoted names taken out, so that one refusal worded for different pins is one
shape. A digest that differs says an answer changed; the shapes and examples
say what kind.

A non-test helper module (leading underscore), so pytest does not collect it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from spicelib import AscEditor

from ltspice_mcp.lib.schematic_ops import sheet_nets
from ltspice_mcp.lib.schematic_scene import scene_of_text, sheet_view
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.lib.symbol_library import SymbolResolver
from ltspice_mcp.tools.schematic_edit import (
    _render_editor_text,  # pyright: ignore[reportPrivateUsage]
)
from tests._schematic_fixtures import open_editor

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


def open_staged(sheet: Path, sandbox: Path) -> AscEditor | None:
    """The editor of ``sheet`` staged afresh in ``sandbox``, or ``None`` if it cannot open it."""
    try:
        return open_editor(stage_afresh(sheet, sandbox))
    except Exception:
        return None


def without_particulars(text: str) -> str:
    """``text`` with what varies from one proposal to the next taken out of it."""
    text = re.sub(r"net:[^\s:,]+", "net:NAME", text)
    text = re.sub(r"'[^']*'", "'…'", text)
    text = re.sub(r"\b[\w+\-]+\.[\w+\-]+", "PIN", text)
    return re.sub(r"-?\d+", "#", text)


def entry_of(asked: Sequence[Asked]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """What the record holds of one sheet's answers, and an example of each shape."""
    shapes: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for one in asked:
        shapes[one.shape] = shapes.get(one.shape, 0) + 1
        examples.setdefault(one.shape, {**one.proposal, **one.answer})
    answers = [one.answer for one in asked]
    digest = hashlib.sha256(json.dumps(answers, sort_keys=True).encode("utf-8")).hexdigest()
    entry = {"proposals": len(asked), "sha256": digest, "shapes": dict(sorted(shapes.items()))}
    return entry, examples


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sheet_says(editor: AscEditor, points: set[Point], resolver: SymbolResolver) -> str:
    """The join rules the sheet as the editor holds it reports at ``points``:
    the word an edit's reply carries of them. Read as ``edit_schematic`` reads
    it (``sheet_findings_of``), with one resolver for every reading of a sheet."""
    view = sheet_view(
        scene_of_text(_render_editor_text(editor), Path(editor.asc_file_path), resolver)
    )
    said = sorted(
        {found.rule for found in findings(view, _JOIN_RULES) if points & set(found.points)}
    )
    return f" | sheet: {', '.join(said)}" if said else ""


def what_is_there(editor: AscEditor) -> Any:
    """For a point and a name: whether the name is already on the sheet, how
    many nets meet at the point, and how many of the nets there carry a name,
    as a sentence and as the two counts.

    The nets that meet are those of the wires through the point, a wire LTspice
    drops between two pins of one part counting as one; with no wire there,
    what sits on the point is one place, and a bare point is none. Worked out
    here from the sheet's nets, apart from the op's own reading of them: it is
    what tells apart answers the op words alike.
    """
    nets = sheet_nets(editor)
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
