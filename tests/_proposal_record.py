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

from ltspice_mcp.lib import symbol_geometry
from ltspice_mcp.lib.schematic_ops import make_editor

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

    Both symbol caches are keyed by name for the whole process, spicelib's by
    file name alone: a symbol of one sheet's folder must not answer for another
    sheet's of the same name.
    """
    AscEditor.symbol_cache = {}
    symbol_geometry._symbol_cache.clear()  # pyright: ignore[reportPrivateUsage]
    return stage(sheet, sandbox)


def every_nth(items: list[_T], most: int) -> list[_T]:
    """``items``, or every n-th of them when there are more than ``most``."""
    return items[:: -(-len(items) // most)] if len(items) > most else items


def open_staged(sheet: Path, sandbox: Path) -> AscEditor | None:
    """The editor of ``sheet`` staged afresh in ``sandbox``, or ``None`` if it cannot open it."""
    try:
        editor = make_editor(stage_afresh(sheet, sandbox))
    except Exception:
        return None
    assert isinstance(editor, AscEditor)
    return editor


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
