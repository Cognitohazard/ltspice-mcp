"""What the editor and the checker each say of every sheet in the suite, held to a record of it.

The two tools report on a sheet through different passes, and the work of
making them one pass moves every check they make. This is what that work is
held to: for every ``.asc`` under ``tests/``, the editor's whole-sheet findings
and wiring counts and the checker's findings, as they were when the record was
made. A difference is either a regression or a rule changed on purpose, and a
rule changed on purpose changes the record in the same commit, where the
difference can be read.

The record is ``fixtures/sheet_findings.json``. To make it again, run this file
with ``LTSPICE_MCP_RECORD_SHEET_FINDINGS=1``; ``git diff`` then shows exactly
which findings of which sheets changed.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from spicelib import AscEditor

from ltspice_mcp.lib import symbol_geometry
from ltspice_mcp.lib.schematic_ops import make_editor, post_op_warnings, wiring_profile
from ltspice_mcp.lib.schematic_scene import build_scene, sheet_view
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import symbol_resolver_for
from ltspice_mcp.tools.verify import (
    VerifyCircuitInput,
    _dropped_wire_findings,  # pyright: ignore[reportPrivateUsage]  # reported only beside an export
    evaluate_verify_circuit,
)
from tests._schematic_fixtures import SUITE_SHEETS, TESTS
from tests._schematic_fixtures import suite_name as _name

_RECORD = TESTS / "fixtures" / "sheet_findings.json"
_RECORDING = os.environ.get("LTSPICE_MCP_RECORD_SHEET_FINDINGS") == "1"


def _stage(sheet: Path, sandbox: Path) -> Path:
    """Copy ``sheet`` into a folder of the sandbox with the symbols kept beside it,
    folders included."""
    staged = sandbox / f"sheet{len(list(sandbox.glob('sheet*')))}"
    staged.mkdir()
    for symbol in sheet.parent.rglob("*.asy"):
        target = staged / symbol.relative_to(sheet.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(symbol, target)
    return Path(shutil.copyfile(sheet, staged / sheet.name))


def _without_folder(finding: dict[str, Any]) -> dict[str, Any]:
    """A checker finding with its sheet named by file name, not by where the test put it."""
    at = dict(finding["at"])
    at["file"] = Path(at["file"]).name
    return {**finding, "at": at}


async def findings_of(sheet: Path, state: SessionState) -> dict[str, Any]:
    """Everything both tools say of ``sheet``, in the order they say it."""
    # Both symbol caches are keyed by name for the whole process, spicelib's
    # by file name alone: without this a sheet is read with whatever symbol
    # of that name the sheet before it left behind.
    AscEditor.symbol_cache = {}
    symbol_geometry._symbol_cache.clear()  # pyright: ignore[reportPrivateUsage]
    staged = _stage(sheet, Path(state.working_dir))
    try:
        editor = make_editor(staged)
        edit: dict[str, Any] = {
            "warnings": post_op_warnings(editor),  # type: ignore[arg-type]
            "wiring": wiring_profile(editor),  # type: ignore[arg-type]
        }
    except Exception:
        # A sheet the editor cannot open is a fact about it too; why is not
        # recorded, since the wording is another library's.
        edit = {"unopenable": True}
    evaluation = await evaluate_verify_circuit(
        VerifyCircuitInput(path=str(staged), checks=["symbols", "layout", "quality"]), state
    )
    view = sheet_view(build_scene(staged, resolver=symbol_resolver_for(staged, state)))
    return {
        "edit": edit,
        "verify": [_without_folder(finding) for finding in evaluation.data["findings"]],
        "dropped_wire": [_without_folder(f) for f in _dropped_wire_findings(view, staged)],
    }


@pytest.fixture(autouse=True)
def _spicelibs_symbols_are_put_back() -> Iterator[None]:
    remembered = dict(AscEditor.symbol_cache)
    try:
        yield
    finally:
        AscEditor.symbol_cache = remembered


def _recorded() -> dict[str, Any]:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _RECORDING, reason="set LTSPICE_MCP_RECORD_SHEET_FINDINGS=1 to record")
async def test_record_the_findings_of_every_sheet(asc_state: SessionState):
    record = {_name(sheet): await findings_of(sheet, asc_state) for sheet in SUITE_SHEETS}
    _RECORD.write_text(
        json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_the_record_covers_the_sheets_the_suite_holds():
    assert sorted(_recorded()) == [_name(sheet) for sheet in SUITE_SHEETS]


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_the_record_holds_findings_of_every_kind_both_tools_report():
    """A record of sheets with nothing wrong on them would hold nothing to."""
    edit: set[str] = set()
    verify: set[str] = set()
    for entry in _recorded().values():
        edit |= {warning["kind"] for warning in entry["edit"].get("warnings", [])}
        verify |= {finding["rule_id"] for finding in entry["verify"] + entry["dropped_wire"]}
    assert edit >= {"floating_pin", "dangling_label", "label_over_component"}
    assert verify >= {
        "floating_pin",
        "dangling_wire_end",
        "symbol_overlap",
        "unresolved_symbol",
        "label_island",
        "dropped_wire",
    }


@pytest.mark.skipif(_RECORDING, reason="recording")
@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=_name)
async def test_both_tools_say_of_a_sheet_what_the_record_says(
    sheet: Path, asc_state: SessionState
):
    said = await findings_of(sheet, asc_state)
    # Through JSON, as the record went: a tuple there is a list here.
    assert json.loads(json.dumps(said)) == _recorded()[_name(sheet)]
