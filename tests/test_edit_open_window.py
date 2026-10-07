"""edit_schematic on a sheet that is open in an LTspice window.

LTspice never reads a file again once it has it open, so the window holds a
copy of its own: an edit has to be shown there or the window's next save
writes the old sheet back, and an edit under a window that differs from the
file would lose whichever side was not saved. These go through the real
transaction against the stand-in bridge (``tests/fake_ltspice_bridge.py``,
itself replayed against a recording of LTspice 26.1.1 in
``test_ltspice_bridge.py``); the sheets LTspice rewrites on opening are the
recorded ones.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib.encoding import decode_spice_bytes, decode_windows_1252
from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests.conftest import LIVENESS_S
from tests.ltspice_bridge_recorder import FIXTURES, INPUTS
from tests.test_ltspice_bridge import fake_command, write_world

PID = 4242
#: The stand-in bridge runs under this interpreter, which is what a message names.
BRIDGE_PROGRAM = Path(sys.executable).name
SET_VALUE = [{"op": "set_component_value", "reference": "R1", "value": "2.2k"}]


def file_bytes(sheet: Path) -> bytes:
    return sheet.read_bytes()


def digest(sheet: Path) -> str:
    return hashlib.sha256(sheet.read_bytes()).hexdigest()


def as_ltspice_reads(sheet: Path) -> str:
    return decode_windows_1252(sheet.read_bytes())


def as_the_server_reads(sheet: Path) -> str:
    return decode_spice_bytes(sheet.read_bytes())


def recorded_pair(name: str, work_dir: Path) -> tuple[Path, str]:
    """A recorded input copied into the sandbox, and LTspice's copy of it."""
    sheet = work_dir / f"{name}.asc"
    shutil.copyfile(INPUTS / f"{name}.asc", sheet)
    return sheet, (FIXTURES / "ltspice26" / "sheets" / f"{name}.asc").read_text(encoding="utf-8")


def store_as_utf8(sheet: Path) -> str:
    """Rewrite ``sheet`` in UTF-8 and return it as LTspice then reads it."""
    stored = decode_windows_1252(sheet.read_bytes()).encode("utf-8")
    sheet.write_bytes(stored)
    return decode_windows_1252(stored)


class Window:
    """One LTspice window, as the stand-in bridge reports it."""

    def __init__(self, state: SessionState, world: Path) -> None:
        self._world = world
        self.shows(None, "")
        state.open_windows = OpenWindows(fake_command(world), timeout=LIVENESS_S)

    def shows(self, sheet: Path | None, text: str, **extra: Any) -> None:
        """Have the window hold ``text`` for ``sheet``; with no sheet, nothing is open."""
        designs = {} if sheet is None else {str(sheet): text}
        write_world(self._world, [{"pid": PID, "version": "26.1.1", "designs": designs}], **extra)

    def closes(self) -> None:
        write_world(self._world, [])

    def text(self, sheet: Path) -> str:
        world = json.loads(self._world.read_text(encoding="utf-8"))
        return world["windows"][0]["designs"][str(sheet)]


@pytest.fixture
def window(asc_state: SessionState, tmp_path: Path) -> Window:
    return Window(asc_state, tmp_path / "world.json")


async def edit(state: SessionState, sheet: Path, ops: list[dict] | None = None, **kw: Any) -> dict:
    arguments: dict[str, Any] = {
        "target": str(sheet),
        "ops": SET_VALUE if ops is None else ops,
        **kw,
    }
    if "expected_sha256" not in kw and not kw.get("dry_run"):
        arguments["expected_sha256"] = digest(sheet)
    result = await handle_edit_schematic(EditSchematicInput.model_validate(arguments), state)
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(se._OUTPUT_SCHEMA).validate(data)
    return data


async def test_a_committed_edit_is_shown_in_the_window_that_has_the_sheet_open(
    asc_state: SessionState, asc_file: Path, window: Window
):
    window.shows(asc_file, as_ltspice_reads(asc_file))

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["open_in_ltspice"] == [{"pid": PID, "version": "26.1.1", "shown": True}]
    # The window holds the sheet that was committed, character for character.
    assert window.text(asc_file) == as_the_server_reads(asc_file)
    assert "SYMATTR Value 2.2k" in window.text(asc_file)
    assert "now shows it" in data["hint"]
    assert data["observations"] == []


async def test_an_edit_is_refused_while_the_window_holds_a_different_sheet(
    asc_state: SessionState, asc_file: Path, window: Window
):
    before = file_bytes(asc_file)
    held = decode_windows_1252(before).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    window.shows(asc_file, held)

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    assert data["error"]["code"] == "open_window_differs"
    assert data["error"]["stage"] == "window_check"
    assert "only in the window: SYMBOL res 128 112 R90" in data["error"]["message"]
    assert data["stages"] == [
        {
            "stage": "window_check",
            "ok": False,
            "error": "the open LTspice window differs from the file",
        }
    ]
    assert "Save the sheet in LTspice (Ctrl+S)" in data["hint"]
    # Neither side was touched.
    assert file_bytes(asc_file) == before
    assert window.text(asc_file) == held
    assert data["sha256"] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize("name", ["older_version", "symbol_attributes"])
async def test_a_sheet_ltspice_rewrote_on_opening_is_not_taken_for_an_unsaved_one(
    asc_state: SessionState, work_dir: Path, window: Window, name: str
):
    """The window's copy differs from the file in text and not in content: the
    recorded pair, an older first line, wires in another order, attributes
    reordered and one written with a micro sign."""
    sheet, held = recorded_pair(name, work_dir)
    assert held != as_the_server_reads(sheet)
    window.shows(sheet, held)

    data = await edit(asc_state, sheet)

    assert data["outcome"] == "complete"
    assert data["open_in_ltspice"] == [{"pid": PID, "version": "26.1.1", "shown": True}]
    assert window.text(sheet) == as_the_server_reads(sheet)


async def test_a_sheet_stored_as_utf8_matches_the_window_that_read_it_as_cp1252(
    asc_state: SessionState, asc_file: Path, window: Window
):
    """LTspice reads a micro sign stored as UTF-8 as two characters, and shows
    those: that window holds the file, not a change to it."""
    held = store_as_utf8(asc_file)
    assert "Âµ" in held
    window.shows(asc_file, held)

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["open_in_ltspice"][0]["shown"] is True


async def test_a_dry_run_reports_a_window_that_differs_and_is_not_refused(
    asc_state: SessionState, asc_file: Path, window: Window
):
    held = as_ltspice_reads(asc_file).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    window.shows(asc_file, held)

    data = await edit(asc_state, asc_file, dry_run=True)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    assert len(data["observations"]) == 1
    assert "holds a different one from the file" in data["observations"][0]
    assert "A commit is refused until it is saved or closed." in data["observations"][0]
    assert window.text(asc_file) == held
    assert "open_in_ltspice" not in data


async def test_a_sheet_no_window_has_open_is_edited_as_before(
    asc_state: SessionState, asc_file: Path, window: Window
):
    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert "open_in_ltspice" not in data
    assert data["hint"].startswith("Committed.")
    assert "LTspice" not in data["hint"]
    assert data["observations"] == []


async def test_a_new_sheet_asks_no_window(asc_state: SessionState, work_dir: Path, window: Window):
    """Nothing can have a file open that does not exist yet, so nothing is asked:
    a bridge that never answers would otherwise stop this build."""
    window.shows(None, "", silent_on="status")

    data = await edit(
        asc_state,
        work_dir / "new.asc",
        ops=[{"op": "add_directive", "instruction": ".op"}],
        base="blank",
        expected_sha256=None,
    )

    assert data["outcome"] == "complete"
    assert data["observations"] == []


async def test_a_window_that_cannot_be_updated_is_reported_and_the_commit_stands(
    asc_state: SessionState, asc_file: Path, window: Window
):
    window.shows(asc_file, as_ltspice_reads(asc_file), silent_on="set_design_content")
    silent = asc_state.open_windows._command
    asc_state.open_windows = OpenWindows(silent, timeout=1.0)  # timing: the deadline under test
    before = file_bytes(asc_file)

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"
    assert file_bytes(asc_file) != before
    assert data["sha256"] == digest(asc_file)
    assert data["open_in_ltspice"] == [
        {
            "pid": PID,
            "version": "26.1.1",
            "shown": False,
            "reason": f"{BRIDGE_PROGRAM} did not answer within 1 s and was ended",
        }
    ]
    assert "could not be updated" in data["hint"]
    assert "saving from it would overwrite this edit" in data["hint"]


async def test_a_bridge_that_cannot_say_does_not_stop_the_edit(
    asc_state: SessionState, asc_file: Path, window: Window
):
    window.shows(asc_file, as_ltspice_reads(asc_file), silent_on="status")
    silent = asc_state.open_windows._command
    asc_state.open_windows = OpenWindows(silent, timeout=1.0)  # timing: the deadline under test

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"
    assert "open_in_ltspice" not in data
    assert data["observations"] == [
        "LTspice could not be asked whether Draft1.asc is open in a window "
        f"({BRIDGE_PROGRAM} did not answer within 1 s and was ended); one that has "
        "it open still shows the sheet as it was."
    ]


async def test_a_window_that_closes_before_the_commit_is_shown_as_not_updated(
    asc_state: SessionState, asc_file: Path, window: Window, monkeypatch: pytest.MonkeyPatch
):
    window.shows(asc_file, as_ltspice_reads(asc_file))
    commit = se._commit_asc

    def close_then_commit(*args: Any) -> Any:
        window.closes()
        return commit(*args)

    monkeypatch.setattr(se, "_commit_asc", close_then_commit)

    data = await edit(asc_state, asc_file)

    assert data["commit_state"] == "committed"
    assert data["open_in_ltspice"] == [
        {
            "pid": PID,
            "version": "26.1.1",
            "shown": False,
            "reason": "no live LTspice instance with that pid",
        }
    ]


async def test_with_the_setting_off_no_window_is_asked(
    asc_state: SessionState, asc_file: Path, window: Window
):
    held = as_ltspice_reads(asc_file).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    window.shows(asc_file, held)
    asc_state.open_windows = OpenWindows.detect([], enabled=False)

    data = await edit(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert "open_in_ltspice" not in data
    assert window.text(asc_file) == held
