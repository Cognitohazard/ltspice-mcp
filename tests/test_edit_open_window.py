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

import ast
import asyncio
import hashlib
import re
import shutil
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import ltspice_window
from ltspice_mcp.lib.encoding import decode_spice_bytes, decode_windows_1252
from ltspice_mcp.lib.ltspice_bridge import BridgeError, BridgeSession
from ltspice_mcp.lib.ltspice_window import OpenSheet, OpenWindows
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests._ltspice_window import (
    PID,
    FakeFrame,
    FrameWhoseSheetChangesAsItAsks,
    a_window,
    as_ltspice_reads,
    digest,
    fake_command,
    follow_open_window_referral,
    put_windows,
    read_world,
    write_world,
)
from tests._source_scan import modules, scopes
from tests.conftest import LIVENESS_S, await_until, written
from tests.fake_ltspice_bridge import silence_of
from tests.ltspice_bridge_recorder import FIXTURES, INPUTS, observed
from tests.ltspice_recorder import NEUTRAL_DIR

#: The stand-in bridge runs under this interpreter, which is what a message names.
BRIDGE_PROGRAM = Path(sys.executable).name
SET_VALUE = [{"op": "set_component_value", "reference": "R1", "value": "2.2k"}]


def file_bytes(sheet: Path) -> bytes:
    return sheet.read_bytes()


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


class _KeptSessions(OpenWindows):
    """The windows behind the stand-in bridge, each bridge session kept as it
    starts, so that a test can have the one in progress pass its deadline."""

    def __init__(self, world: Path) -> None:
        super().__init__(fake_command(world), timeout=LIVENESS_S, frame=FakeFrame(world))
        self.sessions: list[BridgeSession] = []

    def _session(self, waits: float = 0.0) -> BridgeSession:
        session = super()._session(waits)
        self.sessions.append(session)
        return session


class Window:
    """One LTspice window, as the stand-in bridge reports it."""

    def __init__(self, state: SessionState, world: Path) -> None:
        self._state = state
        self._world = world
        put_windows(state, world, [a_window()])

    def shows(self, sheet: Path | None, text: str, **extra: Any) -> None:
        """Have the window hold ``text`` for ``sheet``; with no sheet, nothing is open."""
        write_world(
            self._world, [a_window(None if sheet is None else {str(sheet): text})], **extra
        )

    def answers_within(self, seconds: float) -> None:
        """Give the bridge ``seconds`` for a whole session, for a test of the deadline."""
        self._state.open_windows = OpenWindows(fake_command(self._world), timeout=seconds)

    async def silent_until_the_deadline(self, call: Coroutine[Any, Any, dict]) -> dict:
        """Run ``call`` under the liveness cap, and have the session's deadline
        pass once the bridge has gone silent (``silence_of``), and not before."""
        windows = _KeptSessions(self._world)
        self._state.open_windows = windows
        running = asyncio.create_task(call)
        silent = written(silence_of(self._world), str)
        await await_until(lambda: running.done() or silent(), what="the bridge to go silent")
        if not running.done():
            [waiting] = [session for session in windows.sessions if not session._ended]
            # What the session's watchdog does when its deadline passes.
            await asyncio.to_thread(waiting._expire)
        return await running

    def closes(self) -> None:
        write_world(self._world, [])

    def text(self, sheet: Path) -> str:
        return read_world(self._world)["windows"][0]["designs"][str(sheet)]


@pytest.fixture
def window(asc_state: SessionState, tmp_path: Path) -> Window:
    return Window(asc_state, tmp_path / "world.json")


async def edit(state: SessionState, sheet: Path, ops: list[dict] | None = None, **kw: Any) -> dict:
    arguments: dict[str, Any] = {
        "path": str(sheet),
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
    assert "observations" not in data


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
    assert "open_window='save'" in data["hint"]
    # Neither side was touched.
    assert file_bytes(asc_file) == before
    assert window.text(asc_file) == held
    assert data["sha256"] == hashlib.sha256(before).hexdigest()


async def test_the_call_a_refusal_under_a_different_window_names_is_taken(
    asc_state: SessionState, asc_file: Path, window: Window
):
    """Sent as the hint writes it, with the expected_sha256 the refused call
    sent, the call drops the window's copy, and the edit then goes through."""
    sent = digest(asc_file)
    window.shows(asc_file, a_change(asc_file))
    refused = await edit(asc_state, asc_file, expected_sha256=sent)
    assert refused["error"]["code"] == "open_window_differs"

    settled = await follow_open_window_referral(
        asc_state, refused["hint"], asc_file, "discard", sha256=sent
    )

    assert settled["outcome"] == "complete", settled
    assert window.text(asc_file) == as_ltspice_reads(asc_file)
    assert (await edit(asc_state, asc_file))["commit_state"] == "committed"


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of ``tree``'s strings that stand alone as a statement: the
    docstrings of its module, classes, functions and attributes, which no
    reply sends."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }


def test_every_text_that_sends_the_caller_to_open_window_writes_the_call_out():
    """Replies named edit_schematic's open_window in prose, without the
    expected_sha256 the call needs, and the call they named was refused. Every
    text of the package that names both is written by
    ``_base.open_window_call``, which writes out each argument the call needs,
    and the tests above follow each such reply."""
    offenders = []
    for module, tree in modules().items():
        docstrings = _docstrings(tree)
        for scope, nodes in scopes(tree):
            if (module, scope) == ("tools/_base.py", "open_window_call"):
                continue
            for node in nodes:
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and re.search(r"\bedit_schematic\b", node.value)
                    and re.search(r"\bopen_window\b", node.value)
                ):
                    offenders.append(f"{module}:{node.lineno} ({scope})")
    assert offenders == []


async def test_a_refusal_names_part_of_a_long_difference_and_the_call_that_reads_it_all(
    asc_state: SessionState, asc_file: Path, window: Window
):
    """The refusal names a few entries a side by their first line; the call
    its hint names reads every entry, in full."""
    added = [f"WIRE {x} 2000 {x} 2016" for x in range(0, 16 * 30, 16)]
    window.shows(asc_file, as_ltspice_reads(asc_file) + "".join(f"{w}\n" for w in added))

    data = await edit(asc_state, asc_file)

    assert data["error"]["code"] == "open_window_differs"
    assert "; and 27 more" in data["error"]["message"]
    assert se._DIFFERENCE_IN_ONE in data["hint"]
    queries: Any = [{"kind": "open_in_ltspice", "path": str(asc_file)}]
    read = await handle_inspect(InspectInput(queries=queries), asc_state)
    assert read.structured_content is not None
    (item,) = read.structured_content["results"]
    assert item["data"]["only_in_window"] == sorted(added)
    assert "only_in_file" not in item["data"]
    assert "next_cursor" not in item


def a_change(sheet: Path) -> str:
    """``sheet`` as a window holds it with a resistor value nobody saved."""
    return decode_windows_1252(file_bytes(sheet)).replace("SYMATTR Value 1k", "SYMATTR Value 5k")


def holding_a_change(window: Window, sheet: Path, **extra: Any) -> str:
    """Have the window hold ``a_change`` of ``sheet``; returns that copy."""
    held = a_change(sheet)
    window.shows(sheet, held, **extra)
    return held


def write_protected_box(sheet: Path) -> str:
    """The words of the box LTspice showed when told to save a write-protected
    sheet, as recorded, for ``sheet``."""
    (words,) = observed("ltspice26")[
        "Save of a write-protected sheet leaves the file as it was, takes no input and "
        "shows a box with these words"
    ][2]
    return words.replace(f"{NEUTRAL_DIR}\\locked.asc", str(sheet))


async def save(state: SessionState, sheet: Path, **kw: Any) -> dict:
    return await edit(state, sheet, ops=[], open_window="save", **kw)


async def test_with_the_users_agreement_ltspice_saves_the_window_and_the_edit_goes_on(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """The refusal says to ask the user. Once they agree, LTspice saves the
    sheet from the window, and an edit made against the saved sheet commits."""
    held = holding_a_change(window, asc_file)
    refused = await edit(asc_state, asc_file)
    assert refused["error"]["code"] == "open_window_differs"

    data = await save(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"
    assert data["sha256"] == digest(asc_file)
    assert data["open_in_ltspice"] == [
        {"pid": PID, "version": "26.1.1", "shown": True, "saved": True}
    ]
    assert "Read it again before editing it" in data["hint"]
    # LTspice wrote the window's copy, and was sent its own Save to do it.
    assert decode_windows_1252(file_bytes(asc_file)) == held
    assert read_world(tmp_path / "world.json")["windows"][0]["commands"] == ["Save"]

    edited = await edit(asc_state, asc_file, expected_sha256=data["sha256"])
    assert edited["commit_state"] == "committed"
    assert "SYMATTR Value 2.2k" in window.text(asc_file)


@pytest.mark.parametrize(
    ("open_window", "done", "untouched"),
    [("save", "saved", "commands"), ("discard", "dropped", "closed")],
)
async def test_nothing_is_done_where_no_window_holds_a_different_sheet(
    asc_state: SessionState,
    asc_file: Path,
    window: Window,
    tmp_path: Path,
    open_window: str,
    done: str,
    untouched: str,
):
    before = file_bytes(asc_file)
    window.shows(asc_file, as_ltspice_reads(asc_file))

    data = await edit(asc_state, asc_file, ops=[], open_window=open_window)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    assert "open_in_ltspice" not in data
    assert data["hint"] == (
        f"No LTspice window holds a sheet different from the file, so nothing was {done}."
    )
    assert file_bytes(asc_file) == before
    assert untouched not in read_world(tmp_path / "world.json")["windows"][0]


async def test_a_stale_revision_saves_nothing(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    holding_a_change(window, asc_file)

    data = await save(asc_state, asc_file, expected_sha256="0" * 64)

    assert data["error"]["code"] == "revision_conflict"
    assert "commands" not in read_world(tmp_path / "world.json")["windows"][0]


@pytest.mark.parametrize(
    "other",
    [
        {"ops": SET_VALUE},
        {"ops": [], "dry_run": True},
        {"ops": [], "base": "blank"},
        {"ops": [], "return_views": ["pin_legend"]},
    ],
)
async def test_a_save_is_a_call_of_its_own(
    asc_state: SessionState, asc_file: Path, window: Window, other: dict
):
    """The edit is made in another call, against the saved sheet once read."""
    holding_a_change(window, asc_file)
    with pytest.raises(NetlistError, match="open_window takes only path and expected_sha256"):
        await edit(asc_state, asc_file, open_window="save", **other)


async def test_other_windows_with_the_sheet_open_are_handed_the_saved_one(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """A second LTspice that held the file's sheet would otherwise show the old one."""
    held = a_change(asc_file)
    world = tmp_path / "world.json"
    write_world(
        world,
        [
            a_window({str(asc_file): held}),
            a_window({str(asc_file): as_ltspice_reads(asc_file)}, pid=PID + 1),
        ],
    )

    data = await save(asc_state, asc_file)

    assert data["commit_state"] == "committed"
    assert data["open_in_ltspice"] == [
        {"pid": PID, "version": "26.1.1", "shown": True, "saved": True},
        {"pid": PID + 1, "version": "26.1.1", "shown": True},
    ]
    assert [entry["designs"][str(asc_file)] for entry in read_world(world)["windows"]] == [
        held,
        held,
    ]


async def discard(state: SessionState, sheet: Path, **kw: Any) -> dict:
    return await edit(state, sheet, ops=[], open_window="discard", **kw)


async def test_with_the_users_agreement_the_window_is_closed_and_opened_from_the_file(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """The user changed the sheet in LTspice, so LTspice asks before closing
    it; the answer is No, and the window then holds the file."""
    before = file_bytes(asc_file)
    world = tmp_path / "world.json"
    write_world(world, [a_window({str(asc_file): a_change(asc_file)}, unsaved=[str(asc_file)])])

    data = await discard(asc_state, asc_file)

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "not_committed"
    assert data["sha256"] == hashlib.sha256(before).hexdigest()
    assert data["open_in_ltspice"] == [
        {"pid": PID, "version": "26.1.1", "shown": True, "reopened": True}
    ]
    assert "the changes made in the window are gone" in data["hint"]
    assert file_bytes(asc_file) == before
    entry = read_world(world)["windows"][0]
    assert entry["answers"] == ["no"]
    assert window.text(asc_file) == as_ltspice_reads(asc_file)

    edited = await edit(asc_state, asc_file)
    assert edited["commit_state"] == "committed"


@pytest.mark.parametrize("meanwhile", ["quits LTspice", "closes the sheet"])
async def test_a_window_without_the_sheet_by_the_time_it_is_closed_has_nothing_to_drop(
    asc_state: SessionState,
    asc_file: Path,
    window: Window,
    monkeypatch: pytest.MonkeyPatch,
    meanwhile: str,
):
    """The user's changes went with the window, or with the sheet they closed,
    after the windows were read."""
    window.shows(asc_file, a_change(asc_file))
    windows = asc_state.open_windows
    holding = windows.holding

    def then_gone(sheet: Path) -> list[OpenSheet]:
        held = holding(sheet)
        if meanwhile == "quits LTspice":
            window.closes()
        else:
            window.shows(None, "")
        return held

    monkeypatch.setattr(windows, "holding", then_gone)

    data = await discard(asc_state, asc_file)

    assert data["commit_state"] == "not_committed", data
    assert "open_in_ltspice" not in data
    assert data["hint"] == (
        "No LTspice window holds a sheet different from the file, so nothing was dropped."
    )


def two_windows(tmp_path: Path, sheet: Path) -> str:
    """Two LTspice windows with ``sheet`` open, each holding a copy of its own;
    returns the second window's copy."""
    second = a_change(sheet).replace("5k", "6k")
    write_world(
        tmp_path / "world.json",
        [a_window({str(sheet): a_change(sheet)}), a_window({str(sheet): second}, pid=PID + 1)],
    )
    return second


# A part is named by its first line, so the copies read alike here; the
# open_in_ltspice listing gives each entry whole.
CHANGED_R1 = "only in the window: SYMBOL res 128 112 R90. only in the file: SYMBOL res 128 112 R90"
TWO_COPIES = f"process {PID}: {CHANGED_R1}; process {PID + 1}: {CHANGED_R1}"


async def test_every_window_with_a_different_copy_is_closed_and_opened_from_the_file(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """Unlike a save, a discard leaves no choice between copies: the file's is kept."""
    world = tmp_path / "world.json"
    two_windows(tmp_path, asc_file)

    data = await discard(asc_state, asc_file)

    assert [(row["pid"], row.get("reopened")) for row in data["open_in_ltspice"]] == [
        (PID, True),
        (PID + 1, True),
    ]
    assert {entry["designs"][str(asc_file)] for entry in read_world(world)["windows"]} == {
        as_ltspice_reads(asc_file)
    }


async def test_a_window_changed_while_ltspice_asks_is_left_open(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """The user agreed to drop the changes they had; ones made after the
    window was read are not dropped."""
    world = tmp_path / "world.json"
    held = a_change(asc_file)
    later = held.replace("5k", "7k")
    write_world(world, [a_window({str(asc_file): held}, unsaved=[str(asc_file)])])
    asc_state.open_windows = OpenWindows(
        fake_command(world),
        timeout=LIVENESS_S,
        frame=FrameWhoseSheetChangesAsItAsks(world, asc_file, later),
    )

    data = await discard(asc_state, asc_file)

    assert data["error"]["code"] == "window_not_discarded"
    assert data["error"]["message"] == (
        f"LTspice did not drop the changes to {asc_file.name}: its copy changed after it "
        "was read, so it was left open. Read the sheet again before editing it."
    )
    entry = read_world(world)["windows"][0]
    assert entry["answers"] == ["cancel"]
    assert entry["designs"][str(asc_file)] == later


async def test_two_windows_with_different_copies_save_nothing_until_one_is_chosen(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """Which copy to keep is the user's choice."""
    before = file_bytes(asc_file)
    two_windows(tmp_path, asc_file)

    data = await save(asc_state, asc_file)

    assert data["error"]["code"] == "window_not_chosen"
    assert data["error"]["message"] == (
        f"Nothing was saved: 2 LTspice windows hold different copies ({TWO_COPIES})."
    )
    assert "window_pid=<that window's process>" in data["hint"]
    assert file_bytes(asc_file) == before
    assert all("commands" not in entry for entry in read_world(tmp_path / "world.json")["windows"])


async def test_the_copy_the_user_chose_is_saved_and_the_other_window_is_handed_it(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    second = two_windows(tmp_path, asc_file)

    data = await save(asc_state, asc_file, window_pid=PID + 1)

    assert data["commit_state"] == "committed", data
    assert data["sha256"] == digest(asc_file)
    assert decode_windows_1252(file_bytes(asc_file)) == second
    assert data["open_in_ltspice"] == [
        {"pid": PID + 1, "version": "26.1.1", "shown": True, "saved": True},
        {"pid": PID, "version": "26.1.1", "shown": True},
    ]
    held = [
        entry["designs"][str(asc_file)] for entry in read_world(tmp_path / "world.json")["windows"]
    ]
    assert held == [second, second]


@pytest.mark.parametrize(
    ("chosen", "reason"),
    [(PID + 7, "does not have the sheet open"), (PID + 1, "holds the file's copy")],
)
async def test_a_window_pid_without_a_different_copy_saves_nothing(
    asc_state: SessionState,
    asc_file: Path,
    window: Window,
    tmp_path: Path,
    chosen: int,
    reason: str,
):
    before = file_bytes(asc_file)
    write_world(
        tmp_path / "world.json",
        [
            a_window({str(asc_file): a_change(asc_file)}),
            a_window({str(asc_file): as_ltspice_reads(asc_file)}, pid=PID + 1),
        ],
    )

    data = await save(asc_state, asc_file, window_pid=chosen)

    assert data["error"]["code"] == "window_not_chosen"
    assert data["error"]["message"] == (
        f"Nothing was saved: LTspice process {chosen} {reason}, and one LTspice window "
        f"holds a different copy (process {PID}: {CHANGED_R1})."
    )
    assert file_bytes(asc_file) == before


async def test_a_window_that_does_not_take_the_chosen_copy_is_opened_again_from_the_file(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """The user chose to drop that window's copy, so it is closed and opened
    again from the saved file, though it never held the file's sheet."""
    second = two_windows(tmp_path, asc_file)
    asc_state.open_windows = _DoesNotTakeIt(tmp_path / "world.json")

    data = await save(asc_state, asc_file, window_pid=PID + 1)

    assert data["commit_state"] == "committed", data
    assert data["open_in_ltspice"][1] == {
        "pid": PID,
        "version": "26.1.1",
        "shown": True,
        "reopened": True,
    }
    assert window.text(asc_file) == as_ltspice_reads(asc_file)
    assert decode_windows_1252(file_bytes(asc_file)) == second


@pytest.mark.parametrize("open_window", [None, "discard"])
async def test_window_pid_goes_with_a_save(
    asc_state: SessionState, asc_file: Path, window: Window, open_window: str | None
):
    with pytest.raises(NetlistError, match="window_pid goes with open_window='save'"):
        await edit(asc_state, asc_file, ops=[], open_window=open_window, window_pid=PID)


async def test_an_edit_under_two_different_windows_names_both_and_asks_which_to_keep(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    two_windows(tmp_path, asc_file)

    data = await edit(asc_state, asc_file)

    assert data["error"]["code"] == "open_window_differs"
    assert data["error"]["message"] == (
        f"2 LTspice windows have this sheet open and each holds a different one from the "
        f"file ({TWO_COPIES}). Nothing was written."
    )
    assert data["hint"] == se._WINDOWS_REMEDY


@pytest.mark.parametrize("choice", ["save", "discard"])
async def test_each_call_a_refusal_under_two_different_windows_names_is_taken(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path, choice: str
):
    """Sent as the hint writes it, the save of the copy the user chose names
    that window's process, and the discard of every copy needs none."""
    sent = digest(asc_file)
    second = two_windows(tmp_path, asc_file)
    refused = await edit(asc_state, asc_file, expected_sha256=sent)
    assert refused["error"]["code"] == "open_window_differs"

    settled = await follow_open_window_referral(
        asc_state, refused["hint"], asc_file, choice, sha256=sent, window_pid=PID + 1
    )

    assert settled["outcome"] == "complete", settled
    kept = second if choice == "save" else as_ltspice_reads(asc_file)
    held = {
        entry["designs"][str(asc_file)] for entry in read_world(tmp_path / "world.json")["windows"]
    }
    assert held == {kept}


async def test_the_box_ltspice_shows_instead_of_saving_is_quoted_and_cancelled(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """A write-protected file: LTspice shows a box with only OK, which Cancel
    closes, leaving the window's copy as it was (recorded)."""
    before = file_bytes(asc_file)
    words = write_protected_box(asc_file)
    held = holding_a_change(window, asc_file, save_box=words)

    data = await save(asc_state, asc_file)

    assert data["outcome"] == "failed"
    assert data["commit_state"] == "not_committed"
    assert data["error"]["code"] == "window_not_saved"
    assert data["error"]["message"] == (
        f"LTspice did not save {asc_file.name}; it showed a box: {words}. Read the sheet "
        "again before editing it."
    )
    assert file_bytes(asc_file) == before
    entry = read_world(tmp_path / "world.json")["windows"][0]
    assert entry["answers"] == ["cancel"] and "box_words" not in entry
    assert window.text(asc_file) == held


async def test_a_window_that_does_not_save_is_reported(
    asc_state: SessionState, asc_file: Path, window: Window, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ltspice_window, "_SAVED_S", 0.3)  # timing: the wait under test
    before = file_bytes(asc_file)
    holding_a_change(window, asc_file, frame_ignores=True)

    data = await save(asc_state, asc_file)

    assert data["error"]["code"] == "window_not_saved"
    assert data["error"]["message"].startswith(
        f"LTspice did not save {asc_file.name} within 0.3 s"
    )
    assert file_bytes(asc_file) == before


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
    assert "A commit is refused until the window's copy is saved" in data["observations"][0]
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
    assert "observations" not in data


@pytest.mark.parametrize(
    "ops",
    [[], [{"op": "set_plot_panes", "analysis": "tran", "panes": [{"traces": ["V(out)"]}]}]],
    ids=["no ops", "plot panes only"],
)
async def test_a_call_that_leaves_the_sheet_as_it_is_asks_no_window(
    asc_state: SessionState, asc_file: Path, window: Window, ops: list[dict]
):
    """A window with unsaved changes stops a commit that would replace the
    sheet under it. A read, or a batch that writes only the plot settings
    beside the sheet, replaces nothing, so it is not stopped and not asked."""
    before = file_bytes(asc_file)
    held = decode_windows_1252(before).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    window.shows(asc_file, held)

    data = await edit(asc_state, asc_file, ops=ops)

    assert data["outcome"] == "complete"
    assert "observations" not in data
    assert "open_in_ltspice" not in data
    assert "window_check" not in [stage["stage"] for stage in data.get("stages", [])]
    assert file_bytes(asc_file) == before
    assert window.text(asc_file) == held


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
    assert "observations" not in data


async def test_a_window_that_does_not_take_the_edit_is_closed_and_opened_again(
    asc_state: SessionState, asc_file: Path, window: Window
):
    """The bridge never answers the request that hands the window the committed
    sheet; only that request meets the deadline, not the ones before it. The
    window still holds the sheet as it was, so it is closed and opened again
    from the committed file, and it then shows the edit."""
    window.shows(asc_file, as_ltspice_reads(asc_file), silent_on="set_design_content")
    before = file_bytes(asc_file)

    data = await window.silent_until_the_deadline(edit(asc_state, asc_file))

    assert data["outcome"] == "complete"
    assert data["commit_state"] == "committed"
    assert file_bytes(asc_file) != before
    assert data["sha256"] == digest(asc_file)
    assert data["open_in_ltspice"] == [
        {"pid": PID, "version": "26.1.1", "shown": True, "reopened": True}
    ]
    assert window.text(asc_file) == as_ltspice_reads(asc_file)
    assert "SYMATTR Value 2.2k" in window.text(asc_file)
    assert "closed there and opened again from the file" in data["hint"]


class _DoesNotTakeIt(_KeptSessions):
    """A window whose request to take a committed sheet fails. With
    ``meanwhile``, someone changes the sheet in it first; with ``after_all``,
    the request was carried out all the same."""

    def __init__(
        self, world: Path, meanwhile: str | None = None, *, after_all: bool = False
    ) -> None:
        super().__init__(world)
        self._world = world
        self._meanwhile = meanwhile
        self._after_all = after_all

    def show(self, sheet: OpenSheet, text: str) -> None:
        if self._after_all:
            super().show(sheet, text)
        if self._meanwhile is not None:
            changed = read_world(self._world)
            changed["windows"][0]["designs"][sheet.path] = self._meanwhile
            write_world(self._world, **changed)
        raise BridgeError("LTspice did not take it")


async def test_a_request_carried_out_after_all_is_not_taken_for_a_change(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """A request that timed out may still have been carried out: the window
    then holds the committed sheet, and is left as it is."""
    window.shows(asc_file, as_ltspice_reads(asc_file))
    asc_state.open_windows = _DoesNotTakeIt(tmp_path / "world.json", after_all=True)

    data = await edit(asc_state, asc_file)

    assert data["open_in_ltspice"] == [{"pid": PID, "version": "26.1.1", "shown": True}]
    assert "SYMATTR Value 2.2k" in window.text(asc_file)
    assert "closed" not in read_world(tmp_path / "world.json")["windows"][0]


async def test_a_window_changed_since_it_was_read_is_left_open_and_reported(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """Closing it would lose what was done there since, so it is not closed."""
    held = as_ltspice_reads(asc_file)
    window.shows(asc_file, held)
    meanwhile = held.replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    asc_state.open_windows = _DoesNotTakeIt(tmp_path / "world.json", meanwhile)

    data = await edit(asc_state, asc_file)

    assert data["commit_state"] == "committed"
    assert data["open_in_ltspice"] == [
        {
            "pid": PID,
            "version": "26.1.1",
            "shown": False,
            "reason": (
                "LTspice did not take it; it was not closed and opened again, because it "
                "holds changes made there since"
            ),
        }
    ]
    assert "saving from it would overwrite this edit" in data["hint"]
    assert window.text(asc_file) == meanwhile
    assert "closed" not in read_world(tmp_path / "world.json")["windows"][0]


async def test_the_call_the_note_of_a_window_left_showing_the_old_sheet_names_is_taken(
    asc_state: SessionState, asc_file: Path, window: Window, tmp_path: Path
):
    """The note sends the caller to edit_schematic's open_window once the user
    chooses. Sent as the note writes it, with the sha256 from where it says,
    the call opens the committed file in the window. The note named no
    expected_sha256, and the call it named was refused for want of one."""
    world = tmp_path / "world.json"
    held = as_ltspice_reads(asc_file)
    window.shows(asc_file, held)
    asc_state.open_windows = _DoesNotTakeIt(
        world, held.replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    )
    data = await edit(asc_state, asc_file)
    assert data["commit_state"] == "committed"
    assert data["open_in_ltspice"][0]["shown"] is False
    # The window is reached again as before; only the one request failed.
    put_windows(asc_state, world, read_world(world)["windows"])

    settled = await follow_open_window_referral(
        asc_state, data["hint"], asc_file, "discard", sha256=data["sha256"]
    )

    assert settled["outcome"] == "complete", settled
    assert window.text(asc_file) == as_ltspice_reads(asc_file)
    assert "SYMATTR Value 2.2k" in window.text(asc_file)


async def test_a_window_that_does_not_close_either_is_reported(
    asc_state: SessionState,
    asc_file: Path,
    window: Window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ltspice_window, "_CLOSED_S", 0.3)  # timing: the wait under test
    window.shows(asc_file, as_ltspice_reads(asc_file), frame_keeps_panes=True)
    asc_state.open_windows = _DoesNotTakeIt(tmp_path / "world.json")

    data = await edit(asc_state, asc_file)

    assert data["commit_state"] == "committed"
    (row,) = data["open_in_ltspice"]
    assert row["shown"] is False and "reopened" not in row
    assert row["reason"] == (
        "LTspice did not take it; closing it and opening it again failed too: LTspice "
        f"did not close {asc_file.name} within 0.3 s"
    )


async def test_a_bridge_that_cannot_say_does_not_stop_the_edit(
    asc_state: SessionState, asc_file: Path, window: Window
):
    window.shows(asc_file, as_ltspice_reads(asc_file), silent_on="status")
    # The deadline is the whole session's, its start included, and status is
    # the first request the session makes: a deadline that falls sooner, while
    # the bridge is still starting, ends the session the same way.
    window.answers_within(1.0)  # timing: the deadline under test

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
