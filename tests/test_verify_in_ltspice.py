"""verify_circuit(in_ltspice=True): the checked file opened in the user's LTspice.

After an assistant builds or changes a sheet, the person has to find it and
open it to look. Asked to, the server opens it in the LTspice window that is
already running and puts it in front, and starts LTspice when none is (the
stand-in for that start is ``FakeStart``). These go through the real handler
against the stand-in bridge (``tests/fake_ltspice_bridge.py``, replayed
against a recording of LTspice 26.1.1 in ``test_ltspice_bridge.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import verify
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit
from tests._ltspice_window import (
    PID,
    STARTED_PID,
    FakeStart,
    a_window,
    as_ltspice_reads,
    put_windows,
    read_world,
)


def window_of(world: Path) -> dict[str, Any]:
    return read_world(world)["windows"][0]


def one_window(state: SessionState, world: Path, designs: dict[str, str] | None = None) -> None:
    put_windows(state, world, [a_window(designs)])


async def check(state: SessionState, path: Path, **arguments: Any) -> dict[str, Any]:
    result = await handle_verify_circuit(
        VerifyCircuitInput(path=str(path), checks=["layout"], **arguments), state
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(verify._OUTPUT_SCHEMA).validate(data)
    return data


async def test_the_sheet_is_opened_in_the_window_and_put_in_front(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    one_window(asc_state, world)

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": True,
        "path": str(asc_file),
        "pid": PID,
        "version": "26.1.1",
        "already_open": False,
    }
    window = window_of(world)
    assert list(window["designs"]) == [str(asc_file)]
    assert window["active"] == str(asc_file)
    assert data["hint"].endswith("In front in LTspice.")
    # The checks are made all the same.
    assert data["checks_run"] == ["layout"]


async def test_a_sheet_the_window_already_has_is_put_in_front_as_it_holds_it(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    other = str(asc_file.with_name("other.asc"))
    one_window(
        asc_state, world, {str(asc_file): as_ltspice_reads(asc_file), other: "Version 4.1\n"}
    )
    assert window_of(world).get("active") is None  # the other sheet is the one in front

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": True,
        "path": str(asc_file),
        "pid": PID,
        "version": "26.1.1",
        "already_open": True,
        "differs_from_file": False,
    }
    assert window_of(world)["active"] == str(asc_file)
    assert data["hint"].endswith("In front in LTspice.")


async def test_a_window_showing_another_sheet_than_the_file_is_said_to(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    """LTspice does not read a file again, so the person may be looking at
    something else than was checked."""
    held = as_ltspice_reads(asc_file).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    world = tmp_path / "world.json"
    one_window(asc_state, world, {str(asc_file): held})

    data = await check(asc_state, asc_file, in_ltspice=True)

    shown = data["ltspice"]
    assert (shown["shown"], shown["already_open"], shown["differs_from_file"]) == (
        True,
        True,
        True,
    )
    assert "only in the window: SYMBOL res 128 112 R90" in shown["difference"]
    assert "shows a different one from the file that was checked" in data["hint"]
    assert window_of(world)["designs"][str(asc_file)] == held


async def test_a_netlist_is_opened_too(asc_state: SessionState, work_dir: Path, tmp_path: Path):
    deck = work_dir / "amp.cir"
    deck.write_text("* amp\nR1 in out 1k\nV1 in 0 1\n.op\n.end\n", encoding="utf-8")
    world = tmp_path / "world.json"
    one_window(asc_state, world)

    result = await handle_verify_circuit(
        VerifyCircuitInput(path=str(deck), checks=["syntax"], in_ltspice=True), asc_state
    )

    assert result.structured_content is not None
    assert result.structured_content["ltspice"] == {
        "shown": True,
        "path": str(deck),
        "pid": PID,
        "version": "26.1.1",
        "already_open": False,
    }
    assert list(window_of(world)["designs"]) == [str(deck)]


async def test_with_no_window_open_ltspice_is_started_and_shows_it(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    start = FakeStart(world)
    put_windows(asc_state, world, [], start=start)

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": True,
        "path": str(asc_file),
        "pid": STARTED_PID,
        "version": "26.1.1",
        "already_open": False,
        "started": True,
    }
    assert start.calls == 1
    assert window_of(world)["active"] == str(asc_file)
    assert data["hint"].endswith("LTspice was started, with it in front.")


async def test_a_window_that_is_open_is_used_and_nothing_is_started(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    start = FakeStart(world)
    put_windows(asc_state, world, [a_window()], start=start)

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"]["shown"] is True
    assert data["ltspice"]["pid"] == PID
    assert "started" not in data["ltspice"]
    assert start.calls == 0


async def test_an_ltspice_that_cannot_be_started_is_reported(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    put_windows(asc_state, world, [], start=FakeStart(world, error=OSError("access is denied")))

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": False,
        "path": str(asc_file),
        "reason": "LTspice could not be started: access is denied",
    }
    assert data["outcome"] != "failed"


async def test_an_ltspice_that_opens_no_window_is_reported(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ltspice_mcp.lib import ltspice_window

    monkeypatch.setattr(ltspice_window, "_STARTED_S", 0.3)  # timing: the wait under test
    world = tmp_path / "world.json"
    put_windows(asc_state, world, [], start=FakeStart(world, opens_window=False))

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"]["shown"] is False
    assert data["ltspice"]["reason"] == (
        "LTspice was started and did not open a window within 0.3 s"
    )


async def test_with_starting_turned_off_and_no_window_open_nothing_is_started(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    put_windows(asc_state, world, [])

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": False,
        "path": str(asc_file),
        "reason": "no LTspice window is open, and none is started for this",
    }
    assert "Not opened in LTspice: no LTspice window is open" in data["hint"]
    assert data["checks_run"] == ["layout"]
    assert data["outcome"] != "failed"


async def test_where_windows_cannot_be_reached_it_says_so(asc_state: SessionState, asc_file: Path):
    asc_state.open_windows = OpenWindows(None, unavailable="needs the server on Windows itself")

    data = await check(asc_state, asc_file, in_ltspice=True)

    assert data["ltspice"] == {
        "shown": False,
        "path": str(asc_file),
        "reason": "LTspice windows cannot be reached here: needs the server on Windows itself",
    }


async def test_without_the_argument_ltspice_is_not_asked(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    world = tmp_path / "world.json"
    one_window(asc_state, world)

    data = await check(asc_state, asc_file)

    assert "ltspice" not in data
    assert window_of(world)["designs"] == {}
    assert "LTspice" not in data["hint"]
