"""inspect(kind="open_in_ltspice"): what LTspice has open, asked of LTspice.

Where a request about "this circuit" starts when nobody named a path. Through
the real handler against the stand-in bridge (``tests/fake_ltspice_bridge.py``,
replayed against a recording of LTspice 26.1.1 in ``test_ltspice_bridge.py``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib.ltspice_window import OpenWindows
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import inspect_tools
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests._ltspice_window import (
    PID,
    a_window,
    as_ltspice_reads,
    digest,
    fake_command,
    put_windows,
    write_world,
)


def a_second_sheet(asc_file: Path) -> Path:
    other = asc_file.with_name("other.asc")
    shutil.copyfile(asc_file, other)
    return other


def a_deck(work_dir: Path) -> Path:
    deck = work_dir / "amp.net"
    deck.write_text("* amp\nR1 in out 1k\n.end\n", encoding="utf-8")
    return deck


def a_sheet_outside(elsewhere: Path, asc_file: Path) -> Path:
    shutil.copyfile(asc_file, elsewhere / "private.asc")
    return elsewhere / "private.asc"


async def ask(state: SessionState) -> dict[str, Any]:
    result = await handle_inspect(InspectInput(queries=[{"kind": "open_in_ltspice"}]), state)  # type: ignore[list-item]
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(inspect_tools._OUTPUT_SCHEMA).validate(data)
    (item,) = data["results"]
    return item


async def test_it_lists_what_is_open_and_which_document_is_in_front(
    asc_state: SessionState,
    asc_file: Path,
    work_dir: Path,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    other, deck = a_second_sheet(asc_file), a_deck(work_dir)
    # A directory beside the sandbox, not in it.
    outside = a_sheet_outside(tmp_path_factory.mktemp("elsewhere"), asc_file)
    changed = as_ltspice_reads(other).replace("SYMATTR Value 1k", "SYMATTR Value 5k")
    put_windows(
        asc_state,
        tmp_path / "world.json",
        [
            {
                "pid": PID,
                "version": "26.1.1",
                "designs": {
                    str(asc_file): as_ltspice_reads(asc_file),
                    str(other): changed,
                    str(deck): "* amp\n",
                    str(outside): "Version 4.1\n",
                },
                "active": str(other),
            }
        ],
    )

    item = await ask(asc_state)

    assert item["ok"] is True
    data = item["data"]
    assert (data["windows"], data["total"]) == (1, 4)
    assert data["designs"] == [
        {
            "path": str(asc_file),
            "kind": "schematic",
            "active": False,
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": True,
            "sha256": digest(asc_file),
            "differs_from_file": False,
        },
        {
            "path": str(other),
            "kind": "schematic",
            "active": True,
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": True,
            "sha256": digest(other),
            "differs_from_file": True,
            "difference": (
                "only in the window: SYMBOL res 128 112 R90. "
                "only in the file: SYMBOL res 128 112 R90"
            ),
        },
        # A netlist is listed; only a sheet is compared with its file.
        {
            "path": str(deck),
            "kind": "netlist",
            "active": False,
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": True,
        },
        # Outside the sandbox: named, and neither the file nor the window's copy is read.
        {
            "path": str(outside),
            "kind": "schematic",
            "active": False,
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": False,
        },
    ]
    assert "refused by edit_schematic until it is saved or closed" in data["hint"]
    assert "outside the sandbox is listed and not read" in data["hint"]


async def test_the_digest_it_reports_is_the_one_an_edit_takes(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic

    put_windows(
        asc_state,
        tmp_path / "world.json",
        [
            {
                "pid": PID,
                "version": "26.1.1",
                "designs": {str(asc_file): as_ltspice_reads(asc_file)},
            }
        ],
    )
    (design,) = (await ask(asc_state))["data"]["designs"]
    assert design["active"] is True

    result = await handle_edit_schematic(
        EditSchematicInput.model_validate(
            {
                "target": design["path"],
                "expected_sha256": design["sha256"],
                "ops": [{"op": "set_component_value", "reference": "R1", "value": "2.2k"}],
            }
        ),
        asc_state,
    )

    assert result.structured_content is not None
    assert result.structured_content["commit_state"] == "committed"
    assert result.structured_content["open_in_ltspice"][0]["shown"] is True


async def test_with_ltspice_running_and_nothing_open_the_list_is_empty(
    asc_state: SessionState, tmp_path: Path
):
    put_windows(asc_state, tmp_path / "world.json", [a_window()])

    item = await ask(asc_state)

    assert item["ok"] is True
    assert item["data"] == {"windows": 1, "designs": [], "total": 0}


async def test_with_no_ltspice_running_there_are_no_windows(
    asc_state: SessionState, tmp_path: Path
):
    put_windows(asc_state, tmp_path / "world.json", [])

    item = await ask(asc_state)

    assert item["ok"] is True
    assert item["data"] == {"windows": 0, "designs": [], "total": 0}


async def test_where_windows_cannot_be_read_the_query_fails_and_says_why(asc_state: SessionState):
    """Not an empty list: nothing open and cannot tell are different answers."""
    asc_state.open_windows = OpenWindows(
        None, unavailable="needs the server to run on Windows itself"
    )

    item = await ask(asc_state)

    assert item["ok"] is False
    assert item["error"]["code"] == "open_windows_unavailable"
    assert "needs the server to run on Windows itself" in item["error"]["message"]
    assert "open_window_sync" in item["error"]["message"]


async def test_a_bridge_that_does_not_answer_fails_the_query(
    asc_state: SessionState, tmp_path: Path
):
    write_world(tmp_path / "world.json", [], silent_on="status")
    silent = fake_command(tmp_path / "world.json")
    asc_state.open_windows = OpenWindows(silent, timeout=1.0)  # timing: the deadline under test

    item = await ask(asc_state)

    assert item["ok"] is False
    assert item["error"]["code"] == "open_windows_unreachable"
    assert "did not answer within 1 s" in item["error"]["message"]


@pytest.mark.parametrize("extra", [41, 0])
async def test_a_long_list_is_cut_and_counted(
    asc_state: SessionState,
    asc_file: Path,
    tmp_path: Path,
    extra: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(inspect_tools, "_OPEN_DESIGNS_LIMIT", 3)
    designs = {str(asc_file.with_name(f"deck{n}.net")): "* deck\n" for n in range(3 + extra)}
    put_windows(asc_state, tmp_path / "world.json", [a_window(designs)])

    data = (await ask(asc_state))["data"]

    assert len(data["designs"]) == 3
    assert data["total"] == 3 + extra
