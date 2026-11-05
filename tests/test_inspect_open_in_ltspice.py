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
    FakeStart,
    a_window,
    as_ltspice_reads,
    digest,
    fake_command,
    put_windows,
    write_world,
)
from tests.conftest import SyncApi, dispatched


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


#: The fixture sheet's resistor as a difference lists it: the symbol, then its
#: attributes in a fixed order.
R1 = (
    "SYMBOL res 128 112 R90\n"
    "SYMATTR InstName R1\n"
    "SYMATTR Value {value}\n"
    "WINDOW 0 0 56 VBottom 2\n"
    "WINDOW 3 32 56 VTop 2"
)


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
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": True,
            "sha256": digest(asc_file),
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
            # The sentence names the part; the lists say what changed in it.
            "only_in_window": [R1.format(value="5k")],
            "only_in_file": [R1.format(value="1k")],
        },
        # A netlist is listed; only a sheet is compared with its file.
        {
            "path": str(deck),
            "kind": "netlist",
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": True,
        },
        # Outside the sandbox: named, and neither the file nor the window's copy is read.
        {
            "path": str(outside),
            "kind": "schematic",
            "pid": PID,
            "version": "26.1.1",
            "in_sandbox": False,
        },
    ]
    assert (
        "refused by edit_schematic until the user chooses which to keep and "
        "edit_schematic's open_window saves or drops the window's copy"
    ) in data["hint"]
    assert "outside the sandbox is listed and not read" in data["hint"]


async def test_a_long_difference_is_listed_up_to_a_limit_and_the_rest_counted(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    wires = [f"WIRE {x} 2000 {x} 2016" for x in range(0, 16 * 30, 16)]
    held = as_ltspice_reads(asc_file) + "".join(f"{wire}\n" for wire in wires)
    put_windows(asc_state, tmp_path / "world.json", [a_window({str(asc_file): held})])

    (row,) = (await ask(asc_state))["data"]["designs"]

    assert row["differs_from_file"] is True
    assert row["only_in_window"] == sorted(wires)[:25]
    assert "only_in_file" not in row
    assert row["difference_omitted"] == 5


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
                "path": design["path"],
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
    assert item["data"] == {"windows": 1, "total": 0}


async def test_with_no_ltspice_running_there_are_no_windows(
    asc_state: SessionState, tmp_path: Path
):
    put_windows(asc_state, tmp_path / "world.json", [])

    item = await ask(asc_state)

    assert item["ok"] is True
    assert item["data"] == {"windows": 0, "total": 0}


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


async def test_capabilities_say_whether_ltspice_is_started_on_request(
    asc_state: SessionState, tmp_path: Path
):
    async def reported() -> dict[str, Any]:
        result = await handle_inspect(
            InspectInput.model_validate(
                {"queries": [{"kind": "capabilities", "fields": ["open_window_sync"]}]}
            ),
            asc_state,
        )
        assert result.structured_content is not None
        (item,) = result.structured_content["results"]
        return item["data"]["open_window_sync"]

    world = tmp_path / "world.json"
    put_windows(asc_state, world, [])
    assert await reported() == {"available": True}
    put_windows(asc_state, world, [], start=FakeStart(world))
    assert await reported() == {"available": True, "starts_ltspice": True}


# ---------------------------------------------------------------------------
# A long difference, read to the end
#
# Design principle "Safe edits" (docs/DESIGN.md): a list cut short says how to
# read the rest. A row whose difference is cut has a next_cursor beside
# difference_omitted, which the query's cursor takes back. Through the server's
# own dispatch, as a client calls it, and without run_code.
# ---------------------------------------------------------------------------


def wires(count: int, y: int) -> list[str]:
    """``count`` short wires on the row ``y``, as a difference lists them."""
    return sorted(f"WIRE {x} {y} {x} {y + 16}" for x in range(0, 16 * count, 16))


def lines(entries: list[str]) -> str:
    return "".join(f"{entry}\n" for entry in entries)


def a_long_difference(
    state: SessionState, sheet: Path, world: Path, *, in_window: int, in_file: int
) -> tuple[list[str], list[str]]:
    """Open ``sheet`` in a window whose copy has ``in_window`` wires the file
    lacks, and give the file ``in_file`` wires the copy lacks; both lists."""
    window_only, file_only = wires(in_window, 2000), wires(in_file, 3000)
    held = as_ltspice_reads(sheet) + lines(window_only)
    with sheet.open("a", encoding="cp1252", newline="\n") as file:
        file.write(lines(file_only))
    put_windows(state, world, [a_window({str(sheet): held})])
    return window_only, file_only


def add_a_wire(sheet: Path) -> None:
    """Change the file under a window that has it open."""
    sheet.write_bytes(sheet.read_bytes() + b"WIRE 0 4000 0 4016\n")


def row_for(item: dict[str, Any], sheet: Path) -> dict[str, Any]:
    assert item["ok"] is True, item
    (row,) = [row for row in item["data"]["designs"] if row["path"] == str(sheet)]
    return row


async def listed(state: SessionState, cursor: str | None = None) -> dict[str, Any]:
    """The open_in_ltspice item, asked through the server's dispatch."""
    query: dict[str, Any] = {"kind": "open_in_ltspice"}
    if cursor is not None:
        query["cursor"] = cursor
    (item,) = (await dispatched(state, "inspect", queries=[query]))["results"]
    return item


async def every_entry(state: SessionState, sheet: Path) -> tuple[list[str], list[str], int]:
    """Both sides of ``sheet``'s difference, read by following its cursor, and
    the queries asked."""
    row = row_for(await listed(state), sheet)
    in_window, in_file = list(row.get("only_in_window", [])), list(row.get("only_in_file", []))
    asked = 1
    while "next_cursor" in row:
        row = row_for(await listed(state, row["next_cursor"]), sheet)
        page_window, page_file = row.get("only_in_window", []), row.get("only_in_file", [])
        assert len(page_window) + len(page_file) > 0
        in_window.extend(page_window)
        in_file.extend(page_file)
        asked += 1
    return in_window, in_file, asked


async def test_every_entry_of_a_long_difference_is_read_through_the_cursor(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    window_only, file_only = a_long_difference(
        asc_state, asc_file, tmp_path / "world.json", in_window=60, in_file=30
    )

    in_window, in_file, asked = await every_entry(asc_state, asc_file)

    # 25 a side a reply: the window's 60 take three.
    assert asked == 3
    assert in_window == window_only
    assert in_file == file_only


async def test_each_page_counts_what_is_still_left_out(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    a_long_difference(asc_state, asc_file, tmp_path / "world.json", in_window=60, in_file=30)
    first = row_for(await listed(asc_state), asc_file)
    assert first["difference_omitted"] == (60 - 25) + (30 - 25)

    second = row_for(await listed(asc_state, first["next_cursor"]), asc_file)

    assert second["difference_omitted"] == 60 - 50
    assert len(second["only_in_file"]) == 30 - 25
    # The sentence is about the whole difference, whichever page is listed.
    assert second["difference"] == first["difference"]


async def test_a_cursor_pages_its_own_sheet_and_leaves_the_others_as_they_were(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    other = a_second_sheet(asc_file)
    held = {
        str(sheet): as_ltspice_reads(sheet) + lines(wires(30, 2000)) for sheet in (asc_file, other)
    }
    put_windows(asc_state, tmp_path / "world.json", [a_window(held)])
    first = await listed(asc_state)

    second = await listed(asc_state, row_for(first, asc_file)["next_cursor"])

    assert row_for(second, other) == row_for(first, other)
    assert row_for(second, asc_file)["only_in_window"] == wires(30, 2000)[25:]
    assert "next_cursor" not in row_for(second, asc_file)


@pytest.mark.parametrize("change", ["window", "file", "closed"])
async def test_a_cursor_from_before_the_sheet_changed_is_refused(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, change: str
):
    world = tmp_path / "world.json"
    window_only, _ = a_long_difference(asc_state, asc_file, world, in_window=30, in_file=0)
    first = row_for(await listed(asc_state), asc_file)
    if change == "window":
        held = as_ltspice_reads(asc_file) + lines(window_only[1:])
        write_world(world, [a_window({str(asc_file): held})])
    elif change == "file":
        add_a_wire(asc_file)
    else:
        write_world(world, [a_window({})])

    item = await listed(asc_state, first["next_cursor"])

    assert item["ok"] is False
    assert item["error"]["code"] == "invalid_cursor"
    assert "without cursor" in item["error"]["message"]


def test_the_python_api_lists_every_entry(asc_state: SessionState, asc_file: Path, tmp_path: Path):
    window_only, file_only = a_long_difference(
        asc_state, asc_file, tmp_path / "world.json", in_window=60, in_file=30
    )

    result = SyncApi(asc_state).inspect(queries=[{"kind": "open_in_ltspice"}])

    (row,) = result["results"][0]["data"]["designs"]
    assert row["only_in_window"] == window_only
    assert row["only_in_file"] == file_only
    assert "difference_omitted" not in row
    assert "next_cursor" not in row
