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
from ltspice_mcp.tools._base import difference_in_full
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
from tests.conftest import SyncApi


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


async def ask(state: SessionState, **query: Any) -> dict[str, Any]:
    queries = [{"kind": "open_in_ltspice", **query}]
    result = await handle_inspect(InspectInput(queries=queries), state)  # type: ignore[arg-type]
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
    assert row["hint"] == difference_in_full(window="this pid")


def wires(y: int, count: int) -> list[str]:
    """``count`` short wires along ``y``, as .asc lines."""
    return [f"WIRE {x} {y} {x} {y + 16}" for x in range(0, 16 * count, 16)]


def lines(entries: list[str]) -> str:
    return "".join(f"{entry}\n" for entry in entries)


def a_sheet_and_a_window_far_apart(
    state: SessionState, asc_file: Path, world: Path, in_window: list[str], in_file: list[str]
) -> None:
    """The window holds the sheet with ``in_window`` added; the file, with ``in_file``."""
    sheet = as_ltspice_reads(asc_file)
    put_windows(state, world, [a_window({str(asc_file): sheet + lines(in_window)})])
    with asc_file.open("ab") as file:
        file.write(lines(in_file).encode("ascii"))


async def read_in_full(state: SessionState, **query: Any) -> list[dict[str, Any]]:
    """Every page of an open_in_ltspice query naming a sheet, following next_cursor."""
    pages = [await ask(state, **query)]
    while "next_cursor" in pages[-1]:
        assert len(pages) < 100, "the cursor never ran out"
        pages.append(await ask(state, **query, cursor=pages[-1]["next_cursor"]))
    return pages


async def test_every_entry_of_a_long_difference_is_read_through_the_call_its_hint_names(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The list names 25 entries a side and counts the rest. The call its hint
    names, given the row's path and pid, pages through every one of them."""
    in_window, in_file = wires(2000, 30), wires(3000, 40)
    a_sheet_and_a_window_far_apart(
        asc_state, asc_file, tmp_path / "world.json", in_window, in_file
    )

    (row,) = (await ask(asc_state))["data"]["designs"]
    assert row["difference_omitted"] == (30 - 25) + (40 - 25)
    assert row["hint"] == difference_in_full(window="this pid")
    # A page smaller than either side, so both sides take several pages.
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 7)
    pages = await read_in_full(asc_state, path=row["path"], pid=row["pid"])

    assert all(page["ok"] for page in pages), pages
    assert len(pages) == 6
    for page in pages:
        data = page["data"]
        assert (data["path"], data["pid"], data["version"]) == (str(asc_file), PID, "26.1.1")
        assert data["sha256"] == row["sha256"] == digest(asc_file)
        assert data["differs_from_file"] is True
        assert data["difference"] == row["difference"]
    read = {
        side: [entry for page in pages for entry in page["data"].get(side, [])]
        for side in ("only_in_window", "only_in_file")
    }
    assert read == {"only_in_window": sorted(in_window), "only_in_file": sorted(in_file)}
    assert pages[0]["page"] == {
        "total": 70,
        "returned": 14,
        "truncated": True,
        "collections": {
            "only_in_window": {"total": 30, "returned": 7, "truncated": True},
            "only_in_file": {"total": 40, "returned": 7, "truncated": True},
        },
    }


def test_the_python_api_reads_a_long_difference_whole_in_one_call(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``Api.inspect`` follows the cursor itself, as for every other paged kind."""
    in_window, in_file = wires(2000, 30), wires(3000, 40)
    a_sheet_and_a_window_far_apart(
        asc_state, asc_file, tmp_path / "world.json", in_window, in_file
    )
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 7)

    collected = SyncApi(asc_state).inspect(
        queries=[{"kind": "open_in_ltspice", "path": str(asc_file)}]
    )

    (item,) = collected["results"]
    assert item["data"]["only_in_window"] == sorted(in_window)
    assert item["data"]["only_in_file"] == sorted(in_file)


async def test_a_sheet_open_in_two_windows_is_read_in_the_one_pid_names(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    sheet = as_ltspice_reads(asc_file)
    first, second = wires(2000, 3), wires(4000, 2)
    put_windows(
        asc_state,
        tmp_path / "world.json",
        [
            a_window({str(asc_file): sheet + lines(first)}),
            a_window({str(asc_file): sheet + lines(second)}, pid=PID + 1),
        ],
    )

    unchosen = await ask(asc_state, path=str(asc_file))
    chosen = await ask(asc_state, path=str(asc_file), pid=PID + 1)
    neither = await ask(asc_state, path=str(asc_file), pid=PID + 2)

    assert unchosen["ok"] is False
    assert unchosen["error"]["code"] == "window_not_chosen"
    assert f"(processes {PID}, {PID + 1}): pass one as pid" in unchosen["error"]["message"]
    assert chosen["ok"] is True
    assert chosen["data"]["pid"] == PID + 1
    assert chosen["data"]["only_in_window"] == sorted(second)
    assert "next_cursor" not in chosen
    assert neither["error"]["code"] == "sheet_not_open"
    assert f"open in process {PID}, {PID + 1}: pass one as pid" in neither["error"]["message"]


async def test_a_page_asked_after_the_window_changed_is_refused(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cursor carries offsets into the difference it was minted for; a page
    of a different one would skip or repeat entries, so it is refused."""
    world = tmp_path / "world.json"
    sheet = as_ltspice_reads(asc_file)
    put_windows(asc_state, world, [a_window({str(asc_file): sheet + lines(wires(2000, 10))})])
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 4)
    first = await ask(asc_state, path=str(asc_file))
    write_world(world, [a_window({str(asc_file): sheet + lines(wires(2000, 11))})])

    later = await ask(asc_state, path=str(asc_file), cursor=first["next_cursor"])

    assert later["ok"] is False
    assert later["error"]["code"] == "invalid_cursor"


async def test_a_sheet_whose_window_holds_the_file_says_so_and_lists_nothing(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    put_windows(
        asc_state, tmp_path / "world.json", [a_window({str(asc_file): as_ltspice_reads(asc_file)})]
    )

    item = await ask(asc_state, path=str(asc_file))

    assert item["ok"] is True
    assert item["data"] == {
        "path": str(asc_file),
        "pid": PID,
        "version": "26.1.1",
        "sha256": digest(asc_file),
    }
    assert "next_cursor" not in item


async def test_a_sheet_no_window_has_open_fails_and_says_where_to_look(
    asc_state: SessionState, asc_file: Path, tmp_path: Path
):
    put_windows(asc_state, tmp_path / "world.json", [a_window()])

    item = await ask(asc_state, path=str(asc_file))

    assert item["error"]["code"] == "sheet_not_open"
    assert "inspect(kind='open_in_ltspice') without a path" in item["error"]["message"]


@pytest.mark.parametrize(
    ("query", "said"),
    [
        ({"path": "amp.net"}, "path must be a .asc sheet"),
        ({"pid": PID}, "pid needs path"),
    ],
)
async def test_a_query_that_names_no_sheet_to_compare_is_refused(
    asc_state: SessionState, tmp_path: Path, query: dict[str, Any], said: str
):
    put_windows(asc_state, tmp_path / "world.json", [a_window()])

    item = await ask(asc_state, **query)

    assert item["error"]["code"] == "invalid_query"
    assert said in item["error"]["message"]


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
async def test_every_open_document_is_read_by_following_the_cursor(
    asc_state: SessionState,
    asc_file: Path,
    tmp_path: Path,
    extra: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 3)
    designs = {str(asc_file.with_name(f"deck{n}.net")): "* deck\n" for n in range(3 + extra)}
    put_windows(asc_state, tmp_path / "world.json", [a_window(designs)])

    pages = [await ask(asc_state)]
    while "next_cursor" in pages[-1]:
        assert len(pages) < 100, "the cursor never ran out"
        pages.append(await ask(asc_state, cursor=pages[-1]["next_cursor"]))

    assert all(page["ok"] for page in pages), pages
    assert len(pages[0]["data"]["designs"]) == 3
    assert {page["data"]["total"] for page in pages} == {3 + extra}
    read = [row["path"] for page in pages for row in page["data"]["designs"]]
    assert read == list(designs)


async def test_a_page_of_the_list_asked_after_a_document_opened_is_refused(
    asc_state: SessionState, asc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cursor carries an offset into the documents it was minted for: one
    opened or closed in between would shift the rest, so the page is refused."""
    monkeypatch.setattr(inspect_tools, "_PAGE_SIZE", 3)
    world = tmp_path / "world.json"
    designs = {str(asc_file.with_name(f"deck{n}.net")): "* deck\n" for n in range(5)}
    put_windows(asc_state, world, [a_window(designs)])
    first = await ask(asc_state)
    designs[str(asc_file.with_name("late.net"))] = "* late\n"
    write_world(world, [a_window(designs)])

    later = await ask(asc_state, cursor=first["next_cursor"])
    forged = await ask(asc_state, cursor="x")

    assert later["ok"] is False
    assert later["error"]["code"] == "invalid_cursor"
    assert forged["error"]["code"] == "invalid_cursor"


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
