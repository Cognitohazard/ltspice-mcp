"""What `inspect` answers about every sheet of the suite, held to a record.

`inspect(kind="net")` and `inspect(kind="components")` on an `.asc` are the
queries that read a sheet. This test asks both of every `.asc` under `tests/`
and compares the answers with `fixtures/inspect_sheet_record.json`: the
components at full detail, and the net from each pin, each net label and a
point on each wire, taken from the sheet's scene so the starts do not depend
on the reader under test. A change to how these queries read a sheet is held
to this record, and a change made on purpose changes the record in the same
commit. Record it again with `LTSPICE_MCP_RECORD_INSPECT_SHEETS=1`.

The record keeps the answers themselves, so a diff says what changed. An
error is kept as its code only, since its wording may come from another
library. Like the other records, it does not depend on the LTspice library
of the machine running it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.schematic_scene import build_scene
from ltspice_mcp.lib.sheet import editor_symbol_resolver
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from tests._proposal_record import every_nth, stage_afresh, write_record
from tests._schematic_fixtures import SUITE_SHEETS, TESTS, suite_name

_RECORD = TESTS / "fixtures" / "inspect_sheet_record.json"
_RECORDING = os.environ.get("LTSPICE_MCP_RECORD_INSPECT_SHEETS") == "1"

#: The most net starts asked of one sheet; a sheet with more has every n-th taken.
_MOST_STARTS = 120

#: The most queries one inspect call takes.
_BATCH = 64

#: A Python object's default text, which carries its address and so differs
#: from one run to the next.
_OBJECT = re.compile(r"<[^<>]* object at 0x[0-9a-fA-F]+>")


pytestmark = pytest.mark.usefixtures("suite_symbols_only")


def _starts(sheet: Path) -> list[str | list[int]]:
    """Where to start a net trace: each pin, each label name, a point on each wire."""
    scene = build_scene(sheet, editor_symbol_resolver(sheet))
    starts: list[str | list[int]] = []
    for part in scene.symbols:
        if part.reference and not part.missing:
            starts += [f"{part.reference}.{pin.name}" for pin in part.pins if pin.name]
    starts += [f"net:{name}" for name in dict.fromkeys(flag.text for flag in scene.flags)]
    for wire in scene.wires:
        if wire.y1 == wire.y2 and wire.x1 != wire.x2:
            starts.append([(wire.x1 + wire.x2) // 2, wire.y1])
        elif wire.x1 == wire.x2 and wire.y1 != wire.y2:
            starts.append([wire.x1, (wire.y1 + wire.y2) // 2])
    return every_nth(starts, _MOST_STARTS)


def _plain(value: Any) -> Any:
    """``value`` with object addresses replaced, so a record reads the same every run."""
    return json.loads(_OBJECT.sub("<object>", json.dumps(value)))


def _net(item: dict[str, Any]) -> dict[str, Any]:
    if not item["ok"]:
        return {"error": item["error"]["code"]}
    data = item["data"]
    coordinates = json.dumps(data.get("coordinates", []), sort_keys=True)
    said: dict[str, Any] = {
        "labels": data.get("labels", []),
        "pins": data.get("pins", []),
        "total_pins": data.get("total_pins"),
        "is_shorted": data.get("is_shorted"),
        "total_coordinates": data.get("total_coordinates"),
        "coordinates_sha256": hashlib.sha256(coordinates.encode("utf-8")).hexdigest()[:16],
    }
    for key in ("snapped_to_wire", "warnings"):
        if key in data:
            said[key] = data[key]
    return said


def _components(item: dict[str, Any]) -> dict[str, Any]:
    if not item["ok"]:
        return {"error": item["error"]["code"]}
    data = item["data"]
    return {"total": data["total"], "rows": data["components"]}


async def _ask(state: SessionState, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for at in range(0, len(queries), _BATCH):
        reply = await handle_inspect(
            InspectInput.model_validate({"queries": queries[at : at + _BATCH]}), state
        )
        assert reply.structured_content is not None
        items += reply.structured_content["results"]
    return items


async def answers_of(sheet: Path, state: SessionState, folder: str) -> dict[str, Any]:
    """What `inspect` answers about ``sheet``, staged on its own in ``folder``."""
    sandbox = Path(state.working_dir) / folder
    sandbox.mkdir()
    staged = stage_afresh(sheet, sandbox)
    starts = _starts(staged)
    queries = [{"kind": "components", "path": str(staged), "detail": "full"}]
    queries += [{"kind": "net", "path": str(staged), "at": at} for at in starts]
    components, *nets = await _ask(state, queries)
    return _plain(
        {
            "components": _components(components),
            "nets": [{"at": at, **_net(item)} for at, item in zip(starts, nets, strict=True)],
        }
    )


def _recorded() -> dict[str, Any]:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _RECORDING, reason="set LTSPICE_MCP_RECORD_INSPECT_SHEETS=1 to record")
async def test_record_what_inspect_answers(asc_state: SessionState):
    record = {
        suite_name(sheet): await answers_of(sheet, asc_state, f"sheet{number}")
        for number, sheet in enumerate(SUITE_SHEETS)
    }
    write_record(_RECORD, record)


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_the_record_covers_the_sheets_the_suite_holds():
    assert sorted(_recorded()) == sorted(suite_name(sheet) for sheet in SUITE_SHEETS)


@pytest.mark.skipif(_RECORDING, reason="recording")
def test_the_record_holds_answers_of_every_kind():
    record = _recorded().values()
    nets = [net for entry in record for net in entry["nets"]]
    assert sum(1 for entry in record if "rows" in entry["components"]) >= 50
    assert sum(1 for net in nets if "error" not in net) >= 1000
    assert any(net.get("snapped_to_wire") for net in nets)
    assert any(net.get("is_shorted") for net in nets)
    assert any(net.get("warnings") for net in nets)
    assert any("error" in net for net in nets)
    assert all("rows" in entry["components"] for entry in record)


@pytest.mark.skipif(_RECORDING, reason="recording")
@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=suite_name)
async def test_inspect_answers_as_the_record_says(sheet: Path, asc_state: SessionState):
    said = await answers_of(sheet, asc_state, "sheet")
    recorded = _recorded()[suite_name(sheet)]
    assert said["components"] == recorded["components"]
    assert said["nets"] == recorded["nets"]
