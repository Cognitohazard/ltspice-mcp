"""verify_circuit lists at most 25 findings of a rule, and says how to read the rest.

Design principle "Safe edits" (``docs/DESIGN.md``): a list cut short says how
to read the rest. A rule cut at the cap has a ``findings_omitted`` entry with
the count left out and a ``next_cursor``, which ``findings_cursor`` takes back
to list the rule's next findings. Every call here goes through the server's own
dispatch (``server.call_tool``), as a client's does, and no ``run_code``.

The finding lists are checked against what the files hold, and against the
uncapped list ``api.verify_circuit`` returns: the pages together are that list,
in its order, each finding once.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.cursor_codec import encode_cursor
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import verify
from ltspice_mcp.tools.verify import FINDING_RULE_CAP
from tests.conftest import SyncApi, dispatch, dispatched

#: More findings of one rule than one reply lists, and not a multiple of the
#: cap, so the last page is a short one.
MANY = 2 * FINDING_RULE_CAP + 7


def _netlist(work_dir: Path, values: int = MANY) -> Path:
    """A deck whose every resistor spells its value with an ohm sign: one
    ``value_suffix_nonascii`` finding per resistor, on lines 2 onward. The
    resistors are in parallel, so no node is left on one terminal."""
    cards = "".join(f"R{index} a 0 {index + 1}Ω\n" for index in range(values))
    deck = work_dir / "ohms.cir"
    deck.write_bytes(f"* ohms\n{cards}.end\n".encode())
    return deck


def _sheet(work_dir: Path, labels: int = MANY, more: str = "") -> Path:
    """A sheet of net labels on nothing, one ``dangling_label`` finding each,
    then the lines ``more``."""
    flags = "".join(f"FLAG {64 * index} 0 n{index}\n" for index in range(labels))
    sheet = work_dir / "labels.asc"
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{flags}{more}", encoding="utf-8")
    return sheet


def _of(findings: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    return [finding for finding in findings if finding["rule_id"] == rule]


async def _every_finding(
    state: SessionState, path: Path, rule: str, **arguments: Any
) -> tuple[list[dict[str, Any]], int]:
    """Every ``rule`` finding, read by following its cursor; and the replies read."""
    data = await dispatched(state, "verify_circuit", path=str(path), **arguments)
    read = _of(data["findings"], rule)
    replies = 1
    while rule in data.get("findings_omitted", {}):
        entry = data["findings_omitted"][rule]
        assert entry["total"] == MANY
        assert entry["omitted"] == MANY - len(read), entry
        data = await dispatched(
            state,
            "verify_circuit",
            path=str(path),
            findings_cursor=entry["next_cursor"],
            **arguments,
        )
        page = _of(data["findings"], rule)
        assert 0 < len(page) <= FINDING_RULE_CAP
        read.extend(page)
        replies += 1
    return read, replies


async def _uncapped(state: SessionState, path: Path, rule: str, **arguments: Any) -> list:
    """Every ``rule`` finding as ``api.verify_circuit`` returns it."""
    result = await asyncio.to_thread(
        lambda: SyncApi(state).verify_circuit(path=str(path), **arguments)
    )
    return _of(result["findings"], rule)


async def test_every_finding_of_a_netlist_rule_is_read_through_the_cursor(state_no_sim, work_dir):
    deck = _netlist(work_dir)

    read, replies = await _every_finding(state_no_sim, deck, "value_suffix_nonascii")

    assert replies == 3
    assert [finding["subject"] for finding in read] == [f"{index + 1}Ω" for index in range(MANY)]
    assert [finding["at"]["line"] for finding in read] == list(range(2, MANY + 2))
    assert read == await _uncapped(state_no_sim, deck, "value_suffix_nonascii")


async def test_every_finding_of_a_sheet_rule_is_read_through_the_cursor(state_no_sim, work_dir):
    sheet = _sheet(work_dir)

    read, replies = await _every_finding(state_no_sim, sheet, "dangling_label", checks=["layout"])

    assert replies == 3
    assert [(finding["at"]["x"], finding["at"]["y"]) for finding in read] == [
        (64 * index, 0) for index in range(MANY)
    ]
    assert read == await _uncapped(state_no_sim, sheet, "dangling_label", checks=["layout"])


async def test_the_first_reply_counts_what_it_leaves_out(state_no_sim, work_dir):
    data = await dispatched(state_no_sim, "verify_circuit", path=str(_netlist(work_dir)))

    entry = data["findings_omitted"]["value_suffix_nonascii"]
    assert entry["omitted"] == MANY - FINDING_RULE_CAP
    assert entry["total"] == MANY
    # A summary-line reader learns of the cut, and the way on, from the hint.
    assert f"{MANY - FINDING_RULE_CAP} more value_suffix_nonascii" in data["hint"]
    assert "findings_cursor" in data["hint"]


async def test_a_rule_at_the_cap_or_under_it_has_no_entry(state_no_sim, work_dir):
    deck = _netlist(work_dir, values=FINDING_RULE_CAP)

    data = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    assert len(data["findings"]) == FINDING_RULE_CAP
    assert "findings_omitted" not in data


async def test_a_cursor_pages_its_own_rule_and_leaves_the_others_as_they_were(
    state_no_sim, work_dir
):
    # Two capped rules on one sheet: each label on nothing, and each wire drawn
    # twice. Paging one leaves the other listed, and counted, from its start.
    wires = "".join(f"WIRE {64 * index} 200 {64 * index + 32} 200\n" * 2 for index in range(MANY))
    sheet = _sheet(work_dir, more=wires)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])
    assert set(first["findings_omitted"]) >= {"dangling_label", "duplicate_wire"}

    second = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(sheet),
        checks=["layout"],
        findings_cursor=first["findings_omitted"]["dangling_label"]["next_cursor"],
    )

    assert _of(second["findings"], "duplicate_wire") == _of(first["findings"], "duplicate_wire")
    assert (
        second["findings_omitted"]["duplicate_wire"] == first["findings_omitted"]["duplicate_wire"]
    )
    assert _of(second["findings"], "dangling_label") != _of(first["findings"], "dangling_label")
    assert second["findings_omitted"]["dangling_label"]["omitted"] == MANY - 2 * FINDING_RULE_CAP
    assert f"dangling_label findings {FINDING_RULE_CAP + 1}" in second["hint"]


async def test_a_cursor_from_before_the_file_changed_is_refused(state_no_sim, work_dir):
    deck = _netlist(work_dir)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))
    cursor = first["findings_omitted"]["value_suffix_nonascii"]["next_cursor"]
    # One finding fewer: the cursor's offset would now skip one.
    deck.write_bytes(deck.read_bytes().replace("R0 a 0 1Ω".encode(), b"R0 a 0 1"))

    result = await dispatch(
        state_no_sim, "verify_circuit", {"path": str(deck), "findings_cursor": cursor}
    )

    assert result.is_error
    refusal = result.structured_content
    assert refusal["code"] == "invalid_cursor"
    assert "changed" in refusal["error"]
    assert "without findings_cursor" in refusal["error"]
    # The generic cursor hint would send the caller back to this dead cursor.
    assert "hint" not in refusal


async def test_a_cursor_for_another_file_is_refused(state_no_sim, work_dir):
    first = await dispatched(state_no_sim, "verify_circuit", path=str(_netlist(work_dir)))
    cursor = first["findings_omitted"]["value_suffix_nonascii"]["next_cursor"]
    other = work_dir / "other.cir"
    other.write_bytes((work_dir / "ohms.cir").read_bytes())

    result = await dispatch(
        state_no_sim, "verify_circuit", {"path": str(other), "findings_cursor": cursor}
    )

    assert result.is_error
    assert result.structured_content["code"] == "invalid_cursor"


async def test_a_cursor_whose_check_this_call_does_not_run_is_refused(state_no_sim, work_dir):
    sheet = _sheet(work_dir)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])
    cursor = first["findings_omitted"]["dangling_label"]["next_cursor"]

    result = await dispatch(
        state_no_sim,
        "verify_circuit",
        {"path": str(sheet), "checks": ["quality"], "findings_cursor": cursor},
    )

    assert result.is_error
    refusal = result.structured_content
    assert refusal["code"] == "invalid_cursor"
    # It names the check to add, which is the way round.
    assert "'layout'" in refusal["error"]


async def test_a_cursor_that_is_not_one_is_refused_as_every_tool_refuses_one(
    state_no_sim, work_dir
):
    result = await dispatch(
        state_no_sim,
        "verify_circuit",
        {"path": str(_netlist(work_dir)), "findings_cursor": "o:25"},
    )

    assert result.is_error
    refusal = result.structured_content
    assert refusal["code"] == "invalid_cursor"
    assert "findings_cursor" in refusal["error"]
    assert "next_cursor" in refusal["error"]


class _LTspice:
    """LTspice as the session detects it; its export is ``verify._create_netlist``,
    which each test below replaces with the netlist it exports."""

    spice_exe: ClassVar[list[str]] = ["/fake/LTspice.exe"]


def _exporting(netlist: str):
    """An export that writes ``netlist`` beside the sheet, as LTspice's does."""

    def export(_simulator: Any, asc_path: Path, timeout: float = 0) -> Path:
        net = Path(asc_path).with_suffix(".net")
        net.write_bytes(netlist.encode())
        return net

    return export


def _writing_nothing(_simulator: Any, asc_path: Path, timeout: float = 0) -> Path:
    """An export that exits without writing the netlist."""
    return Path(asc_path).with_suffix(".net")


async def _export_cursor(
    config: ServerConfig, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SessionState, Path, str]:
    """A session with LTspice, a sheet whose export spells every value with an
    ohm sign, and the cursor a real call returned for those findings, which
    the export check reports."""
    cards = "".join(f"R{index} a 0 {index + 1}Ω\n" for index in range(MANY))
    monkeypatch.setattr(verify, "_create_netlist", _exporting(f"* ohms\n{cards}.end\n"))
    state = SessionState.create(config, available={"ltspice": _LTspice})
    sheet = _sheet(work_dir, labels=1)
    first = await dispatched(state, "verify_circuit", path=str(sheet), checks=["export"])
    return state, sheet, first["findings_omitted"]["value_suffix_nonascii"]["next_cursor"]


async def _refusal(state: SessionState, sheet: Path, cursor: str, **arguments: Any) -> dict:
    result = await dispatch(
        state, "verify_circuit", {"path": str(sheet), "findings_cursor": cursor, **arguments}
    )
    assert result.is_error, result.structured_content
    assert result.structured_content["code"] == "invalid_cursor"
    return result.structured_content


async def test_a_cursor_whose_check_could_not_run_says_why(config, work_dir, monkeypatch):
    # The export the cursor's findings came from cannot run in this call, so
    # the refusal names the skipped check and its reason rather than reading
    # as "no more findings".
    state, sheet, cursor = await _export_cursor(config, work_dir, monkeypatch)
    del state.available_simulators["ltspice"]

    refusal = await _refusal(state, sheet, cursor)

    assert (
        "the export check that reports them did not run (LTspice not detected)"
        in (refusal["error"])
    )
    assert "without findings_cursor" in refusal["error"]


async def test_a_cursor_whose_check_failed_names_the_failure(config, work_dir, monkeypatch):
    state, sheet, cursor = await _export_cursor(config, work_dir, monkeypatch)
    monkeypatch.setattr(verify, "_create_netlist", _writing_nothing)

    refusal = await _refusal(state, sheet, cursor)

    assert "produced no .net file" in refusal["error"]


async def test_a_cursor_whose_export_lists_other_findings_is_refused(
    config, work_dir, monkeypatch
):
    # The sheet is unchanged; what its export says is not. The cursor's place
    # in the old list would serve a page of the new one.
    state, sheet, cursor = await _export_cursor(config, work_dir, monkeypatch)
    cards = "".join(f"R{index} a 0 {index + 1001}Ω\n" for index in range(MANY))
    monkeypatch.setattr(verify, "_create_netlist", _exporting(f"* ohms\n{cards}.end\n"))

    refusal = await _refusal(state, sheet, cursor)

    assert "changed" in refusal["error"]


#: A two-pin part, kept beside the sheet that uses it.
_TWO_PINS = (
    "Version 4\nSymbolType BLOCK\nRECTANGLE Normal -16 -16 16 16\n"
    "PIN -16 0 LEFT 8\nPINATTR PinName A\nPINATTR SpiceOrder 1\n"
    "PIN 16 0 RIGHT 8\nPINATTR PinName B\nPINATTR SpiceOrder 2\n"
)


async def test_a_cursor_whose_symbol_changed_is_refused(state_no_sim, work_dir):
    # The sheet is unchanged; the symbol its parts use loses a pin, so each
    # part has one floating pin where it had two.
    symbol = work_dir / "twopin.asy"
    symbol.write_text(_TWO_PINS, encoding="utf-8")
    parts = "".join(
        f"SYMBOL twopin {128 * index} 0 R0\nSYMATTR InstName X{index}\n" for index in range(30)
    )
    sheet = work_dir / "parts.asc"
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{parts}", encoding="utf-8")
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])
    assert first["findings_omitted"]["floating_pin"]["total"] == 60
    cursor = first["findings_omitted"]["floating_pin"]["next_cursor"]
    symbol.write_text(_TWO_PINS.split("PIN 16 0")[0], encoding="utf-8")

    refusal = await _refusal(state_no_sim, sheet, cursor, checks=["layout"])

    assert "changed" in refusal["error"]


async def test_a_render_only_call_names_the_render_mode_to_leave(state_no_sim, work_dir):
    sheet = _sheet(work_dir)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])
    cursor = first["findings_omitted"]["dangling_label"]["next_cursor"]

    refusal = await _refusal(
        state_no_sim, sheet, cursor, checks=["layout"], render={"format": "svg", "mode": "only"}
    )

    assert "render.mode" in refusal["error"]
    assert "among the checks" not in refusal["error"]


async def test_the_hint_names_the_cut_beside_a_failure(state_no_sim, work_dir):
    # A compare against a sheet needs LTspice to export it, so the compare
    # fails; the findings that were listed are still cut, and the hint says so.
    deck = _netlist(work_dir)
    reference = _sheet(work_dir, labels=1)
    arguments = {"compare": {"reference": str(reference)}}
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck), **arguments)
    assert first["failures"]
    assert f"{MANY - FINDING_RULE_CAP} more value_suffix_nonascii" in first["hint"]

    second = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        findings_cursor=first["findings_omitted"]["value_suffix_nonascii"]["next_cursor"],
        **arguments,
    )

    assert f"value_suffix_nonascii findings {FINDING_RULE_CAP + 1}" in second["hint"]


async def test_a_cursor_naming_no_rule_is_refused_as_a_cursor(state_no_sim, work_dir):
    # The codec's checksum is not a secret, so any body can be encoded.
    crafted = encode_cursor({"rule": ["floating_pin"], "offset": FINDING_RULE_CAP, "of": None})

    await _refusal(state_no_sim, _netlist(work_dir), crafted)
