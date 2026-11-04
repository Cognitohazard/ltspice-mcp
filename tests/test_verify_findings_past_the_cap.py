"""verify_circuit lists at most 25 findings of a rule, and says how to read the rest.

Design principle "Safe edits" (``docs/DESIGN.md``): a list cut short says how
to read the rest. A rule cut at the cap has a ``findings_omitted`` entry with
the count left out and a ``next_cursor``, which ``findings_cursor`` takes back
to list the rule's next findings. Every call here goes through the server's own
dispatch (``server.call_tool``), as a client's does, and no ``run_code``.

The finding lists are checked against what the files hold, and against the
uncapped evaluation the Python API returns (``evaluate_verify_circuit``): the
pages together are that list, in its order, each finding once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import as_sent
from ltspice_mcp.tools.verify import (
    FINDING_RULE_CAP,
    FindingsCursor,
    VerifyCircuitInput,
    evaluate_verify_circuit,
    file_revision,
)
from tests.conftest import dispatch, dispatched

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


def _sheet(work_dir: Path, labels: int = MANY) -> Path:
    """A sheet of net labels on nothing: one ``dangling_label`` finding each."""
    flags = "".join(f"FLAG {64 * index} 0 n{index}\n" for index in range(labels))
    sheet = work_dir / "labels.asc"
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{flags}", encoding="utf-8")
    return sheet


async def _every_finding(
    state: SessionState, path: Path, rule: str, **arguments: Any
) -> tuple[list[dict[str, Any]], int]:
    """Every ``rule`` finding, read by following its cursor; and the replies read."""
    data = await dispatched(state, "verify_circuit", path=str(path), **arguments)
    read = [finding for finding in data["findings"] if finding["rule_id"] == rule]
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
        page = [finding for finding in data["findings"] if finding["rule_id"] == rule]
        assert 0 < len(page) <= FINDING_RULE_CAP
        read.extend(page)
        replies += 1
    return read, replies


async def _uncapped(state: SessionState, path: Path, rule: str, **arguments: Any) -> list:
    """Every ``rule`` finding as the Python API returns it: the evaluation, as sent."""
    evaluation = await evaluate_verify_circuit(
        VerifyCircuitInput(path=str(path), **arguments), state
    )
    findings = as_sent(evaluation.data["findings"], "verify_circuit")
    return [finding for finding in findings if finding["rule_id"] == rule]


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
    flags = "".join(f"FLAG {64 * index} 0 n{index}\n" for index in range(MANY))
    wires = "".join(f"WIRE {64 * index} 200 {64 * index + 32} 200\n" * 2 for index in range(MANY))
    sheet = work_dir / "busy.asc"
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{flags}{wires}", encoding="utf-8")
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])
    assert set(first["findings_omitted"]) >= {"dangling_label", "duplicate_wire"}

    second = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(sheet),
        checks=["layout"],
        findings_cursor=first["findings_omitted"]["dangling_label"]["next_cursor"],
    )

    def of(data: dict, rule: str) -> list:
        return [finding for finding in data["findings"] if finding["rule_id"] == rule]

    assert of(second, "duplicate_wire") == of(first, "duplicate_wire")
    assert (
        second["findings_omitted"]["duplicate_wire"] == first["findings_omitted"]["duplicate_wire"]
    )
    assert of(second, "dangling_label") != of(first, "dangling_label")
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


async def test_a_cursor_that_is_not_one_is_refused_as_an_argument(state_no_sim, work_dir):
    result = await dispatch(
        state_no_sim,
        "verify_circuit",
        {"path": str(_netlist(work_dir)), "findings_cursor": "o:25"},
    )

    assert result.is_error
    refusal = result.structured_content
    assert refusal["code"] == "invalid_arguments"
    assert "findings_cursor" in refusal["error"]
    assert "next_cursor" in refusal["error"]


async def test_a_cursor_whose_check_could_not_run_says_why(state_no_sim, work_dir):
    # A sheet's value-suffix findings are read off its export. A cursor for one,
    # on an unchanged sheet, in a session where the export cannot run (no
    # LTspice) reaches no finding, and the refusal names the skipped check and
    # its reason rather than reading as "no more findings".
    sheet = _sheet(work_dir, labels=1)
    revision = file_revision(state_no_sim.resolve_path(str(sheet)))
    cursor = FindingsCursor("value_suffix_micro_sign", "export", FINDING_RULE_CAP, revision)

    result = await dispatch(
        state_no_sim, "verify_circuit", {"path": str(sheet), "findings_cursor": cursor.encode()}
    )

    assert result.is_error
    refusal = result.structured_content
    assert refusal["code"] == "invalid_cursor"
    assert "this call found 0" in refusal["error"]
    assert "the export check did not run (LTspice not detected)" in refusal["error"]
    assert "without findings_cursor" in refusal["error"]
