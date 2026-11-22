"""verify_circuit sends every finding when its reply fits, and stores the
answer and says how to read the rest when it does not.

Design principle "Jobs and results are stored on disk and requests are repeatable" (``docs/DESIGN.md``):
the findings are cut by the presenter (``lib/presenter.py``) only when the
reply is over the reply target (``[analysis] default_budget``), never at a
number per rule. A cut reply stores the complete answer and names three ways
to read the rest: the stored file, the call with ``findings_cursor``, and its
Python form. Reading on slices the stored answer: no check runs again, and a
page read after the file changed still reads, carrying the snapshot the
answer was computed from. Every call here goes through the server's own
dispatch (``server.call_tool``), as a client's does.

The findings read are checked against what the files hold, and against the
complete list ``api.verify_circuit`` returns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import verify
from tests.conftest import SyncApi, dispatch, dispatched

#: More findings of one rule than the old cap of 25 listed.
FITTING_DECK = 30
FITTING_SHEET = 50

#: A reply target those findings fit on every platform. Each finding carries
#: the checked file's path, which is longer on Windows and JSON-escaped there,
#: so how many fit the 4000-token default depends on the machine: a deck's
#: value-suffix finding weighs about 105 tokens on Linux, a sheet's dangling
#: label about 60, and more under a Windows temp folder.
ROOMY = 20_000

#: Enough findings that a reply holding them all is over the default target.
OVER = 400


def _netlist(work_dir: Path, values: int, name: str = "ohms.cir") -> Path:
    """A deck whose every resistor spells its value with an ohm sign: one
    ``value_suffix_nonascii`` finding per resistor, on lines 2 onward. The
    resistors are in parallel, so no node is left on one terminal."""
    cards = "".join(f"R{index} a 0 {index + 1}Ω\n" for index in range(values))
    deck = work_dir / name
    deck.write_bytes(f"* ohms\n{cards}.end\n".encode())
    return deck


def _sheet(work_dir: Path, labels: int) -> Path:
    """A sheet of net labels on nothing, one ``dangling_label`` finding each."""
    flags = "".join(f"FLAG {64 * index} 0 n{index}\n" for index in range(labels))
    sheet = work_dir / "labels.asc"
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{flags}", encoding="utf-8")
    return sheet


def _stored_file(reply: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(reply["stored_answer"]["path"]).read_text(encoding="utf-8"))


def _remove_stored_file(reply: dict[str, Any]) -> None:
    Path(reply["stored_answer"]["path"]).unlink()


def _stored_file_exists(reply: dict[str, Any]) -> bool:
    return Path(reply["stored_answer"]["path"]).exists()


def _of(findings: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    return [finding for finding in findings if finding["rule_id"] == rule]


async def _complete(state: SessionState, path: Path, **arguments: Any) -> list[dict[str, Any]]:
    """Every finding, as ``api.verify_circuit`` returns it."""
    result = await asyncio.to_thread(
        lambda: SyncApi(state).verify_circuit(path=str(path), **arguments)
    )
    return result["findings"]


async def _read_on(
    state: SessionState, path: Path, first: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every finding, read from ``first`` by following its cursor, and the
    pages read."""
    read = list(first["findings"])
    pages = []
    entry = first["cut"]["/findings"]
    while "next" in entry:
        page = await dispatched(
            state, "verify_circuit", path=str(path), findings_cursor=entry["next"]
        )
        pages.append(page)
        entry = page["cut"]["/findings"]
        assert entry["offset"] == len(read)
        read.extend(page["findings"])
        assert entry["returned"] == len(page["findings"])
    return read, pages


# ---------------------------------------------------------------------------
# A reply that fits
# ---------------------------------------------------------------------------


def _roomy(config: ServerConfig) -> SessionState:
    """A session whose reply target the fitting findings fit on any machine."""
    config.default_budget = ROOMY
    return SessionState.create(config, available={})


def _assert_fits(complete: list[dict[str, Any]]) -> None:
    """The test's premise: the complete findings are within the target, so a
    reply that cuts them cuts by something else than size."""
    assert estimate_tokens(complete, as_sent=False) < ROOMY // 2


async def test_findings_that_fit_all_go_out_past_25_a_rule(config, work_dir):
    state_no_sim = _roomy(config)
    deck = _netlist(work_dir, FITTING_DECK)
    _assert_fits(await _complete(state_no_sim, deck))

    data = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    found = _of(data["findings"], "value_suffix_nonascii")
    assert [finding["subject"] for finding in found] == [f"{i + 1}Ω" for i in range(FITTING_DECK)]
    assert data["findings"] == await _complete(state_no_sim, deck)
    for field in ("findings_omitted", "cut", "stored_answer"):
        assert field not in data
    assert not list(state_no_sim.store.results_dir.glob("sa_*.json"))


async def test_a_sheet_rule_that_fits_goes_out_whole(config, work_dir):
    state_no_sim = _roomy(config)
    sheet = _sheet(work_dir, FITTING_SHEET)
    _assert_fits(await _complete(state_no_sim, sheet, checks=["layout"]))

    data = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])

    assert [(f["at"]["x"], f["at"]["y"]) for f in _of(data["findings"], "dangling_label")] == [
        (64 * index, 0) for index in range(FITTING_SHEET)
    ]
    assert "cut" not in data


async def test_no_target_sends_every_finding(config, work_dir):
    config.default_budget = 0
    state = SessionState.create(config, available={})
    deck = _netlist(work_dir, OVER)

    data = await dispatched(state, "verify_circuit", path=str(deck))

    assert len(data["findings"]) == OVER
    assert "cut" not in data


# ---------------------------------------------------------------------------
# A reply that does not fit
# ---------------------------------------------------------------------------


async def test_a_reply_over_the_target_is_cut_and_names_three_ways_on(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)

    data = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    entry = data["cut"]["/findings"]
    assert entry["total"] == OVER
    assert entry["returned"] == len(data["findings"]) < OVER
    assert estimate_tokens(data, as_sent=False) <= state_no_sim.config.default_budget
    stored = data["stored_answer"]
    assert stored["snapshot"] == {
        "path": str(deck),
        "sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
    }
    on_disk = _stored_file(data)
    assert on_disk["answer"]["findings"] == await _complete(state_no_sim, deck)
    # The three ways on: the stored file, the call, and its Python form.
    hint = data["hint"]
    assert stored["path"] in hint
    assert "verify_circuit(path=" in hint and "findings_cursor=" in hint
    assert "api.verify_circuit(" in hint and "raw_page=True" in hint
    # The hint counts every finding the check found, not those the reply holds.
    assert f"{OVER} warning(s): value_suffix_nonascii" in hint


async def test_findings_omitted_is_filled_from_the_cut(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)

    data = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    entry = data["cut"]["/findings"]
    assert data["findings_omitted"] == {
        "value_suffix_nonascii": {
            "omitted": OVER - entry["returned"],
            "total": OVER,
            "next_cursor": entry["next"],
        }
    }


async def test_every_finding_is_read_through_the_cursor(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    read, pages = await _read_on(state_no_sim, deck, first)

    assert pages
    assert [finding["subject"] for finding in read] == [f"{i + 1}Ω" for i in range(OVER)]
    assert [finding["at"]["line"] for finding in read] == list(range(2, OVER + 2))
    assert read == await _complete(state_no_sim, deck)
    assert "findings_omitted" not in pages[-1]
    assert all(page["stored_answer"] == first["stored_answer"] for page in pages)


async def test_every_finding_of_a_sheet_rule_is_read_through_the_cursor(state_no_sim, work_dir):
    sheet = _sheet(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(sheet), checks=["layout"])

    read, _ = await _read_on(state_no_sim, sheet, first)

    assert [(f["at"]["x"], f["at"]["y"]) for f in _of(read, "dangling_label")] == [
        (64 * index, 0) for index in range(OVER)
    ]
    assert read == await _complete(state_no_sim, sheet, checks=["layout"])


async def test_a_continuation_runs_no_check(state_no_sim, work_dir, monkeypatch):
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    async def no_check(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a page ran the check")

    monkeypatch.setattr(verify, "evaluate_verify_circuit", no_check)
    page = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        findings_cursor=first["cut"]["/findings"]["next"],
    )

    assert page["findings"][0]["subject"] == f"{first['cut']['/findings']['returned'] + 1}Ω"


async def test_a_page_after_the_file_changed_reads_the_older_snapshot(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)
    before = hashlib.sha256(deck.read_bytes()).hexdigest()
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))
    returned = first["cut"]["/findings"]["returned"]
    # Every value fixed: a check now finds nothing.
    deck.write_bytes(deck.read_bytes().replace("Ω".encode(), b""))

    page = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        findings_cursor=first["cut"]["/findings"]["next"],
    )

    assert page["findings"][0]["subject"] == f"{returned + 1}Ω"
    assert page["stored_answer"]["snapshot"]["sha256"] == before
    assert before != hashlib.sha256(deck.read_bytes()).hexdigest()
    assert "compare stored_answer.snapshot" in page["hint"]


async def test_a_page_holds_only_the_list_it_reads_and_what_names_the_reply(
    state_no_sim, work_dir
):
    """A page holds its list, what names the page (``page_of``, ``cut`` and
    ``stored_answer``) and the reply's text fields; the lists the first reply
    sent whole are that reply's to give (``presenter.page_skeleton``)."""
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))

    page = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        findings_cursor=first["cut"]["/findings"]["next"],
    )

    for field in ("path", "kind", "outcome"):
        assert page.get(field) == first.get(field), field
    assert page["page_of"] == "/findings"
    assert set(page["cut"]) == {"/findings"}
    for field in ("checks_run", "rules_run", "observations"):
        assert field not in page, field


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def _refusal(state: SessionState, path: Path, cursor: str) -> dict[str, Any]:
    result = await dispatch(
        state, "verify_circuit", {"path": str(path), "findings_cursor": cursor}
    )
    assert result.is_error, result.structured_content
    refusal = result.structured_content
    assert refusal["code"] == "invalid_cursor"
    return refusal


async def test_a_missing_stored_answer_is_refused_with_the_fix(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))
    _remove_stored_file(first)

    refusal = await _refusal(state_no_sim, deck, first["cut"]["/findings"]["next"])

    assert "is gone" in refusal["error"]
    assert "again without findings_cursor" in refusal["error"]
    # The generic cursor hint would send the caller back to this dead cursor.
    assert "hint" not in refusal


def _expire(reply: dict[str, Any]) -> None:
    """Move the stored answer's expiry into the past, as time would."""
    path = Path(reply["stored_answer"]["path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["expires_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(record), encoding="utf-8")


async def test_an_expired_stored_answer_is_refused_with_the_fix(state_no_sim, work_dir):
    state = state_no_sim
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state, "verify_circuit", path=str(deck))
    _expire(first)

    refusal = await _refusal(state, deck, first["cut"]["/findings"]["next"])

    assert "expired" in refusal["error"]
    assert "hint" not in refusal
    assert "again without findings_cursor" in refusal["error"]
    assert not _stored_file_exists(first)


async def test_a_cursor_for_another_file_is_refused(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)
    first = await dispatched(state_no_sim, "verify_circuit", path=str(deck))
    other = _netlist(work_dir, OVER, name="other.cir")

    refusal = await _refusal(state_no_sim, other, first["cut"]["/findings"]["next"])

    assert str(deck) in refusal["error"]


@pytest.mark.parametrize("cursor", ["o:25", ""])
async def test_a_cursor_that_is_not_one_is_refused_as_every_tool_refuses_one(
    state_no_sim, work_dir, cursor
):
    refusal = await _refusal(state_no_sim, _netlist(work_dir, 3), cursor)

    assert "next" in refusal["error"]


# ---------------------------------------------------------------------------
# The Python API
# ---------------------------------------------------------------------------


async def test_the_python_api_returns_the_complete_answer(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)

    found = await _complete(state_no_sim, deck)

    assert len(_of(found, "value_suffix_nonascii")) == OVER


async def test_a_raw_page_is_the_reply_a_client_reads(state_no_sim, work_dir):
    deck = _netlist(work_dir, OVER)
    api = SyncApi(state_no_sim)

    first = await asyncio.to_thread(lambda: api.verify_circuit(path=str(deck), raw_page=True))
    page = await asyncio.to_thread(
        lambda: api.verify_circuit(
            path=str(deck), findings_cursor=first["cut"]["/findings"]["next"], raw_page=True
        )
    )

    assert first["cut"]["/findings"]["returned"] < OVER
    assert page["cut"]["/findings"]["offset"] == first["cut"]["/findings"]["returned"]
