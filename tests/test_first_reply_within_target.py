"""A session's first reply holds the read-the-guide reminder and still fits
the reply target.

The reminder goes on the first reply of a session that has not read the
guide (``server.GUIDE_REMINDER``). The presenter (``lib/presenter.py``) is
the one place that holds a reply to its target, so the reminder is in the
reply it measures and cuts: a first reply over the target with the reminder
has its lists cut further, and the reminder is never what is cut. Nothing is
added to a reply after the presenter has cut it.

Each target here is measured from the same call in a session that gets no
reminder, so the reply without it fills the target exactly and the
reminder's own tokens decide the outcome. The checked file's path is in
every finding, so each case runs once under a short folder and once under a
long one the test makes, whatever the machine's temporary folder.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.response_budget import NOTE_RESERVE_TOKENS, estimate_tokens
from ltspice_mcp.server import GUIDE_REMINDER
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, dispatch, dispatched

#: Findings enough that the reply is cut under the default target.
OVER = 400

#: Findings that go out whole under the default target.
FEW = 12

#: A folder name long enough that each finding's path weighs more than on any
#: temporary folder the suite meets, and short enough for Windows' 260
#: character path limit under one.
LONG = "a-folder-name-long-enough-to-weigh-in-every-finding-that-carries-the-path-" + "x" * 30

FOLDERS = pytest.mark.parametrize("folder", ["s", LONG], ids=["short_path", "long_path"])


def _deck(work_dir: Path, folder: str, values: int) -> Path:
    """A deck with one ``value_suffix_nonascii`` finding per resistor."""
    home = work_dir / folder
    home.mkdir()
    cards = "".join(f"R{index} a 0 {index + 1}Ω\n" for index in range(values))
    deck = home / "ohms.cir"
    deck.write_bytes(f"* ohms\n{cards}.end\n".encode())
    return deck


def _session(config: ServerConfig, target: int, *, guide_read: bool = False) -> SessionState:
    state = SessionState.create(dataclasses.replace(config, default_budget=target), available={})
    state.guide_read = guide_read
    return state


def _size(reply: dict[str, Any]) -> int:
    return estimate_tokens(reply, as_sent=False)


async def _unreminded(config: ServerConfig, deck: Path, target: int) -> dict[str, Any]:
    """The reply to ``verify_circuit`` of ``deck`` in a session that read the
    guide, so it carries no reminder."""
    return await dispatched(
        _session(config, target, guide_read=True), "verify_circuit", path=str(deck)
    )


@FOLDERS
async def test_a_first_reply_over_the_target_fits_it_with_the_reminder(config, work_dir, folder):
    deck = _deck(work_dir, folder, OVER)
    plain = await _unreminded(config, deck, config.default_budget)
    assert "cut" in plain
    # The target the reply without the reminder fills, to the token.
    target = _size(plain)
    state = _session(config, target)

    first = await dispatched(state, "verify_circuit", path=str(deck))

    assert GUIDE_REMINDER in first["hint"]
    assert _size(first) <= target
    assert first["cut"]["/findings"]["returned"] < plain["cut"]["/findings"]["returned"]
    # The rest is still read on from the stored answer the hint names.
    assert first["stored_answer"]["path"] in first["hint"]


@FOLDERS
async def test_the_second_reply_carries_no_reminder_and_fits(config, work_dir, folder):
    deck = _deck(work_dir, folder, OVER)
    target = _size(await _unreminded(config, deck, config.default_budget))
    state = _session(config, target)
    first = await dispatched(state, "verify_circuit", path=str(deck))

    second = await dispatched(state, "verify_circuit", path=str(deck))

    assert GUIDE_REMINDER in first["hint"]
    assert GUIDE_REMINDER not in second["hint"]
    assert _size(second) <= target
    # The same answer is stored, with or without the reminder on the reply.
    assert second["stored_answer"] == first["stored_answer"]


@FOLDERS
async def test_a_first_reply_the_reminder_takes_over_the_target_is_cut(config, work_dir, folder):
    deck = _deck(work_dir, folder, FEW)
    whole = await _unreminded(config, deck, config.default_budget)
    assert "cut" not in whole
    target = _size(whole)
    state = _session(config, target)

    first = await dispatched(state, "verify_circuit", path=str(deck))

    assert GUIDE_REMINDER in first["hint"]
    assert _size(first) <= target
    entry = first["cut"]["/findings"]
    assert entry["total"] == FEW
    assert entry["returned"] == len(first["findings"]) < FEW
    # It says how to read the rest.
    assert first["stored_answer"]["path"] in first["hint"]
    assert "findings_cursor=" in first["hint"]


async def test_a_first_reply_that_is_an_error_carries_the_reminder_once(config, work_dir):
    # An error result is not presented: it has no lists to cut and no target,
    # so the reminder goes on its hint as the server writes it.
    state = _session(config, config.default_budget)

    failed = await dispatch(state, "verify_circuit", {"path": str(work_dir / "missing.cir")})
    after = await dispatch(state, "verify_circuit", {"path": str(work_dir / "missing.cir")})

    assert failed.is_error
    assert failed.structured_content is not None
    assert GUIDE_REMINDER in failed.structured_content["hint"]
    assert failed.structured_content["hint"].count(GUIDE_REMINDER) == 1
    assert GUIDE_REMINDER not in (after.structured_content or {}).get("hint", "")


async def test_a_python_api_result_never_carries_the_reminder(config, work_dir):
    deck = _deck(work_dir, "s", OVER)
    state = _session(config, config.default_budget)

    result = await asyncio.to_thread(lambda: SyncApi(state).verify_circuit(path=str(deck)))

    assert GUIDE_REMINDER not in result.get("hint", "")
    # The Python API is no MCP reply: the next tool reply is still the first.
    first = await dispatched(state, "verify_circuit", path=str(deck))
    assert GUIDE_REMINDER in first["hint"]


async def test_a_first_inspect_reply_under_a_caller_budget_fits_it_with_the_reminder(
    config, work_dir
):
    # inspect holds a caller's ``budget`` with its own ladder, which measures
    # each envelope with the reminder in it too.
    deck = work_dir / "ladder.cir"
    deck.write_text(
        "* ladder\n" + "".join(f"R{n} a n{n} 1k\n" for n in range(40)) + ".end\n",
        encoding="utf-8",
    )
    query = {"kind": "components", "path": str(deck)}
    plain = await dispatched(
        _session(config, config.default_budget, guide_read=True),
        "inspect",
        queries=[query],
        budget=100_000,
    )
    # The budget the reply without the reminder fills exactly.
    budget = estimate_tokens(plain) + NOTE_RESERVE_TOKENS
    state = _session(config, config.default_budget)

    first = await dispatched(state, "inspect", queries=[query], budget=budget)

    assert GUIDE_REMINDER in first["hint"]
    # The reply less the budget's own notes fits the room the ladder leaves them.
    body = {key: value for key, value in first.items() if key != "observations"}
    assert estimate_tokens(body) <= budget - NOTE_RESERVE_TOKENS
    assert len(first["results"][0]["data"]["components"]) < len(
        plain["results"][0]["data"]["components"]
    )
