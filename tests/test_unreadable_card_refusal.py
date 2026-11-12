"""A card no reader can split into its parts is refused the same way by every
path that meets it: the file, the line, the card as written, what is wrong
and the fix, as one error type (``spice_lex_views.unreadable_card``).

Three decks hold a card left open: a ``.model`` whose brace is left open
inside its parameter list, which the card's own split once read as whole, and
a transistor whose ``W=`` brace is left open. A Monte Carlo run reached each
through its own reader and ended with the tokenizer's bare message, naming
neither the card nor its file; the checks passed the first two as having no
problem at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import UnreadableCardError
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.montecarlo import parse_model_params
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_validator import validate_netlist_directive_refs
from ltspice_mcp.state import SessionState
from tests.conftest import FakeSim, SyncApi, dispatch, fake_simulator
from tests.test_inspect_tools import _run

_FIX = "Check for a missing } matching the {."

#: Each deck, the card left open, its line, and the random rule that reads it.
_DECKS: dict[str, tuple[str, str, int, dict[str, Any]]] = {
    "model brace inside its parameter list": (
        "* diode\nV1 a 0 1\nD1 a 0 D\n.model D D(is={1e-14)\n.op\n.end\n",
        ".model D D(is={1e-14)",
        4,
        {"rule": "model", "target": "D", "param": "IS", "tolerance": 0.1},
    ),
    "mismatch model brace inside its parameter list": (
        "* fet\nV1 d 0 1\nV2 g 0 1\nM1 d g 0 0 nm W=1u L=1u\n"
        ".model nm NMOS(level=1 vto={0.7 kp=1e-4)\n.op\n.end\n",
        ".model nm NMOS(level=1 vto={0.7 kp=1e-4)",
        5,
        {"rule": "mismatch", "prefix": "M", "AVT": 3e-3},
    ),
    "mismatch instance brace": (
        "* fet\nV1 d 0 1\nV2 g 0 1\nM1 d g 0 0 nm W={1u L=1u\n"
        ".model nm NMOS(level=1 vto=0.7 kp=1e-4)\n.op\n.end\n",
        "M1 d g 0 0 nm W={1u L=1u",
        4,
        {"rule": "mismatch", "prefix": "M", "AVT": 3e-3},
    ),
}

_SIMULATORS = {"ngspice": "NGspiceSimulator", "ltspice": "LTspice"}


def _assert_refusal(message: str, card: str, line: int) -> None:
    """The one refusal: the file, the line, the card as written, the fix."""
    assert f"Cannot read the card on line {line} of " in message, message
    assert "d.cir" in message, message
    assert repr(card) in message, message
    assert "unterminated {...}" in message, message
    assert _FIX in message, message


def _deck(work_dir: Path, case: str) -> Path:
    path = work_dir / "d.cir"
    path.write_text(_DECKS[case][0], encoding="utf-8")
    return path


def _payload(path: Path, rule: dict[str, Any], lint: str) -> dict[str, Any]:
    return {
        "request_id": f"unreadable-{lint}",
        "circuits": [{"path": str(path), "id": "dut"}],
        "execution": {"wait_s": 30},
        "lint": lint,
        "variations": [{"kind": "random", "runs": 1, "seed": 1, "rules": [rule]}],
    }


@pytest.mark.parametrize("simulator", list(_SIMULATORS))
@pytest.mark.parametrize("case", list(_DECKS))
async def test_a_random_run_refuses_the_card_by_its_file_and_line(
    config, work_dir: Path, monkeypatch: pytest.MonkeyPatch, case: str, simulator: str
):
    _, card, line, rule = _DECKS[case]
    state = SessionState.create(
        config, available={simulator: type(_SIMULATORS[simulator], (FakeSim,), {})}
    )
    submitted: list[str] = []
    fake_simulator(monkeypatch, submitted)
    path = _deck(work_dir, case)

    for lint in ("off", "warn"):
        result = await dispatch(state, "run_experiments", _payload(path, rule, lint))
        data = result.structured_content
        assert data is not None, result.content
        (failure,) = data["failures"]
        assert failure["code"] == "card_unreadable", failure
        _assert_refusal(failure["message"], card, line)
    assert submitted == []

    # The linter's finding names the same card, by its line.
    lint_rows = [
        finding
        for block in data.get("lint", [])
        for finding in block["findings"]
        if finding["rule_id"] == "unreadable-card"
    ]
    assert [row["at"]["line"] for row in lint_rows] == [line], data.get("lint")


@pytest.mark.parametrize("case", list(_DECKS))
async def test_the_checks_report_the_card(state_no_sim, work_dir: Path, case: str):
    """verify_circuit, the validator and the linter each name the card: the
    first two decks passed every check before."""
    _, card, line, _rule = _DECKS[case]
    path = _deck(work_dir, case)

    data = (await dispatch(state_no_sim, "verify_circuit", {"path": str(path)})).structured_content
    assert data is not None
    rows = [f for f in data.get("findings", []) if f["rule_id"] == "unreadable_card"]
    assert [(row["at"]["line"], row["subject"]) for row in rows] == [(line, card)], data

    cards = lex_deck(path.read_text(encoding="utf-8")).cards
    # The validator leaves the card out of its node checks rather than raising.
    validate_netlist_directive_refs(cards)
    findings = lint_deck(path.read_text(encoding="utf-8"), path, "ltspice", "ltspice")
    assert [f["at"]["line"] for f in findings if f["rule_id"] == "unreadable-card"] == [line]


@pytest.mark.parametrize("case", list(_DECKS))
async def test_hierarchy_discovery_refuses_the_card(state_no_sim, work_dir: Path, case: str):
    _, card, line, _rule = _DECKS[case]
    path = _deck(work_dir, case)

    (row,) = await _run(
        state_no_sim, [{"kind": "hierarchy", "path": str(path), "simulator": "ltspice"}]
    )
    assert not row["ok"], row
    assert row["error"]["code"] == "card_unreadable", row
    _assert_refusal(row["error"]["message"], card, line)

    with pytest.raises(UnreadableCardError) as excinfo:
        load_hierarchy(str(path), [work_dir], SemanticProfile("ltspice", None))
    _assert_refusal(str(excinfo.value), card, line)


@pytest.mark.parametrize("case", list(_DECKS))
def test_the_python_api_refuses_the_card(
    config, work_dir: Path, monkeypatch: pytest.MonkeyPatch, case: str
):
    _, card, line, rule = _DECKS[case]
    state = SessionState.create(config, available={"ltspice": type("LTspice", (FakeSim,), {})})
    fake_simulator(monkeypatch, [])
    path = _deck(work_dir, case)
    api = SyncApi(state)

    receipt = api.run_experiments(**_payload(path, rule, "off"))
    (failure,) = receipt["failures"]
    assert failure["code"] == "card_unreadable", failure
    _assert_refusal(failure["message"], card, line)

    checked = api.verify_circuit(path=str(path))
    assert [f["at"]["line"] for f in checked["findings"]] == [line], checked


def test_the_model_reader_refuses_a_brace_left_open_in_its_parameters():
    """The Monte Carlo nominal reader goes through the model view, whose
    refusal is the one error type."""
    with pytest.raises(UnreadableCardError, match=r"line 1, '\.model D D\(is=\{1e-14\)'"):
        parse_model_params(".model D D(is={1e-14)\n")
