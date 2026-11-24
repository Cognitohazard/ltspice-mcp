"""Which components a deck holds when one reference is on more than one card.

Every reader of a deck's cards by reference gives the same answer: one entry
per card, in deck order, and the repeat stated with its reference, lines and
scope. ``spice_lex_views.repeated_references`` decides what a repeat is, and
these routes ask it: the components and net queries of ``inspect`` (and of the
Python API), ``verify_circuit``'s quality check and both comparison modes, the
``run_experiments`` lint, and the Monte Carlo reader and rewriter of
transistors.

What a simulator does with such a deck is recorded only for two top-level
cards of one spelling (``log/err_duplicate_name``), so nothing here asserts a
claim about it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.montecarlo import extract_mosfet_instances, rewrite_instance_model
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, dispatched

ROUTES = pytest.mark.parametrize("route", ["mcp", "api"])

TWO_TOP = "* t\nV1 a 0 1\nR1 a 0 1k\nR1 a b 2k\n.op\n.end\n"
ONE_TOP = "* t\nV1 a 0 1\nR1 a 0 1k\n.op\n.end\n"
TOP_AND_BODY = "* t\nV1 a 0 1\nR1 a 0 1k\n.subckt A p\nR1 p 0 2k\n.ends\nX1 a A\n.op\n.end\n"
TWICE_IN_BODY = (
    "* t\nV1 a 0 1\n.subckt A p\nR1 p 0 1k\nR1 p 0 2k\n.ends\nX1 a A\n.op\n.end\n"
)
TWO_SPELLINGS = "* t\nV1 a 0 1\nR1 a 0 1k\nr1 a b 2k\n.op\n.end\n"
CONTINUED = "* t\nV1 a 0 1\nR1 a 0\n+ 1k\nR2 a b 2k\n.op\n.end\n"


async def _call(state: SessionState, route: str, tool: str, **arguments: Any) -> dict[str, Any]:
    if route == "mcp":
        return await dispatched(state, tool, **arguments)
    api = SyncApi(state)
    return await asyncio.to_thread(lambda: getattr(api, tool)(**arguments))


async def _query(state: SessionState, route: str, query: dict[str, Any]) -> dict[str, Any]:
    reply = await _call(state, route, "inspect", queries=[query])
    (result,) = reply["results"]
    return result


async def _components(state: SessionState, route: str, deck: Path) -> dict[str, Any]:
    result = await _query(
        state, route, {"kind": "components", "path": str(deck), "detail": "full"}
    )
    assert result["ok"], result
    return result["data"]


def _deck(work_dir: Path, text: str, name: str = "deck.cir") -> Path:
    path = work_dir / name
    path.write_text(text)
    return path


def _lint(text: str, work_dir: Path) -> list[dict[str, Any]]:
    findings = lint_deck(text, work_dir / "deck.cir", None, "LTspice")
    return [f for f in findings if f["rule_id"] == "repeated-reference"]


async def _verify_quality(state: SessionState, route: str, deck: Path) -> list[dict[str, Any]]:
    reply = await _call(state, route, "verify_circuit", path=str(deck), checks=["quality"])
    return [f for f in reply.get("findings", []) if f["rule_id"] == "shared_reference"]


# --- two top-level cards of one reference -----------------------------------


@ROUTES
async def test_two_top_level_cards_of_one_reference_are_two_rows_and_a_stated_repeat(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    data = await _components(state_no_sim, route, _deck(work_dir, TWO_TOP))
    rows = [(r["reference"], r.get("value"), tuple(r["nodes"])) for r in data["components"]]
    # The listing is ordered by reference; the cards of one reference keep
    # their deck order.
    assert [(ref, nodes) for ref, _, nodes in rows] == [
        ("R1", ("a", "0")),
        ("R1", ("a", "b")),
        ("V1", ("a", "0")),
    ]
    assert data["repeated_references"] == [{"reference": "R1", "lines": [3, 4]}]


@ROUTES
async def test_a_net_query_lists_every_card_of_a_repeated_reference(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    result = await _query(
        state_no_sim, route, {"kind": "net", "path": str(_deck(work_dir, TWO_TOP)), "at": "a"}
    )
    assert result["ok"], result
    data = result["data"]
    assert [m["reference"] for m in data["members"]].count("R1") == 2
    assert data["repeated_references"] == [{"reference": "R1", "lines": [3, 4]}]


@ROUTES
async def test_a_pin_of_a_repeated_reference_is_refused_naming_its_lines(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    result = await _query(
        state_no_sim, route, {"kind": "net", "path": str(_deck(work_dir, TWO_TOP)), "at": "R1.2"}
    )
    assert not result["ok"]
    assert result["error"]["code"] == "invalid_at"
    assert "lines 3, 4" in result["error"]["message"]


@ROUTES
async def test_verify_reports_a_repeated_reference_with_its_lines(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    (finding,) = await _verify_quality(state_no_sim, route, _deck(work_dir, TWO_TOP))
    assert finding["severity"] == "warning"
    assert finding["evidence"]["reference"] == "R1"
    assert finding["evidence"]["lines"] == [3, 4]


def test_the_run_lint_reports_a_repeated_reference_with_its_lines(work_dir: Path):
    (finding,) = _lint(TWO_TOP, work_dir)
    assert finding["evidence"]["reference"] == "R1"
    assert finding["evidence"]["lines"] == [3, 4]
    assert finding["at"]["line"] == 3


@ROUTES
async def test_equivalence_names_the_repeat_and_is_not_equivalent(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    # The reference holds R1 once with the value of the candidate's first card:
    # pairing that card silently would call the decks equivalent.
    candidate = _deck(work_dir, TWO_TOP)
    reference = _deck(work_dir, ONE_TOP, "ref.cir")
    reply = await _call(
        state_no_sim,
        route,
        "verify_circuit",
        path=str(candidate),
        checks=["compare"],
        compare={"reference": str(reference)},
    )
    comparison = reply["comparison"]
    assert comparison["equivalent"] is False
    assert comparison["repeated_references"] == [
        {"reference": "R1", "lines": [3, 4], "side": "candidate"}
    ]


@ROUTES
async def test_structural_diff_leaves_a_repeated_reference_uncompared_and_reaches_no_verdict(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    candidate = _deck(work_dir, TWO_TOP)
    reference = _deck(work_dir, ONE_TOP, "ref.cir")
    reply = await _call(
        state_no_sim,
        route,
        "verify_circuit",
        path=str(candidate),
        checks=["compare"],
        compare={"reference": str(reference), "mode": "structural_diff"},
    )
    comparison = reply["comparison"]
    assert comparison["equivalent"] is None
    assert any("R1" in note and "lines 3, 4" in note for note in comparison["not_compared"])


def test_monte_carlo_leaves_a_repeated_transistor_unsized_and_refuses_to_rewrite_it():
    deck = (
        "* t\n.model N nmos\nM1 d g 0 0 N W=1u L=1u\nM1 d g 0 0 N W=2u L=1u\n"
        "M2 d g 0 0 N W=1u L=1u\n.op\n.end\n"
    )
    found = extract_mosfet_instances(deck, deck=True)
    assert [i.ref for i in found.sized] == ["M2"]
    (unsized,) = found.unsized
    assert unsized.ref == "M1"
    assert "lines 3, 4" in unsized.reason
    with pytest.raises(ValueError, match="lines 3, 4"):
        rewrite_instance_model(deck, "M1", "N2", deck=True)


# --- scopes, spellings and continuation lines -------------------------------


@ROUTES
async def test_one_reference_at_top_level_and_in_a_body_is_not_a_repeat(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = _deck(work_dir, TOP_AND_BODY)
    data = await _components(state_no_sim, route, deck)
    assert [(tuple(r.get("subcircuit", ())), r["reference"]) for r in data["components"]] == [
        ((), "R1"),
        ((), "V1"),
        ((), "X1"),
        (("A",), "R1"),
    ]
    assert "repeated_references" not in data
    assert _lint(TOP_AND_BODY, work_dir) == []
    assert await _verify_quality(state_no_sim, route, deck) == []


@ROUTES
async def test_one_reference_twice_in_one_body_is_a_repeat_named_with_its_subcircuit(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = _deck(work_dir, TWICE_IN_BODY)
    data = await _components(state_no_sim, route, deck)
    body_rows = [r for r in data["components"] if r.get("subcircuit") == ["A"]]
    assert [r["reference"] for r in body_rows] == ["R1", "R1"]
    expected = {"reference": "R1", "subcircuit": ["A"], "lines": [4, 5]}
    assert data["repeated_references"] == [expected]
    (lint,) = _lint(TWICE_IN_BODY, work_dir)
    assert {k: lint["evidence"][k] for k in expected} == expected
    (finding,) = await _verify_quality(state_no_sim, route, deck)
    assert {k: finding["evidence"][k] for k in expected} == expected


@ROUTES
async def test_two_spellings_of_one_reference_are_one_reference_under_fold_name(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = _deck(work_dir, TWO_SPELLINGS)
    data = await _components(state_no_sim, route, deck)
    assert [r["reference"] for r in data["components"]] == ["R1", "V1", "r1"]
    (repeat,) = data["repeated_references"]
    assert repeat["lines"] == [3, 4]
    assert len(_lint(TWO_SPELLINGS, work_dir)) == 1
    assert len(await _verify_quality(state_no_sim, route, deck)) == 1


@ROUTES
async def test_a_continuation_line_is_not_a_second_card(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = _deck(work_dir, CONTINUED)
    data = await _components(state_no_sim, route, deck)
    assert [r["reference"] for r in data["components"]] == ["R1", "R2", "V1"]
    assert "repeated_references" not in data
    assert _lint(CONTINUED, work_dir) == []
    assert await _verify_quality(state_no_sim, route, deck) == []


@ROUTES
async def test_a_deck_with_no_repeat_carries_no_new_field(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = _deck(work_dir, ONE_TOP)
    reference = _deck(work_dir, ONE_TOP, "ref.cir")
    data = await _components(state_no_sim, route, deck)
    assert "repeated_references" not in data
    net = await _query(state_no_sim, route, {"kind": "net", "path": str(deck), "at": "a"})
    assert "repeated_references" not in net["data"]
    for mode in ("equivalence", "structural_diff"):
        reply = await _call(
            state_no_sim,
            route,
            "verify_circuit",
            path=str(deck),
            checks=["compare"],
            compare={"reference": str(reference), "mode": mode},
        )
        comparison = reply["comparison"]
        assert comparison["equivalent"] is True
        assert "repeated_references" not in comparison
        assert not any("more than one card" in n for n in comparison.get("not_compared", []))
