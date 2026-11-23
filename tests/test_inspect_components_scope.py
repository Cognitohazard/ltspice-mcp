"""Which card a reference names, in which scope, on every route that answers it.

A netlist's components are the cards of its top level; a card inside a
``.subckt`` body belongs to that body, names the body as its ``subcircuit``,
and never replaces a top-level card of the same reference.
``spice_lex_views.instances_by_scope`` decides it, and these readers of a
deck's cards by reference ask it: the components listing of ``inspect`` and of
the Python API, the members of a netlist net query, and the transistors a
Monte Carlo run rewrites.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.montecarlo import extract_mosfet_instances, rewrite_instance_model
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, dispatched

TOP_Q1 = "Q1 c1 b1 e1 QTOP\n"
BODY = ".subckt CE in c\nQ1 c b e QBODY\nRE e 0 1k\n.ends\n"


async def _listing(state: SessionState, deck: Path, route: str) -> dict[str, Any]:
    query = {"kind": "components", "path": str(deck), "detail": "full"}
    if route == "mcp":
        reply = await dispatched(state, "inspect", queries=[query])
    else:
        reply = await asyncio.to_thread(lambda: SyncApi(state).inspect(queries=[query]))
    (result,) = reply["results"]
    assert result["ok"], result
    return result["data"]


async def _components(state: SessionState, deck: Path, route: str) -> list[dict[str, Any]]:
    return (await _listing(state, deck, route))["components"]


def _rows(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (tuple(r.get("subcircuit", ())), r["reference"], tuple(r["nodes"]), r.get("model"))
        for r in rows
    ]


ROUTES = pytest.mark.parametrize("route", ["mcp", "api"])


@ROUTES
@pytest.mark.parametrize(
    "deck_text",
    [
        f"* t\n{BODY}{TOP_Q1}X1 a c1 CE\n.end\n",
        f"* t\n{TOP_Q1}{BODY}X1 a c1 CE\n.end\n",
    ],
    ids=["subckt-before", "subckt-after"],
)
async def test_a_body_card_does_not_replace_the_top_level_card_of_its_reference(
    route: str, deck_text: str, work_dir: Path, state_no_sim: SessionState
):
    deck = work_dir / "deck.cir"
    deck.write_text(deck_text)
    rows = await _components(state_no_sim, deck, route)
    assert _rows(rows) == [
        ((), "Q1", ("c1", "b1", "e1"), "QTOP"),
        ((), "X1", ("a", "c1"), "CE"),
        (("CE",), "Q1", ("c", "b", "e"), "QBODY"),
        (("CE",), "RE", ("e", "0"), None),
    ]


@ROUTES
async def test_one_reference_in_two_bodies_and_the_top_level_is_three_rows(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = work_dir / "deck.cir"
    deck.write_text(
        "* t\n.subckt A p\nQ1 p b e QA\n.ends\n"
        f"{TOP_Q1}"
        ".subckt B p\nQ1 p b e QB\n.ends\n.end\n"
    )
    rows = await _components(state_no_sim, deck, route)
    assert _rows(rows) == [
        ((), "Q1", ("c1", "b1", "e1"), "QTOP"),
        (("A",), "Q1", ("p", "b", "e"), "QA"),
        (("B",), "Q1", ("p", "b", "e"), "QB"),
    ]


@ROUTES
async def test_a_nested_body_repeating_its_outer_bodys_reference_keeps_both(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = work_dir / "deck.cir"
    deck.write_text(
        "* t\n.subckt OUTER p\nQ1 p ob oe QO\n"
        ".subckt INNER q\nQ1 q ib ie QI\n.ends INNER\n"
        ".ends OUTER\n"
        f"{TOP_Q1}.end\n"
    )
    rows = await _components(state_no_sim, deck, route)
    assert _rows(rows) == [
        ((), "Q1", ("c1", "b1", "e1"), "QTOP"),
        (("OUTER",), "Q1", ("p", "ob", "oe"), "QO"),
        (("OUTER", "INNER"), "Q1", ("q", "ib", "ie"), "QI"),
    ]


@ROUTES
async def test_an_included_files_body_does_not_replace_a_top_level_card(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    """The listing reads the file as written and does not follow the include,
    so the included body's ``Q1`` cannot replace the deck's; this pins that."""
    (work_dir / "parts.lib").write_text(BODY)
    deck = work_dir / "deck.cir"
    deck.write_text(f"* t\n.include parts.lib\n{TOP_Q1}X1 a c1 CE\n.end\n")
    rows = await _components(state_no_sim, deck, route)
    assert _rows(rows) == [
        ((), "Q1", ("c1", "b1", "e1"), "QTOP"),
        ((), "X1", ("a", "c1"), "CE"),
    ]


@ROUTES
async def test_a_deck_with_no_subcircuit_lists_no_scope(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = work_dir / "deck.cir"
    deck.write_text("* t\nR2 a 0 2k\nR1 a b 1k\n.end\n")
    rows = await _components(state_no_sim, deck, route)
    assert [{k: r[k] for k in ("reference", "value", "nodes")} for r in rows] == [
        {"reference": "R1", "value": "1k", "nodes": ["a", "b"]},
        {"reference": "R2", "value": "2k", "nodes": ["a", "0"]},
    ]
    assert not any("subcircuit" in r for r in rows)


@ROUTES
async def test_a_body_defined_twice_in_one_scope_is_named_not_merged(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    """Both definitions share one scope, so no reference there names one card:
    neither body's cards are listed, and a warning names the subcircuit."""
    deck = work_dir / "deck.cir"
    deck.write_text(
        "* t\n.subckt CE p\nQ1 p b e QA\nR1 p 0 1k\n.ends\n"
        ".subckt CE p\nQ1 p b e QB\nR2 p 0 2k\n.ends\n"
        f"{TOP_Q1}X1 a CE\n.end\n"
    )
    data = await _listing(state_no_sim, deck, route)
    assert _rows(data["components"]) == [
        ((), "Q1", ("c1", "b1", "e1"), "QTOP"),
        ((), "X1", ("a",), "CE"),
    ]
    assert [w for w in data["warnings"] if "CE" in w] == [
        "subcircuit CE is defined 2 times in one scope, so its bodies' cards are not "
        "listed: which card a reference names there is not known. Rename or delete "
        "all but one definition of CE."
    ]


# The other readers of a deck's cards by reference ------------------------------


@ROUTES
async def test_a_net_querys_members_are_the_top_level_card_not_the_body_card(
    route: str, work_dir: Path, state_no_sim: SessionState
):
    deck = work_dir / "deck.cir"
    deck.write_text(f"* t\n{TOP_Q1}{BODY}X1 a c1 CE\n.end\n")
    query = {"kind": "net", "path": str(deck), "at": "c1"}
    if route == "mcp":
        reply = await dispatched(state_no_sim, "inspect", queries=[query])
    else:
        reply = await asyncio.to_thread(lambda: SyncApi(state_no_sim).inspect(queries=[query]))
    (result,) = reply["results"]
    assert result["ok"], result
    assert {(m["reference"], m["terminal"]) for m in result["data"]["members"]} == {
        ("Q1", 1),
        ("X1", 2),
    }


def test_a_monte_carlo_rewrite_and_read_take_the_top_level_transistor():
    text = (
        "* t\n.subckt CE in d\nM1 d in 0 0 NBODY w=9u l=9u\n.ends\n"
        "M1 d g 0 0 NTOP w=1u l=2u\nX1 g d CE\n.end\n"
    )
    found = extract_mosfet_instances(text, deck=True)
    assert [(m.ref, m.model_name, m.width_m) for m in found.sized] == [("M1", "NTOP", 1e-6)]
    rewritten = rewrite_instance_model(text, "M1", "NNEW", deck=True)
    assert "M1 d g 0 0 NNEW w=1u l=2u" in rewritten
    assert "M1 d in 0 0 NBODY w=9u l=9u" in rewritten
