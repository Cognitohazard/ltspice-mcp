"""What a parameter expression evaluates to, on every route that answers it.

The question is the expression-value grid's (``[grid.expression-value]`` in
``inputs/cases.toml``; ``docs/TESTING.md``, "Grids"). The server evaluates an
expression with ``hierarchy_expr.evaluate``, its powers with ``_power``, and a
deck's parameters and an element's value with ``Environment``, which
``load_hierarchy`` builds for ``inspect(kind="hierarchy")``, the run's
mismatch and variation plans, and the reader's check against ngspice.

A cell is one expression as a deck writes it, on one source whose node
voltage in the ``.op`` raw is what LTspice computed. Every recorded cell, on
each build, is read through every route: ``evaluate`` with the deck's
parameters, ``load_hierarchy``, and ``inspect(kind="hierarchy")`` through the
server's dispatch and through the Python API, where the reader does not know
which build runs the deck, and ``Environment`` and ``load_hierarchy`` told the
build that ran it, where a power the builds compute differently is decided by
it. A route either takes LTspice's
number (within ``SAME_VALUE_RTOL``: a raw stores a node voltage in four bytes)
or declines the expression; one that takes another number, or takes one where
LTspice refused the deck, is a difference, declared on its cell (``differs``)
and counted in ``KNOWN_DIFFERENCES``. The same expressions are computed live by
ngspice in each compatibility mode and compared with the server's reading for
ngspice.
"""

from __future__ import annotations

import ast
import asyncio
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import hierarchy_expr
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import SAME_VALUE_RTOL
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_expr import Environment
from ltspice_mcp.lib.ngspice_mode import run_batch
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.spice_lex import lex_deck
from ltspice_mcp.lib.spice_lex_views import read_instance
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._recorded_grids import Cell, grid
from tests.conftest import LIVENESS_S, SyncApi, dispatched
from tests.ltspice_recorder import INPUTS

GRID = grid("expression-value")

#: How many routes answer otherwise than a recorded cell, per build, each
#: declared on its cell (``differs``, with the routes and the reason). A count
#: may only fall: a fix lowers it here in the same change.
KNOWN_DIFFERENCES: dict[str, int] = {"ltspice26": 0, "ltspice17": 0}


#: How each recorded build names itself in a run's output, so the reader is
#: told which build runs the deck (``SemanticProfile.of``).
_PRODUCERS = {
    "ltspice26": Producer("ltspice", "LTspice 26.1.1 for Windows", None),
    "ltspice17": Producer("ltspice", "Linear Technology Corporation LTspice XVII", None),
}


def _number(cell: Cell, build: str) -> float | None:
    """The number a cell declares for ``build``, None where it declares none."""
    number = cell.extra.get("number")
    if isinstance(number, dict):
        return number.get(build)
    return number


def _recorded(build: str, case_id: str) -> bool:
    return case_id in rec.manifest(build)["cases"]


def _source_of(case_id: str) -> Path:
    return INPUTS / rec.CASES.case(case_id).source


def _cases() -> list[str]:
    return sorted({cell.case for cell in GRID.cells})


def _runs() -> list[Any]:
    runs = []
    for case_id in _cases():
        pending = rec.CASES.behaviours[rec.CASES.case(case_id).behaviour].pending
        for build in rec.BUILDS:
            if _recorded(build, case_id):
                runs.append(pytest.param(build, case_id, id=f"{build}/{case_id}"))
            elif pending:
                runs.append(
                    pytest.param(
                        build, case_id, id=f"{build}/{case_id}", marks=pytest.mark.skip(pending)
                    )
                )
    return runs


# The index, held to the decks -------------------------------------------------


def _deck(case_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """A deck's ``.param`` assignments, by name, and each probe node's value."""
    params: dict[str, str] = {}
    values: dict[str, str] = {}
    for card in lex_deck(read_spice_text(_source_of(case_id))).cards:
        if card.kind == "param":
            for assignment in card.body.split()[1:]:
                name, _, expression = assignment.partition("=")
                params[name] = expression
        line = read_instance(card) if card.kind == "instance" else None
        if line is not None and line.ref[:1].upper() in ("R", "V"):
            values[line.nodes[0].lower()] = line.value or ""
    return params, values


#: The parameters of the grid's decks, by the sign they stand for in a cell.
_SIGN_OF = {
    "lpos": "positive",
    "lneg": "negative",
    "lzero": "zero",
    "rpos": "positive",
    "rneg": "negative",
    "rzero": "zero",
    "bpos": "positive",
    "bneg": "negative",
    "bzero": "zero",
}
_EXPONENT_OF = {"ewhole": "whole", "eneg": "negative_whole", "ezero": "zero", "efrac": "fraction"}
_OPERATOR_OF = {"+": "plus", "-": "minus", "*": "times", "/": "divide"}
#: Expressions of more than one operator, by the chain each is.
_CHAINS = {
    "{10-4-3}": "minus_minus",
    "{64/8/2}": "divide_divide",
    "{2**3**2}": "power_power_star",
    "{2^3^2}": "power_power_caret",
    "{-2**2}": "negated_power_star",
    "{-2^2}": "negated_power_caret",
    "{bpos**-ewhole}": "power_negated_name",
    "{2+3*4}": "precedence",
    "{(2+3)*4}": "parentheses",
    "{(2^3)+5}": "parentheses",
}


def _form(item: str) -> dict[str, str]:
    """An expression's coordinates, read from how it is written."""
    at: dict[str, str] = {}
    if item.startswith("{{"):
        at["wrap"], body = "nested_braces", item[2:-2]
    elif item.startswith("{"):
        at["wrap"], body = "braces", item[1:-1]
    elif item.startswith("'"):
        at["wrap"], body = "quotes", item[1:-1]
    elif "=" in item:
        at["wrap"], body = "param", item.partition("=")[2]
    else:
        at["wrap"], body = "bare", item
    at["operands"] = "parameter" if re.search(r"[a-z]", body) else "literal"
    if item in _CHAINS:
        at["chain"] = _CHAINS[item]
        return at
    if match := re.fullmatch(r"([a-z]+)(\*\*|\^)([a-z]+)", body):
        at["power"] = "star" if match[2] == "**" else "caret"
        at["base"], at["exponent"] = _SIGN_OF[match[1]], _EXPONENT_OF[match[3]]
    elif match := re.fullmatch(r"(\d)(\*\*|\^)(\d)", body):
        at["power"] = "star" if match[2] == "**" else "caret"
        at["base"], at["exponent"] = "positive", "whole"
    elif match := re.fullmatch(r"([-+])([a-z]+)", body):
        at["unary"] = "minus" if match[1] == "-" else "plus"
        at["operand"] = _SIGN_OF[match[2]]
    elif match := re.fullmatch(r"([a-z]+)([-+*/])([a-z]+)", body):
        at["operator"] = _OPERATOR_OF[match[2]]
        at["left"], at["right"] = _SIGN_OF[match[1]], _SIGN_OF[match[3]]
    if at["wrap"] != "braces":
        # A wrap other than braces is told by one sum; its operator is not crossed.
        for axis in ("operator", "left", "right"):
            at.pop(axis, None)
    return at


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_is_the_expression_its_deck_writes_on_its_node_and_has_its_form(cell: Cell):
    params, values = _deck(cell.case)
    written = values.get(cell.extra["node"])
    if cell.coordinates["wrap"] == "param":
        name, _, expression = cell.item.partition("=")
        assert written == "{" + name + "}", cell.id
        assert params.get(name) == expression, cell.id
    else:
        assert written == cell.item, (cell.id, written)
    declared = {axis: value for axis, value in cell.coordinates.items() if value != ""}
    assert declared == _form(cell.item), cell.id


def _arithmetic(cell: Cell) -> float | None:
    """What plain arithmetic over the deck's parameters gives for a cell of
    one operator, or None for any other."""
    params, _ = _deck(cell.case)
    numbers = {
        name: float(value) for name, value in params.items() if re.fullmatch(r"-?[\d.]+", value)
    }
    at = cell.coordinates
    body = cell.item.strip("{}'")
    if at["operator"] and at["wrap"] == "braces":
        match = re.fullmatch(r"([a-z]+)([-+*/])([a-z]+)", body)
        assert match is not None, cell.id
        left, symbol, right = match.groups()
        a, b = numbers[left], numbers[right]
        if symbol == "/" and b == 0:
            return None
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else math.nan}[symbol]
    if at["unary"]:
        sign, name = body[0], body[1:]
        return -numbers[name] if sign == "-" else numbers[name]
    if at["power"] == "star" and at["base"] == "positive" and at["operands"] == "parameter":
        base, exponent = body.split("**")
        return numbers[base] ** numbers[exponent]
    return None


@pytest.mark.parametrize("cell", GRID.cells, ids=lambda cell: cell.id)
def test_a_cell_of_one_plain_operator_declares_what_arithmetic_gives(cell: Cell):
    """A cell of one operator that every SPICE computes as arithmetic (a sum,
    a difference, a product, a quotient by a number that is not zero, a sign,
    a positive base to a power) declares the number arithmetic gives, worked
    here from the deck's parameters, never from the evaluator."""
    expected = _arithmetic(cell)
    if expected is not None:
        for build in rec.BUILDS:
            if build not in cell.refused:
                assert _number(cell, build) == pytest.approx(expected), (cell.id, build)


def test_every_operator_the_evaluator_takes_is_a_value_of_an_axis():
    """The evaluator's binary operators, its power and its unary operators are
    each an axis value of the grid, so a new operator it learns needs cells."""
    by_class = {ast.Add: "plus", ast.Sub: "minus", ast.Mult: "times", ast.Div: "divide"}
    assert set(hierarchy_expr._BINARY) == set(by_class)
    assert set(by_class.values()) == set(GRID.axes["operator"].values)
    assert set(GRID.axes["power"].values) == {"star", "caret"}
    assert set(GRID.axes["unary"].values) == {"minus", "plus"}
    source = Path(hierarchy_expr.__file__).read_text(encoding="utf-8")
    accepted = set(re.findall(r"ast\.(UAdd|USub|Pow|Constant|Name|UnaryOp|BinOp)\b", source))
    assert accepted == {"UAdd", "USub", "Pow", "Constant", "Name", "UnaryOp", "BinOp"}


def test_a_recorded_cell_has_no_prediction_left():
    for cell in GRID.cells:
        if "not_established" in cell.extra:
            recorded = [build for build in rec.BUILDS if _recorded(build, cell.case)]
            assert not recorded, (
                f"{cell.id} is recorded on {recorded}: settle its number against the "
                "recording and remove not_established"
            )


def test_the_count_of_known_differences_only_falls():
    for build in rec.BUILDS:
        count = sum(
            len(entry["routes"])
            for cell in GRID.cells
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
        )
        assert count <= KNOWN_DIFFERENCES[build], (build, count)
        assert count == KNOWN_DIFFERENCES[build], f"lower the pin for {build} to {count}"


# Every route -------------------------------------------------------------------


def _same(taken: float | None, ltspice: float | None) -> bool:
    if taken is None:
        return True
    if ltspice is None:
        return False
    return math.isclose(taken, ltspice, rel_tol=SAME_VALUE_RTOL, abs_tol=1e-12)


def _environment_value(
    case_id: str,
    node: str,
    *,
    simulator: str,
    mode: str | None,
    ltspice_build: str | None = None,
) -> float | None:
    """The value ``Environment`` gives the probe node's source, over the
    deck's ``.param`` assignments."""
    params, values = _deck(case_id)
    environment = Environment(
        params, simulator=simulator, ngbehavior=mode, ltspice_build=ltspice_build
    )
    return environment.fact(values[node], "V", element_value=True).value


async def _hierarchy_values(
    state: SessionState, deck: Path, build: str
) -> dict[str, dict[str, float | None]]:
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}
    reply = await dispatched(state, "inspect", queries=[query])
    api_reply = await asyncio.to_thread(lambda: SyncApi(state).inspect(queries=[query]))
    found: dict[str, dict[str, float | None]] = {}
    for route, data in (("inspect", reply), ("api inspect", api_reply)):
        (result,) = data["results"]
        assert result["ok"], result
        for row in result["data"]["instances"]:
            if row["instance"][0][:1].upper() in ("R", "V"):
                found.setdefault(route, {})[row["nodes"][0]["name"].lower()] = row["value"].get(
                    "value"
                )
    for route, profile in (
        ("load_hierarchy", SemanticProfile("ltspice")),
        ("load_hierarchy, build known", SemanticProfile.of(_PRODUCERS[build], None)),
    ):
        for row in load_hierarchy(str(deck), None, profile).instances:
            if row.reference[:1].upper() in ("R", "V"):
                found.setdefault(route, {})[row.nodes[0].name.lower()] = row.value.value
    return found


@pytest.mark.parametrize(("build", "case_id"), _runs())
async def test_every_recorded_cell_evaluates_as_ltspice_computed_it_on_every_route(
    build: str, case_id: str, tmp_path: Path, state_no_sim: SessionState, work_dir: Path
):
    entry = rec.entry(build, case_id)
    voltages = (
        rec.operating_point(build, case_id, tmp_path)
        if entry["exit_code"] == 0 and rec.has(build, f"{case_id}.raw")
        else None
    )
    deck = work_dir / _source_of(case_id).name
    deck.write_bytes(_source_of(case_id).read_bytes())
    hierarchy = await _hierarchy_values(state_no_sim, deck, build)
    known = SemanticProfile.of(_PRODUCERS[build], None).ltspice_build
    assert known is not None
    found: set[tuple[str, str]] = set()
    declared: set[tuple[str, str]] = set()
    for cell in GRID.cells_of(case_id):
        node = cell.extra["node"]
        if voltages is None or build in cell.refused:
            # A build that stops on an expression can still write a raw for
            # the rest of the deck: its log is what says it refused.
            assert build in cell.refused, f"{cell.id}: {build} took no number"
            assert _number(cell, build) is None, f"{cell.id}: {build} took no number"
            ltspice = None
        else:
            assert build not in cell.refused, f"{cell.id}: {build} ran the deck"
            ltspice = voltages[f"v({node})"]
            assert _number(cell, build) == pytest.approx(ltspice, rel=SAME_VALUE_RTOL), cell.id
        answers = {
            "Environment": _environment_value(case_id, node, simulator="ltspice", mode=None),
            "Environment, build known": _environment_value(
                case_id, node, simulator="ltspice", mode=None, ltspice_build=known
            ),
            **{route: values[node] for route, values in hierarchy.items()},
        }
        found |= {
            (cell.item, route) for route, taken in answers.items() if not _same(taken, ltspice)
        }
        declared |= {
            (cell.item, route)
            for entry in cell.extra.get("differs", ())
            if entry["build"] == build
            for route in entry["routes"]
        }
    assert found == declared, (
        f"answered otherwise, not declared: {sorted(found - declared)}; "
        f"declared, not found: {sorted(declared - found)}"
    )


# ngspice, live ----------------------------------------------------------------

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

#: The compatibility modes the server models (``lib/ngspice_compat.py``); a
#: power's sign follows ``hs`` and ``lt`` (``signed_power``).
NGSPICE_MODES = ("", "hsa", "kiltpsa", "lt", "ps", "spe")

#: Expressions where ngspice computes another number than the server's
#: reading for ngspice, with the reason. Not asserted to differ: another
#: ngspice version may agree.
NGSPICE_DIFFERS: dict[str, str] = {
    "lpos+rpos": (
        "ngspice-42 stops on an element's value written as an expression with no "
        "braces; the server's reading evaluates it (evaluate reads a value that is "
        "not one number as an expression) and takes the sum"
    ),
}

_PRINTED = re.compile(r"^(\w+)\s*=\s*(\S+)", re.M)


def _ngspice_computes(mode: str, case_id: str, folder: Path) -> dict[str, float | None]:
    """What ngspice computes for each probe node of a deck, as its voltage,
    None where it stops on the deck. Each expression is run in its own deck
    with the grid deck's parameters."""
    params, values = _deck(case_id)
    plain = [f"{k}={v}" for k, v in params.items() if k != "psum"]
    cards = [".param " + " ".join(plain)] if plain else []
    if "psum" in params:
        cards.append(f".param psum={params['psum']}")
    taken: dict[str, float | None] = {}
    for node, value in values.items():
        deck = "\n".join(
            [
                "* expression",
                *cards,
                f"V1 {node} 0 {value}",
                ".control",
                "set numdgt=12",
                "op",
                "print all",
                ".endc",
                ".end",
                "",
            ]
        )
        log = run_batch(
            ("ngspice",),
            mode,
            deck.encode("ascii"),
            cwd=folder,
            clean_startup=True,
            timeout=LIVENESS_S,
        )
        printed = dict(_PRINTED.findall(log))
        taken[node] = float(printed[node]) if node in printed else None
    return taken


@needs_ngspice
@pytest.mark.parametrize("mode", NGSPICE_MODES)
def test_the_server_evaluates_each_expression_for_ngspice_as_ngspice_does(
    mode: str, tmp_path: Path
):
    """Each expression of the grid, with its deck's parameters, computed by
    the ngspice on PATH in ``mode``: where the server's reading for ngspice
    (``Environment`` and ``load_hierarchy`` for ngspice in that mode) takes a
    number, ngspice computes the same one."""
    wrong = []
    for case_id in _cases():
        taken = _ngspice_computes(mode, case_id, tmp_path)
        deck = tmp_path / _source_of(case_id).name
        deck.write_bytes(_source_of(case_id).read_bytes())
        read = {
            row.nodes[0].name.lower(): row.value.value
            for row in load_hierarchy(str(deck), None, SemanticProfile("ngspice", mode)).instances
        }
        for cell in GRID.cells_of(case_id):
            node = cell.extra["node"]
            if cell.item in NGSPICE_DIFFERS:
                continue
            for route, server in (
                ("Environment", _environment_value(case_id, node, simulator="ngspice", mode=mode)),
                ("load_hierarchy", read[node]),
            ):
                if not _same(server, taken[node]):
                    wrong.append((cell.item, route, server, taken[node]))
    assert not wrong, wrong
    assert set(NGSPICE_DIFFERS) <= {cell.item for cell in GRID.cells}
    assert all(NGSPICE_DIFFERS.values())
