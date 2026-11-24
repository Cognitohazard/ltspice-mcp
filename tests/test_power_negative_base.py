"""What a power with a negative base or a negated exponent, set in a
``.param``, evaluates to, on every route that answers it.

The question is the power-negative-base behaviour's (``inputs/cases.toml``):
each deck sets one power in a ``.param`` and puts it on a source, so the node
voltage in the ``.op`` raw is what LTspice computed. The server answers it in
``hierarchy_expr._power``, reached through ``Environment`` and through
``load_hierarchy`` (``inspect(kind="hierarchy")``). A route told which build
runs the deck takes that build's number, or declines where the build refused
the deck; a route that is not told takes the number where both builds agree
and declines where they differ.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ltspice_mcp.lib.format import SAME_VALUE_RTOL
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_expr import Environment
from ltspice_mcp.lib.log_parser import printed_measurements
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import dispatched
from tests.ltspice_recorder import INPUTS

BEHAVIOUR = "power-negative-base"

#: How each recorded build names itself in a run's output.
_PRODUCERS = {
    "ltspice26": Producer("ltspice", "LTspice 26.1.1 for Windows", None),
    "ltspice17": Producer("ltspice", "Linear Technology Corporation LTspice XVII", None),
}

#: What each build computed, by case and node, worked by hand from the
#: recordings: None where the build refused the deck, with the reason its log
#: gives. LTspice binds a sign written before a number to the number, so
#: ``-2**2`` is 4, as ``(-2)**2`` is.
COMPUTED: dict[str, dict[str, dict[str, float | None]]] = {
    "deck/pow_neg_base_whole": {
        "ltspice26": {"e01": 4.0, "e02": 4.0},
        "ltspice17": {"e01": 4.0, "e02": 4.0},
    },
    "deck/pow_neg_base_negated_exponent": {
        "ltspice26": {"e01": 0.125},
        "ltspice17": {"e01": 0.125},
    },
    "deck/pow_neg_base_cube_root": {
        "ltspice26": {"e01": 0.0},
        "ltspice17": {"e01": None},
    },
    "deck/pow_neg_base_square_root": {
        "ltspice26": {"e01": 0.0},
        "ltspice17": {"e01": None},
    },
}

#: The line of each refusal in the build's log.
REFUSED_BECAUSE = {
    ("ltspice17", "deck/pow_neg_base_cube_root"): "WARNING: Can't resolve .param p1=(-8)**(1/3)",
    ("ltspice17", "deck/pow_neg_base_square_root"): "WARNING: Can't resolve .param p1=(-2)**0.5",
}


def test_every_case_of_the_behaviour_is_worked():
    assert set(rec.cases_of(BEHAVIOUR)) == set(COMPUTED)


def _same(taken: float | None, computed: float | None) -> bool:
    if taken is None or computed is None:
        return taken is computed
    return math.isclose(taken, computed, rel_tol=SAME_VALUE_RTOL, abs_tol=1e-12)


def _deck(case_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """The deck's ``.param`` assignments and each source's value, by node."""
    params: dict[str, str] = {}
    values: dict[str, str] = {}
    for card in rec.deck_cards(rec.CASES.case(case_id).source):
        words = card.body.split()
        if card.kind == "param":
            for word in words[1:]:
                name, _, value = word.partition("=")
                params[name] = value
        elif card.kind == "instance" and words[0][:1].upper() == "V":
            values[words[1].lower()] = words[3]
    return params, values


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(sorted(COMPUTED))))
async def test_a_power_reads_as_the_build_computed_it_on_every_route(
    build: str, case_id: str, tmp_path: Path, state_no_sim: SessionState, work_dir: Path
):
    computed = COMPUTED[case_id][build]
    log = rec.recorded(build, f"{case_id}.log").read_bytes()
    text = log.decode("utf-16-le" if b"\x00" in log else "utf-8").replace("\x00", "")
    if (build, case_id) in REFUSED_BECAUSE:
        assert REFUSED_BECAUSE[build, case_id] in text
        assert set(computed.values()) == {None}
    else:
        voltages = rec.operating_point(build, case_id, tmp_path)
        for node, number in computed.items():
            assert number == pytest.approx(voltages[f"v({node})"], rel=SAME_VALUE_RTOL)

    other = next(b for b in COMPUTED[case_id] if b != build)
    agreed = {
        node: number if _same(number, COMPUTED[case_id][other][node]) else None
        for node, number in computed.items()
    }
    known = SemanticProfile.of(_PRODUCERS[build], None)
    assert known.ltspice_build is not None
    params, values = _deck(case_id)

    deck = work_dir / Path(rec.CASES.case(case_id).source).name
    deck.write_bytes((INPUTS / rec.CASES.case(case_id).source).read_bytes())
    query = {"kind": "hierarchy", "path": str(deck), "simulator": "ltspice"}
    (inspected,) = (await dispatched(state_no_sim, "inspect", queries=[query]))["results"]
    assert inspected["ok"], inspected
    by_inspect = {
        row["nodes"][0]["name"].lower(): row["value"].get("value")
        for row in inspected["data"]["instances"]
    }

    for node, value in values.items():
        answers = {
            "Environment": (
                Environment(params, simulator="ltspice", ngbehavior=None)
                .fact(value, "V", element_value=True)
                .value,
                agreed[node],
            ),
            "Environment, build known": (
                Environment(
                    params,
                    simulator="ltspice",
                    ngbehavior=None,
                    ltspice_build=known.ltspice_build,
                )
                .fact(value, "V", element_value=True)
                .value,
                computed[node],
            ),
            "inspect": (by_inspect[node], agreed[node]),
        }
        for route, profile, expected in (
            ("load_hierarchy", SemanticProfile("ltspice"), agreed[node]),
            ("load_hierarchy, build known", known, computed[node]),
        ):
            rows = load_hierarchy(str(deck), None, profile).instances
            (row,) = [row for row in rows if row.nodes[0].name.lower() == node]
            answers[route] = (row.value.value, expected)
        wrong = {
            route: taken
            for route, (taken, expected) in answers.items()
            if not _same(taken, expected)
        }
        assert not wrong, (case_id, build, node, wrong)


def test_a_warning_line_of_a_refused_power_is_no_measurement_but_a_lower_case_name_is():
    """XVII prints ``WARNING: Can't resolve .param p1=(-8)**(1/3)`` for the
    cube root; that line is a diagnostic, and no measurement named warning.
    Both builds print a measurement's name in lower case (recorded:
    log/meas_grid_name_upper), so a measurement named ``error`` stays one."""
    log = rec.recorded("ltspice17", "deck/pow_neg_base_cube_root.log").read_bytes()
    text = log.decode("utf-16-le" if b"\x00" in log else "utf-8").replace("\x00", "")
    assert "WARNING: Can't resolve .param p1=(-8)**(1/3)" in text
    assert printed_measurements(text) == {}
    assert set(printed_measurements("error: v(out)=1 at 0.001\n")) == {"error"}
