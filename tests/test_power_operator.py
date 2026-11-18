"""What a power in a parameter expression evaluates to, for the simulator and
the ngspice mode that read the deck (``hierarchy_expr.evaluate``).

Measured on ngspice-42, a ``.param`` read by a source's value
(``test_ngspice_computes_a_power_as_the_table_says``):

==============  ===========  ========  =======
Expression      native       kiltpsa   hsa
==============  ===========  ========  =======
``(-2)^3``      8            -8        -8
``(-2)^0.5``    1.414        0         -2
``0^0``         1            1         0
``0^(-1)``      fails        fails     0
==============  ===========  ========  =======

A positive base gives the same value in every mode, and the reader computes a
power only where its value is the one ngspice computes: a positive base, and
a negative base with an integer exponent in kiltpsa and hsa. Everywhere else
it declines, with the reason, rather than model ngspice's arithmetic.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.hierarchy_expr import Environment, evaluate

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")


def _no_names(name: str) -> float:
    raise ValueError(f"unknown {name}")


#: What the reader answers for each expression and mode: a value, or the
#: words its refusal holds.
READER = {
    ("2^3", ""): 8.0,
    ("2^3", "hsa"): 8.0,
    ("2^3", "kiltpsa"): 8.0,
    ("0.5^-1", ""): 2.0,
    ("(-2)^3", ""): "magnitude",
    ("(-2)^3", "kiltpsa"): -8.0,
    ("(-2)^3", "hsa"): -8.0,
    ("(-2)^2", "hsa"): 4.0,
    ("(-2)^0.5", ""): "magnitude",
    ("(-2)^0.5", "kiltpsa"): "fractional",
    ("(-2)^0.5", "hsa"): "fractional",
    ("0^0", ""): "zero",
    ("0^0", "kiltpsa"): "zero",
    ("0^0", "hsa"): "zero",
    ("0^(-1)", ""): "zero",
    ("0^(-1)", "hsa"): "zero",
    ("0^2", "hsa"): 0.0,
    ("(-2)**3", ""): "magnitude",
    ("(-2)**3", "hsa"): -8.0,
}


@pytest.mark.parametrize(("expression", "mode"), list(READER))
def test_the_reader_computes_a_power_only_where_its_value_is_pinned(expression: str, mode: str):
    expected = READER[expression, mode]
    if isinstance(expected, float):
        assert evaluate(expression, _no_names, simulator="ngspice", ngbehavior=mode) == expected
        return
    with pytest.raises(ValueError, match=expected):
        evaluate(expression, _no_names, simulator="ngspice", ngbehavior=mode)


def test_a_negated_name_straight_after_the_operator_is_declined():
    """ngspice-42 refuses ``3^-b`` in every mode ("wrongly determined
    negation") and computes ``3^(-b)`` and ``3^-1``."""

    def lookup(name: str) -> float:
        return -2.0

    for mode in ("", "hsa", "kiltpsa"):
        with pytest.raises(ValueError, match="negated name"):
            evaluate("3^-b", lookup, simulator="ngspice", ngbehavior=mode)
        with pytest.raises(ValueError, match="negated name"):
            evaluate("3**-b", lookup, simulator="ngspice", ngbehavior=mode)
        assert evaluate("3^(-b)", lookup, simulator="ngspice", ngbehavior=mode) == 9.0
        assert evaluate("3^-1", lookup, simulator="ngspice", ngbehavior=mode) == 1 / 3


def test_ngspice_needs_its_mode():
    with pytest.raises(ValueError, match="mode"):
        evaluate("2^3", _no_names, simulator="ngspice", ngbehavior=None)
    with pytest.raises(ValueError, match="mode"):
        Environment({}, simulator="ngspice", ngbehavior=None)


def test_ltspice_caret_stays_declined():
    with pytest.raises(ValueError, match="LTspice"):
        evaluate("2^3", _no_names, simulator="ltspice", ngbehavior=None)


def _hierarchy_value(tmp_path: Path, expression: str, mode: str):
    deck = tmp_path / "power.cir"
    deck.write_text(
        f"* power\n.param p={{{expression}}}\nR1 a 0 {{p}}\nV1 a 0 1\n.op\n.end\n",
        encoding="utf-8",
    )
    hierarchy = load_hierarchy(str(deck), [tmp_path], SemanticProfile("ngspice", mode))
    (resistor,) = [row for row in hierarchy.instances if row.reference == "R1"]
    return resistor.value


def test_hierarchy_declines_a_negative_power_in_native_mode(tmp_path: Path):
    """``inspect`` marked the reader's -8 as resolved where native ngspice
    computes 8."""
    fact = _hierarchy_value(tmp_path, "(-2)^3", "")
    assert fact.status == "unresolved"
    assert fact.value is None
    assert fact.reason is not None and "magnitude" in fact.reason
    assert _hierarchy_value(tmp_path, "(-2)^3", "hsa").value == -8.0


@needs_ngspice
async def test_inspect_declines_a_negative_power_in_native_mode(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    from tests.test_hierarchy_check import _hierarchy, _session

    state = _session(work_dir, "hsa", monkeypatch)
    deck = work_dir / "power.cir"
    deck.write_text(
        "* power\n.param p={(-2)^3}\nR1 a 0 {p}\nV1 a 0 1\n.op\n.end\n", encoding="utf-8"
    )
    result = await _hierarchy(state, deck, ngbehavior="")
    assert result["ok"], result
    (row,) = [row for row in result["data"]["instances"] if row["reference"] == "R1"]
    assert row["value"]["status"] == "unresolved", row
    assert "magnitude" in row["value"]["reason"]


def test_a_run_draws_no_mismatch_from_a_multiplier_ngspice_reads_otherwise(tmp_path: Path):
    """Under hsa ngspice reads ``0^0`` as 0, where the reader read 1: the
    mismatch area of XA's devices was drawn for one copy."""
    from ltspice_mcp.lib.variations import VariationError
    from tests.test_nested_targeting import _nested_draw

    with pytest.raises(VariationError) as exc:
        _nested_draw(
            tmp_path / "zero",
            "ngspice",
            replacements=[
                ("XA a g block\n", "XA a g block m={0^0}\n"),
                ("Xleaf p g leaf\n", "Xleaf p g leaf m=1\n"),
            ],
        )
    assert exc.value.code == "mismatch_geometry_unresolved"
    assert "zero" in str(exc.value)


def test_the_mismatch_plan_reads_a_power_as_native_ngspice():
    """The plan for a deck with no semantic profile (QSPICE, Xyce) reads an
    expression in ngspice's native mode, the strictest: it computes a power
    of a positive base, and leaves a negative one unresolved."""
    from ltspice_mcp.lib.subckt_mismatch import _environment

    scope = _environment({"p": "(-2)^3", "q": "2^3"})
    assert scope.fact("p").value is None
    assert "magnitude" in (scope.fact("p").reason or "")
    assert scope.fact("q").value == 8.0


# --------------------------------------------------------------------------
# Live: ngspice computes each power as the table says
# --------------------------------------------------------------------------

#: What ngspice-42 computes, by expression and mode; None where the deck fails.
LIVE_TABLE = {
    "(-2)^3": {"": 8.0, "kiltpsa": -8.0, "hsa": -8.0},
    "(-2)^0.5": {"": 2**0.5, "kiltpsa": 0.0, "hsa": -2.0},
    "0^0": {"": 1.0, "kiltpsa": 1.0, "hsa": 0.0},
    "0^(-1)": {"": None, "kiltpsa": None, "hsa": 0.0},
    "2^3": {"": 8.0, "kiltpsa": 8.0, "hsa": 8.0},
    "0.5^-1": {"": 2.0, "kiltpsa": 2.0, "hsa": 2.0},
    "(-2)^2": {"": 4.0, "kiltpsa": 4.0, "hsa": 4.0},
    "2^3^2": {"": 64.0, "kiltpsa": 64.0, "hsa": 64.0},
    "3^-y": {"": None, "kiltpsa": None, "hsa": None},
    "3^(-y)": {"": 9.0, "kiltpsa": 9.0, "hsa": 9.0},
    "0^2": {"": 0.0, "kiltpsa": 0.0, "hsa": 0.0},
    "(-2)^(-1)": {"": 0.5, "kiltpsa": -0.5, "hsa": -0.5},
}


def _ngspice_version() -> str:
    out = subprocess.run(
        ["ngspice", "-v"], capture_output=True, text=True, timeout=60, check=False
    ).stdout
    match = re.search(r"ngspice-\S+", out)
    return match.group(0) if match else out.strip()


@needs_ngspice
@pytest.mark.parametrize("mode", ["", "kiltpsa", "hsa"])
@pytest.mark.parametrize("expression", list(LIVE_TABLE))
def test_ngspice_computes_a_power_as_the_table_says(expression: str, mode: str, tmp_path: Path):
    print(f"ngspice that ran: {_ngspice_version()}")
    deck = tmp_path / "power.cir"
    deck.write_text(
        f"* power\n.param y=-2\n.param x={{{expression}}}\nV1 a 0 {{x}}\nR1 a 0 1\n.op\n"
        ".control\nop\nprint v(a)\n.endc\n.end\n",
        encoding="utf-8",
    )
    command = ["ngspice", "-n", "-b", *(["-D", f"ngbehavior={mode}"] if mode else []), deck.name]
    out = subprocess.run(
        command, capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False
    )
    match = re.search(r"v\(a\) = (\S+)", out.stdout + out.stderr)
    measured = float(match.group(1)) if match else None
    expected = LIVE_TABLE[expression][mode]
    if expected is None:
        assert measured is None
    else:
        assert measured == pytest.approx(expected, rel=1e-5, abs=1e-12)
    # The reader agrees where it answers, and declines everywhere else.
    try:
        value = evaluate(expression, lambda name: -2.0, simulator="ngspice", ngbehavior=mode)
    except ValueError:
        return
    assert measured is not None and value == pytest.approx(measured, rel=1e-5, abs=1e-12)
