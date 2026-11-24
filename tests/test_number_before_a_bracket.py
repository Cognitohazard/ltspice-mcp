"""A number written directly before a closing parenthesis or a comma, on every
route that reads one.

The question is which number a token such as ``1e3)`` names. A closing
parenthesis or a comma is no part of a number, and both LTspice builds read
the number in front of the parenthesis that closes a source function: the
recorded ``raw/compress_1mv`` deck drives ``PWL(0 0 1u 1m)``, and its RC
reaches 0.632 mV at one time constant on both builds, so ``1m)`` was 1 mV.
``format.parse_spice_value`` decides it; every other reader asks it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.format import parse_plain_value, parse_spice_value, plain_number
from ltspice_mcp.lib.metrics import parse_freq, parse_time
from ltspice_mcp.lib.montecarlo import parse_value
from ltspice_mcp.lib.netlist_graph import values_equal
from ltspice_mcp.lib.spice_validator import leading_numeric_tokens
from ltspice_mcp.lib.step_values import step_number
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import dispatched, stage_recorded_fixture


def _readings(token: str) -> dict[str, float | None]:
    """What each reader of a single value takes ``token`` for; None where it
    refuses it."""

    def refused_as_none(read):
        try:
            return read(token)
        except Exception:
            return None

    return {
        "parse_spice_value": refused_as_none(parse_spice_value),
        "parse_plain_value": refused_as_none(parse_plain_value),
        "plain_number": plain_number(token),
        "leading_numeric_tokens": (leading_numeric_tokens([token]) or [None])[0],
        "step_number": step_number(token),
        "metrics.parse_time": refused_as_none(lambda t: parse_time(t, "t")),
        "metrics.parse_freq": refused_as_none(parse_freq),
        "montecarlo.parse_value": parse_value(token),
    }


@pytest.mark.parametrize(
    ("token", "number"),
    [
        ("1e3)", 1e3),
        ("1e6)", 1e6),
        ("1k)", 1e3),
        ("1Meg)", 1e6),
        ("2.5e3)", 2.5e3),
        ("1m)", 1e-3),
        ("1e3,", 1e3),
        ("1k,", 1e3),
        # No punctuation: read as before.
        ("1e3", 1e3),
        ("1k", 1e3),
        ("1Meg", 1e6),
    ],
)
def test_every_reader_reads_the_number_before_a_bracket_or_a_comma(token: str, number: float):
    readings = _readings(token)
    wrong = {route: taken for route, taken in readings.items() if taken != pytest.approx(number)}
    assert not wrong, (token, wrong)


@pytest.mark.parametrize(("token", "number"), [("4R7", 4.7), ("1k5", 1.5e3)])
def test_digits_after_a_suffix_read_as_before(token: str, number: float):
    """``parse_plain_value`` refuses these by design (a name such as 1N4148 is
    no number there); every other reader reads them as before."""
    readings = _readings(token)
    assert readings.pop("parse_plain_value") is None
    wrong = {route: taken for route, taken in readings.items() if taken != pytest.approx(number)}
    assert not wrong, (token, wrong)


@pytest.mark.parametrize("token", ["1e3%", ")", ",", "1e3,2e3", "1e3))"])
def test_a_token_that_is_no_number_is_refused_by_every_reader(token: str):
    readings = _readings(token)
    assert set(readings.values()) == {None}, (token, readings)


def test_the_recording_shows_both_builds_read_the_number_before_the_bracket():
    """``PWL(0 0 1u 1m)`` drives an RC of 1 ms; at 1 ms its output is
    1 mV * (1 - 1/e), so the build took ``1m)`` as 1 mV."""
    deck = rec.CASES.case("raw/compress_1mv").source
    assert "PWL(0 0 1u 1m)" in (rec.INPUTS / deck).read_text()
    for build in rec.BUILDS:
        log = rec.recorded(build, "raw/compress_1mv.log").read_bytes()
        text = log.decode("utf-16-le" if b"\x00" in log else "utf-8").replace("\x00", "")
        (line,) = [line for line in text.splitlines() if line.startswith("vat:")]
        at_one_ms = float(line.split("=")[1].split()[0])
        assert at_one_ms == pytest.approx(1e-3 * (1 - 1 / 2.718281828459045), rel=5e-3), build


@pytest.mark.parametrize("token", ["{1/0}", "{(}", "{1e999*1e999}"])
def test_a_compare_of_a_braced_value_that_is_no_constant_compares_its_text(token: str):
    assert values_equal(token, token, 1e-6)
    assert not values_equal(token, "1", 1e-6)


def test_a_braced_constant_the_simulators_compute_differently_compares_as_text():
    """LTspice binds the sign to the number (``{-2**2}`` is 4, recorded:
    deck/expr_grid_negated_power_star); ngspice takes -4. A compare does not
    know which simulator runs the deck."""
    assert not values_equal("{-2**2}", "4", 1e-6)
    assert not values_equal("{-2**2}", "-4", 1e-6)
    assert values_equal("{2**2}", "4", 1e-6)


def test_a_comma_separated_list_reads_each_number():
    assert values_equal("1e3, 2", "1000, 2", 1e-6)
    assert not values_equal("1e3, 2", "1e6, 2", 1e-6)


SINE_DECK = "* a source\nV1 in 0 {source}\nR1 in 0 1k\n.tran 1m\n.end\n"


@pytest.mark.parametrize(
    ("deck_value", "reference_value", "equivalent"),
    [
        ("SINE(0 1 1e3)", "SINE(0 1 1e6)", False),
        ("SINE(0 1 1k)", "SINE(0 1 1Meg)", False),
        ("SINE(0 1 1e3)", "SINE(0 1 1k)", True),
        ("SINE(0 1 1e3)", "SINE(0 1 1000)", True),
    ],
)
async def test_verify_circuit_compares_the_number_before_the_bracket(
    deck_value: str,
    reference_value: str,
    equivalent: bool,
    state_no_sim: SessionState,
    work_dir: Path,
):
    deck = work_dir / "deck.cir"
    deck.write_text(SINE_DECK.format(source=deck_value), encoding="utf-8")
    reference = work_dir / "reference.cir"
    reference.write_text(SINE_DECK.format(source=reference_value), encoding="utf-8")
    data = await dispatched(
        state_no_sim,
        "verify_circuit",
        path=str(deck),
        checks=["compare"],
        compare={"reference": str(reference)},
    )
    comparison = data["comparison"]
    assert comparison["equivalent"] is equivalent, comparison
    if not equivalent:
        (mismatch,) = comparison["value_mismatches"]
        assert "V1" in str(mismatch), mismatch


async def test_an_analyze_results_bound_before_a_bracket_reads_as_on_the_other_routes(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    values = {}
    for at in ("900u", "900u)", "0.9e-3)", "9e-4"):
        data = await dispatched(
            state_no_sim,
            "analyze_results",
            sources=[{"path": str(raw)}],
            recipes=[{"key": "v", "metric": "value", "expr": "V(out)", "at": at}],
        )
        (row,) = data["results"]["v"]["values"]
        values[at] = row["value"]
    assert all(value == values["900u"] for value in values.values()), values
    assert parse_time("0.9e-3)", "at") == parse_time("9e-4", "at")
