"""Where a trace crosses a level between its samples, found by one rule.

``signal_analysis.level_crossings`` is the rule: a strict change of sign
between two samples, or a sample on the level that the next one leaves,
placed by linear interpolation. Transient edges and timing, AC crossings
(in log frequency) and the corner of a phase lobe all use it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib.ac_analysis import detect_crossings
from ltspice_mcp.lib.ac_structure import _f0_at_half_swing
from ltspice_mcp.lib.signal_analysis import _interp_crossings, level_crossings

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp"


@pytest.mark.parametrize(
    ("y", "crossings"),
    [
        ([0.0, 2.0], [(0, 0.5, True)]),
        ([2.0, 0.0], [(0, 0.5, False)]),
        # A sample on the level is one crossing, from where it leaves.
        ([0.0, 1.0, 2.0], [(1, 1.0, True)]),
        ([2.0, 1.0, 0.0], [(1, 1.0, False)]),
        # Touching the level and turning back is a crossing out of it.
        ([0.0, 1.0, 0.0], [(1, 1.0, False)]),
        ([1.0, 1.0, 1.0], []),
        ([1.0, 1.0, 2.0], [(1, 1.0, True)]),
        ([0.0, 0.5, 0.9], []),
    ],
)
def test_a_crossing_is_found_once(y: list[float], crossings: list[tuple[int, float, bool]]):
    x = np.arange(len(y), dtype=float)
    index, at, rising = level_crossings(x, np.asarray(y), 1.0)
    assert list(zip(index.tolist(), at.tolist(), rising.tolist(), strict=True)) == crossings
    assert _interp_crossings(x, np.asarray(y), 1.0, "any") == [c[1] for c in crossings]
    found = detect_crossings(10.0**x, np.asarray(y), 1.0)
    assert [c["frequency_hz"] for c in found] == pytest.approx([10.0 ** c[1] for c in crossings])
    assert [c["direction"] == "rising" for c in found] == [c[2] for c in crossings]


@pytest.mark.parametrize("on_a_sample", [True, False])
def test_a_phase_lobes_corner_is_the_crossing_of_its_half_swing(on_a_sample: bool):
    """A -180 degree lobe whose -90 degree point falls on a sample, or
    between two."""
    count = 21 if on_a_sample else 20
    x = np.linspace(2.0, 4.0, count)
    phase = np.linspace(0.0, -180.0, count)
    assert (-90.0 in phase.tolist()) is on_a_sample

    corner = _f0_at_half_swing(x, phase, 0, count - 1, -180.0)

    (crossing,) = detect_crossings(10.0**x, phase, -90.0)
    assert corner == pytest.approx(crossing["frequency_hz"], rel=1e-12)
    assert corner == pytest.approx(1e3, rel=1e-12)


#: Sign changes that are not a level crossing: the passband ripple check
#: counts where a response's slope turns.
_NOT_CROSSINGS = {("ac_analysis.py", "np.diff(np.sign(diffs))")}


def _sign_change_tests() -> list[tuple[str, str]]:
    """Comparisons that test consecutive samples for a change of sign:
    ``d[:-1] * d[1:]`` or ``np.diff(np.sign(...))``."""
    found = []
    for path in sorted((SOURCE / "lib").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Compare):
                text = ast.unparse(node.left)
                if ("[:-1] * " in text and "[1:]" in text) or "np.diff(np.sign(" in text:
                    found.append((path.name, text))
    return found


def test_only_the_crossing_rule_tests_samples_for_a_sign_change():
    found = [entry for entry in _sign_change_tests() if entry not in _NOT_CROSSINGS]
    assert [module for module, _ in found] == ["signal_analysis.py"]
