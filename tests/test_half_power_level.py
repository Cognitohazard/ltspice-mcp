"""The level a "-3 dB" cutoff, bandwidth or resonance flank is read at.

Every reader of one uses ``HALF_POWER_DB``, the true half-power level
(20·log10(1/sqrt(2))), so the filter recipe, the summary's bandwidth and a
crossing at that level agree with each other and with LTspice's own ``.meas``
of the same corner.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib.ac_analysis import HALF_POWER_DB, compute_resonances
from ltspice_mcp.lib.raw_parser import compute_ac_bandwidth_metrics
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ltspice_mcp" / "lib"
#: The modules that read an AC response's -3 dB points.
READERS = ("ac_analysis.py", "metrics.py", "raw_parser.py")


def test_the_level_is_the_half_power_point():
    half_power = 20 * math.log10(1 / math.sqrt(2))
    assert abs(HALF_POWER_DB - half_power) < 1e-12


def _rc_reply(state: SessionState, raw: Path, recipes: list[dict]) -> dict[str, dict]:
    reply = SyncApi(state).analyze_results(
        sources=[{"path": str(raw), "label": "rc"}],
        recipes=recipes,
        include={"fields": ["value"]},
    )
    assert reply["failures"] == []
    return {key: result["values"][0]["value"] for key, result in reply["results"].items()}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_reader_of_the_rc_corner_agrees_with_ltspices_meas(
    build: str, state_no_sim: SessionState, work_dir: Path, tmp_path: Path
):
    """``raw/ac``: a 1 kHz RC low-pass swept at 5 points a decade, with
    ``.meas ac fc WHEN mag(V(out))=0.70710678`` in the deck."""
    measured = rec.decode_log(build, "raw/ac", tmp_path).value("measurements")
    corner = measured["measurements"]["fc"]["at"]
    assert corner == pytest.approx(1000.0, rel=1e-6)

    raw = rec.stage_run(build, "raw/ac", work_dir)
    first = _rc_reply(
        state_no_sim,
        raw,
        [
            {"key": "filter", "metric": "bode_filter", "signal": "V(out)"},
            {
                "key": "crossing",
                "metric": "bode_crossing",
                "signal": "V(out)",
                "level_db": HALF_POWER_DB,
            },
        ],
    )
    filtered, crossed = first["filter"], first["crossing"]
    bandwidth = compute_ac_bandwidth_metrics(
        rec.decode(build, "raw/ac", tmp_path / "decoded").raw, "V(out)"
    )["bandwidth_3db"]
    assert crossed["first_crossing_hz"] == pytest.approx(corner, rel=1e-4)
    assert bandwidth == pytest.approx(corner, rel=1e-4)

    # The filter's cutoff is the half-power level below its passband, which
    # it reads off the sweep's first samples: -0.0011 dB at 5 points a decade.
    assert filtered["ref_db"] == HALF_POWER_DB
    passband = filtered["passband_gain_db"]
    assert passband == pytest.approx(-0.0011, abs=1e-4)
    below_passband = _rc_reply(
        state_no_sim,
        raw,
        [
            {
                "key": "crossing",
                "metric": "bode_crossing",
                "signal": "V(out)",
                "level_db": passband + HALF_POWER_DB,
            }
        ],
    )["crossing"]
    assert filtered["cutoff_high_hz"] == pytest.approx(
        below_passband["first_crossing_hz"], rel=1e-9
    )
    assert filtered["cutoff_high_hz"] == pytest.approx(corner, rel=3e-4)


def test_a_resonance_flank_is_read_at_the_half_power_level():
    """A second-order band-pass of Q 5 at 10 kHz: its half-power bandwidth is
    exactly f0/Q, 2 kHz."""
    freqs = np.logspace(3, 5, 4001)
    s = 1j * freqs / 10e3
    q = 5.0
    response = (s / q) / (s**2 + s / q + 1)
    (peak,) = compute_resonances(freqs, response)["peaks"]
    assert peak["frequency_hz"] == pytest.approx(10e3, rel=1e-3)
    assert peak["bandwidth_3db_hz"] == pytest.approx(10e3 / q, rel=1e-3)
    assert peak["q_factor"] == pytest.approx(q, rel=1e-3)


def _three_db_literals(path: Path) -> list[int]:
    """Lines where 3.0 is used as a level in dB: added to or subtracted from
    something, or given as a default, outside the definition of HALF_POWER_DB."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[int] = []

    def is_three(node: ast.AST) -> bool:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            node = node.operand
        return isinstance(node, ast.Constant) and node.value == 3.0 and type(node.value) is float

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            if is_three(node.left) or is_three(node.right):
                found.append(node.lineno)
        elif isinstance(node, ast.arguments):
            defaults = [*node.defaults, *(d for d in node.kw_defaults if d is not None)]
            args = [*node.posonlyargs, *node.args][-len(node.defaults) :] if node.defaults else []
            named = list(zip(args, node.defaults, strict=True)) + [
                (arg, default)
                for arg, default in zip(node.kwonlyargs, node.kw_defaults, strict=True)
                if default is not None
            ]
            assert len(named) == len(defaults)
            found += [
                default.lineno
                for arg, default in named
                if is_three(default) and arg.arg.endswith("_db") and "prominence" not in arg.arg
            ]
    return found


@pytest.mark.parametrize("module", READERS)
def test_no_reader_writes_its_own_three_db_level(module: str):
    assert _three_db_literals(SOURCE / module) == []
