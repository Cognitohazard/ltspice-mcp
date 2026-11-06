"""The level a "-3 dB" cutoff, bandwidth or resonance flank is read at.

Every reader of one uses ``HALF_POWER_DB``, the true half-power level
(20·log10(1/sqrt(2))), so the filter recipe, the summary's bandwidth and a
crossing at that level agree with each other and with LTspice's own ``.meas``
of the same corner.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib.ac_analysis import HALF_POWER_DB, compute_resonances
from ltspice_mcp.lib.raw_parser import compute_ac_bandwidth_metrics
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._source_scan import PACKAGE, parsed
from tests.conftest import SyncApi


def test_the_level_is_the_half_power_point():
    half_power = 20 * math.log10(1 / math.sqrt(2))
    assert abs(HALF_POWER_DB - half_power) < 1e-12


def _rc_reply(state: SessionState, raw: Path, recipes: list[dict]) -> dict[str, dict]:
    reply = SyncApi(state).analyze_results(
        sources=[{"path": str(raw), "label": "rc"}],
        recipes=recipes,
        include={"fields": ["value"]},
    )
    assert reply.get("failures", []) == []
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


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("ref_hz", ["10", 100.0, "2.5k"])
def test_a_crossing_below_the_gain_at_a_frequency_takes_one_call(
    build: str, ref_hz: str | float, state_no_sim: SessionState, work_dir: Path
):
    """``ref_hz`` measures ``level_db`` from the gain at that frequency: the
    half-power point below the gain at 10 Hz (the sweep's first point), at
    100 Hz (between points) or at 2.5 kHz (past the corner) is the crossing at
    the absolute level bode_point reads there, plus ``HALF_POWER_DB``. Two
    calls before; one now."""
    raw = rec.stage_run(build, "raw/ac", work_dir)
    reference = _rc_reply(
        state_no_sim,
        raw,
        [{"key": "ref", "metric": "bode_point", "signal": "V(out)", "at_hz": ref_hz}],
    )["ref"]
    absolute, relative = (
        _rc_reply(state_no_sim, raw, [{"key": key, **recipe}])[key]
        for key, recipe in (
            (
                "absolute",
                {
                    "metric": "bode_crossing",
                    "signal": "V(out)",
                    "level_db": reference["magnitude_db"] + HALF_POWER_DB,
                },
            ),
            (
                "relative",
                {
                    "metric": "bode_crossing",
                    "signal": "V(out)",
                    "level_db": HALF_POWER_DB,
                    "ref_hz": ref_hz,
                },
            ),
        )
    )
    assert relative["first_crossing_hz"] == pytest.approx(absolute["first_crossing_hz"], rel=1e-12)
    assert relative["level"] == pytest.approx(absolute["level"], rel=1e-12)
    assert relative["reference"]["magnitude_db"] == reference["magnitude_db"]
    assert relative["reference"]["level_db"] == HALF_POWER_DB


@pytest.mark.parametrize(
    ("recipe", "refused"),
    [
        ({"ref_hz": "1meg", "level_db": -3.0}, "outside the sweep"),
        ({"ref_hz": "1k", "level_deg": -45.0}, "a phase level is not relative"),
    ],
)
def test_a_reference_the_crossing_cannot_be_measured_from_is_refused(
    recipe: dict, refused: str, state_no_sim: SessionState, work_dir: Path
):
    raw = rec.stage_run(rec.BUILDS[0], "raw/ac", work_dir)
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"path": str(raw), "label": "rc"}],
        recipes=[{"key": "c", "metric": "bode_crossing", "signal": "V(out)", **recipe}],
    )
    assert refused in json.dumps(reply)


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


def _is_three(node: ast.AST) -> bool:
    """``3`` or ``-3``, written as an int or a float."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        node = node.operand
    if not isinstance(node, ast.Constant):
        return False
    value = node.value
    return isinstance(value, int | float) and not isinstance(value, bool) and abs(value) == 3


def _is_float(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, float)


def _says_db(word: str) -> bool:
    """Whether a name or key says it holds decibels: a ``db`` part, as in
    ``level_db``, ``HALF_POWER_DB`` or ``"magnitude_db"``."""
    return "db" in word.lower().split("_")


def _mentions_db(node: ast.AST) -> bool:
    """Whether ``node`` reads a name, attribute or key that holds decibels."""
    return any(
        (isinstance(inner, ast.Name) and _says_db(inner.id))
        or (isinstance(inner, ast.Attribute) and _says_db(inner.attr))
        or (
            isinstance(inner, ast.Constant)
            and isinstance(inner.value, str)
            and _says_db(inner.value)
        )
        for inner in ast.walk(node)
    )


def _gives_three(node: ast.AST) -> bool:
    """``node`` is 3 dB, or a call given it (``Field(default=-3.0)``)."""
    if isinstance(node, ast.Call):
        return any(_is_three(given) for given in [*node.args, *(k.value for k in node.keywords)])
    return _is_three(node)


#: Names given 3 dB that is not a half-power level, with what it is instead.
_OTHER_THREE_DB: dict[str, str] = {
    "min_prominence_db": "how far a peak must stand above its surroundings to be one",
    "_NOTCH_UNDERSAMPLED_GAP_DB": (
        "how far above a notch's sampled null a neighbouring sample may lie before "
        "the null is reported as falling between samples"
    ),
}


def _three_db_levels_in(tree: ast.AST) -> list[int]:
    """The lines of ``tree`` that write 3 dB (``3`` or ``-3``, int or float) where
    a dB level is read: added to, subtracted from or compared with something
    that holds decibels; bound to, defaulted for or passed as a name that holds
    decibels; or given under such a key in a dict. A float ``3.0`` added or
    subtracted is read as a dB level whatever it is added to."""
    found: list[int] = []

    def named(name: str, value: ast.expr | None) -> None:
        if (
            value is not None
            and _says_db(name)
            and name not in _OTHER_THREE_DB
            and _gives_three(value)
        ):
            found.append(value.lineno)

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
            pair = (node.left, node.right)
            if any(
                _is_three(one) and (_mentions_db(other) or _is_float(one))
                for one, other in (pair, pair[::-1])
            ):
                found.append(node.lineno)
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(_is_three(side) for side in sides) and any(map(_mentions_db, sides)):
                found.append(node.lineno)
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name | ast.Attribute):
                    named(target.id if isinstance(target, ast.Name) else target.attr, node.value)
        elif isinstance(node, ast.arguments):
            positional = [*node.posonlyargs, *node.args]
            for arg, default in zip(
                positional[len(positional) - len(node.defaults) :], node.defaults, strict=True
            ):
                named(arg.arg, default)
            for arg, default in zip(node.kwonlyargs, node.kw_defaults, strict=True):
                named(arg.arg, default)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            named(node.arg, node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    named(key.value, value)
    return sorted(set(found))


def test_no_module_writes_its_own_three_db_level():
    found = [
        f"{path.relative_to(PACKAGE).as_posix()}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line in _three_db_levels_in(parsed(path))
    ]
    assert found == []


def test_the_three_db_rule_catches_what_it_names():
    tree = ast.parse(
        "def offset(peak_db):\n"
        "    return peak_db - 3\n"  # 2
        "def added(ref):\n"
        "    return ref['magnitude_db'] + -3.0\n"  # 4
        "def compared(gain_db):\n"
        "    return gain_db < -3\n"  # 6
        "LEVEL_DB = -3\n"  # 7
        "class Recipe:\n"
        "    level_db: float = Field(default=-3.0)\n"  # 9
        "async def read(signal, cutoff_db=-3.0, *, floor_db=3):\n"  # 10
        "    return crossing(signal, level_db=-3)\n"  # 11
        "SPEC = {'ref_db': 3.0}\n"  # 12
        "def floated(magnitude):\n"
        "    return magnitude - 3.0\n"  # 14
        "def unrelated(index, times):\n"
        "    return index - 3, times[3], 3 + index, {'count': 3}, level(db=None)\n"
    )
    assert _three_db_levels_in(tree) == [2, 4, 6, 7, 9, 10, 11, 12, 14]
