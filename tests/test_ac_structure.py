"""Tests for the AC structural reader (lib + tool handler).

The LIB tests call ``analyze_ac_structure`` directly on synthesized complex
``H(jw)`` over a log-spaced sweep; the TOOL tests drive the real
``handle_ac_structure`` handler with an AC raw mock injected into the result
cache (reusing the mock helpers from ``test_analysis_tools``); the REJECTION
test confirms a transient raw is refused.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.ac_structure import analyze_ac_structure
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import schema_from_typeddict
from ltspice_mcp.tools.analysis import (
    AcStructureInput,
    AcStructureResponse,
    handle_ac_structure,
)
from tests.test_analysis_tools import _inject_raw_mock, _make_raw_mock

# Shared log-spaced sweep: 1 Hz .. 10 MHz, dense enough to read corners.
FREQS = np.logspace(0, 7, 351)


# ---- Analytic transfer functions ------------------------------------------


def one_pole(fp: float) -> np.ndarray:
    return 1.0 / (1.0 + 1j * FREQS / fp)


def rlc(f0: float, Q: float) -> np.ndarray:
    return 1.0 / (1.0 - (FREQS / f0) ** 2 + 1j * (FREQS / f0) / Q)


def rhp_zero(fz: float, fp: float) -> np.ndarray:
    return (1.0 - 1j * FREQS / fz) / (1.0 + 1j * FREQS / fp)


def type2(fz: float, fp: float, fc: float = 1.0) -> np.ndarray:
    # Integrator (pole at origin) + a zero + a higher pole.
    return (fc / (1j * FREQS)) * (1.0 + 1j * FREQS / fz) / (1.0 + 1j * FREQS / fp)


def notch(fz: float, fp: float, Q: float) -> np.ndarray:
    # Complex ZERO pair at fz (on the jw axis) over a complex POLE pair at fp.
    return (1.0 - (FREQS / fz) ** 2) / (1.0 - (FREQS / fp) ** 2 + 1j * (FREQS / fp) / Q)


def _has_review_obs(result) -> bool:
    return any(o["code"] == "review_against_plot" for o in result["observations"])


def _corner_near(result, f_hz: float, *, decades: float, kind: str | None = None):
    """First located corner whose [f_lo, f_hi] center sits within ``decades`` of
    ``f_hz`` (optionally filtered by kind), or None."""
    for c in result["corners"]:
        if kind is not None and c["kind"] != kind:
            continue
        center = float(np.sqrt(c["f_lo"] * c["f_hi"]))
        if abs(np.log10(center / f_hz)) <= decades:
            return c
    return None


# ---- A. LIB: analyze_ac_structure on synthesized signals ------------------


class TestAcStructureLib:
    def test_one_pole_order_and_corner(self):
        result = analyze_ac_structure(FREQS, one_pole(1e3))
        assert result["net_order"] == 1
        assert result["non_minimum_phase"] is False
        assert _corner_near(result, 1e3, decades=0.25) is not None
        assert _has_review_obs(result)

    def test_rlc_complex_pair_with_q(self):
        result = analyze_ac_structure(FREQS, rlc(1e4, 5.0))
        assert result["net_order"] == 2
        assert result["non_minimum_phase"] is False
        corner = _corner_near(result, 1e4, decades=0.25, kind="complex_pair")
        assert corner is not None
        assert corner["q"] is not None
        assert 2.0 <= corner["q"] <= 10.0
        assert _has_review_obs(result)

    def test_rhp_zero_is_non_minimum_phase(self):
        result = analyze_ac_structure(FREQS, rhp_zero(5e3, 500.0))
        assert result["non_minimum_phase"] is True
        assert result["phase_residual_deg"] is not None
        assert result["phase_residual_deg"] > 15.0
        assert _has_review_obs(result)

    def test_type2_integrator(self):
        result = analyze_ac_structure(FREQS, type2(1e3, 20e3))
        assert result["integrator"] is True
        assert result["net_order"] == 1
        assert result["non_minimum_phase"] is False
        assert _has_review_obs(result)

    def test_minimum_phase_short_sweep_not_flagged(self):
        # A plain single-pole RC lowpass is minimum-phase. With a short sweep
        # (3..6 points) the residual edge-trim floored to zero, leaving the
        # Bode-kernel convolution ring in the residual and misreading the most
        # basic minimum-phase response as non-minimum-phase (with a spurious
        # transport delay).
        for n in (3, 4, 5, 6, 7):
            f = np.logspace(0, n - 1, n)
            H = 1.0 / (1.0 + 1j * f / 100.0)
            result = analyze_ac_structure(f, H)
            assert result["non_minimum_phase"] is False, (
                f"n={n}: residual={result['phase_residual_deg']}"
            )
            assert result["transport_delay_s"] is None, f"n={n}"

    def test_complex_zero_pair_not_mislabeled_as_pole(self):
        # A notch has a complex ZERO pair (~5 kHz) over a complex POLE pair (~2 kHz).
        # The zero pair must be labeled complex_zero_pair, never a (pole) complex_pair.
        result = analyze_ac_structure(FREQS, notch(5e3, 2e3, 3.0))
        zero = _corner_near(result, 5e3, decades=0.3, kind="complex_zero_pair")
        assert zero is not None, [c["kind"] for c in result["corners"]]
        assert _corner_near(result, 5e3, decades=0.3, kind="complex_pair") is None


# ---- B. TOOL: real handler path with an AC raw mock injected --------------


def _inject_ac(state: SessionState, work_dir: Path, name: str, H: np.ndarray) -> str:
    """Inject an AC complex raw at ``work_dir/name`` and return the file name."""
    raw_file = work_dir / name
    raw = _make_raw_mock(
        plotname="AC Analysis",
        trace_names=["frequency", "V(out)"],
        waves={"frequency": FREQS, "V(out)": H},
        axis=FREQS,
    )
    _inject_raw_mock(state, raw_file, raw)
    return raw_file.name


@pytest.mark.asyncio
class TestAcStructureTool:
    async def test_handler_returns_structured_facts(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        name = _inject_ac(state_no_sim, work_dir, "ac.raw", rlc(1e4, 5.0))
        result = await handle_ac_structure(
            AcStructureInput(raw_file=name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        for key in ("net_order", "corners", "non_minimum_phase", "method", "observations"):
            assert key in sc
        assert sc["signal"] == "V(out)"
        assert "AC structure" in result.content[0].text

    async def test_handler_flags_non_minimum_phase(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        name = _inject_ac(state_no_sim, work_dir, "rhp.raw", rhp_zero(5e3, 500.0))
        result = await handle_ac_structure(
            AcStructureInput(raw_file=name, signal="V(out)"),
            state_no_sim,
        )
        assert result.structuredContent["non_minimum_phase"] is True
        assert "Out-of-phase zero / delay" in result.content[0].text


# ---- B2. OBSERVATION SHAPE: doctrine fields declared + relayed ------------


class TestAcStructureObservationShape:
    """The observations list mixes the reader's own facts (code/detail) with
    facts relayed from the simulator (code/kind/detail/severity/evidence). The
    declared schema must document that full doctrine shape."""

    def test_schema_declares_doctrine_observation_fields(self):
        schema = schema_from_typeddict(AcStructureResponse)
        item = schema["properties"]["observations"]["items"]
        assert {"code", "kind", "detail", "severity", "evidence"} <= set(item["properties"])
        # total=False Observation → no field required, so the reader's own
        # code/detail-only facts validate alongside the relayed ones.
        assert not item.get("required")

    @pytest.mark.asyncio
    async def test_relayed_solve_failure_has_full_shape(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        import ltspice_mcp.tools.analysis as analysis_mod

        async def _fake_solve_failures(raw_path):
            return ["singular matrix: node V(x) has no DC path"]

        monkeypatch.setattr(analysis_mod, "_solve_failures", _fake_solve_failures)
        name = _inject_ac(state_no_sim, work_dir, "relay.raw", one_pole(1e3))
        # format="json" routes through json_response, so the autouse conformance
        # hook validates this mixed-shape observations list against the schema.
        result = await handle_ac_structure(
            AcStructureInput(raw_file=name, signal="V(out)", format="json"),
            state_no_sim,
        )
        relayed = [o for o in result.structuredContent["observations"] if o.get("kind") == "relay"]
        assert relayed, "expected a relayed solve-failure observation"
        r = relayed[0]
        assert r["severity"] == "error"
        assert "evidence" in r
        assert set(r) <= {"code", "kind", "detail", "severity", "evidence"}


# ---- C. REJECTION: a transient raw must be refused ------------------------


@pytest.mark.asyncio
class TestAcStructureRejection:
    async def test_transient_raw_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "tran.raw"
        t = np.linspace(0, 1e-3, 200)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": np.sin(2 * np.pi * 1e3 * t)},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="AC analysis"):
            await handle_ac_structure(
                AcStructureInput(raw_file=raw_file.name, signal="V(out)"),
                state_no_sim,
            )


# ---- D. DEGENERATE: zero-magnitude inputs stay finite / JSON-valid ---------


class TestAcStructureDegenerate:
    """A valid AC sweep can contain exact zeros (an all-zero degenerate trace,
    or a notch sampled exactly at its null). ``log|H|`` must not drive the
    phase residual to NaN/inf — structuredContent has to stay JSON-valid."""

    @staticmethod
    def _assert_json_valid(result) -> None:
        # allow_nan=False raises on any NaN/inf that would corrupt the wire JSON.
        json.dumps(result, allow_nan=False)

    def test_all_zero_trace_is_finite(self):
        result = analyze_ac_structure(FREQS, np.zeros(FREQS.shape, dtype=complex))
        self._assert_json_valid(result)
        resid = result["phase_residual_deg"]
        assert resid is None or np.isfinite(resid)

    def test_notch_with_exactly_sampled_null_is_finite(self):
        # FREQS[200] == 1e4 exactly, so the on-axis zero makes |H| hit exactly 0.
        fz = float(FREQS[200])
        H = (1.0 - (FREQS / fz) ** 2) / (1.0 - (FREQS / 2e3) ** 2 + 1j * (FREQS / 2e3) / 3.0)
        assert np.any(np.abs(H) == 0.0)  # the null is exactly sampled
        result = analyze_ac_structure(FREQS, H)
        self._assert_json_valid(result)
