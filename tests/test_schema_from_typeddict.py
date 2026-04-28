"""Unit tests for the TypedDict → JSON Schema generator + drift checks.

The generator is small and covers only what this repo uses. We test each
type construct once plus two end-to-end drift cases that catch the
classic failure mode: "lib returns key X but schema doesn't mention it"
(or vice versa).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import numpy as np
import pytest

from ltspice_mcp.lib.ac_analysis import (
    FilterMetricsOutput,
    ResonancesOutput,
    RollOffOutput,
    StabilityMetricsOutput,
    compute_filter_metrics,
    compute_resonances,
    compute_roll_off,
    compute_stability_metrics,
)
from ltspice_mcp.lib.signal_analysis import (
    EdgeMetricsOutput,
    PeriodicMetricsOutput,
    PulseResponseOutput,
    TimingBetweenOutput,
    analyze_edge,
    analyze_periodic,
    analyze_pulse_response,
    analyze_timing_between,
    window_and_clean,
)
from ltspice_mcp.tools._base import schema_from_typeddict


class TestPrimitives:
    def test_str(self):
        class M(TypedDict):
            x: str

        s = schema_from_typeddict(M)
        assert s["properties"]["x"] == {"type": "string"}
        assert s["required"] == ["x"]

    def test_int_float_bool(self):
        class M(TypedDict):
            i: int
            f: float
            b: bool

        s = schema_from_typeddict(M)
        assert s["properties"]["i"] == {"type": "integer"}
        assert s["properties"]["f"] == {"type": "number"}
        assert s["properties"]["b"] == {"type": "boolean"}

    def test_any(self):
        class M(TypedDict):
            x: Any

        s = schema_from_typeddict(M)
        assert s["properties"]["x"] == {}


class TestLiteral:
    def test_string_enum(self):
        class M(TypedDict):
            kind: Literal["a", "b", "c"]

        s = schema_from_typeddict(M)
        assert s["properties"]["kind"] == {"enum": ["a", "b", "c"]}


class TestNullableUnion:
    def test_optional_primitive_drops_required(self):
        class M(TypedDict):
            x: float | None

        s = schema_from_typeddict(M)
        assert s["properties"]["x"] == {"type": ["number", "null"]}
        assert "required" not in s or "x" not in s["required"]

    def test_nonnullable_stays_required(self):
        class M(TypedDict):
            x: float
            y: float | None

        s = schema_from_typeddict(M)
        assert s["required"] == ["x"]


class TestContainers:
    def test_list_of_primitive(self):
        class M(TypedDict):
            items: list[str]

        s = schema_from_typeddict(M)
        assert s["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_dict_of_primitive(self):
        class M(TypedDict):
            mapping: dict[str, int]

        s = schema_from_typeddict(M)
        assert s["properties"]["mapping"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }


class _NestedInner(TypedDict):
    name: str
    value: int


class _NestedOuter(TypedDict):
    inner: _NestedInner
    many: list[_NestedInner]


class TestNested:
    def test_nested_typeddict(self):
        s = schema_from_typeddict(_NestedOuter)
        assert s["properties"]["inner"]["type"] == "object"
        assert s["properties"]["inner"]["properties"]["name"] == {"type": "string"}
        assert s["properties"]["many"]["items"]["properties"]["value"] == {"type": "integer"}


class _Opaque:
    pass


class _Unsupported(TypedDict):
    x: _Opaque


class TestRejectsUnsupported:
    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported type"):
            schema_from_typeddict(_Unsupported)

    def test_non_typeddict_rejected(self):
        with pytest.raises(TypeError, match="Expected TypedDict"):
            schema_from_typeddict(dict)


# ---------------------------------------------------------------------------
# Drift checks: generated schema keys == lib return keys
#
# These are the real payoff. If someone adds a key to the TypedDict but
# the lib function still returns the old shape (or vice versa), the test
# fails.
# ---------------------------------------------------------------------------


def _log_freqs(lo: int, hi: int, n: int) -> np.ndarray:
    return np.logspace(lo, hi, n)


def _lpf_1pole(freqs: np.ndarray, fc: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    return wc / (s + wc)


def _two_pole_loop(freqs: np.ndarray, A: float, p1: float, p2: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    return A / ((1 + s / (2 * np.pi * p1)) * (1 + s / (2 * np.pi * p2)))


class TestSchemaMatchesActualOutput:
    """Call the real lib functions and assert ``set(result.keys()) == set(schema keys)``.

    These are drift checks — they only verify key presence, not values — so
    sweep sizes are kept small. Physical realism isn't needed; the lib just
    has to execute cleanly.
    """

    _SMALL_SWEEP = 100

    def test_filter_metrics_keys_match(self):
        f = _log_freqs(0, 6, self._SMALL_SWEEP)
        H = _lpf_1pole(f, 1000.0)
        result = compute_filter_metrics(f, H)
        schema = schema_from_typeddict(FilterMetricsOutput)
        assert set(result.keys()) == set(schema["properties"].keys()), (
            "Drift: lib return and TypedDict fields disagree. "
            f"lib-only: {set(result.keys()) - set(schema['properties'].keys())} ; "
            f"schema-only: {set(schema['properties'].keys()) - set(result.keys())}"
        )

    def test_stability_metrics_keys_match(self):
        f = _log_freqs(0, 8, self._SMALL_SWEEP)
        H = _two_pole_loop(f, 1000.0, 1000.0, 100000.0)
        result = compute_stability_metrics(f, H)
        schema = schema_from_typeddict(StabilityMetricsOutput)
        assert set(result.keys()) == set(schema["properties"].keys()), (
            "Drift: lib return and TypedDict fields disagree. "
            f"lib-only: {set(result.keys()) - set(schema['properties'].keys())} ; "
            f"schema-only: {set(schema['properties'].keys()) - set(result.keys())}"
        )

    def test_roll_off_keys_match(self):
        f = _log_freqs(0, 8, self._SMALL_SWEEP)
        H = _lpf_1pole(f, 100.0)
        result = compute_roll_off(f, H, f_low=1e4, f_high=1e6)
        schema = schema_from_typeddict(RollOffOutput)
        assert set(result.keys()) == set(schema["properties"].keys())

    def test_resonances_keys_match(self):
        f = _log_freqs(0, 6, self._SMALL_SWEEP)
        H = _lpf_1pole(f, 1000.0)  # no peaks — tests the empty-peaks path
        result = compute_resonances(f, H)
        schema = schema_from_typeddict(ResonancesOutput)
        assert set(result.keys()) == set(schema["properties"].keys())

    def _transient_edge(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a simple rising-edge waveform for transient-lib drift checks."""
        t = np.linspace(0, 1e-6, 101)
        y = np.where(t < 0.5e-6, 0.0, 1.0)
        return window_and_clean(t, y, None, None)[:2]

    def test_edge_metrics_keys_match(self):
        t, y = self._transient_edge()
        result = analyze_edge(t, y)
        schema = schema_from_typeddict(EdgeMetricsOutput)
        assert set(result.keys()) == set(schema["properties"].keys())

    def test_pulse_response_keys_match(self):
        t, y = self._transient_edge()
        result = analyze_pulse_response(t, y)
        schema = schema_from_typeddict(PulseResponseOutput)
        assert set(result.keys()) == set(schema["properties"].keys())

    def test_timing_between_keys_match(self):
        t, ya = self._transient_edge()
        _, yb = self._transient_edge()
        result = analyze_timing_between(t, ya, yb)
        schema = schema_from_typeddict(TimingBetweenOutput)
        assert set(result.keys()) == set(schema["properties"].keys())

    def test_periodic_metrics_keys_match(self):
        # Need multiple periods; 200 points over 5 cycles is plenty.
        t = np.linspace(0, 5e-6, 201)
        y = np.where(np.sin(2 * np.pi * 1e6 * t) > 0, 1.0, 0.0)
        result = analyze_periodic(t, y)
        schema = schema_from_typeddict(PeriodicMetricsOutput)
        assert set(result.keys()) == set(schema["properties"].keys())
