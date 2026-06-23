"""Tests for raw_parser operating-point trace classification."""

from __future__ import annotations

from typing import cast

import numpy as np
from spicelib import RawRead

from ltspice_mcp.lib.raw_parser import (
    extract_operating_point,
    nearest_index,
    trace_unit,
    whattype_unit,
)


class TestNearestIndex:
    """``nearest_index`` must handle a descending sweep axis (e.g. .dc Vg 1.8 0
    -0.01) — searchsorted alone lands every lookup at an endpoint and silently
    returns the wrong sample."""

    def test_ascending(self):
        ax = np.array([0.0, 1.0, 2.0, 3.0])
        assert nearest_index(ax, 2.0) == 2
        assert nearest_index(ax, 1.4) == 1
        assert nearest_index(ax, -5.0) == 0
        assert nearest_index(ax, 99.0) == 3

    def test_descending(self):
        ax = np.array([3.0, 2.0, 1.0, 0.0])
        assert nearest_index(ax, 2.0) == 1  # value 2.0 sits at index 1
        assert nearest_index(ax, 1.4) == 2  # nearest is 1.0 at index 2
        assert nearest_index(ax, 99.0) == 0  # largest value at index 0
        assert nearest_index(ax, -5.0) == 3  # smallest value at index 3


class _FakeRaw:
    """Minimal stub exposing the RawRead interface extract_operating_point uses.

    extract_operating_point only calls ``get_trace_names()`` and
    ``get_wave(trace, step=...)``, so this stub implements exactly those.
    """

    def __init__(self, waves: dict[str, float]):
        self._waves = waves

    def get_trace_names(self) -> list[str]:
        return list(self._waves)

    def get_wave(self, trace: str, step: int = 0) -> np.ndarray:
        del step
        return np.array([self._waves[trace]])


def test_operating_point_classifies_device_terminal_currents():
    """Device terminal currents must land in the currents dict.

    Covers the absence-class gap: existing .op fixtures only contain
    two-terminal element currents (e.g. I(R1), I(I1)), so multi-terminal
    device terminal currents like Ic/Ib/Ie(Q1) (BJT) never exercised the
    classifier. A bare startswith("I(") test silently dropped them.
    """
    raw = _FakeRaw(
        {
            "V(c)": 5.0,
            "V(b)": 0.7,
            "Ic(Q1)": 1e-3,
            "Ib(Q1)": 1e-5,
            "Ie(Q1)": 1.01e-3,
            "I(RC)": 1e-3,
        }
    )

    result = extract_operating_point(cast(RawRead, raw))

    assert set(result["voltages"]) == {"V(c)", "V(b)"}
    assert set(result["currents"]) == {"Ic(Q1)", "Ib(Q1)", "Ie(Q1)", "I(RC)"}
    assert result["voltages"]["V(c)"] == 5.0
    assert result["currents"]["Ic(Q1)"] == 1e-3


def test_operating_point_classifies_device_internals():
    """Device small-signal / model parameters (@dev[param]) get their own bucket.

    Covers the absence-class gap: every .op fixture was pure V()/I(), so the
    classifier never saw ngspice's device internals. A bare '@m1[gm]' fell
    through both buckets (dropped), and a v-wrapped 'v(@m1[vth])' was filed
    under node voltages (mislabeled). The '@' marker must win over the V(/I(
    wrapping.
    """
    raw = _FakeRaw(
        {
            "V(d)": 1.8,
            "I(Vd)": 5.9e-4,
            "@m1[gm]": 1.58e-3,
            "@m1[gds]": 3.2e-6,
            "v(@m1[vth])": 0.4,
            "i(@m1[id])": 5.9e-4,
        }
    )

    result = extract_operating_point(cast(RawRead, raw))

    assert set(result["voltages"]) == {"V(d)"}
    assert set(result["currents"]) == {"I(Vd)"}
    assert set(result["device_internals"]) == {
        "@m1[gm]",
        "@m1[gds]",
        "v(@m1[vth])",
        "i(@m1[id])",
    }
    # The mislabel regression: vth is a parameter, not a node voltage.
    assert "v(@m1[vth])" not in result["voltages"]
    assert result["device_internals"]["@m1[gm]"] == 1.58e-3


class TestTraceUnit:
    """Units come from the simulator's declared ``whattype`` (relayed, not
    invented); name-prefix is only a fallback, and a parameter name never gets a
    guessed unit."""

    def test_whattype_unit_known_and_unknown(self):
        assert whattype_unit("voltage") == "V"
        assert whattype_unit("device_current") == "A"
        assert whattype_unit("frequency") == "Hz"
        assert whattype_unit("admittance") == "S"
        assert whattype_unit("notype") is None
        assert whattype_unit(None) is None

    def test_trace_unit_falls_back_to_name_prefix(self):
        class _NoTraceRaw:
            def get_trace(self, name):  # simulator gives no type info
                raise KeyError(name)

        raw = cast(RawRead, _NoTraceRaw())
        assert trace_unit(raw, "V(out)") == "V"
        assert trace_unit(raw, "Id(M1)") == "A"
        assert trace_unit(raw, "I(R1)") == "A"
        # A device-internal parameter name is NEVER assigned a guessed unit.
        assert trace_unit(raw, "@m1[gm]") is None

    def test_trace_unit_prefers_declared_whattype(self):
        class _Trace:
            whattype = "admittance"

        class _TypedRaw:
            def get_trace(self, name):
                return _Trace()

        # The simulator typed @m1[gm] as an admittance -> relay S, don't fall
        # through to "no unit".
        assert trace_unit(cast(RawRead, _TypedRaw()), "@m1[gm]") == "S"
