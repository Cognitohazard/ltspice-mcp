"""Tests for raw_parser operating-point trace classification."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
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


def test_operating_point_classifies_device_op_points():
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
    assert set(result["device_op_points"]) == {
        "@m1[gm]",
        "@m1[gds]",
        "v(@m1[vth])",
        "i(@m1[id])",
    }
    # The mislabel regression: vth is a parameter, not a node voltage.
    assert "v(@m1[vth])" not in result["voltages"]
    assert result["device_op_points"]["@m1[gm]"] == 1.58e-3


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


class TestOffsetAwareRawRead:
    """LTspice stores a windowed .tran (``.tran 0 202u 196u``) with the time
    axis rebased to 0 and the true start in the header's ``Offset:`` field.
    The offset-aware reader adds it back so every consumer works in deck time."""

    @staticmethod
    def _write_ascii_raw(path, *, plotname: str, offset: float) -> None:
        path.write_text(
            "Title: * windowed\n"
            "Date: Thu Jul 10 12:00:00 2026\n"
            f"Plotname: {plotname}\n"
            "Flags: real\n"
            "No. Variables: 2\n"
            "No. Points: 3\n"
            f"Offset: {offset:.16e}\n"
            "Variables:\n"
            "\t0\ttime\ttime\n"
            "\t1\tV(out)\tvoltage\n"
            "Values:\n"
            "0\t0.0000000000000000e+00\n"
            "\t1.0\n"
            "1\t1.0000000000000000e-06\n"
            "\t2.0\n"
            "2\t2.0000000000000000e-06\n"
            "\t3.0\n"
        )

    def test_windowed_tran_axis_rebased_to_deck_time(self, tmp_path):
        from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead

        raw_file = tmp_path / "windowed.raw"
        self._write_ascii_raw(raw_file, plotname="Transient Analysis", offset=1.96e-4)
        raw = OffsetAwareRawRead(str(raw_file), traces_to_read="*", dialect="ltspice")
        axis = np.asarray(raw.get_axis())
        assert axis[0] == pytest.approx(1.96e-4)
        assert axis[-1] == pytest.approx(1.98e-4)
        # Trace data itself is untouched.
        assert list(np.asarray(raw.get_trace("V(out)").get_wave())) == [1.0, 2.0, 3.0]

    def test_zero_offset_axis_unchanged(self, tmp_path):
        from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead

        raw_file = tmp_path / "plain.raw"
        self._write_ascii_raw(raw_file, plotname="Transient Analysis", offset=0.0)
        raw = OffsetAwareRawRead(str(raw_file), traces_to_read="*", dialect="ltspice")
        axis = np.asarray(raw.get_axis())
        assert axis[0] == pytest.approx(0.0)
        assert axis[-1] == pytest.approx(2.0e-06)

    def test_non_transient_offset_ignored(self, tmp_path):
        from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead

        raw_file = tmp_path / "dc.raw"
        self._write_ascii_raw(raw_file, plotname="DC transfer characteristic", offset=1.96e-4)
        raw = OffsetAwareRawRead(str(raw_file), traces_to_read="*", dialect="ltspice")
        assert np.asarray(raw.get_axis())[0] == pytest.approx(0.0)


class TestMultiPlotNoiseRaw:
    """An ngspice ``.noise`` raw is two ASCII plots (spectral density, then
    integrated noise) with no blank line between them. Stock spicelib's
    trailing-empty-line skip infinite-loops on the second plot's header,
    which — parsed synchronously — hangs the whole server. The install-time
    guard must break that loop and still read both plots.
    """

    def test_guard_breaks_reread_of_next_plot_header(self):
        # Simulate the trailing-skip loop's pathological move directly: read a
        # non-empty line, seek back to it, read again. The guard must return a
        # one-shot empty read the second time so the loop can break.
        import io

        from ltspice_mcp.lib.raw_parser import _MultiPlotAsciiGuard

        buf = io.BytesIO(b"Title: plot 2\nDate: ...\n")
        g = _MultiPlotAsciiGuard(buf)
        cursor = g.tell()
        first = g.readline()
        assert first.strip()  # non-empty (the next plot's header)
        g.seek(cursor)  # trailing-skip loop rewinds onto it
        second = g.readline()
        assert second == b""  # guard breaks the loop instead of re-reading forever
        # After the one-shot break the cursor is left on the header for the
        # next plot's reader (position unchanged, header not consumed).
        assert g.tell() == cursor

    def test_two_plot_noise_raw_parses_without_hanging(self):
        # Integration: the real captured artifact that used to wedge the server.
        # Run the parse in a worker thread with a hard deadline so a regression
        # fails the test fast instead of hanging the whole suite.
        import threading

        from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
        from tests.conftest import FIXTURES_DIR

        fixture = FIXTURES_DIR / "ngspice_noise_2plot.raw"
        result: dict = {}

        def _parse() -> None:
            raw = OffsetAwareRawRead(str(fixture), dialect="ngspice")
            result["plots"] = len(raw.plots)
            result["traces"] = raw.get_trace_names()

        t = threading.Thread(target=_parse, daemon=True)
        t.start()
        t.join(timeout=20)
        assert not t.is_alive(), "parsing the two-plot noise raw hung (guard regressed)"
        assert result["plots"] == 2  # both plots preserved, not just the first
        assert "onoise_spectrum" in result["traces"]
