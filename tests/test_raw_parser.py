"""Tests for raw_parser: reading raws, classifying traces, and the run summary."""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from spicelib import RawRead

from ltspice_mcp.lib import log_parser, raw_parser, services
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    is_ac_analysis,
    is_dc_analysis,
    nearest_index,
    query_point_value,
    read_partial_raw_progress,
    sample_to_dict,
    trace_unit,
    whattype_unit,
)
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    make_raw_mock,
    ngspice_binary_raw,
    stage_recorded_fixture,
)
from tests.test_log_decode import fourier_text
from tests.test_summary_log_facts import captured_log_facts


def _recorded(name: str) -> RawRead:
    """One of the recorded LTspice fixtures, every trace read."""
    return RawRead(str(FIXTURES_DIR / f"{name}.raw"), traces_to_read="*", dialect="ltspice")


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

    ``get_trace`` answers an untyped trace on purpose: a real raw declares a
    type for every variable, and these cases are about what the *name* decides
    when no type is available. Recorded fixtures cover declared types.
    """

    def __init__(self, waves: dict[str, float]):
        self._waves = waves

    def get_trace_names(self) -> list[str]:
        return list(self._waves)

    def get_wave(self, trace: str, step: int = 0) -> np.ndarray:
        del step
        return np.array([self._waves[trace]])

    def get_trace(self, trace: str) -> SimpleNamespace:
        del trace
        return SimpleNamespace(whattype=None)


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

    def test_stall_backstop_aborts_a_nonprogressing_loop(self):
        # A file-like whose position never advances models any pathological
        # ASCII loop the specific break above doesn't recognize. The guard must
        # fail THIS parse after a bounded number of no-progress reads instead of
        # spinning forever — the categorical "bound untrusted work" backstop.
        from ltspice_mcp.lib.raw_parser import _MultiPlotAsciiGuard

        class _StuckFile:
            def tell(self) -> int:
                return 0  # never advances — a stalled reader

            def readline(self, *args: object) -> bytes:
                return b"data 1.0 2.0\n"  # always non-empty, no forward motion

        g = _MultiPlotAsciiGuard(_StuckFile())

        def _spin() -> None:
            for _ in range(_MultiPlotAsciiGuard._STALL_LIMIT + 5):
                g.readline()

        with pytest.raises(RuntimeError, match="no forward progress"):
            _spin()

    def test_stall_backstop_allows_a_large_forward_read(self):
        # The backstop counts NON-advancing reads, so a legitimately long raw
        # (every read advances) must never trip it, regardless of length.
        import io

        from ltspice_mcp.lib.raw_parser import _MultiPlotAsciiGuard

        big = b"".join(b"%d 1.0 2.0\n" % i for i in range(_MultiPlotAsciiGuard._STALL_LIMIT * 10))
        g = _MultiPlotAsciiGuard(io.BytesIO(big))
        reads = 0
        while g.readline():
            reads += 1
        assert reads == _MultiPlotAsciiGuard._STALL_LIMIT * 10  # no false abort

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
        t.join(timeout=LIVENESS_S)
        assert not t.is_alive(), "parsing the two-plot noise raw hung (guard regressed)"
        assert result["plots"] == 2  # both plots preserved, not just the first
        assert "onoise_spectrum" in result["traces"]


class TestSummarySurfacesParserFaults:
    """A parser that raises must leave a fact behind, not just a missing key.

    Every field below is optional on the wire, so an exception used to produce
    exactly the same summary as a run that legitimately had nothing to report:
    no measurements, no errors, no Fourier block, and no way for the consumer
    to tell the two apart.
    """

    @staticmethod
    def _tran_fixture() -> tuple[RawRead, Path]:
        return _recorded("ltspice_tran_rc"), FIXTURES_DIR / "ltspice_tran_rc.log"

    @staticmethod
    def _raiser(exc: Exception):
        return MagicMock(side_effect=exc)

    def test_the_fixture_reports_these_fields_when_nothing_raises(self, tmp_path):
        """Baseline: the fields the fault tests remove are really there."""
        raw, log = self._tran_fixture()
        summary = raw_parser.build_simulation_summary(raw, captured_log_facts(tmp_path, log))
        assert "vfinal" in {k.lower() for k in summary["measurements"]}
        assert summary.get("warnings") is None

    def test_measurement_parse_failure_is_named(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        raw, log = self._tran_fixture()
        fault = self._raiser(ValueError("bad measure block"))
        monkeypatch.setattr(log_parser, "parse_measurements", fault)
        summary = raw_parser.build_simulation_summary(raw, captured_log_facts(tmp_path, log))
        fault.assert_called_once()
        assert "measurements" not in summary
        joined = " ".join(summary["warnings"])
        assert "measurements" in joined
        assert "ValueError" in joined
        assert "bad measure block" in joined

    def test_log_diagnostics_failure_is_named(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """The diagnostics channel itself: no errors list must not be able to
        mean "the error scan crashed"."""
        raw, log = self._tran_fixture()
        fault = self._raiser(RuntimeError("walker died"))
        monkeypatch.setattr(log_parser, "extract_log_diagnostics", fault)
        summary = raw_parser.build_simulation_summary(raw, captured_log_facts(tmp_path, log))
        fault.assert_called_once()
        assert "errors" not in summary
        joined = " ".join(summary["warnings"])
        assert "log diagnostics" in joined
        assert "RuntimeError" in joined
        assert "walker died" in joined

    def test_fourier_parse_failure_is_named(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        raw, _ = self._tran_fixture()
        fault = self._raiser(KeyError("harmonics"))
        monkeypatch.setattr(log_parser, "parse_fourier_data", fault)
        summary = raw_parser.build_simulation_summary(
            raw, captured_log_facts(tmp_path, text=fourier_text())
        )
        fault.assert_called_once()
        assert "fourier" not in summary
        joined = " ".join(summary["warnings"])
        assert "fourier" in joined
        assert "KeyError" in joined

    def test_unreadable_log_names_both_fields_it_costs(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A log no reader can open costs measurements AND Fourier data."""
        from ltspice_mcp.errors import ResultError
        from ltspice_mcp.lib import log_parser

        raw, log = self._tran_fixture()
        fault = self._raiser(ResultError("Could not parse log file"))
        monkeypatch.setattr(log_parser, "make_log_reader", fault)
        summary = raw_parser.build_simulation_summary(raw, captured_log_facts(tmp_path, log))
        fault.assert_called_once()
        assert "measurements" not in summary
        assert "fourier" not in summary
        joined = " ".join(summary["warnings"])
        assert "measurements" in joined and "fourier" in joined
        assert "ResultError" in joined

    def test_an_axis_read_fault_is_named_but_an_axis_less_raw_is_not(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """The two reasons for a missing range are different facts.

        A stepped ``.op`` raw has no axis at all — spicelib says so with a
        RuntimeError, and that is the documented degenerate shape, not a
        problem. Any other read fault means the range is missing because the
        file could not be read, which the caller should be told."""
        from spicelib.raw.raw_classes import SpiceReadException

        raw, log = self._tran_fixture()
        monkeypatch.setattr(
            raw, "get_axis", self._raiser(RuntimeError("This RAW file does not have an axis."))
        )
        quiet = raw_parser.build_simulation_summary(raw, captured_log_facts(tmp_path, log))
        assert quiet["range"] == {}
        assert not [w for w in (quiet.get("warnings") or []) if "axis" in w]

        raw2, log2 = self._tran_fixture()
        monkeypatch.setattr(
            raw2, "get_axis", self._raiser(SpiceReadException("Not enough data in the binary"))
        )
        loud = raw_parser.build_simulation_summary(raw2, captured_log_facts(tmp_path, log2))
        assert loud["range"] == {}
        joined = " ".join(loud["warnings"])
        assert "axis" in joined
        assert "SpiceReadException" in joined

    def test_a_trace_the_value_scan_cannot_read_is_reported(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A narrowed scan must not look like a complete one."""
        raw, log = self._tran_fixture()
        real_get_wave = raw.get_wave

        def refuse_one(trace, step=0):
            if str(trace).lower() == "v(out)":
                raise IndexError(f'does not contain trace "{trace}"')
            return real_get_wave(trace, step)

        monkeypatch.setattr(raw, "get_wave", refuse_one)
        summary = raw_parser.build_simulation_summary(
            raw, captured_log_facts(tmp_path, log), value_scan=True
        )
        joined = " ".join(summary["warnings"])
        assert "value scan" in joined
        assert "V(out)" in joined


class TestAcBandwidthMetricsSurfaceFaults:
    """``bandwidth_3db``/``unity_gain_freq`` are None both when the response has
    no such crossing and when computing it raised. Only the second is a fact
    the caller can act on, so it gets said."""

    @staticmethod
    def _ac_raw() -> RawRead:
        return _recorded("ltspice_ac_rc")

    def test_unity_gain_failure_is_named(self, monkeypatch: pytest.MonkeyPatch):
        from ltspice_mcp.lib import ac_analysis

        def boom(*args, **kwargs):
            raise ZeroDivisionError("empty sweep")

        monkeypatch.setattr(ac_analysis, "compute_stability_metrics", boom)
        metrics = raw_parser.compute_ac_bandwidth_metrics(self._ac_raw(), "V(out)")
        assert metrics["unity_gain_freq"] is None
        joined = " ".join(metrics["warnings"])
        assert "unity_gain_freq" in joined
        assert "ZeroDivisionError" in joined

    def test_a_missing_trace_names_the_trace(self):
        metrics = raw_parser.compute_ac_bandwidth_metrics(self._ac_raw(), "V(nope)")
        assert metrics["bandwidth_3db"] is None
        assert metrics["unity_gain_freq"] is None
        assert "V(nope)" in " ".join(metrics["warnings"])

    def test_a_clean_run_carries_no_warnings_key(self):
        metrics = raw_parser.compute_ac_bandwidth_metrics(self._ac_raw(), "V(out)")
        assert "warnings" not in metrics
        assert metrics["bandwidth_3db"] is not None


def _headerless_ngspice_raw(directory: Path) -> Path:
    """An ngspice raw as versions before 44 wrote them: no ``Command:`` field.

    That is the whole defect in one file. spicelib names the writer from
    ``Command:`` and refuses the file outright without it, and a raw handed
    over as a bare path has no job to ask instead.
    """
    raw = directory / "headerless.raw"
    raw.write_text(
        "Title: * divider\n"
        "Date: Tue Sep  8 01:02:20 2026\n"
        "Plotname: Operating Point\n"
        "Flags: real\n"
        "No. Variables: 2\n"
        "No. Points: 1\n"
        "Variables:\n"
        "\t0\tv(in)\tvoltage\n"
        "\t1\tv(out)\tvoltage\n"
        "Values:\n"
        "0\t1.0000000000000000e+00\n"
        "\t5.0000000000000000e-01\n"
    )
    return raw


class TestSniffRawDialect:
    """Naming a raw's writer from its own bytes, when no job can say."""

    def test_an_ltspice_raw_is_named_from_its_utf16_header(self) -> None:
        assert raw_parser.sniff_raw_dialect(FIXTURES_DIR / "ltspice_tran_rc.raw") == "ltspice"

    @pytest.mark.parametrize("bom", [b"", b"\xff\xfe"])
    def test_recovery_and_sniffing_accept_the_same_utf16_headers(
        self, tmp_path: Path, bom: bytes
    ) -> None:
        path = tmp_path / "recorded.raw"
        path.write_bytes(bom + (FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes())

        assert raw_parser.has_valid_raw_header(path)
        assert raw_parser.sniff_raw_dialect(path) == "ltspice"

    def test_a_raw_that_names_its_own_writer_is_left_to_spicelib(self) -> None:
        """ngspice 44 and later, qspice and xyce all write ``Command:``.

        spicelib reads the field and names the writer exactly; a guess here
        could only be worse, so the sniff declines.
        """
        assert raw_parser.sniff_raw_dialect(FIXTURES_DIR / "ngspice_noise_2plot.raw") is None

    def test_an_ascii_raw_with_no_writer_field_is_ngspice(self, tmp_path: Path) -> None:
        assert raw_parser.sniff_raw_dialect(_headerless_ngspice_raw(tmp_path)) == "ngspice"

    def test_a_file_that_is_not_a_raw_is_not_guessed_at(self, tmp_path: Path) -> None:
        other = tmp_path / "notes.txt"
        other.write_text("Command: ngspice-46\nnot a raw at all\n")
        assert raw_parser.sniff_raw_dialect(other) is None
        assert raw_parser.sniff_raw_dialect(tmp_path / "absent.raw") is None


def _ltspice_transfer_function_raw(directory: Path) -> Path:
    """An LTspice ``.tf`` result, with the types LTspice actually declares.

    Transcribed from a live run: the trace names carry no ``V(``/``I(`` prefix
    and the types are LTspice's own words for what the numbers are. ngspice
    writes the same three quantities as ``v(...)``/``voltage``, so this file is
    the one that shows whether the declared type is being read at all.
    """
    raw = directory / "ltspice_tf.raw"
    raw.write_text(
        "Title: * divider\n"
        "Date: Tue Sep  8 01:19:35 2026\n"
        "Plotname: Transfer Function\n"
        "Flags: real\n"
        "No. Variables: 3\n"
        "No. Points: 1\n"
        "Offset: 0.0000000000000000e+00\n"
        "Command: Linear Technology Corporation LTspice\n"
        "Variables:\n"
        "\t0\ttransfer_function\ttransfer\n"
        "\t1\tV1#input_impedance\timpedance\n"
        "\t2\toutput_impedance_at_v(out)\timpedance\n"
        "Values:\n"
        "0\t5.0000000000000000e-01\n"
        "\t2.0000000000000000e+03\n"
        "\t5.0000000000000000e+02\n"
    )
    return raw


class TestBiasPointBucketing:
    """Traces are sorted by the type the simulator declared, then by name."""

    def test_a_trace_typed_neither_voltage_nor_current_is_kept(self, tmp_path: Path) -> None:
        """It used to be discarded, and the run read back as holding nothing.

        The name test had no else branch, so a trace matching neither prefix
        left no trace of itself — a caller could not tell an empty run from an
        unrecognised one.
        """
        raw = RawRead(str(_ltspice_transfer_function_raw(tmp_path)))

        op = extract_operating_point(raw)

        assert op["voltages"] == {}
        assert op["currents"] == {}
        assert op["other"]["transfer_function"] == pytest.approx(0.5)
        assert op["other"]["V1#input_impedance"] == pytest.approx(2000.0)

    def test_the_declared_type_decides_before_the_name(self, tmp_path: Path) -> None:
        """``V1#input_impedance`` starts with a V but is not a node voltage.

        It does not start with ``V(``, so the name test alone would drop it;
        what places it is the simulator having typed it ``impedance``.
        """
        raw = RawRead(str(_ltspice_transfer_function_raw(tmp_path)))

        assert raw_parser.declared_type(raw, "V1#input_impedance") == "impedance"
        assert raw_parser.trace_unit(raw, "V1#input_impedance") == "Ω"
        assert raw_parser.trace_unit(raw, "transfer_function") is None

    def test_recorded_voltage_and_device_current_types_keep_their_buckets(self) -> None:
        """The recorded LTspice fixture declares both voltage and device_current."""
        raw = RawRead(str(FIXTURES_DIR / "op_extreme_node.raw"))

        op = extract_operating_point(raw)

        assert op["voltages"]
        assert op["currents"]
        assert op["other"] == {}


# ---------------------------------------------------------------------------
# How far a stopped run got
# ---------------------------------------------------------------------------

_LTSPICE_DATA_MARK = "Binary:\n".encode("utf-16-le")


def _ltspice_layout(path: Path) -> tuple[bytes, int, int, np.ndarray]:
    """A recorded LTspice raw, where its data starts, its record size and axis.

    The record size is the data length over the point count of the finished
    file, and the axis is spicelib's own parse of it, so neither depends on
    the reader under test.
    """
    data = path.read_bytes()
    start = data.index(_LTSPICE_DATA_MARK) + len(_LTSPICE_DATA_MARK)
    raw = RawRead(str(path), verbose=False)
    axis = raw_parser.real_axis(np.asarray(raw.get_trace(0).data))
    if raw.get_trace(0).name == "time":
        axis = np.abs(axis)
    assert (len(data) - start) % len(axis) == 0
    return data, start, (len(data) - start) // len(axis), axis


class TestPartialRawProgress:
    """A stopped run's raw is read for how far it got, not for its values.

    spicelib cannot open one: ngspice leaves ``No. Points`` at 0 until a plot
    ends, which spicelib rejects, and LTspice leaves a count that lags the
    records behind it. Every expectation here comes from a finished file's
    own length and spicelib's parse of it, never from the reader's layout.
    """

    @pytest.mark.parametrize(
        "fixture",
        [
            "ltspice_tran_rc.raw",
            "ltspice_ac_rc.raw",
            "ltspice_step_tran.raw",
            "ltspice_dc_div.raw",
            "ltspice_noise_rc.raw",
        ],
    )
    def test_truncated_ltspice_raw_counts_complete_records(self, tmp_path: Path, fixture: str):
        data, start, record, axis = _ltspice_layout(FIXTURES_DIR / fixture)
        for points in (1, len(axis) // 2, len(axis) - 1):
            for torn in (0, record // 2):
                cut = tmp_path / f"{points}_{torn}.raw"
                cut.write_bytes(data[: start + points * record + torn])

                progress = read_partial_raw_progress(cut)

                assert progress is not None
                assert progress.header_complete
                assert progress.points == points
                assert progress.last_axis_value == pytest.approx(axis[points - 1])

    def test_a_declared_count_behind_the_records_does_not_cap_them(self, tmp_path: Path):
        """LTspice rewrites ``No. Points`` only now and then while it runs.

        Killed part way (observed on LTspice 26.1), the header said 1821697
        and the file held 1822135 complete records, every one past the count
        a valid, rising time. The count is a floor, not the answer.
        """
        data, _start, _record, axis = _ltspice_layout(FIXTURES_DIR / "ltspice_tran_rc.raw")
        final = "No. Points:          221".encode("utf-16-le")
        lagging = "No. Points:          100".encode("utf-16-le")
        assert data.count(final) == 1
        raw = tmp_path / "lagging.raw"
        raw.write_bytes(data.replace(final, lagging))

        progress = read_partial_raw_progress(raw)

        assert progress is not None
        assert progress.points == len(axis) == 221
        assert progress.last_axis_value == pytest.approx(axis[-1])

    def test_negative_stored_time_and_window_offset_read_in_deck_time(self, tmp_path: Path):
        data, start, record, axis = _ltspice_layout(FIXTURES_DIR / "ltspice_tran_rc.raw")
        points = 50
        stored = bytearray(data[: start + points * record])
        at = start + (points - 1) * record
        struct.pack_into("<d", stored, at, -axis[points - 1])
        zero = "Offset:    0.0000000000000000e+00".encode("utf-16-le")
        window = "Offset:    1.0000000000000000e-03".encode("utf-16-le")
        raw = tmp_path / "windowed.raw"
        raw.write_bytes(bytes(stored).replace(zero, window))

        progress = read_partial_raw_progress(raw)

        assert progress is not None
        assert progress.last_axis_value == pytest.approx(axis[points - 1] + 1e-3)

    def test_stepped_flag_is_reported(self, tmp_path: Path):
        progress = read_partial_raw_progress(FIXTURES_DIR / "ltspice_step_tran.raw")

        assert progress is not None
        assert progress.stepped

    def test_header_cut_short_reports_no_points(self, tmp_path: Path):
        data, start, _record, _axis = _ltspice_layout(FIXTURES_DIR / "ltspice_tran_rc.raw")
        raw = tmp_path / "header.raw"
        raw.write_bytes(data[: start - 40])

        progress = read_partial_raw_progress(raw)

        assert progress is not None
        assert not progress.header_complete
        assert progress.points == 0
        assert progress.last_axis_value is None
        assert progress.plot == "Transient Analysis"

    def test_ngspice_binary_with_an_unpatched_count(self, tmp_path: Path):
        rows = [(1e-9 * index, 0.5 * index, 2.0) for index in range(7)]
        raw = tmp_path / "ngspice.raw"
        raw.write_bytes(ngspice_binary_raw(rows, ["time", "v(in)", "v(out)"], tail=b"\x01" * 13))

        progress = read_partial_raw_progress(raw, "ngspice")

        assert progress is not None
        assert (progress.plot, progress.axis) == ("Transient Analysis", "time")
        assert progress.points == 7
        assert progress.last_axis_value == pytest.approx(6e-9)

    def test_finished_plot_is_stepped_over_to_the_one_in_progress(self, tmp_path: Path):
        """ngspice writes one plot per analysis: ``.op`` then ``.tran`` is two."""
        op = ngspice_binary_raw(
            [(1.0, 2.0)], ["v(in)", "v(out)"], plot="Operating Point", declared=1
        )
        tran = ngspice_binary_raw(
            [(0.0, 1.0, 2.0), (1e-6, 1.1, 2.1), (2e-6, 1.2, 2.2)], ["time", "v(in)", "v(out)"]
        )
        raw = tmp_path / "two_plots.raw"
        raw.write_bytes(op + tran)

        progress = read_partial_raw_progress(raw, "ngspice")

        assert progress is not None
        assert progress.plot == "Transient Analysis"
        assert progress.points == 3
        assert progress.last_axis_value == pytest.approx(2e-6)

    def test_ascii_values_stop_at_the_last_complete_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """What ngspice left when killed writing ``filetype=ascii``: a torn line."""
        header = (
            "Title: synthesized\nDate: x\nPlotname: Transient Analysis\nFlags: real\n"
            "No. Variables: 3\nNo. Points: 0       \nVariables:\n"
            "\t0\ttime\ttime\n\t1\tv(in)\tvoltage\n\t2\tv(out)\tvoltage\nValues:\n"
        )
        points = "".join(
            f"{index}\t\t{index * 1e-3:.15e}\n\t{index * 0.1:.15e}\n\t{index * 0.2:.15e}\n"
            for index in range(40)
        )
        raw = tmp_path / "ascii.raw"
        raw.write_text(header + points + "40\t\t4.0e-02\n\t4.00000", encoding="ascii")
        # A window smaller than one point, so finding it takes the growing read.
        monkeypatch.setattr(raw_parser, "_ASCII_TAIL_START", 16)
        monkeypatch.setattr(raw_parser, "_ASCII_TAIL_PER_VARIABLE", 0)

        progress = read_partial_raw_progress(raw, "ngspice")

        assert progress is not None
        assert progress.points == 40
        assert progress.last_axis_value == pytest.approx(39e-3)

    def test_ascii_complex_axis_reads_its_real_part(self, tmp_path: Path):
        header = (
            "Title: ac\nDate: x\nPlotname: AC Analysis\nFlags: complex\n"
            "No. Variables: 2\nNo. Points: 0       \nVariables:\n"
            "\t0\tfrequency\tfrequency\tgrid=3\n\t1\tv(out)\tvoltage\nValues:\n"
        )
        body = (
            "0\t\t1.000000000000000e+00,4.645981770173875e-310\n"
            "\t9.999605231408795e-01,-6.282937266758386e-03\n"
            "1\t\t1.584893192461113e+00,4.645981770173875e-310\n"
            "\t9.999008445312640e-01,-9.957190212550916e-03\n"
        )
        raw = tmp_path / "ac.raw"
        raw.write_text(header + body, encoding="ascii")

        progress = read_partial_raw_progress(raw, "ngspice")

        assert progress is not None
        assert progress.points == 2
        assert progress.last_axis_value == pytest.approx(1.584893192461113)

    def test_a_following_plot_out_of_reach_is_not_guessed_at(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Past the skip cap, the last plot's lines would be read with the first
        plot's variable count, so the count is reported as unknown instead."""
        monkeypatch.setattr(raw_parser, "_ASCII_SKIP_CAP", 64)

        progress = read_partial_raw_progress(FIXTURES_DIR / "ngspice_noise_2plot.raw", "ngspice")

        assert progress is not None
        assert progress.points is None
        assert progress.last_axis_value is None

    def test_recorded_two_plot_ascii_raw_reports_its_last_plot(self):
        progress = read_partial_raw_progress(FIXTURES_DIR / "ngspice_noise_2plot.raw", "ngspice")

        assert progress is not None
        assert progress.plot == "Integrated Noise"
        assert progress.axis is None
        assert progress.points == 1

    def test_a_file_that_is_not_a_raw_reads_as_none(self, tmp_path: Path):
        other = tmp_path / "other.raw"
        other.write_bytes(b"\x00\x01 not a raw")

        assert read_partial_raw_progress(other) is None
        with pytest.raises(FileNotFoundError):
            read_partial_raw_progress(tmp_path / "missing.raw")


# ---------------------------------------------------------------------------
# Raw metadata, point reads, AC bandwidth and the run summary
# ---------------------------------------------------------------------------


class TestDetectSimType:
    @pytest.mark.parametrize(
        ("fixture", "plotname"),
        [("ltspice_tran_rc", "Transient Analysis"), ("ltspice_ac_rc", "AC Analysis")],
    )
    def test_reads_the_plotname_of_a_recorded_raw(self, fixture: str, plotname: str):
        assert detect_sim_type(_recorded(fixture)) == plotname

    def test_fallback_on_error(self):
        # ValueError is what spicelib raises for a property the raw doesn't
        # carry. The fallback is for that shape, not for an arbitrary fault:
        # anything else propagates rather than being reported as "Unknown".
        raw = MagicMock()
        raw.get_raw_property.side_effect = ValueError("no property")
        assert detect_sim_type(raw) == "Unknown"


class TestIsAcAnalysis:
    def test_ac_variants(self):
        assert is_ac_analysis("AC Analysis") is True
        assert is_ac_analysis("ac analysis") is True

    def test_non_ac(self):
        assert is_ac_analysis("Transient Analysis") is False
        assert is_ac_analysis("DC sweep") is False


class TestIsDcAnalysis:
    def test_dc_analysis_variants(self):
        assert is_dc_analysis("DC transfer characteristic") is True
        assert is_dc_analysis("DC sweep") is True

    def test_dc_analysis_non_dc(self):
        assert is_dc_analysis("Transient Analysis") is False
        assert is_dc_analysis("AC Analysis") is False
        assert is_dc_analysis("Noise Spectral Density") is False

    def test_dc_analysis_word_boundary(self):
        # "dc" appearing only inside a word (substring present, no word
        # boundary) must not match — these discriminate \bDC\b from `"dc" in s`.
        assert is_dc_analysis("abcdc") is False
        assert is_dc_analysis("adc") is False


class TestGetStepCount:
    @pytest.mark.parametrize(
        ("fixture", "steps"), [("ltspice_tran_rc", 1), ("ltspice_step_tran", 3)]
    )
    def test_counts_the_steps_of_a_recorded_raw(self, fixture: str, steps: int):
        # ltspice_step_tran stepped r over 1, 22 and 680 (see its .log).
        assert get_step_count(_recorded(fixture)) == steps

    def test_error_returns_1(self):
        # A lookup miss inside the raw, not an arbitrary fault — see the note
        # on detect_sim_type's fallback above.
        raw = MagicMock()
        raw.get_steps.side_effect = IndexError("no steps")
        assert get_step_count(raw) == 1


class TestSteppedTransientAxes:
    """Per-step structure of a real stepped ``.tran`` raw."""

    def test_each_step_has_a_distinct_time_vector(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A stepped ``.tran`` run stores a different time vector per step.

        LTspice's adaptive timestep yields a different sample count for each
        ``.step`` value, so the steps cannot share one x-axis. This pins the
        contract behind writing stepped runs in a tidy/long layout (one row
        per step+sample) instead of a wide shared-x table. Recorded from a
        real stepped-damping RLC transient (underdamped -> overdamped).
        """
        raw_path = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        raw = services.load_raw_sync(
            services.source_for_raw_path(raw_path, state_no_sim), state_no_sim
        )

        n_steps = get_step_count(raw)
        assert n_steps > 1, "fixture must be a multi-step run"

        lengths = [len(np.asarray(raw.get_axis(step=s))) for s in range(n_steps)]
        assert len(set(lengths)) > 1, (
            f"per-step time vectors should differ in length; got {lengths}"
        )


class TestQueryPointValue:
    def test_exact_match(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0, 40.0])
        raw = make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 2.0)
        assert result["actual_x"] == pytest.approx(2.0)
        assert result["value"] == pytest.approx(30.0)
        assert result["trace"] == "V(out)"

    def test_nearest_neighbor(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0, 40.0])
        raw = make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 1.3)
        assert result["actual_x"] == pytest.approx(1.0)
        assert result["value"] == pytest.approx(20.0)

    def test_beyond_range_start(self):
        axis = np.array([1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0])
        raw = make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 0.0)
        assert result["actual_x"] == pytest.approx(1.0)

    def test_beyond_range_end(self):
        axis = np.array([1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0])
        raw = make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 100.0)
        assert result["actual_x"] == pytest.approx(3.0)

    def test_complex_returns_db_and_phase(self):
        axis = np.array([100.0, 1000.0, 10000.0])
        # Unity gain at all freqs, 0 phase
        wave = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        raw = make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 1000.0)
        assert "magnitude_db" in result
        assert result["magnitude_db"] == pytest.approx(0.0, abs=0.01)
        assert "phase_deg" in result
        assert "value" not in result  # complex path doesn't set "value"


class TestSampleToDict:
    def test_complex_sample_has_magnitude_linear(self):
        d = sample_to_dict(complex(0.0, 1.0))
        assert d["magnitude_linear"] == pytest.approx(1.0)
        assert d["magnitude_db"] == pytest.approx(0.0, abs=1e-9)
        assert d["phase_deg"] == pytest.approx(90.0)

    def test_real_sample_unchanged(self):
        d = sample_to_dict(3.5)
        assert d == {"value": 3.5}
        assert "magnitude_linear" not in d


class TestComputeAcBandwidthMetrics:
    def test_lowpass_bandwidth(self):
        """The -3 dB bandwidth of a 1-pole RC lowpass is its corner frequency."""
        freqs = np.logspace(0, 6, 1000)  # 1Hz to 1MHz
        fc = 1000  # 1kHz cutoff
        wave = 1 / (1 + 1j * freqs / fc)
        raw = make_raw_mock(["V(out)"], freqs, {"V(out)": wave})

        metrics = compute_ac_bandwidth_metrics(raw, "V(out)")
        # Referenced to the 1 Hz sample (a 1e-6 power ratio below DC); at 166
        # points/decade the log-linear crossing lands within ~3e-5 of fc.
        assert metrics["bandwidth_3db"] == pytest.approx(fc, rel=1e-4)

    def test_unity_gain_freq(self):
        """Lowpass with DC gain > 1 should have a unity gain frequency."""
        freqs = np.logspace(0, 8, 2000)
        fc = 1000
        gain = 100  # 40dB DC gain
        wave = gain / (1 + 1j * freqs / fc)
        raw = make_raw_mock(["V(out)"], freqs, {"V(out)": wave})

        metrics = compute_ac_bandwidth_metrics(raw, "V(out)")
        # |H(f)| = 1 at f = fc * sqrt(gain^2 - 1); the response is a straight
        # -20 dB/decade line there, so 250 points/decade interpolate it closely.
        assert metrics["unity_gain_freq"] == pytest.approx(fc * np.sqrt(gain**2 - 1), rel=1e-4)


class TestOpSteppingFailureRawGate:
    """An OP 'gmin/source stepping failed' error is a recoverable ladder rung.
    The log-only converged-check keys on LTspice's success wording, so an
    ngspice run that recovered via an unannounced fallback leaves a false hard
    error — gated on raw validity: finite node data demotes it to a warning, a
    rail-pinned/NaN raw keeps it an error, and always-terminal failures don't
    qualify at all."""

    def _summary(self, tmp_path: Path, node_wave: np.ndarray, phrase: str) -> dict:
        log = tmp_path / "op.log"
        # No recognized LTspice success line follows, so extract_log_diagnostics
        # classifies the phrase as an error before the raw gate runs.
        log.write_text(f"ngspice-42\n{phrase}\n")
        axis = np.array([0.0, 1e-3, 2e-3])
        raw = make_raw_mock(["time", "v(out)"], axis, {"time": axis, "v(out)": node_wave})
        return build_simulation_summary(raw, captured_log_facts(tmp_path, log))

    @staticmethod
    def _op_raw(trace: str) -> MagicMock:
        # A real .op raw has no axis — get_axis raises "does not have an axis".
        raw = make_raw_mock(
            [trace], np.array([0.0]), {trace: np.array([1.0])}, plotname="Operating Point"
        )
        raw.get_axis.side_effect = RuntimeError("This RAW file does not have an axis.")
        return raw

    def test_finite_data_demotes_to_warning(self, tmp_path: Path):
        s = self._summary(tmp_path, np.array([1.0, 1.01, 0.99]), "gmin stepping failed")
        assert "errors" not in s
        assert any("gmin stepping failed" in w for w in s.get("warnings", []))

    def test_railed_data_keeps_error(self, tmp_path: Path):
        s = self._summary(tmp_path, np.array([1e30, 1e30, 1e30]), "source stepping failed")
        assert any("source stepping failed" in e for e in s.get("errors", []))
        assert not any("source stepping failed" in w for w in s.get("warnings", []))

    def test_iteration_limit_never_demoted(self, tmp_path: Path):
        # Always-terminal — not a stepping-failure candidate even with clean data.
        s = self._summary(tmp_path, np.array([1.0, 1.0, 1.0]), "iteration limit reached")
        assert any("iteration limit" in e for e in s.get("errors", []))

    def test_stepped_op_keeps_error_for_later_step(self, tmp_path: Path):
        # A stepped .op solves the bias point per step but LTspice writes only
        # step 0 to the .raw. Step 0's finite data can't clear a stepping failure
        # that belongs to a later step the raw never carries — keep it an error.
        log = tmp_path / "op.log"
        log.write_text(".step v1=1\n.step v1=2\nngspice-42\ngmin stepping failed\n")
        s = build_simulation_summary(self._op_raw("v(out)"), captured_log_facts(tmp_path, log))
        assert any("gmin stepping failed" in e for e in s.get("errors", []))
        assert any("Stepped .op detected" in w for w in s.get("warnings", []))

    def test_stepped_op_later_step_fails_without_step_markers(self, tmp_path: Path):
        # LTspice stepped .op emits no ".step name=value" markers — the only
        # signal is the per-step "Direct Newton iteration" line. Step 0 succeeds
        # (one line) and step 1 fails (a second attempt + gmin failure): two
        # solve blocks, so step 0's finite raw can't vouch for step 1's failure.
        log = tmp_path / "op.log"
        log.write_text(
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Direct Newton iteration failed to find operating point.\n"
            "gmin stepping failed\n"
        )
        s = build_simulation_summary(self._op_raw("v(out)"), captured_log_facts(tmp_path, log))
        assert any("gmin stepping failed" in e for e in s.get("errors", []))

    def test_current_only_raw_keeps_error(self, tmp_path: Path):
        # A finite branch current can't vouch for a solved bias point: a failing
        # .op may still write i(V1) while the node voltage sits at NaN/rail. Only
        # a finite node VOLTAGE demotes; a current-only raw keeps the error.
        log = tmp_path / "op.log"
        log.write_text("ngspice-42\ngmin stepping failed\n")
        s = build_simulation_summary(self._op_raw("i(v1)"), captured_log_facts(tmp_path, log))
        assert any("gmin stepping failed" in e for e in s.get("errors", []))

    def test_single_step_op_still_demotes(self, tmp_path: Path):
        # An unstepped .op (one bias point) with finite data and no success line
        # still demotes — the guard must not over-suppress the single-block case.
        log = tmp_path / "op.log"
        log.write_text("ngspice-42\ngmin stepping failed\n")
        s = build_simulation_summary(self._op_raw("v(out)"), captured_log_facts(tmp_path, log))
        assert "errors" not in s
        assert any("gmin stepping failed" in w for w in s.get("warnings", []))


class TestBuildSimulationSummary:
    def test_transient_summary(self):
        axis = np.linspace(0, 0.01, 1000)  # 10ms transient
        wave = np.sin(2 * np.pi * 1000 * axis)
        raw = make_raw_mock(
            ["time", "V(out)", "I(R1)"],
            axis,
            {"V(out)": wave, "I(R1)": wave * 0.001, "time": axis},
            plotname="Transient Analysis",
        )

        summary = build_simulation_summary(raw, logs=None)

        assert summary["sim_type"] == "Transient Analysis"
        assert summary["point_count"] == 1000
        assert summary["step_count"] == 1
        assert summary["signals"] == ["time", "V(out)", "I(R1)"]
        assert summary["range"] == {"time_start": 0.0, "time_end": pytest.approx(0.01)}
        # No log provided — no measurements, warnings, or fourier
        assert "measurements" not in summary
        assert "warnings" not in summary
        assert "fourier" not in summary

    def test_ac_summary(self):
        freqs = np.logspace(0, 6, 500)
        fc = 1000
        wave = 1 / (1 + 1j * freqs / fc)
        raw = make_raw_mock(
            ["frequency", "V(out)"],
            freqs,
            {"V(out)": wave, "frequency": freqs},
            plotname="AC Analysis",
        )

        summary = build_simulation_summary(raw, logs=None)

        assert summary["sim_type"] == "AC Analysis"
        assert summary["point_count"] == 500
        assert summary["range"] == {"freq_start": 1.0, "freq_end": pytest.approx(1e6)}

    def test_dc_sweep_summary(self):
        sweep = np.linspace(0, 5, 100)
        wave = sweep * 2  # linear gain
        raw = make_raw_mock(
            ["V(in)", "V(out)"],
            sweep,
            {"V(in)": sweep, "V(out)": wave},
            plotname="DC sweep",
        )

        summary = build_simulation_summary(raw, logs=None)

        assert summary["sim_type"] == "DC sweep"
        assert summary["range"] == {"sweep_start": 0.0, "sweep_end": 5.0}

    def test_summary_with_duration(self):
        axis = np.linspace(0, 0.001, 100)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
        )

        summary = build_simulation_summary(raw, logs=None, duration=1.23)

        assert summary["duration"] == pytest.approx(1.23)

    def test_summary_with_log_measurements(self, work_dir: Path):
        """Summary includes .MEAS results when the log file has measurements."""
        axis = np.linspace(0, 0.01, 100)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.sin(2 * np.pi * 100 * axis), "time": axis},
        )
        log = work_dir / "sim.log"
        log.write_text(
            "Circuit: test.cir\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "vpk: MAX(V(out) )=0.9997 FROM 0 TO 0.01\n"
            "Total elapsed time: 0.5 seconds.\n"
        )

        summary = build_simulation_summary(raw, captured_log_facts(work_dir, log))

        assert list(summary["measurements"]) == ["vpk"]
        entry = summary["measurements"]["vpk"]
        assert entry["values"][0] == pytest.approx(0.9997)
        assert entry.get("range_to") == pytest.approx(0.01)

    def test_summary_with_log_but_no_measurements(self, work_dir: Path):
        """A log that parses but holds no .MEAS lines adds no measurements key."""
        axis = np.linspace(0, 0.01, 100)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.sin(2 * np.pi * 100 * axis), "time": axis},
        )
        log = work_dir / "sim.log"
        log.write_text(
            "Circuit: test.cir\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "Total elapsed time: 0.5 seconds.\n"
        )

        summary = build_simulation_summary(raw, captured_log_facts(work_dir, log))

        assert "measurements" not in summary
        assert "warnings" not in summary

    def test_summary_with_log_warnings(self, work_dir: Path):
        """Summary includes warnings from log file."""
        axis = np.linspace(0, 0.01, 100)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
        )

        log = work_dir / "warn.log"
        log.write_text(
            "Circuit: test.cir\n"
            "Warning: node N001 is floating\n"
            "Warning: less than 2 connections to node VCC\n"
            "Total elapsed time: 0.1 seconds.\n"
        )

        summary = build_simulation_summary(raw, captured_log_facts(work_dir, log))

        assert len(summary["warnings"]) == 2
        assert any("N001" in w for w in summary["warnings"])
        assert any("VCC" in w for w in summary["warnings"])

    def test_summary_multi_step(self):
        axis = np.linspace(0, 0.01, 100)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
            steps=[0, 1, 2],
        )

        summary = build_simulation_summary(raw, logs=None)

        assert summary["step_count"] == 3

    def test_all_values_are_python_types(self):
        """Ensure no numpy scalars leak into the summary."""
        axis = np.linspace(0, 0.005, 50)
        raw = make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.sin(axis * 1000), "time": axis},
        )

        summary = build_simulation_summary(raw, logs=None)

        # Check numeric values are Python types
        assert type(summary["point_count"]) is int
        assert type(summary["step_count"]) is int
        assert type(summary["range"]["time_start"]) is float
        assert type(summary["range"]["time_end"]) is float
