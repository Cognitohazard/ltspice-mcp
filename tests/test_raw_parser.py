"""Tests for raw_parser operating-point trace classification."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from spicelib import RawRead

from ltspice_mcp.lib import raw_parser
from ltspice_mcp.lib.raw_parser import (
    extract_operating_point,
    nearest_index,
    trace_unit,
    whattype_unit,
)
from tests.conftest import FIXTURES_DIR


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
        t.join(timeout=20)
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
        from tests.conftest import FIXTURES_DIR

        raw = RawRead(
            str(FIXTURES_DIR / "ltspice_tran_rc.raw"), traces_to_read="*", dialect="ltspice"
        )
        return raw, FIXTURES_DIR / "ltspice_tran_rc.log"

    @staticmethod
    def _raiser(exc: Exception):
        def boom(*args, **kwargs):
            raise exc

        return boom

    def test_the_fixture_reports_these_fields_when_nothing_raises(self):
        """Baseline: the fields the fault tests remove are really there."""
        raw, log = self._tran_fixture()
        summary = raw_parser.build_simulation_summary(raw, log)
        assert "vfinal" in {k.lower() for k in summary["measurements"]}
        assert summary.get("warnings") is None

    def test_measurement_parse_failure_is_named(self, monkeypatch: pytest.MonkeyPatch):
        raw, log = self._tran_fixture()
        monkeypatch.setattr(
            raw_parser, "parse_measurements", self._raiser(ValueError("bad measure block"))
        )
        summary = raw_parser.build_simulation_summary(raw, log)
        assert "measurements" not in summary
        joined = " ".join(summary["warnings"])
        assert "measurements" in joined
        assert "ValueError" in joined
        assert "bad measure block" in joined

    def test_log_diagnostics_failure_is_named(self, monkeypatch: pytest.MonkeyPatch):
        """The diagnostics channel itself: no errors list must not be able to
        mean "the error scan crashed"."""
        raw, log = self._tran_fixture()
        monkeypatch.setattr(
            raw_parser, "extract_log_diagnostics", self._raiser(RuntimeError("walker died"))
        )
        summary = raw_parser.build_simulation_summary(raw, log)
        assert "errors" not in summary
        joined = " ".join(summary["warnings"])
        assert "log diagnostics" in joined
        assert "RuntimeError" in joined
        assert "walker died" in joined

    def test_fourier_parse_failure_is_named(self, monkeypatch: pytest.MonkeyPatch):
        raw, log = self._tran_fixture()
        monkeypatch.setattr(raw_parser, "parse_fourier_data", self._raiser(KeyError("harmonics")))
        summary = raw_parser.build_simulation_summary(raw, log)
        assert "fourier" not in summary
        joined = " ".join(summary["warnings"])
        assert "fourier" in joined
        assert "KeyError" in joined

    def test_unreadable_log_names_both_fields_it_costs(self, monkeypatch: pytest.MonkeyPatch):
        """A log no reader can open costs measurements AND Fourier data."""
        from ltspice_mcp.errors import ResultError
        from ltspice_mcp.lib import log_parser

        raw, log = self._tran_fixture()
        monkeypatch.setattr(
            log_parser, "make_log_reader", self._raiser(ResultError("Could not parse log file"))
        )
        summary = raw_parser.build_simulation_summary(raw, log)
        assert "measurements" not in summary
        assert "fourier" not in summary
        joined = " ".join(summary["warnings"])
        assert "measurements and fourier" in joined
        assert "ResultError" in joined

    def test_an_axis_read_fault_is_named_but_an_axis_less_raw_is_not(
        self, monkeypatch: pytest.MonkeyPatch
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
        quiet = raw_parser.build_simulation_summary(raw, log)
        assert quiet["range"] == {}
        assert not [w for w in (quiet.get("warnings") or []) if "axis" in w]

        raw2, log2 = self._tran_fixture()
        monkeypatch.setattr(
            raw2, "get_axis", self._raiser(SpiceReadException("Not enough data in the binary"))
        )
        loud = raw_parser.build_simulation_summary(raw2, log2)
        assert loud["range"] == {}
        joined = " ".join(loud["warnings"])
        assert "axis" in joined
        assert "SpiceReadException" in joined

    def test_a_trace_the_value_scan_cannot_read_is_reported(self, monkeypatch: pytest.MonkeyPatch):
        """A narrowed scan must not look like a complete one."""
        raw, log = self._tran_fixture()
        real_get_wave = raw.get_wave

        def refuse_one(trace, step=0):
            if str(trace).lower() == "v(out)":
                raise IndexError(f'does not contain trace "{trace}"')
            return real_get_wave(trace, step)

        monkeypatch.setattr(raw, "get_wave", refuse_one)
        summary = raw_parser.build_simulation_summary(raw, log, value_scan=True)
        joined = " ".join(summary["warnings"])
        assert "value scan" in joined
        assert "V(out)" in joined


class TestAcBandwidthMetricsSurfaceFaults:
    """``bandwidth_3db``/``unity_gain_freq`` are None both when the response has
    no such crossing and when computing it raised. Only the second is a fact
    the caller can act on, so it gets said."""

    @staticmethod
    def _ac_raw() -> RawRead:
        from tests.conftest import FIXTURES_DIR

        return RawRead(
            str(FIXTURES_DIR / "ltspice_ac_rc.raw"), traces_to_read="*", dialect="ltspice"
        )

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
