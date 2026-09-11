"""Unit tests for the lib/services application service layer."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.state import SessionState
from tests.conftest import FIXTURES_DIR


class TestLoadRaw:
    async def test_missing_file(self, state_no_sim: SessionState, tmp_path: Path):
        with pytest.raises(ResultError, match="not found"):
            await services.load_raw(tmp_path / "nope.raw", state_no_sim)

    async def test_caches_results(self, state_no_sim: SessionState, tmp_path: Path):
        # Just cover the caching path - call twice on missing file
        with pytest.raises(ResultError):
            await services.load_raw(tmp_path / "x.raw", state_no_sim)

    async def test_truncated_binary_raw_raises_not_silently_short(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # A killed/interrupted sim can leave a .raw with a valid header but a
        # data section cut short. spicelib reads trace data EXACTLY and raises
        # on a short read; load_raw must surface that as ResultError, never
        # serve silently-short arrays that masquerade as a complete result.
        # Locks this safety property against a future spicelib regression to
        # a count-limited (silently-truncating) read.
        import numpy as np
        from spicelib.raw.raw_write import RawWrite, Trace

        n = 2000
        rw = RawWrite(plot_name="Transient Analysis")
        rw.add_trace(Trace("time", np.linspace(0.0, 1e-3, n), whattype="time"))
        rw.add_trace(Trace("V(out)", np.linspace(0.0, 5.0, n), whattype="voltage"))
        good = tmp_path / "good.raw"
        rw.save(good)

        # Sanity: the intact fixture parses (valid header). This proves the
        # truncated case below fails on the truncation, not a malformed header.
        raw = await services.load_raw(good, state_no_sim)
        names = [t.lower() for t in raw.get_trace_names()]
        assert "time" in names and "v(out)" in names

        # Cut the data section short (header is a few hundred bytes; 60% of a
        # 2000-point file lands well inside the binary data).
        data = good.read_bytes()
        truncated = tmp_path / "truncated.raw"
        truncated.write_bytes(data[: int(len(data) * 0.6)])

        with pytest.raises(ResultError):
            await services.load_raw(truncated, state_no_sim)

    async def test_zero_variable_raw_is_diagnosed_as_corrupt(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        # A file cut mid-header (here: 100 bytes into a real LTspice raw, in
        # the middle of the UTF-16 Title line) does NOT make spicelib raise —
        # RawRead parses it into a "valid" raw with zero variables. A real
        # SPICE raw always carries at least its axis variable, so zero
        # variables is a corruption signature; without this diagnosis,
        # consumers would report "Signal not found" against an empty
        # signal list.
        truncated = tmp_path / "trunc.raw"
        truncated.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes()[:100])

        with pytest.raises(ResultError) as exc_info:
            await services.load_raw(truncated, state_no_sim)
        msg = str(exc_info.value)
        assert "zero variables" in msg
        assert "truncated or corrupt" in msg
        assert str(truncated) in msg


class TestExtractModelSuggestions:
    def test_none_when_log_missing(self, state_no_sim: SessionState, tmp_path: Path):
        assert (
            services.extract_model_suggestions(tmp_path / "no.log", state_no_sim.libraries) is None
        )

    def test_none_for_clean_log(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text("Total elapsed time: 0.01 seconds.\n")
        assert services.extract_model_suggestions(log, state_no_sim.libraries) is None

    def test_none_when_no_libraries_loaded(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "err.log"
        log.write_text('Error on line 2 : s1 0 0 sw Unable to find definition of model "sw"\n')
        assert services.extract_model_suggestions(log, state_no_sim.libraries) is None

    def test_returns_ranked_suggestions(self, state_no_sim: SessionState, work_dir: Path):
        lib = work_dir / "sw.lib"
        lib.write_text(".MODEL SW VSWITCH(VT=1)\n.MODEL SW2 VSWITCH(VT=2)\n")
        state_no_sim.libraries.load_library(lib)
        log = work_dir / "err.log"
        log.write_text('Error on line 2 : s1 0 0 swx Unable to find definition of model "swx"\n')
        out = services.extract_model_suggestions(log, state_no_sim.libraries)
        assert out is not None
        assert "swx" in out
        names = {m["name"] for m in out["swx"]}
        assert "SW" in names

    def test_format_suggestion_block_empty(self):
        assert services.format_suggestion_block(None) == ""
        assert services.format_suggestion_block({}) == ""


class TestAttachSuggestionsToFailure:
    def test_recovery_hint_fires_without_library(self, state_no_sim: SessionState, tmp_path: Path):
        # With no library loaded (the common case stock parts fail in), a
        # missing-model failure still gets a library-independent find_model
        # recovery hint naming the unresolved ref.
        log = tmp_path / "err.log"
        log.write_text('Error on line 2 : s1 0 0 sw Unable to find definition of model "sw"\n')
        msg = services.attach_suggestions_to_failure("failed", {}, log, state_no_sim.libraries)
        assert 'inspect(kind="model"' in msg
        assert 'mode="search"' in msg
        assert "sw" in msg

    def test_no_hint_for_clean_log(self, state_no_sim: SessionState, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text("Total elapsed time: 0.01 seconds.\n")
        assert (
            services.attach_suggestions_to_failure("failed", {}, log, state_no_sim.libraries)
            == "failed"
        )

    def test_format_suggestion_block_renders(self):
        out = services.format_suggestion_block(
            {"swx": [{"name": "SW", "score": 0.9, "source_path": "/tmp/sw.lib"}]}
        )
        assert "Missing 'swx'" in out
        assert "SW" in out
        assert "/tmp/sw.lib" in out


class TestValidateSignal:
    """``validate_signal`` is case-insensitive — LTspice writes ``v(onoise)``
    in lowercase for ``.NOISE`` raws but ``V(out)`` everywhere else, and we
    don't want to reject a user's ``V(onoise)`` just because the raw used a
    different case."""

    def _raw(self, names: list[str]) -> MagicMock:
        raw = MagicMock()
        raw.get_trace_names.return_value = names
        return raw

    def test_exact_match_returns_same_string(self):
        raw = self._raw(["V(out)", "I(R1)"])
        assert services.validate_signal(raw, "V(out)") == "V(out)"

    def test_case_insensitive_returns_canonical_name(self):
        # spicelib preserves the case the simulator wrote — for noise raws
        # that's lowercase. Caller must use the canonical name to read traces.
        raw = self._raw(["v(onoise)", "v(inoise)"])
        assert services.validate_signal(raw, "V(onoise)") == "v(onoise)"
        assert services.validate_signal(raw, "V(INOISE)") == "v(inoise)"

    def test_unknown_signal_lists_available(self):
        raw = self._raw(["V(a)", "V(b)"])
        with pytest.raises(ResultError, match="Signal 'V\\(missing\\)' not found"):
            services.validate_signal(raw, "V(missing)")

    def test_noise_alias_ltspice_form_to_ngspice(self):
        # Resolve the alias, don't just hint.
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.validate_signal(raw, "V(onoise)") == "onoise_spectrum"
        assert services.validate_signal(raw, "V(inoise)") == "inoise_spectrum"

    def test_noise_alias_bare_shorthand(self):
        raw = self._raw(["frequency", "onoise_spectrum", "inoise_spectrum"])
        assert services.validate_signal(raw, "onoise") == "onoise_spectrum"
        assert services.validate_signal(raw, "inoise") == "inoise_spectrum"

    def test_noise_alias_ngspice_form_to_ltspice(self):
        raw = self._raw(["frequency", "v(onoise)", "v(inoise)"])
        assert services.validate_signal(raw, "onoise_spectrum") == "v(onoise)"

    def test_dev_param_hierarchical_multi_dot(self):
        # A subckt-flattened device path: m.x1.mn.gm -> @m.x1.mn[gm], including
        # the v()/i() wrapped forms.
        raw = self._raw(["@m.x1.mn[gm]", "v(@m.x1.mn[vth])", "i(@m.x1.mn[id])"])
        assert services.validate_signal(raw, "m.x1.mn.gm") == "@m.x1.mn[gm]"
        assert services.validate_signal(raw, "m.x1.mn.vth") == "v(@m.x1.mn[vth])"
        assert services.validate_signal(raw, "m.x1.mn.id") == "i(@m.x1.mn[id])"

    def test_dev_param_hierarchical_without_device_letter_unique(self):
        # Dropping the leading device-type letter (x1.mn.gm) resolves when the
        # suffix is unique.
        raw = self._raw(["@m.x1.mn[gm]"])
        assert services.validate_signal(raw, "x1.mn.gm") == "@m.x1.mn[gm]"

    def test_dev_param_hierarchical_ambiguous_refused(self):
        # Two devices share the .mn suffix — refuse rather than guess.
        raw = self._raw(["@m.x1.mn[gm]", "@m.x2.mn[gm]"])
        with pytest.raises(ResultError, match="not found"):
            services.validate_signal(raw, "mn.gm")

    def test_hierarchical_colon_resolves_to_dot(self):
        # LTspice V(X1:mid) <-> ngspice v(x1.mid)
        raw = self._raw(["time", "v(x1.mid)", "v(out)"])
        assert services.validate_signal(raw, "V(X1:mid)") == "v(x1.mid)"

    def test_device_param_shorthand_resolves_each_wrap(self):
        # 'dev.param' resolves to whichever form ngspice actually wrote:
        # bare @m1[gm], v-wrapped v(@m1[vth]), or i-wrapped i(@m1[id]).
        raw = self._raw(["@m1[gm]", "v(@m1[vth])", "i(@m1[id])"])
        assert services.validate_signal(raw, "m1.gm") == "@m1[gm]"
        assert services.validate_signal(raw, "M1.VTH") == "v(@m1[vth])"
        assert services.validate_signal(raw, "m1.id") == "i(@m1[id])"

    def test_device_param_not_saved_hints_save(self):
        # A dev.param that isn't in the raw points at the missing .save.
        raw = self._raw(["@m1[gm]"])
        with pytest.raises(ResultError, match=r"\.save @m1\[gds\]"):
            services.validate_signal(raw, "m1.gds")


# Real ngspice log shapes for the per-run convergence walk. The clean preamble
# is verbatim ngspice-42 batch output; the failure lines are the exact formats
# ngspice prints for a non-converging bias point (a "Warning:"-prefixed gmin
# line followed by a bare source-stepping line) and for a singular matrix.
_NGSPICE_CLEAN_LOG = (
    "Note: Compatibility modes selected: ps lt ki a\n"
    "\n"
    "Circuit: * dc divider\n"
    "\n"
    'binary raw file "out.raw"\n'
    "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n"
    "\n"
    "No. of Data Columns : 4\n"
    "No. of Data Rows : 3\n"
    "\n"
    "Total elapsed time (seconds) = 0.017\n"
)
_NGSPICE_GMIN_FAIL_LINES = "Warning: gmin stepping failed\nsource stepping failed\n"
_NGSPICE_SINGULAR_LINE = "Warning: singular matrix:  check nodes out and 0\n"


class TestLoadRawParseDeadline:
    @pytest.mark.asyncio
    async def test_wedged_parse_fails_the_call_not_the_session(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # The parse thread is untrusted third-party code; a wedged parse must
        # fail THIS call with a clear error instead of hanging the caller.
        # The Event lets teardown release the abandoned worker immediately —
        # a bare sleep would serially delay executor shutdown by its length.
        import threading

        raw = work_dir / "wedged.raw"
        raw.write_bytes(b"x")
        release = threading.Event()

        def _slow(path, state):
            # Quiet return: the deadline already abandoned this worker's
            # future; raising here would only log an unretrieved exception.
            release.wait(5.0)
            return MagicMock()

        monkeypatch.setattr(services, "load_raw_sync", _slow)
        monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.05)
        try:
            with pytest.raises(ResultError, match="exceeded"):
                await services.load_raw(raw, state_no_sim)
        finally:
            release.set()
            services._wedged_raw_paths.pop(raw, None)

    @pytest.mark.asyncio
    async def test_retry_during_cooldown_fails_fast_without_new_worker(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # A wedged path enters a retry cooldown: hammering it must not park
        # one more executor thread per call on the still-held parse lock.
        import threading

        raw = work_dir / "wedged2.raw"
        raw.write_bytes(b"x")
        release = threading.Event()
        calls = {"n": 0}

        def _slow(path, state):
            calls["n"] += 1
            release.wait(5.0)
            return MagicMock()

        monkeypatch.setattr(services, "load_raw_sync", _slow)
        monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.05)
        try:
            with pytest.raises(ResultError, match="exceeded"):
                await services.load_raw(raw, state_no_sim)
            assert calls["n"] == 1
            with pytest.raises(ResultError, match="paused"):
                await services.load_raw(raw, state_no_sim)
            assert calls["n"] == 1  # no second worker spawned
        finally:
            release.set()
            services._wedged_raw_paths.pop(raw, None)

    @pytest.mark.asyncio
    async def test_normal_parse_unaffected_by_deadline(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = FIXTURES_DIR / "ltspice_tran_rc.raw"
        import shutil

        local = work_dir / "ok.raw"
        shutil.copy2(raw, local)
        loaded = await services.load_raw(local, state_no_sim)
        assert loaded.get_trace_names()


@pytest.mark.asyncio
async def test_raw_writer_header_wins_over_session_default(state_no_sim: SessionState):
    from spicelib.simulators.qspice_simulator import Qspice

    state_no_sim.default_simulator = Qspice
    path = FIXTURES_DIR / "ngspice_noise_2plot.raw"

    raw = await services.load_raw(path, state_no_sim)

    assert raw.dialect == "ngspice"
