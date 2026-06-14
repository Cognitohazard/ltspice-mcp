"""Tests for plot_waveform — interactive HTML charts opened on the desktop.

Pure-helper unit tests (downsample, HTML/XSS, client classification, opener
branch selection, union-x padding) plus handler integration through recorded
LTspice fixtures with ``open=False`` (or an injected opener) so no browser is
launched. The AC dual-panel, .step overlay, and noise cases are load-bearing.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import desktop
from ltspice_mcp.lib.plot_html import build_plot_html
from ltspice_mcp.lib.signal_analysis import downsample_minmax
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    PlotWaveformInput,
    _union_panel,
    handle_plot_waveform,
)
from tests.conftest import make_sim_job, stage_recorded_fixture


def _read(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _data_blob(html: str) -> dict:
    """Extract and parse the embedded plot-data JSON from a rendered page."""
    start = html.index('type="application/json">') + len('type="application/json">')
    end = html.index("</script>", start)
    return json.loads(html[start:end].replace("<\\/", "</"))


async def _plot(state: SessionState, **kwargs) -> dict:
    kwargs.setdefault("open", False)
    result = await handle_plot_waveform(PlotWaveformInput(**kwargs), state)
    assert result.structuredContent is not None
    return result.structuredContent


# --- pure helpers ----------------------------------------------------------


class TestDownsampleMinmax:
    def test_preserves_spike(self):
        x = np.arange(10_000, dtype=float)
        y = np.zeros(10_000)
        y[4321] = 999.0  # a one-sample spike
        _, ys = downsample_minmax(x, y, 200)
        assert len(ys) <= 220
        assert max(ys) == pytest.approx(999.0)  # spike amplitude survives

    def test_roughly_target_size(self):
        x = np.arange(100_000, dtype=float)
        y = np.sin(x / 100.0)
        _, ys = downsample_minmax(x, y, 1000)
        assert 900 <= len(ys) <= 1100


class TestUnionPanel:
    def test_shared_x_not_unioned(self):
        x = np.array([0.0, 1.0, 2.0])
        panel, unioned = _union_panel([(x, x * 2, "a"), (x, x * 3, "b")], "linear", "t", "v")
        assert unioned is False
        assert panel["data"][0] == [0.0, 1.0, 2.0]
        assert len(panel["series"]) == 2

    def test_differing_x_padded_with_nulls(self):
        panel, unioned = _union_panel(
            [
                (np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0]), "a"),
                (np.array([0.0, 2.0]), np.array([20.0, 22.0]), "b"),
            ],
            "linear",
            "t",
            "v",
        )
        assert unioned is True
        assert panel["data"][0] == [0.0, 1.0, 2.0]  # union
        # series b has no sample at x=1.0 -> null gap there
        assert panel["data"][2] == [20.0, None, 22.0]

    def test_refuses_oversized_union_before_padding(self, monkeypatch):
        # Distinct axes inflate the union; the cap must trip (stage 2) before the
        # padded arrays are materialized.
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_PLOT_MAX_CELLS", 10)
        s = [
            (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), "a"),
            (np.array([0.5, 1.5]), np.array([5.0, 6.0]), "b"),
        ]
        with pytest.raises(ResultError, match="cells"):
            _union_panel(s, "linear", "t", "v")

    def test_refuses_long_series_before_concat(self, monkeypatch):
        # Many long series must trip the cap (stage 1) before concatenating.
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_PLOT_MAX_CELLS", 10)
        big = np.arange(6.0)
        with pytest.raises(ResultError, match="cells"):
            _union_panel([(big, big, "a"), (big, big, "b")], "linear", "t", "v")


class TestBuildPlotHtml:
    def _spec(self, label="V(out)"):
        return {
            "analysis_type": "transient",
            "bode": False,
            "panels": [
                {
                    "x_scale": "linear",
                    "x_label": "Time (s)",
                    "y_label": label,
                    "series": [{"label": label}],
                    "data": [[0.0, 1.0], [0.1, 0.2]],
                }
            ],
        }

    def test_inlines_uplot_and_roundtrips_data(self):
        html = build_plot_html(self._spec(), title="t", summary="s")
        assert "uPlot" in html  # the library is inlined
        assert 'id="plot-data"' in html
        blob = _data_blob(html)
        assert blob["panels"][0]["data"] == [[0.0, 1.0], [0.1, 0.2]]

    def test_neutralizes_script_breakout(self):
        evil = "V(</script><img src=x onerror=alert(1)>)"
        html = build_plot_html(self._spec(label=evil), title="t")
        # the raw breakout sequence must not appear unescaped in the document
        assert "</script><img" not in html
        # but the label round-trips intact through the JSON blob
        assert _data_blob(html)["panels"][0]["series"][0]["label"] == evil

    def test_escapes_title_chrome(self):
        html = build_plot_html(self._spec(), title="<b>x</b>")
        assert "<b>x</b>" not in html
        assert "&lt;b&gt;x&lt;/b&gt;" in html

    def test_nan_in_data_raises_not_silent(self):
        spec = self._spec()
        spec["panels"][0]["data"] = [[0.0, 1.0], [0.1, float("nan")]]
        with pytest.raises(ValueError, match="JSON compliant"):
            build_plot_html(spec, title="t")


class TestOpenInDesktop:
    def test_wsl_uses_explorer_with_windows_path(self, monkeypatch):
        monkeypatch.setattr(desktop, "is_wsl", lambda: True)
        monkeypatch.setattr(desktop, "to_windows_path", lambda p: "C:\\plot.html")
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/x/plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "explorer.exe"
        assert calls == [["explorer.exe", "C:\\plot.html"]]

    def test_linux_uses_xdg_open(self, monkeypatch):
        monkeypatch.setattr(desktop, "is_wsl", lambda: False)
        monkeypatch.setattr(desktop.sys, "platform", "linux")
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/x/plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "xdg-open"
        assert calls == [["xdg-open", "/x/plot.html"]]

    def test_failure_degrades(self, monkeypatch):
        monkeypatch.setattr(desktop, "is_wsl", lambda: False)
        monkeypatch.setattr(desktop.sys, "platform", "linux")

        def boom(*a, **k):
            raise OSError("no opener")

        assert desktop.open_in_desktop(Path("/x/p.html"), spawn=boom) == (False, None)


# --- input model -----------------------------------------------------------


class TestInputModel:
    def test_defaults(self):
        m = PlotWaveformInput()
        assert m.signals == "all"
        assert m.open is True
        assert m.step is None
        assert m.max_points is None

    def test_strict_rejects_unknown(self):
        with pytest.raises(ValidationError):
            PlotWaveformInput(bogus=1)  # type: ignore[call-arg]


# --- handler error paths ---------------------------------------------------


@pytest.mark.asyncio
class TestErrors:
    async def test_requires_one_source(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_plot_waveform(PlotWaveformInput(), state_no_sim)

    async def test_empty_signals_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="at least one signal"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals=[]), state_no_sim
            )

    async def test_axis_as_signal_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="sweep axis"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals=["time"]), state_no_sim
            )

    async def test_op_raw_refused(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "op_extreme_node")
        with pytest.raises(ResultError, match="operating_point"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals="all"), state_no_sim
            )


# --- handler integration via recorded fixtures -----------------------------


@pytest.mark.asyncio
class TestRender:
    async def test_transient_single_panel(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["analysis_type"] == "transient"
        assert data["panels"] == 1
        assert data["opened"] is False  # open=False
        out = Path(data["path"])
        assert (work_dir / ".ltspice-mcp" / "plots") in out.parents
        html = _read(out)
        assert "uPlot" in html
        blob = _data_blob(html)
        assert blob["bode"] is False and len(blob["panels"]) == 1
        assert any(o["code"] == "open_skipped" for o in data["observations"])

    async def test_ac_bode_dual_panel(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["analysis_type"] == "ac"
        assert data["panels"] == 2  # stacked magnitude + phase
        blob = _data_blob(_read(Path(data["path"])))
        assert blob["bode"] is True
        assert blob["panels"][0]["y_label"] == "Magnitude (dB)"
        assert blob["panels"][1]["y_label"] == "Phase (deg)"
        assert blob["panels"][0]["x_scale"] == "log"
        assert any(o["code"] == "phase_unwrapped" for o in data["observations"])

    async def test_dc_sweep(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_dc_div")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["analysis_type"] == "dc"
        assert _data_blob(_read(Path(data["path"])))["panels"][0]["x_scale"] == "linear"

    async def test_noise_log_axis(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_noise_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals="all")
        assert data["analysis_type"] == "noise"
        assert _data_blob(_read(Path(data["path"])))["panels"][0]["x_scale"] == "log"

    async def test_step_overlay_unions_distinct_axes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["n_steps"] > 1
        assert data["steps_plotted"] == data["n_steps"]
        assert data["series_count"] == data["n_steps"]  # one trace per step
        # the step_tran fixture has distinct per-step time vectors -> union-x
        assert any(o["code"] == "step_axis_unioned" for o in data["observations"])

    async def test_oversized_stepped_plot_refused(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Many distinct-axis steps must trip the global cell cap before allocating
        # / padding the full panel (the union-padding blowup guard).
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_PLOT_MAX_CELLS", 100)
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        with pytest.raises(ResultError, match="cells"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False),
                state_no_sim,
            )

    async def test_single_step_selection(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], step=1)
        assert data["steps_plotted"] == 1
        assert data["series_count"] == 1

    async def test_max_points_downsamples_and_observes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], max_points=20)
        assert data["downsampled"] is True
        assert all(n <= 22 for n in data["points_per_series"])
        assert any(o["code"] == "downsampled" for o in data["observations"])

    async def test_json_format_passes_schema(self, state_no_sim: SessionState, work_dir: Path):
        # The autouse conformance hook validates structuredContent vs output_schema.
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], format="json")
        assert {"path", "analysis_type", "opened", "observations"} <= data.keys()


# --- delivery / opener / security ------------------------------------------


@pytest.mark.asyncio
class TestDeliveryAndSecurity:
    async def test_opener_invoked_when_open_true(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        seen = {}

        def fake_open(path):
            seen["path"] = path
            return True, "explorer.exe"

        monkeypatch.setattr(desktop, "open_in_desktop", fake_open)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], open=True)
        assert data["opened"] is True
        assert data["opener"] == "explorer.exe"
        assert seen["path"] == Path(data["path"])

    async def test_symlinked_sidecar_refused(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        outside = work_dir.parent / "plot_outside_target"
        outside.mkdir(exist_ok=True)
        (work_dir / ".ltspice-mcp").symlink_to(outside)
        with pytest.raises(ResultError, match="outside the circuit directory"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False),
                state_no_sim,
            )

    async def test_job_id_plots_next_to_circuit(self, state_no_sim: SessionState, work_dir: Path):
        raw_dir = work_dir / "elsewhere"
        raw_dir.mkdir()
        raw = stage_recorded_fixture(raw_dir, "ltspice_tran_rc")
        netlist = work_dir / "circuit.cir"
        job = make_sim_job(
            "jp", status="completed", netlist=netlist, raw_file=raw, simulator="ltspice"
        )
        state_no_sim.add_job(job)
        data = await _plot(state_no_sim, job_id="jp", run_index=0, signals=["V(out)"])
        out = Path(data["path"])
        assert (work_dir / ".ltspice-mcp" / "plots") in out.parents
        assert raw_dir not in out.parents
