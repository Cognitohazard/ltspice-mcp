"""Tests for plot_waveform — interactive HTML charts opened on the desktop.

Pure-helper unit tests (downsample, HTML/XSS, client classification, opener
branch selection, union-x padding) plus handler integration through recorded
LTspice fixtures with ``open=False`` (or an injected opener) so no browser is
launched. The AC dual-panel, .step overlay, and noise cases are load-bearing.
"""

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import types
from pydantic import ValidationError

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import desktop, raster, services
from ltspice_mcp.lib.ac_analysis import prepare_ac_arrays
from ltspice_mcp.lib.metrics import guarded_axis
from ltspice_mcp.lib.plot_html import build_plot_html
from ltspice_mcp.lib.raster import raster_available
from ltspice_mcp.lib.raw_parser import safe_magnitude_db
from ltspice_mcp.lib.signal_analysis import compute_signal_stats, downsample_minmax
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    PlotWaveformInput,
    _union_panel,
    handle_plot_waveform,
)
from tests.conftest import make_experiment_job, stage_recorded_fixture, symlink_or_skip


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
    assert result.structured_content is not None
    return result.structured_content


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

    def test_descending_axis_not_collapsed(self):
        # A high->low sweep (a .dc 5 0 or descending .noise) used to read
        # x_end <= x_start and collapse to a single bucket — two points for the
        # whole curve. It must bucket like an ascending axis and stay in
        # descending (caller) order.
        x = np.linspace(5.0, 0.0, 10_000)  # descending
        y = np.sin(x) * x
        xs, ys = downsample_minmax(x, y, 1000)
        assert 900 <= len(ys) <= 1100  # not collapsed to 2
        assert xs[0] > xs[-1]  # preserved descending order
        # The spike-preservation property still holds (global extremes survive).
        assert max(ys) == pytest.approx(float(np.max(y)), rel=1e-6)
        assert min(ys) == pytest.approx(float(np.min(y)), rel=1e-6)


class TestUnionPanel:
    def test_shared_x_not_unioned(self):
        x = np.array([0.0, 1.0, 2.0])
        panel, unioned = _union_panel([(x, x * 2, "a"), (x, x * 3, "b")], "linear", "t", "v")
        assert unioned is False
        assert panel["data"][0] == [0.0, 1.0, 2.0]
        assert len(panel["series"]) == 2

    def test_shared_x_with_duplicate_timepoints_not_unioned(self):
        # Solver restarts emit duplicate x samples. np.unique would collapse
        # them, making each series look mismatched and wrongly flagging a
        # single-run multi-signal panel as step-axis-unioned. The shared axis
        # must be used verbatim, dups and all.
        x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        panel, unioned = _union_panel([(x, x * 2, "a"), (x, x * 3, "b")], "linear", "t", "v")
        assert unioned is False
        assert len(panel["data"][0]) == len(x)

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

    def test_padded_series_are_marked_and_keep_their_own_gaps(self):
        # A .step overlay whose steps have distinct time vectors interleaves
        # every series with the others' padding. The renderers must join a line
        # across padding (else each sample is isolated and nothing is drawn)
        # but still break it at the series' own non-finite samples, so the
        # spec says which nulls are which.
        panel, unioned = _union_panel(
            [
                (np.array([0.0, 1.0, 2.0]), np.array([10.0, np.nan, 12.0]), "a"),
                (np.array([0.0, 0.5, 2.0]), np.array([20.0, 21.0, 22.0]), "b"),
            ],
            "linear",
            "t",
            "v",
        )
        assert unioned is True
        assert panel["data"][0] == [0.0, 0.5, 1.0, 2.0]
        assert panel["data"][1] == [10.0, None, None, 12.0]
        assert panel["series"][0] == {"label": "a", "padded": True, "gaps": [2]}
        assert panel["series"][1] == {"label": "b", "padded": True, "gaps": []}

    def test_shared_x_series_carry_no_padding_marks(self):
        x = np.array([0.0, 1.0, 2.0])
        panel, _ = _union_panel([(x, np.array([1.0, np.nan, 3.0]), "a")], "linear", "t", "v")
        assert panel["series"] == [{"label": "a"}]
        assert panel["data"][1] == [1.0, None, 3.0]  # a real gap stays a gap

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

    def test_render_js_has_annotation_draw_hook(self):
        # The shared render core carries the canvas draw-hook that paints AC
        # corner markers + the out-of-phase-zero / delay tag.
        html = build_plot_html(self._spec(), title="t")
        assert "annotPlugin" in html
        assert "hooks: { draw:" in html
        assert "spec.annotations" in html
        assert "OUT-OF-PHASE ZERO / DELAY" in html

    def test_every_multi_panel_chart_shares_one_x_cursor(self):
        html = build_plot_html(
            {
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
                    for label in ("V(out) (V)", "I(C1) (A)", "V(sense) (V)")
                ],
            },
            title="t",
        )
        # The cursor sync used to be wired only for a two-panel Bode pair.
        assert "spec.bode && spec.panels.length === 2" not in html
        assert "spec.panels.length > 1" in html

    def test_render_core_joins_lines_across_union_padding(self):
        # uPlot breaks a line only at a strict null and joins across undefined,
        # so the render core must turn a padded series' padding into undefined.
        html = build_plot_html(
            {
                "analysis_type": "transient",
                "bode": False,
                "panels": [
                    {
                        "x_scale": "linear",
                        "x_label": "Time (s)",
                        "y_label": "V(out) (V)",
                        "series": [{"label": "V(out)", "padded": True, "gaps": []}],
                        "data": [[0.0, 1.0], [0.1, None]],
                    }
                ],
            },
            title="t",
        )
        assert ".padded" in html and "undefined" in html

    def test_annotations_roundtrip_into_blob(self):
        spec = self._spec()
        spec["bode"] = True
        spec["annotations"] = [{"x": 1234.0, "label": "pole ~1.2k", "kind": "real_pole"}]
        spec["nmp"] = True
        blob = _data_blob(build_plot_html(spec, title="t"))
        assert blob["annotations"][0]["label"] == "pole ~1.2k"
        assert blob["nmp"] is True


def _ui_caps() -> types.ClientCapabilities:
    """Client capabilities advertising MCP Apps (ui://) support per SEP-1865.

    ``extensions`` is an extra field (the model is ``extra="allow"``), so inject it
    via ``model_validate`` rather than a constructor kwarg.
    """
    return types.ClientCapabilities.model_validate(
        {
            "extensions": {
                "io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}
            }
        }
    )


class TestDeliveryChannel:
    def test_ui_when_extension_advertised(self):
        caps = _ui_caps()
        assert desktop.client_supports_ui(caps) is True
        assert desktop.resolve_delivery_channel(caps) == "ui"

    def test_terminal_when_no_extension(self):
        caps = types.ClientCapabilities()
        assert desktop.client_supports_ui(caps) is False
        assert desktop.resolve_delivery_channel(caps) == "terminal"

    def test_terminal_when_caps_none(self):
        assert desktop.client_supports_ui(None) is False
        assert desktop.resolve_delivery_channel(None) == "terminal"

    def test_unrelated_extension_does_not_count(self):
        caps = types.ClientCapabilities.model_validate({"extensions": {"some.other/ext": {}}})
        assert desktop.resolve_delivery_channel(caps) == "terminal"


class TestOpenInDesktop:
    def test_wsl_uses_explorer_with_windows_path(self, monkeypatch):
        # No Chromium available -> falls back to the OS default opener.
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: [])
        monkeypatch.setattr(desktop, "is_wsl", lambda: True)
        monkeypatch.setattr(desktop, "to_windows_path", lambda p: "C:\\plot.html")
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/x/plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "explorer.exe"
        assert calls == [["explorer.exe", "C:\\plot.html"]]

    @pytest.mark.skipif(os.name == "nt", reason="stands in for Linux with a POSIX path")
    def test_linux_uses_xdg_open(self, monkeypatch):
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: [])
        monkeypatch.setattr(desktop, "is_wsl", lambda: False)
        monkeypatch.setattr(sys, "platform", "linux")
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/x/plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "xdg-open"
        assert calls == [["xdg-open", "/x/plot.html"]]

    @pytest.mark.skipif(os.name == "nt", reason="stands in for Linux with a POSIX path")
    def test_failure_degrades(self, monkeypatch):
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: [])
        monkeypatch.setattr(desktop, "is_wsl", lambda: False)
        monkeypatch.setattr(sys, "platform", "linux")

        def boom(*a, **k):
            raise OSError("no opener")

        assert desktop.open_in_desktop(Path("/x/p.html"), spawn=boom) == (False, None)

    def test_app_window_preferred_on_wsl_with_unc_url(self, monkeypatch):
        # A chromeless Edge app window over the \\wsl.localhost UNC file URL is
        # tried before explorer.exe when Chromium is present.
        monkeypatch.setattr(desktop, "is_wsl", lambda: True)
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: ["/mnt/c/edge/msedge.exe"])
        monkeypatch.setattr(
            desktop, "to_windows_path", lambda p: "\\\\wsl.localhost\\Claude\\x\\plot.html"
        )
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/x/plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "msedge.exe"
        assert calls == [
            [
                "/mnt/c/edge/msedge.exe",
                "--app=file:////wsl.localhost/Claude/x/plot.html",
                "--window-size=1100,860",
            ]
        ]

    @pytest.mark.skipif(os.name == "nt", reason="stands in for Linux with a POSIX path")
    def test_app_window_falls_back_when_spawn_fails(self, monkeypatch):
        # Chromium present but its spawn raises -> fall through to xdg-open.
        monkeypatch.setattr(desktop, "is_wsl", lambda: False)
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: ["/usr/bin/chromium"])
        calls = []

        def spawn(argv, **kw):
            if "--app" in str(argv):
                raise OSError("app mode unavailable")
            calls.append(argv)

        opened, method = desktop.open_in_desktop(Path("/x/plot.html"), spawn=spawn)
        assert opened is True and method == "xdg-open"
        assert calls == [["xdg-open", "/x/plot.html"]]

    def test_app_window_drive_path_url_is_well_formed(self, monkeypatch):
        # A /mnt/c-backed workspace yields a Windows DRIVE path from wslpath -w,
        # which must become file:///C:/... (3 slashes), not file://C:/... — and
        # spaces must be percent-encoded so the browser isn't handed a bad URI.
        monkeypatch.setattr(desktop, "is_wsl", lambda: True)
        monkeypatch.setattr(desktop, "_chromium_exes", lambda: ["/mnt/c/edge/msedge.exe"])
        monkeypatch.setattr(desktop, "to_windows_path", lambda p: "C:\\Temp\\my plot.html")
        calls = []
        opened, method = desktop.open_in_desktop(
            Path("/mnt/c/Temp/my plot.html"), spawn=lambda argv, **kw: calls.append(argv)
        )
        assert opened is True and method == "msedge.exe"
        assert calls == [
            [
                "/mnt/c/edge/msedge.exe",
                "--app=file:///C:/Temp/my%20plot.html",
                "--window-size=1100,860",
            ]
        ]


# --- input model -----------------------------------------------------------


class TestInputModel:
    def test_defaults(self):
        m = PlotWaveformInput()
        assert m.signals == "all"
        # None defers to the [analysis] open_plot / attach_plot settings.
        assert m.open is None
        assert m.attach_plot is None
        assert m.panels is None
        assert m.step is None
        assert m.max_points is None

    def test_strict_rejects_unknown(self):
        with pytest.raises(ValidationError):
            PlotWaveformInput(bogus=1)  # type: ignore[call-arg]

    def test_empty_panel_is_refused(self):
        with pytest.raises(ValidationError):
            PlotWaveformInput(panels=[["V(out)"], []])


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

    async def test_ac_annotate_emits_corner_and_nmp(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Single-trace AC + annotate=True -> the spec carries corner markers near
        # the RC corner plus a non-minimum-phase flag.
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], annotate=True)
        blob = _data_blob(_read(Path(data["path"])))
        assert blob["bode"] is True
        anns = blob["annotations"]
        assert isinstance(anns, list) and len(anns) >= 1
        assert "nmp" in blob and isinstance(blob["nmp"], bool)
        # The RC fixture has a single real pole somewhere in the swept decade(s).
        xs = [a["x"] for a in anns]
        lo = blob["panels"][0]["data"][0][0]
        hi = blob["panels"][0]["data"][0][-1]
        assert any(lo <= x <= hi for x in xs)
        assert all(isinstance(a["label"], str) and a["label"] for a in anns)
        # Each marker is classified pole (drawn as a cross) or zero (a circle).
        assert all(a.get("marker") in ("pole", "zero") for a in anns)

    async def test_ac_annotate_off_omits_markers(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], annotate=False)
        blob = _data_blob(_read(Path(data["path"])))
        # No annotation keys when annotate is off (or an empty list at most).
        assert not blob.get("annotations")
        assert "nmp" not in blob

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
        blob = _data_blob(_read(Path(data["path"])))
        assert all(s.get("padded") for s in blob["panels"][0]["series"])

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
        symlink_or_skip(work_dir / ".ltspice-mcp", outside)
        with pytest.raises(ResultError, match="outside the destination directory"):
            await handle_plot_waveform(
                PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False),
                state_no_sim,
            )

    async def test_experiment_job_id_plots_a_case(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A run_experiments job_id plots like any other job: by run_index or case_id.

        Before this, the only job kind run_experiments produces was refused with
        an error naming an internal type, and the plot landed nowhere.
        """
        raw_dir = work_dir / "elsewhere"
        raw_dir.mkdir()
        raw = stage_recorded_fixture(raw_dir, "ltspice_tran_rc")
        make_experiment_job(state_no_sim, job_id="ex1", count=2, raw=raw)
        by_index = await _plot(state_no_sim, job_id="ex1", run_index=1, signals=["V(out)"])
        by_case = await _plot(state_no_sim, job_id="ex1", case_id="case-0001", signals=["V(out)"])
        for data in (by_index, by_case):
            out = Path(data["path"])
            assert out.is_file()  # noqa: ASYNC240
            # Next to the circuit (the experiment's source deck), not the raw.
            assert (work_dir / ".ltspice-mcp" / "plots") in out.parents
            assert raw_dir not in out.parents
        assert "run1" in Path(by_case["path"]).name

    async def test_case_id_needs_a_job_that_has_cases(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        # Never silently dropped beside a raw_file: a case id names a run of a
        # job, and a bare raw path is not one.
        with pytest.raises(ResultError, match="case_id"):
            await _plot(state_no_sim, raw_file=str(raw), case_id="case-0000", signals=["V(out)"])


def _widget_spec(result) -> dict | None:
    """Parse the widget chart spec from the result's _meta (the hidden channel).

    Also asserts the spec is NOT leaked into model-visible content (no content
    block parses to a chart spec)."""
    from ltspice_mcp.lib.plot_html import WIDGET_SPEC_META_KEY

    for c in result.content:
        text = getattr(c, "text", None)
        if not text:
            continue
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            continue
        assert not (isinstance(obj, dict) and isinstance(obj.get("panels"), list)), (
            "chart spec must not appear in model-visible content"
        )
    meta = result.meta
    if not meta or WIDGET_SPEC_META_KEY not in meta:
        return None
    return json.loads(meta[WIDGET_SPEC_META_KEY])


@pytest.mark.asyncio
class TestWidgetDelivery:
    """On an MCP Apps host the chart spec rides in the result _meta (hidden from the
    model) for the host to render via the predeclared ui:// renderer, and the local
    open is skipped; on a plain client there is no spec and the file is opened."""

    async def test_ui_host_pipes_spec_and_skips_open(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        opens: list = []
        monkeypatch.setattr(
            desktop, "open_in_desktop", lambda p: (opens.append(p), (True, "x"))[1]
        )
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        # open=True, but a UI host must NOT trigger the local opener.
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=True), state_no_sim
        )
        assert opens == []  # no local open on a UI host

        # The compact chart spec rides in _meta (NOT content, NOT inline HTML).
        spec = _widget_spec(result)
        assert spec is not None and spec["bode"] is False
        assert not any(isinstance(c, types.EmbeddedResource) for c in result.content)
        # The full-fidelity HTML file is still written.
        assert "uPlot" in _read(Path(result.structured_content["path"]))

        sc = result.structured_content
        assert sc["delivery"] == "ui"
        assert sc["opened"] is False
        assert any(o["code"] == "widget_delivered" for o in sc["observations"])

    async def test_ui_widget_spec_is_decimated_small(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The widget spec is capped to a small per-series budget so the _meta
        # payload stays small even when the file keeps full fidelity.
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        monkeypatch.setattr(desktop, "open_in_desktop", lambda p: (True, "x"))
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False), state_no_sim
        )
        spec = _widget_spec(result)
        assert spec is not None
        longest = max(len(p["data"][0]) for p in spec["panels"])
        assert longest <= 4_000

    async def test_ui_widget_respects_lower_max_points(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A caller-lowered max_points must bound the widget spec too (min of the
        # two), not be ignored in favor of the widget budget.
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        monkeypatch.setattr(desktop, "open_in_desktop", lambda p: (True, "x"))
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], max_points=500, open=False),
            state_no_sim,
        )
        spec = _widget_spec(result)
        assert spec is not None
        longest = max(len(p["data"][0]) for p in spec["panels"])
        assert longest <= 500

    async def test_ui_widget_oversize_falls_back_to_terminal(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # If the widget spec exceeds the byte budget, deliver the file locally
        # instead of shipping a huge _meta payload — surfaced as a fact.
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_WIDGET_MAX_BYTES", 100)
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        opens: list = []
        monkeypatch.setattr(
            desktop, "open_in_desktop", lambda p: (opens.append(p), (True, "x"))[1]
        )
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=True), state_no_sim
        )
        assert _widget_spec(result) is None  # no widget
        assert result.structured_content["delivery"] == "terminal"  # fell back
        assert opens  # opened locally instead
        assert any(
            o["code"] == "widget_unavailable" for o in result.structured_content["observations"]
        )

    async def test_terminal_host_no_widget(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", lambda: None)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False), state_no_sim
        )
        assert _widget_spec(result) is None
        assert result.meta is None
        assert not any(isinstance(c, types.EmbeddedResource) for c in result.content)
        assert result.structured_content["delivery"] == "terminal"


class TestWidgetTemplateAndResource:
    """The predeclared ui:// renderer: a static template served via resources/read,
    referenced by the tool declaration's _meta (canonical SEP-1865 wiring)."""

    def test_widget_template_inlines_runtimes(self):
        from ltspice_mcp.lib.plot_html import WIDGET_SPEC_META_KEY, build_widget_html

        html_doc = build_widget_html()
        assert "uPlot" in html_doc  # chart library inlined
        assert "globalThis.ExtApps" in html_doc  # ext-apps runtime globalized + inlined
        assert "ontoolresult" in html_doc  # receives the piped chart spec
        assert "renderSpec" in html_doc  # shared render core
        assert WIDGET_SPEC_META_KEY in html_doc  # reads the spec from result _meta
        # Self-contained for the iframe CSP: no external script/style/link tags
        # (URL strings inside the minified bundle, e.g. the SVG namespace, are fine).
        assert 'src="http' not in html_doc
        assert "<link" not in html_doc

    def test_globalize_rewrites_export(self):
        from ltspice_mcp.lib.plot_html import _globalize_ext_apps

        rewritten = _globalize_ext_apps("var a=1;export{a as App,b as Other};")
        assert rewritten == "var a=1;globalThis.ExtApps={App:a,Other:b};"

    def test_resource_read_serves_template(self, state_no_sim: SessionState):
        from ltspice_mcp.lib.plot_html import WIDGET_RESOURCE_URI
        from ltspice_mcp.resources import handle_read_resource

        result = handle_read_resource(WIDGET_RESOURCE_URI, state_no_sim)
        entry = result.contents[0]
        assert entry.mime_type == "text/html;profile=mcp-app"
        assert "globalThis.ExtApps" in getattr(entry, "text", "")

    def test_widget_resource_is_listed(self):
        from ltspice_mcp.lib.plot_html import WIDGET_RESOURCE_URI
        from ltspice_mcp.resources import get_static_resources

        uris = {str(r.uri) for r in get_static_resources()}
        assert WIDGET_RESOURCE_URI in uris

    def test_tool_declares_ui_resource(self):
        from ltspice_mcp.lib.plot_html import WIDGET_RESOURCE_URI
        from ltspice_mcp.tools._base import registry

        defs, _ = registry.get_tools()
        plot = next(d for d in defs if d.name == "plot_waveform")
        assert plot.meta == {"ui": {"resourceUri": WIDGET_RESOURCE_URI}}


# --- what the model reads: per-trace summary, panel layout, attached image ---


async def _raw_series(state: SessionState, raw_path: Path, name: str, step: int = 0):
    """One trace straight from the raw, independent of the plot path."""
    raw = await services.load_raw(raw_path, state)
    return guarded_axis(raw, step), np.asarray(raw.get_wave(name, step=step))


def _series_labels(blob: dict) -> list[list[str]]:
    return [[s["label"] for s in p["series"]] for p in blob["panels"]]


@pytest.mark.asyncio
class TestTraceSummary:
    """Every reply carries each plotted trace's extremes, ends and mean, so a
    model that cannot see the chart still learns what it shows."""

    async def test_transient_summary_is_computed_over_every_sample(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        # max_points=20 decimates the plotted series; the summary must still be
        # read from the full-resolution data, not the decimated copy.
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], max_points=20)
        assert data["downsampled"] is True
        t, y = await _raw_series(state_no_sim, raw, "V(out)")
        want = compute_signal_stats(t, y)
        (trace,) = data["traces"]
        assert trace["signal"] == "V(out)"
        assert trace["unit"] == "V"
        assert trace["panel"] == 0
        assert "step" not in trace
        assert trace["min"] == pytest.approx(want["min"])
        assert trace["max"] == pytest.approx(want["max"])
        assert trace["x_at_min"] == pytest.approx(want["t_at_min"])
        assert trace["x_at_max"] == pytest.approx(want["t_at_max"])
        assert trace["mean"] == pytest.approx(want["mean"])
        assert trace["initial"] == pytest.approx(float(y[0]))
        assert trace["final"] == pytest.approx(float(y[-1]))
        assert data["x_unit"] == "s"
        assert data["traces_total"] == 1

    async def test_summary_covers_only_the_window(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        t, y = await _raw_series(state_no_sim, raw, "V(out)")
        mid = len(t) // 2
        t_end = f"{float(t[mid]):.12g}"
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], t_end=t_end)
        (trace,) = data["traces"]
        hi = int(np.searchsorted(t, float(t_end), side="right"))
        want = compute_signal_stats(t[:hi], y[:hi])
        assert trace["final"] == pytest.approx(float(y[hi - 1]))
        assert trace["mean"] == pytest.approx(want["mean"])
        assert trace["max"] == pytest.approx(want["max"])

    async def test_text_channel_carries_the_summary(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False), state_no_sim
        )
        text = result.content[0].text
        final = result.structured_content["traces"][0]["final"]
        assert "V(out)" in text
        assert f"final {final:.4g}" in text

    async def test_each_step_is_its_own_trace(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert [t["step"] for t in data["traces"]] == list(range(data["n_steps"]))
        _, y1 = await _raw_series(state_no_sim, raw, "V(out)", step=1)
        assert data["traces"][1]["final"] == pytest.approx(float(y1[-1]))
        assert data["traces"][1]["max"] == pytest.approx(float(np.max(y1)))

    async def test_summary_is_capped_and_says_so(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_TRACE_SUMMARY_MAX", 2)
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["n_steps"] > 2
        assert len(data["traces"]) == 2
        assert data["traces_total"] == data["n_steps"]
        (obs,) = [o for o in data["observations"] if o["code"] == "trace_summary_truncated"]
        assert f"2 of {data['n_steps']}" in obs["detail"]

    async def test_ac_summary_reads_magnitude_in_db_and_phase_ends(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        axis, wave = await _raw_series(state_no_sim, raw, "V(out)")
        f, h = prepare_ac_arrays(axis, wave)
        mag = safe_magnitude_db(h)
        phase = np.degrees(np.unwrap(np.angle(h)))
        (trace,) = data["traces"]
        assert trace["unit"] == "dB"
        assert "mean" not in trace  # an average over a log sweep is not a figure of merit
        assert trace["max"] == pytest.approx(float(np.max(mag)))
        assert trace["x_at_max"] == pytest.approx(float(f[int(np.argmax(mag))]))
        assert trace["min"] == pytest.approx(float(np.min(mag)))
        assert trace["initial"] == pytest.approx(float(mag[0]))
        assert trace["final"] == pytest.approx(float(mag[-1]))
        assert trace["phase_initial_deg"] == pytest.approx(float(phase[0]), abs=1e-6)
        assert trace["phase_final_deg"] == pytest.approx(float(phase[-1]), abs=1e-6)
        assert data["x_unit"] == "Hz"

    async def test_sweeps_report_no_mean(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_dc_div")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        (trace,) = data["traces"]
        assert "mean" not in trace
        _, y = await _raw_series(state_no_sim, raw, "V(out)")
        assert trace["final"] == pytest.approx(float(y[-1]))
        assert data["x_unit"] == "V"  # the swept source's declared unit

    async def test_ui_delivery_carries_the_summary_too(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
        )
        assert result.structured_content["delivery"] == "ui"
        (trace,) = result.structured_content["traces"]
        assert trace["signal"] == "V(out)" and trace["final"] is not None


@pytest.mark.asyncio
class TestPanelLayout:
    """Traces of different units never share a y-axis; a caller can group
    same-unit traces of very different size apart by naming the panels."""

    async def test_volts_and_amps_get_separate_panels(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)", "I(C1)", "V(in)"])
        assert data["panels"] == 2
        blob = _data_blob(_read(Path(data["path"])))
        assert _series_labels(blob) == [["V(out)", "V(in)"], ["I(C1)"]]
        assert blob["panels"][0]["y_label"].endswith("(V)")
        assert blob["panels"][1]["y_label"].endswith("(A)")
        assert {t["signal"]: t["panel"] for t in data["traces"]} == {
            "V(out)": 0,
            "V(in)": 0,
            "I(C1)": 1,
        }
        assert {t["signal"]: t["unit"] for t in data["traces"]}["I(C1)"] == "A"

    async def test_one_unit_is_one_panel(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(in)", "V(out)"])
        assert data["panels"] == 1

    async def test_explicit_panels_are_drawn_as_named(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _plot(
            state_no_sim, raw_file=str(raw), panels=[["V(out)"], ["V(in)", "I(C1)"]]
        )
        assert data["signals"] == ["V(out)", "V(in)", "I(C1)"]
        assert data["panels"] == 2
        blob = _data_blob(_read(Path(data["path"])))
        assert _series_labels(blob) == [["V(out)"], ["V(in)", "I(C1)"]]
        # A panel the caller mixed keeps both units in its label.
        assert blob["panels"][1]["y_label"].endswith("(V, A)")

    async def test_panels_beside_a_signal_list_is_refused(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="panels"):
            await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)"], panels=[["V(in)"]])

    async def test_a_signal_in_two_panels_is_refused(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="more than one panel"):
            await _plot(state_no_sim, raw_file=str(raw), panels=[["V(out)"], ["v(out)", "V(in)"]])

    async def test_ac_gets_a_bode_pair_per_unit(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals=["V(out)", "I(C1)"])
        assert data["panels"] == 4
        blob = _data_blob(_read(Path(data["path"])))
        assert blob["bode"] is True
        assert _series_labels(blob) == [["V(out)"], ["V(out)"], ["I(C1)"], ["I(C1)"]]
        labels = [p["y_label"] for p in blob["panels"]]
        assert "Magnitude (dB)" in labels[0] and "Phase (deg)" in labels[1]
        assert "Magnitude (dB)" in labels[2] and "Phase (deg)" in labels[3]
        assert {t["signal"]: t["panel"] for t in data["traces"]} == {"V(out)": 0, "I(C1)": 2}

    async def test_noise_gain_is_kept_off_the_density_axis(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_noise_rc")
        data = await _plot(state_no_sim, raw_file=str(raw), signals="all")
        units = {t["signal"]: t["unit"] for t in data["traces"]}
        # The raw spells its noise traces in lower case.
        assert units["v(onoise)"] == "V/√Hz"
        assert units["gain"] is None
        panel_of = {t["signal"]: t["panel"] for t in data["traces"]}
        assert panel_of["gain"] != panel_of["v(onoise)"]
        # No deck beside this raw to read the .NOISE source from: the input-
        # referred unit is the simulator's declared one, and the reply says so.
        assert any(o["code"] == "noise_input_unit_unverified" for o in data["observations"])


needs_raster = pytest.mark.skipif(
    not raster_available(), reason="optional 'raster' extra (cairosvg) not installed"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _images(result) -> list[types.ImageContent]:
    return [c for c in result.content if isinstance(c, types.ImageContent)]


@pytest.mark.asyncio
class TestAttachedImage:
    """``attach_plot`` adds a static PNG a vision model can read, rendered from
    the same panels; without the raster extra it is skipped and reported."""

    @needs_raster
    async def test_attach_plot_returns_a_png(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(
                raw_file=str(raw), signals=["V(out)", "I(C1)"], open=False, attach_plot=True
            ),
            state_no_sim,
        )
        (image,) = _images(result)
        assert image.mime_type == "image/png"
        png = base64.b64decode(image.data)
        assert png.startswith(_PNG_MAGIC)
        sc = result.structured_content
        assert sc["image"]["image_format"] == "png"
        assert sc["image"]["width"] > 0 and sc["image"]["height"] > 0
        assert sc["image"]["estimated_tokens"] > 0
        on_disk = Path(sc["image_path"])
        assert on_disk.suffix == ".png"
        assert on_disk.parent == Path(sc["path"]).parent
        assert on_disk.read_bytes() == png  # noqa: ASYNC240

    async def test_no_image_unless_asked(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False), state_no_sim
        )
        assert _images(result) == []
        assert "image" not in result.structured_content
        assert "image_path" not in result.structured_content

    @needs_raster
    async def test_config_default_attaches_and_a_call_can_decline(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        state_no_sim.config.attach_plot = True
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        attached = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False), state_no_sim
        )
        assert len(_images(attached)) == 1
        declined = await handle_plot_waveform(
            PlotWaveformInput(
                raw_file=str(raw), signals=["V(out)"], open=False, attach_plot=False
            ),
            state_no_sim,
        )
        assert _images(declined) == []

    async def test_missing_extra_is_reported_not_raised(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(raster, "_load_cairosvg", lambda: None)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False, attach_plot=True),
            state_no_sim,
        )
        assert _images(result) == []
        # The SVG fallback is not passed on as text: path data is no use to a model.
        assert not any("<svg" in getattr(c, "text", "") for c in result.content)
        sc = result.structured_content
        assert "image" not in sc and "image_path" not in sc
        (obs,) = [o for o in sc["observations"] if o["code"] == "image_unavailable"]
        assert "raster" in obs["detail"]
        assert sc["traces"]  # the numbers still arrive

    @needs_raster
    async def test_ui_host_gets_the_widget_and_the_image(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ltspice_mcp.server.get_client_capabilities", _ui_caps)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], attach_plot=True),
            state_no_sim,
        )
        assert _widget_spec(result) is not None
        assert len(_images(result)) == 1


@pytest.mark.asyncio
class TestOpenDefault:
    """``[analysis] open_plot`` lets a terminal session stop the browser windows."""

    async def test_config_can_turn_the_local_open_off(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        opens: list = []
        monkeypatch.setattr(
            desktop, "open_in_desktop", lambda p: (opens.append(p), (True, "x"))[1]
        )
        state_no_sim.config.open_plot = False
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        result = await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
        )
        assert opens == []
        sc = result.structured_content
        assert sc["opened"] is False
        (obs,) = [o for o in sc["observations"] if o["code"] == "open_skipped"]
        assert "open_plot" in obs["detail"]

    async def test_a_call_overrides_the_config(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        opens: list = []
        monkeypatch.setattr(
            desktop, "open_in_desktop", lambda p: (opens.append(p), (True, "x"))[1]
        )
        state_no_sim.config.open_plot = False
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=True), state_no_sim
        )
        assert len(opens) == 1

    async def test_default_config_still_opens(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        opens: list = []
        monkeypatch.setattr(
            desktop, "open_in_desktop", lambda p: (opens.append(p), (True, "x"))[1]
        )
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        await handle_plot_waveform(
            PlotWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
        )
        assert len(opens) == 1
