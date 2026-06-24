"""Tests for plot_waveform — interactive HTML charts opened on the desktop.

Pure-helper unit tests (downsample, HTML/XSS, client classification, opener
branch selection, union-x padding) plus handler integration through recorded
LTspice fixtures with ``open=False`` (or an injected opener) so no browser is
launched. The AC dual-panel, .step overlay, and noise cases are load-bearing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import types
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
        with pytest.raises(ResultError, match="outside the destination directory"):
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
        assert "uPlot" in _read(Path(result.structuredContent["path"]))

        sc = result.structuredContent
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
        assert result.structuredContent["delivery"] == "terminal"  # fell back
        assert opens  # opened locally instead
        assert any(
            o["code"] == "widget_unavailable" for o in result.structuredContent["observations"]
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
        assert result.structuredContent["delivery"] == "terminal"


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
        assert entry.mimeType == "text/html;profile=mcp-app"
        assert "globalThis.ExtApps" in getattr(entry, "text", "")

    def test_widget_resource_is_listed(self):
        from ltspice_mcp.lib.plot_html import WIDGET_RESOURCE_URI
        from ltspice_mcp.resources import get_static_resources

        uris = {str(r.uri) for r in get_static_resources()}
        assert WIDGET_RESOURCE_URI in uris

    def test_tool_declares_ui_resource(self):
        from ltspice_mcp.lib.plot_html import WIDGET_RESOURCE_URI
        from ltspice_mcp.tools._base import registry

        defs, _ = registry.get_for_profile("full")
        plot = next(d for d in defs if d.name == "plot_waveform")
        assert plot.meta == {"ui": {"resourceUri": WIDGET_RESOURCE_URI}}
