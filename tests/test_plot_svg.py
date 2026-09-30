"""Tests for the static waveform renderer behind plot_waveform's attached image.

The renderer takes the same plot spec the interactive chart does (plain arrays
plus labels) and returns SVG markup; ``lib/raster.py`` turns that into the PNG a
vision model reads. These tests parse the markup rather than grepping it, so a
label that broke out of its element, or a coordinate that came out as ``nan``,
fails as a malformed document or a wrong count.
"""

import re
from xml.etree import ElementTree as ET

import pytest

from ltspice_mcp.lib.plot_svg import render_plot_svg, tick_labels

_NS = {"svg": "http://www.w3.org/2000/svg"}


def _panel(data, labels, *, x_scale="linear", x_label="Time (s)", y_label="V(out) (V)"):
    return {
        "x_scale": x_scale,
        "x_label": x_label,
        "y_label": y_label,
        "series": [{"label": label} for label in labels],
        "data": data,
    }


def _spec(*panels, **extra):
    return {"analysis_type": "transient", "bode": False, "panels": list(panels), **extra}


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _texts(root: ET.Element) -> list[str]:
    return ["".join(t.itertext()) for t in root.iter("{http://www.w3.org/2000/svg}text")]


def _series_paths(root: ET.Element) -> list[ET.Element]:
    return [p for p in root.iter("{http://www.w3.org/2000/svg}path") if p.get("class") == "trace"]


class TestRenderPlotSvg:
    def test_one_plot_area_per_panel(self):
        spec = _spec(
            _panel([[0.0, 1.0, 2.0], [0.0, 1.0, 0.5]], ["V(out)"]),
            _panel([[0.0, 1.0, 2.0], [1e-3, 2e-3, 0.0]], ["I(C1)"], y_label="I(C1) (A)"),
        )
        root = _parse(render_plot_svg(spec, title="rc — transient"))
        panels = [
            g for g in root.iter("{http://www.w3.org/2000/svg}g") if g.get("class") == "panel"
        ]
        assert len(panels) == 2
        assert len(_series_paths(root)) == 2
        texts = _texts(root)
        assert "rc — transient" in texts
        assert "V(out) (V)" in texts and "I(C1) (A)" in texts

    def test_markup_in_labels_stays_text(self):
        evil = 'V(</text><script>alert(1)</script>)&"'
        root = _parse(render_plot_svg(_spec(_panel([[0.0, 1.0], [0.0, 1.0]], [evil])), title=evil))
        assert not list(root.iter("{http://www.w3.org/2000/svg}script"))
        assert evil in _texts(root)

    def test_null_samples_break_the_line(self):
        spec = _spec(_panel([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, None, 3.0, 4.0]], ["V(out)"]))
        (path,) = _series_paths(_parse(render_plot_svg(spec, title="t")))
        assert path.get("d", "").count("M") == 2

    def test_all_null_series_is_skipped(self):
        spec = _spec(_panel([[0.0, 1.0], [None, None], [1.0, 2.0]], ["gap", "V(out)"]))
        root = _parse(render_plot_svg(spec, title="t"))
        assert len(_series_paths(root)) == 1

    def test_constant_series_gets_a_finite_range(self):
        svg = render_plot_svg(
            _spec(_panel([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]], ["V(out)"])), title="t"
        )
        assert "nan" not in svg.lower() and "inf" not in svg.lower()
        (path,) = _series_paths(_parse(svg))
        assert path.get("d")

    def test_log_axis_is_labelled_at_decades(self):
        freqs = [10.0 * 10 ** (i / 10) for i in range(41)]  # 10 Hz .. 100 kHz
        spec = _spec(
            _panel([freqs, [0.0] * 41], ["V(out)"], x_scale="log", x_label="Frequency (Hz)")
        )
        texts = _texts(_parse(render_plot_svg(spec, title="t")))
        for decade in ("10", "100", "1k", "10k", "100k"):
            assert decade in texts

    def test_linear_ticks_use_engineering_prefixes(self):
        spec = _spec(_panel([[0.0, 1e-3, 2e-3], [0.0, 2.5e-6, 5e-6]], ["I(C1)"]))
        texts = _texts(_parse(render_plot_svg(spec, title="t")))
        assert "1.0m" in texts and "1.5m" in texts
        assert any(re.fullmatch(r"-?\d+(\.\d+)?µ", t) for t in texts)

    def test_tick_labels_on_one_axis_are_distinct(self):
        # Rounding every tick to a fixed precision printed 0.5 ms and 1 ms both
        # as "0.001"; one prefix per axis with the step's decimals keeps them apart.
        assert tick_labels([0.0, 5e-4, 1e-3, 1.5e-3, 2e-3]) == [
            "0",
            "0.5m",
            "1.0m",
            "1.5m",
            "2.0m",
        ]
        assert tick_labels([-0.02, 0.0, 0.02]) == ["-20m", "0", "20m"]
        assert tick_labels([-40.0, -20.0, 0.0]) == ["-40", "-20", "0"]

    def test_padding_is_joined_across_and_real_gaps_break_the_line(self):
        panel = _panel([[0.0, 0.5, 1.0, 1.5, 2.0], [1.0, None, 2.0, None, 3.0]], ["V(out)"])
        panel["series"][0].update(padded=True, gaps=[])
        (path,) = _series_paths(_parse(render_plot_svg(_spec(panel), title="t")))
        assert path.get("d", "").count("M") == 1
        panel["series"][0]["gaps"] = [3]
        (path,) = _series_paths(_parse(render_plot_svg(_spec(panel), title="t")))
        assert path.get("d", "").count("M") == 2

    def test_descending_axis_renders_finite_coordinates(self):
        spec = _spec(_panel([[5.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]], ["V(out)"]))
        svg = render_plot_svg(spec, title="t")
        (path,) = _series_paths(_parse(svg))
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", path.get("d", ""))]
        assert numbers and all(n == n for n in numbers)

    def test_annotations_are_drawn_on_a_log_panel(self):
        freqs = [10.0 * 10 ** (i / 10) for i in range(41)]
        spec = _spec(
            _panel([freqs, [0.0] * 41], ["V(out)"], x_scale="log"),
            annotations=[{"x": 1000.0, "label": "pole ~1k", "marker": "pole"}],
            nmp=True,
        )
        texts = _texts(_parse(render_plot_svg(spec, title="t")))
        assert "pole ~1k" in texts
        assert any("OUT-OF-PHASE" in t for t in texts)

    def test_legend_is_capped_and_says_how_many_it_left_out(self):
        n = 30
        data = [[0.0, 1.0]] + [[float(i), float(i + 1)] for i in range(n)]
        spec = _spec(_panel(data, [f"V(out) [step {i}]" for i in range(n)]))
        root = _parse(render_plot_svg(spec, title="t"))
        texts = _texts(root)
        assert len(_series_paths(root)) == n  # every trace is drawn
        shown = [t for t in texts if t.startswith("V(out) [step")]
        assert 0 < len(shown) < n
        assert f"+{n - len(shown)} more" in texts

    def test_empty_panel_list_is_refused(self):
        with pytest.raises(ValueError, match="panel"):
            render_plot_svg(_spec(), title="t")
