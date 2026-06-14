"""Self-contained interactive-plot HTML assembly (uPlot, vendored).

:func:`build_plot_html` turns a plain ``PlotSpec`` dict (arrays + labels,
ignorant of job/RunRef internals) into one offline HTML file with uPlot and the
data inlined — no CDN, no network, no template engine. ``plot_waveform`` writes
the result to disk and opens it on the desktop.

Security: signal/node names flow into the page, so chart data + labels are
emitted as a single JSON blob in a ``<script type="application/json">`` element
(``</`` neutralized so a crafted name cannot close the script early), and the
page-chrome strings (title, summary) are ``html.escape``-d. The init script is a
static literal that reads the JSON blob — no user data is ever interpolated into
markup attributes or executable script.
"""

import html
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

_ASSET_PKG = "ltspice_mcp"


@lru_cache(maxsize=1)
def _uplot_js() -> str:
    return (files(_ASSET_PKG) / "assets" / "uplot" / "uPlot.iife.min.js").read_text("utf-8")


@lru_cache(maxsize=1)
def _uplot_css() -> str:
    return (files(_ASSET_PKG) / "assets" / "uplot" / "uPlot.min.css").read_text("utf-8")


_PAGE_CSS = (
    "body{font:14px system-ui,-apple-system,sans-serif;margin:16px;color:#1a1a1a;"
    "background:#fff}h1{font-size:17px;margin:0 0 4px}#summary{color:#555;"
    "font-size:12px;margin:0 0 12px;white-space:pre-wrap}.uplot{margin-bottom:18px}"
)

# Static init script — NO user data is interpolated; it parses the JSON blob and
# builds one uPlot per panel. A 'bode' spec stacks two panels sharing a synced
# log-frequency x cursor/zoom. All presentation (palette, sizes) lives here.
_INIT_JS = """
(function () {
  var spec = JSON.parse(document.getElementById('plot-data').textContent);
  var root = document.getElementById('panels');
  var palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b',
                 '#e377c2','#7f7f7f','#bcbd22','#17becf'];
  var charts = [];
  var H = spec.bode ? 300 : 440;
  function width() { return Math.max(640, Math.floor(window.innerWidth - 40)); }
  function mkOpts(panel, syncKey) {
    var series = [{}];
    for (var i = 0; i < panel.series.length; i++) {
      series.push({ label: panel.series[i].label,
                    stroke: palette[i % palette.length],
                    points: { show: false } });
    }
    var opts = {
      width: width(), height: H, title: panel.y_label,
      scales: { x: panel.x_scale === 'log' ? { distr: 3, log: 10 } : { time: false } },
      axes: [{ label: panel.x_label }, { label: panel.y_label }],
      series: series,
    };
    if (syncKey) opts.cursor = { sync: { key: syncKey, scales: ['x', null] } };
    return opts;
  }
  function add(panel, syncKey) {
    var u = new uPlot(mkOpts(panel, syncKey), panel.data, root);
    charts.push(u);
    return u;
  }
  if (spec.bode && spec.panels.length === 2) {
    var s = uPlot.sync('bode');
    s.sub(add(spec.panels[0], s.key));
    s.sub(add(spec.panels[1], s.key));
  } else {
    for (var p = 0; p < spec.panels.length; p++) add(spec.panels[p], null);
  }
  window.addEventListener('resize', function () {
    var w = width();
    for (var i = 0; i < charts.length; i++) charts[i].setSize({ width: w, height: H });
  });
})();
"""


def build_plot_html(spec: dict[str, Any], *, title: str, summary: str = "") -> str:
    """Assemble a single self-contained HTML file from a PlotSpec.

    ``spec`` carries only chart data (``panels``/``bode``/``analysis_type``) — the
    ``title``/``summary`` chrome is passed separately so neither can collide with
    the data blob. ``allow_nan=False``: non-finite samples must already be JSON
    ``null`` (the worker's job); a stray ``NaN`` raises here (fail loud) rather
    than emitting a ``NaN`` token the browser's ``JSON.parse`` would reject —
    which would silently blank the chart.
    """
    blob = json.dumps(spec, ensure_ascii=True, allow_nan=False).replace("</", "<\\/")
    esc_title = html.escape(title)
    esc_summary = html.escape(summary)
    # Positional concatenation (not an f-string / .replace template): each piece
    # is inserted exactly once, so user text in title/summary/labels can never
    # introduce a placeholder that later expansion would substitute into.
    return "".join(
        [
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            "<title>",
            esc_title,
            "</title>\n<style>",
            _uplot_css(),
            "</style>\n",
            "<style>",
            _PAGE_CSS,
            "</style>\n</head><body>\n",
            "<h1>",
            esc_title,
            '</h1>\n<div id="summary">',
            esc_summary,
            "</div>\n",
            '<div id="panels"></div>\n',
            '<script id="plot-data" type="application/json">',
            blob,
            "</script>\n",
            "<script>",
            _uplot_js(),
            "</script>\n",
            "<script>",
            _INIT_JS,
            "</script>\n</body></html>\n",
        ]
    )
