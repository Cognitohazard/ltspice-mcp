"""Interactive-plot HTML assembly (uPlot, vendored) — two delivery shapes.

:func:`build_plot_html` turns a plain ``PlotSpec`` dict (arrays + labels,
ignorant of job/RunRef internals) into one offline HTML file with uPlot and the
data inlined — no CDN, no network, no template engine. ``plot_waveform`` writes
it to disk and opens it on the desktop (terminal clients).

:func:`build_widget_html` builds the *generic* MCP Apps renderer template
(SEP-1865): uPlot + the vendored ext-apps ``App`` runtime inlined, NO data baked
in. It is served once as the stable ``ui://`` resource the tool references; the
host renders it in a sandboxed iframe and pipes the per-call chart spec in via
``app.ontoolresult``. Both shapes share the same :data:`_RENDER_JS` core.

Security: signal/node names flow into the page. For the file, chart data + labels
are emitted as a single JSON blob in a ``<script type="application/json">``
element (``</`` neutralized so a crafted name cannot close the script early) and
chrome strings (title, summary) are ``html.escape``-d. The widget template is a
static literal with NO user data interpolated at all — the per-call spec arrives
at runtime via the host bridge, never as markup.
"""

import html
import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any

_ASSET_PKG = "ltspice_mcp"

# MIME type SEP-1865 mandates for an HTML UI resource — how an apps-capable host
# knows to render the resource as an interactive iframe, not show its source.
WIDGET_MIME_TYPE = "text/html;profile=mcp-app"

# Stable, predeclared ``ui://`` URI for the one generic plot renderer. The tool
# references it via ``_meta.ui.resourceUri``; the host fetches it with
# ``resources/read``. One renderer serves every plot — the per-call data is piped
# in at runtime, so the URI never changes (it stays prefetchable/cacheable).
WIDGET_RESOURCE_URI = "ui://ltspice-mcp/plot"

# Namespaced key under the tool result's ``_meta`` carrying the per-call chart
# spec (a JSON string) for the widget. ``_meta`` is not shown to the model, so the
# plot returns no numbers to it; the widget reads this in ``app.ontoolresult``.
WIDGET_SPEC_META_KEY = "ltspice/plotSpec"


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

# Shared render core — NO user data is interpolated. Defines renderSpec(spec, root):
# one uPlot per panel; a 'bode' spec stacks two panels sharing a synced
# log-frequency x cursor/zoom. All presentation (palette, sizes) lives here. Used
# by both the offline file and the in-chat widget so they render identically.
_RENDER_JS = """
function renderSpec(spec, root) {
  var palette = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b',
                 '#e377c2','#7f7f7f','#bcbd22','#17becf'];
  var charts = [];
  var H = spec.bode ? 300 : 440;
  function width() { return Math.max(640, Math.floor(window.innerWidth - 40)); }
  function fmtHz(v) { var a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6) + 'M'; if (a >= 1e3) return (v / 1e3) + 'k'; return '' + v; }
  // Log-axis tick label: only label exact decades (10/100/1k/...), leave minor
  // ticks and uPlot's null padding-slots blank (returning null, NOT the string
  // "null"), so the frequency axis stays readable.
  function hzTick(v) {
    if (v == null || !isFinite(v)) return null;
    var l = Math.log10(v);
    return Math.abs(l - Math.round(l)) > 1e-6 ? null : fmtHz(v);
  }
  // Draw detected AC corner markers (dashed vertical lines + labels) and an
  // out-of-phase-zero / delay tag straight onto each panel's canvas. Pure canvas — no
  // extra deps. Reads spec.annotations / spec.nmp from the renderSpec closure;
  // a no-op when neither is present (transient/DC and un-annotated AC unchanged).
  function annotPlugin() {
    return { hooks: { draw: function (u) {
      var anns = spec.annotations || [];
      if (!anns.length && !spec.nmp) return;
      var ctx = u.ctx;
      // uPlot draws in DEVICE pixels (valToPos/bbox are device px), so every size
      // must scale by the pixel ratio or it renders tiny on a hi-DPI display.
      var dpr = (typeof uPlot !== 'undefined' && uPlot.pxRatio) || window.devicePixelRatio || 1;
      var L = u.bbox.left, T = u.bbox.top, W = u.bbox.width, Hh = u.bbox.height;
      var xs = u.data[0], ys = u.data[1] || [];
      var fs = Math.round(13 * dpr), r = 6 * dpr;
      ctx.save();
      ctx.font = fs + 'px system-ui,-apple-system,sans-serif';
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      function chip(text, lx, ly, fill) {  // white-backed text for legibility over the curve
        var w = ctx.measureText(text).width;
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fillRect(lx - 3 * dpr, ly - 2 * dpr, w + 6 * dpr, fs + 5 * dpr);
        ctx.fillStyle = fill;
        ctx.fillText(text, lx, ly);
      }
      for (var i = 0; i < anns.length; i++) {
        var ax = anns[i].x;
        var x = u.valToPos(ax, 'x', true);
        if (x < L || x > L + W) continue;
        // faint dashed guide line down the panel at the corner frequency
        ctx.strokeStyle = 'rgba(120,120,120,0.4)';
        ctx.lineWidth = 1 * dpr;
        ctx.setLineDash([3 * dpr, 3 * dpr]);
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + Hh); ctx.stroke();
        ctx.setLineDash([]);
        // marker placed ON this panel's curve at the nearest sample: a cross for
        // a pole, a circle for a zero (the control-theory convention).
        var j = 0, best = Infinity;
        for (var k = 0; k < xs.length; k++) {
          var d = Math.abs(xs[k] - ax);
          if (d < best) { best = d; j = k; }
        }
        var yv = ys.length ? ys[j] : null;
        var my = (yv !== null && isFinite(yv)) ? u.valToPos(yv, 'y', true) : T + 16 * dpr;
        ctx.lineWidth = 2.5 * dpr;
        if (anns[i].marker === 'zero') {
          ctx.strokeStyle = '#2ca02c';
          ctx.beginPath(); ctx.arc(x, my, r, 0, 6.2832); ctx.stroke();
        } else {
          ctx.strokeStyle = '#d62728';
          ctx.beginPath();
          ctx.moveTo(x - r, my - r); ctx.lineTo(x + r, my + r);
          ctx.moveTo(x - r, my + r); ctx.lineTo(x + r, my - r);
          ctx.stroke();
        }
        // label lifted clear of the marker and the curve, with a white backing
        var ly = Math.max(T + 3 * dpr, my - r - fs - 6 * dpr);
        chip(anns[i].label, x + 9 * dpr, ly, 'rgba(20,20,20,0.97)');
      }
      if (anns.length) {
        // Legend in the usually-empty top-left (low-frequency, flat) corner.
        chip(String.fromCharCode(215) + ' pole   ' + String.fromCharCode(9675) + ' zero',
          L + 6 * dpr, T + 4 * dpr, 'rgba(70,70,70,0.95)');
      }
      if (spec.nmp) {
        ctx.font = 'bold ' + Math.round(15 * dpr) + 'px system-ui,-apple-system,sans-serif';
        var tag = 'OUT-OF-PHASE ZERO / DELAY';
        var tw = ctx.measureText(tag).width;
        var tx = L + W - tw - 6 * dpr;
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fillRect(tx - 4 * dpr, T + 3 * dpr, tw + 8 * dpr, Math.round(15 * dpr) + 6 * dpr);
        ctx.fillStyle = '#d62728';
        ctx.fillText(tag, tx, T + 5 * dpr);
      }
      ctx.restore();
    } } };
  }
  function mkOpts(panel, syncKey) {
    var logx = panel.x_scale === 'log';
    // x series (index 0): format the cursor-legend readout as Hz on a log axis.
    var series = [logx ? { value: function (u, v) { return v == null ? '' : fmtHz(v) + ' Hz'; } } : {}];
    for (var i = 0; i < panel.series.length; i++) {
      series.push({ label: panel.series[i].label,
                    stroke: palette[i % palette.length],
                    points: { show: false } });
    }
    var opts = {
      width: width(), height: H, title: panel.y_label,
      // time:false on BOTH branches — a log frequency axis must not be formatted
      // as epoch time (uPlot's x-scale default), which renders Hz values as dates.
      scales: { x: logx ? { distr: 3, log: 10, time: false } : { time: false } },
      axes: [
        { label: panel.x_label, values: logx ? function (u, sp) { return sp.map(hzTick); } : null },
        { label: panel.y_label },
      ],
      series: series,
      plugins: [annotPlugin()],
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
}
"""

# Offline-file driver: reads the inlined JSON blob and renders once.
_FILE_INIT_JS = """
(function () {
  var spec = JSON.parse(document.getElementById('plot-data').textContent);
  renderSpec(spec, document.getElementById('panels'));
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
            _RENDER_JS,
            _FILE_INIT_JS,
            "</script>\n</body></html>\n",
        ]
    )


def _globalize_ext_apps(bundle: str) -> str:
    """Rewrite the ext-apps ESM ``export{…}`` tail into a ``globalThis`` assign.

    The vendored ``app-with-deps`` bundle is minified ESM ending in
    ``export{a as App,…}``; an inline ``<script type="module">`` can't re-export,
    so turn the trailing export list into ``globalThis.ExtApps={App:a,…}``. Done
    with ``str`` slicing (not ``re.sub``) because the minified body is full of
    ``$`` sequences a regex replacement would mangle.
    """
    m = re.search(r"export\{([^}]+)\};?\s*$", bundle)
    if m is None:  # pragma: no cover - shape guard; vendored bundle always matches
        return bundle
    pairs = []
    for pair in m.group(1).split(","):
        names = [s.strip() for s in pair.split(" as ")]
        local = names[0]
        exported = names[1] if len(names) > 1 else local
        pairs.append(f"{exported}:{local}")
    return bundle[: m.start()] + "globalThis.ExtApps={" + ",".join(pairs) + "};"


@lru_cache(maxsize=1)
def _ext_apps_bundle() -> str:
    raw = (files(_ASSET_PKG) / "assets" / "ext-apps" / "app-with-deps.js").read_text("utf-8")
    return _globalize_ext_apps(raw)


# Light/dark variables for the widget (the host advertises its theme; the file
# uses the fixed light _PAGE_CSS since a browser tab has no host theme to follow).
_WIDGET_THEME_CSS = (
    ":root{color-scheme:light}"
    "html.dark{color-scheme:dark}html.dark body{background:#1f2428;color:#e6e6e6}"
    "#status{color:#666;font-size:12px;margin:0 0 8px}"
)

# Widget driver — set ontoolresult BEFORE connect (the result can arrive
# immediately). ``ontoolresult`` receives the full CallToolResult; the chart spec
# rides in ``_meta["ltspice/plotSpec"]`` (a JSON string) — a non-model-visible
# channel, so the plot returns no numbers to the model. NO user data is in this
# template; the per-call spec arrives at runtime via the host bridge.
_WIDGET_INIT_JS = """
(async function () {
  var App = globalThis.ExtApps.App;
  var root = document.getElementById('panels');
  var status = document.getElementById('status');
  function applyTheme(t) { document.documentElement.classList.toggle('dark', t === 'dark'); }
  function draw(result) {
    var raw = result && result._meta && result._meta['__SPEC_KEY__'];
    if (!raw) { status.textContent = 'No plot data in this result.'; return; }
    var spec;
    try { spec = JSON.parse(raw); } catch (e) { status.textContent = 'Could not parse plot data.'; return; }
    status.style.display = 'none';
    root.innerHTML = '';
    renderSpec(spec, root);
  }
  var app = new App({ name: 'ltspice-plot', version: '1.0.0' }, {});
  app.onhostcontextchanged = function (ctx) { applyTheme(ctx && ctx.theme); };
  app.ontoolresult = draw;
  await app.connect();
  var ctx = app.getHostContext ? app.getHostContext() : null;
  applyTheme(ctx && ctx.theme);
})();
""".replace("__SPEC_KEY__", WIDGET_SPEC_META_KEY)


@lru_cache(maxsize=1)
def build_widget_html() -> str:
    """The generic MCP Apps renderer template (static; no per-call data).

    Inlines uPlot + the ext-apps ``App`` runtime; receives the chart spec at
    runtime via ``app.ontoolresult`` and renders it with the shared
    :data:`_RENDER_JS` core. Served as the stable ``ui://`` resource that the
    ``plot_waveform`` tool references via ``_meta.ui.resourceUri``. Cached: it
    never varies, so it is built once.
    """
    return "".join(
        [
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            "<title>LTspice plot</title>\n<style>",
            _uplot_css(),
            "</style>\n<style>",
            _PAGE_CSS,
            _WIDGET_THEME_CSS,
            "</style>\n</head><body>\n",
            '<div id="status">Waiting for plot data…</div>\n',
            '<div id="panels"></div>\n',
            "<script>",
            _uplot_js(),
            "</script>\n",
            '<script type="module">\n',
            _ext_apps_bundle(),
            "\n",
            _RENDER_JS,
            _WIDGET_INIT_JS,
            "</script>\n</body></html>\n",
        ]
    )
