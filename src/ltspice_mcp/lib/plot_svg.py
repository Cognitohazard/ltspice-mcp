"""Static waveform rendering: one plot spec drawn as SVG for a model to look at.

``plot_waveform``'s interactive chart (:mod:`ltspice_mcp.lib.plot_html`) is for a
person: zoom, pan and hover do nothing for a model, which reads one still
frame. This module draws that frame from the same plot spec the chart takes —
plain arrays plus labels, ignorant of jobs and runs — and returns SVG markup;
:mod:`ltspice_mcp.lib.raster` turns it into the PNG attached to the reply.

It is plain string building with no plotting dependency, so the one optional
piece is the rasterizer the schematic renders already use (the ``raster``
extra). Every panel is drawn over the same x range, so a feature lines up
vertically across panels the way the chart's shared cursor lines it up.

Security: every label is XML-escaped and stripped of characters XML cannot
carry; no spec value is emitted as markup.
"""

import html
import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

# Same palette, in the same order, as the interactive chart, so a trace keeps
# its color between the image and the HTML file.
_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"
_WIDTH = 960
_MARGIN_LEFT = 64
_LEGEND_WIDTH = 200
_TITLE_HEIGHT = 30
_PANEL_TITLE_HEIGHT = 20
_X_TICK_HEIGHT = 18
_PANEL_GAP = 12
_X_LABEL_HEIGHT = 22
_LEGEND_ROW = 16
_LEGEND_CHARS = 30

# SI prefixes for tick labels, largest first.
_PREFIXES = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "µ"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)

# XML 1.0 cannot carry these at all, escaped or not.
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _esc(text: object) -> str:
    return html.escape(_XML_INVALID.sub("", str(text)), quote=True)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _panel_height(n_panels: int) -> int:
    """Plot-area height per panel: shorter as panels are added, so a tall stack
    does not multiply the image's pixel cost."""
    if n_panels <= 2:
        return 220
    if n_panels <= 4:
        return 180
    return 150


def _prefix_for(magnitude: float) -> tuple[float, str]:
    return next(
        ((scale, prefix) for scale, prefix in _PREFIXES if magnitude >= scale * (1 - 1e-9)),
        _PREFIXES[-1],
    )


def tick_labels(ticks: Sequence[float]) -> list[str]:
    """Labels for evenly stepped linear ticks: one SI prefix for the whole axis
    (from its largest tick) and as many decimals as the step needs, so
    neighbouring ticks never print alike — ``0, 0.5m, 1.0m, 1.5m``."""
    top = max((abs(t) for t in ticks), default=0.0)
    if not top:
        return ["0" for _ in ticks]
    scale, prefix = _prefix_for(top)
    steps = [abs(b - a) for a, b in pairwise(ticks) if b != a]
    step = min(steps) if steps else top
    decimals = max(0, min(6, -math.floor(math.log10(step / scale) + 1e-9)))
    return ["0" if abs(t) < step * 1e-9 else f"{t / scale:.{decimals}f}{prefix}" for t in ticks]


def _eng(value: float) -> str:
    """Engineering-notation label for one value (a log axis's decades):
    ``1.5m``, ``20k``, ``-2µ``, ``0``."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else ""
    scale, prefix = _prefix_for(abs(value))
    return f"{value / scale:.3g}{prefix}"


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round-numbered ticks (1, 2, 2.5, 5 x 10^n steps) inside ``[lo, hi]``."""
    span = hi - lo
    if not span > 0:
        return [lo]
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [0.0 if k == 0 else k * step for k in range(first, last + 1)]


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Decade ticks inside ``[lo, hi]``, with 2x and 5x added when a span holds
    fewer than two decades."""
    first = math.ceil(math.log10(lo) - 1e-9)
    last = math.floor(math.log10(hi) + 1e-9)
    ticks = [10.0**k for k in range(first, last + 1)]
    if len(ticks) < 2:
        ticks = sorted(
            v
            for k in range(first - 1, last + 1)
            for m in (1.0, 2.0, 5.0)
            if lo <= (v := m * 10.0**k) <= hi
        )
    return ticks


def _finite(values: Sequence[Any], *, positive: bool = False) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if math.isfinite(f) and (not positive or f > 0):
            out.append(f)
    return out


def _x_range(panels: Sequence[Mapping[str, Any]], log: bool) -> tuple[float, float]:
    xs = [v for p in panels for v in _finite(p["data"][0] if p["data"] else [], positive=log)]
    if not xs:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = min(xs), max(xs)
    if hi > lo:
        return lo, hi
    return (lo / 2, lo * 2) if log else (lo - 1.0, lo + 1.0)


def _y_range(rows: Sequence[Sequence[Any]]) -> tuple[float, float]:
    ys = [v for row in rows for v in _finite(row)]
    if not ys:
        return 0.0, 1.0
    lo, hi = min(ys), max(ys)
    if hi == lo:
        pad = abs(lo) * 0.05 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


class _Axis:
    """Maps a data value to a pixel along one axis (linear or log10)."""

    def __init__(self, lo: float, hi: float, p0: float, p1: float, log: bool) -> None:
        self.log = log
        self.lo = math.log10(lo) if log else lo
        self.hi = math.log10(hi) if log else hi
        self.p0, self.p1 = p0, p1

    def __call__(self, value: float) -> float:
        v = math.log10(value) if self.log else value
        return self.p0 + (v - self.lo) / (self.hi - self.lo) * (self.p1 - self.p0)


def _trace_path(
    xs: Sequence[Any],
    ys: Sequence[Any],
    sx: _Axis,
    sy: _Axis,
    series: Mapping[str, Any],
) -> str:
    """SVG path data for one series; a null or non-finite sample breaks the line.

    A series ``padded`` onto a shared x has no sample at the other series' x
    values; its line is joined across those, and broken only at its own
    non-finite samples, which ``gaps`` lists by index.
    """
    padded = bool(series.get("padded"))
    gaps = set(series.get("gaps") or ())
    parts: list[str] = []
    pen_down = False
    for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
        if y is None and padded and i not in gaps:
            continue
        if x is None or y is None:
            pen_down = False
            continue
        fx, fy = float(x), float(y)
        if not (math.isfinite(fx) and math.isfinite(fy)) or (sx.log and fx <= 0):
            pen_down = False
            continue
        px, py = sx(fx), sy(fy)
        if pen_down:
            parts.append(f"L{px:.1f},{py:.1f}")
        else:
            # A one-sample segment still shows: the round cap draws a dot.
            parts.append(f"M{px:.1f},{py:.1f} L{px:.1f},{py:.1f}")
            pen_down = True
    return " ".join(parts)


def _text(x: float, y: float, text: str, *, size: int = 11, anchor: str = "start", **attrs) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{extra}>'
        f"{_esc(text)}</text>"
    )


def render_plot_svg(spec: Mapping[str, Any], *, title: str) -> str:
    """Draw a plot spec (the one :func:`build_plot_html` takes) as an SVG document.

    One stacked plot area per panel, all over the same x range, each with its
    own y range, gridlines, tick labels, title (the panel's ``y_label``) and a
    legend in the right-hand column. A legend with more series than fit beside
    its panel lists as many as fit and says how many it left out. AC
    ``annotations`` are drawn as dashed guide lines across every panel, labelled
    on the first, and ``nmp`` as a tag on the first panel.
    """
    panels: list[Mapping[str, Any]] = list(spec.get("panels") or [])
    if not panels:
        raise ValueError("A plot spec needs at least one panel to draw.")
    log_x = panels[0].get("x_scale") == "log"
    x_lo, x_hi = _x_range(panels, log_x)

    ph = _panel_height(len(panels))
    block = _PANEL_TITLE_HEIGHT + ph + _X_TICK_HEIGHT + _PANEL_GAP
    height = _TITLE_HEIGHT + len(panels) * block + _X_LABEL_HEIGHT
    left = _MARGIN_LEFT
    right = _WIDTH - _LEGEND_WIDTH - 12
    sx_ticks = _log_ticks(x_lo, x_hi) if log_x else _nice_ticks(x_lo, x_hi, target=8)
    sx_labels = [_eng(t) for t in sx_ticks] if log_x else tick_labels(sx_ticks)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{height}" '
        f'viewBox="0 0 {_WIDTH} {height}" font-family="{_FONT}" fill="#1a1a1a">',
        f'<rect x="0" y="0" width="{_WIDTH}" height="{height}" fill="#ffffff"/>',
        _text(left, 20, title, size=15, font_weight="bold"),
    ]

    for i, panel in enumerate(panels):
        top = _TITLE_HEIGHT + i * block + _PANEL_TITLE_HEIGHT
        bottom = top + ph
        sx = _Axis(x_lo, x_hi, left, right, log_x)
        rows = list(panel["data"][1:])
        y_lo, y_hi = _y_range(rows)
        sy = _Axis(y_lo, y_hi, bottom, top, False)

        out.append('<g class="panel">')
        out.append(
            _text(left, top - 6, str(panel.get("y_label", "")), size=12, font_weight="bold")
        )
        out.append(
            f'<clipPath id="clip{i}"><rect x="{left}" y="{top}" width="{right - left}" height="{ph}"/></clipPath>'
        )

        grid: list[str] = []
        for tx, label in zip(sx_ticks, sx_labels, strict=True):
            px = sx(tx)
            grid.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}"/>')
            out.append(_text(px, bottom + 13, label, size=10, anchor="middle", fill="#444"))
        y_ticks = _nice_ticks(y_lo, y_hi)
        for ty, label in zip(y_ticks, tick_labels(y_ticks), strict=True):
            py = sy(ty)
            grid.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}"/>')
            out.append(_text(left - 5, py + 3.5, label, size=10, anchor="end", fill="#444"))
        out.append(f'<g stroke="#e3e3e3" stroke-width="1">{"".join(grid)}</g>')

        xs = panel["data"][0]
        series = list(panel.get("series") or [])
        out.append(
            f'<g clip-path="url(#clip{i})" fill="none" stroke-width="1.5" '
            'stroke-linejoin="round" stroke-linecap="round">'
        )
        for j, row in enumerate(rows):
            d = _trace_path(xs, row, sx, sy, series[j] if j < len(series) else {})
            if d:
                color = _PALETTE[j % len(_PALETTE)]
                out.append(f'<path class="trace" stroke="{color}" d="{d}"/>')
        out.append("</g>")

        annotations = spec.get("annotations") or []
        for ann in annotations:
            ax = ann.get("x")
            if ax is None or not math.isfinite(ax) or (log_x and ax <= 0):
                continue
            if not min(x_lo, x_hi) <= ax <= max(x_lo, x_hi):
                continue
            px = sx(float(ax))
            out.append(
                f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}" '
                'stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            if i == 0:
                out.append(
                    _text(px + 4, top + 12, str(ann.get("label", "")), size=10, fill="#222")
                )
        if i == 0 and spec.get("nmp"):
            out.append(
                _text(
                    right - 6,
                    top + 14,
                    "OUT-OF-PHASE ZERO / DELAY",
                    size=12,
                    anchor="end",
                    fill="#d62728",
                    font_weight="bold",
                )
            )

        out.append(
            f'<rect x="{left}" y="{top}" width="{right - left}" height="{ph}" '
            'fill="none" stroke="#666" stroke-width="1"/>'
        )

        fits = max(1, ph // _LEGEND_ROW)
        shown = series if len(series) <= fits else series[: fits - 1]
        lx = right + 14
        for j, entry in enumerate(shown):
            ly = top + 10 + j * _LEGEND_ROW
            color = _PALETTE[j % len(_PALETTE)]
            out.append(
                f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 16}" y2="{ly - 4}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            out.append(
                _text(lx + 22, ly, _clip(str(entry.get("label", "")), _LEGEND_CHARS), size=11)
            )
        if len(shown) < len(series):
            ly = top + 10 + len(shown) * _LEGEND_ROW
            out.append(
                _text(lx + 22, ly, f"+{len(series) - len(shown)} more", size=11, fill="#555")
            )
        out.append("</g>")

    x_label = str(panels[-1].get("x_label", ""))
    out.append(_text((left + right) / 2, height - 6, x_label, size=12, anchor="middle"))
    out.append("</svg>")
    return "\n".join(out)
