"""Render a :class:`~ltspice_mcp.lib.schematic_scene.Scene` to SVG.

SVG is the only backend — no matplotlib, no raster step. Output is deterministic
for identical source + library set: elements are emitted in the stable order the
scene stores them, coordinates are integers, and nothing time- or
environment-dependent is written. The image is cropped to the scene's content
bounding box plus a fixed margin.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_scene import (
    FLAG_LABEL_SIZE,
    LINE_GAP,
    UNCONNECTED_MARKER,
    DrawEllipse,
    DrawLine,
    DrawPolyline,
    DrawRect,
    Graphic,
    NetFlag,
    Scene,
    decode_text_lines,
    flag_label_anchor,
    font_px,
    ground_polygon,
)

_STYLE = """
.bg { fill: #ffffff; }
.wire { stroke: #0a4bd8; stroke-width: 1.5; fill: none; }
.sym { stroke: #111111; stroke-width: 1.5; fill: none; }
.pin { stroke: #b00020; stroke-width: 1; fill: none; }
.gnd { stroke: #0a4bd8; stroke-width: 1.5; fill: none; }
.unconnected { stroke: #0a4bd8; stroke-width: 1; fill: none; }
text { font-family: 'DejaVu Sans Mono', monospace; }
.t-attr { fill: #101010; }
.t-flag { fill: #0a4bd8; }
.t-directive { fill: #106010; }
.t-comment { fill: #505050; }
.t-placeholder { fill: #b00020; }
.missing { stroke: #b00020; stroke-width: 1.5; stroke-dasharray: 4 3; fill: none; }
""".strip()


def _fmt(n: int) -> str:
    return str(int(n))


def _graphic_svg(g: Graphic, cls: str) -> str:
    if isinstance(g, DrawLine):
        return (
            f'<line class="{cls}" x1="{_fmt(g.x1)}" y1="{_fmt(g.y1)}" '
            f'x2="{_fmt(g.x2)}" y2="{_fmt(g.y2)}"/>'
        )
    if isinstance(g, DrawRect):
        # Normalize corners: a source RECTANGLE may list them reversed, which
        # would give a negative width/height and SVG would render nothing.
        x, y = min(g.x1, g.x2), min(g.y1, g.y2)
        return (
            f'<rect class="{cls}" x="{_fmt(x)}" y="{_fmt(y)}" '
            f'width="{_fmt(abs(g.x2 - g.x1))}" height="{_fmt(abs(g.y2 - g.y1))}"/>'
        )
    if isinstance(g, DrawEllipse):
        cx = (g.x1 + g.x2) / 2
        cy = (g.y1 + g.y2) / 2
        rx = abs(g.x2 - g.x1) / 2
        ry = abs(g.y2 - g.y1) / 2
        return (
            f'<ellipse class="{cls}" cx="{_fmt(round(cx))}" cy="{_fmt(round(cy))}" '
            f'rx="{_fmt(round(rx))}" ry="{_fmt(round(ry))}"/>'
        )
    if isinstance(g, DrawPolyline):
        pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in g.pts)
        return f'<polyline class="{cls}" points="{pts}"/>'
    return ""


def _flag_svg(flag: NetFlag) -> list[str]:
    out: list[str] = []
    x, y = flag.x, flag.y
    if flag.unconnected:
        # LTspice's small square marking a flag that connects to nothing.
        half = UNCONNECTED_MARKER // 2
        out.append(
            f'<rect class="unconnected" x="{_fmt(x - half)}" y="{_fmt(y - half)}" '
            f'width="{_fmt(UNCONNECTED_MARKER)}" height="{_fmt(UNCONNECTED_MARKER)}"/>'
        )
    if flag.is_ground:
        # Triangle with its base through the connection point and its apex on
        # the side away from the wire.
        pts = " ".join(f"{_fmt(px_)},{_fmt(py_)}" for px_, py_ in ground_polygon(flag))
        out.append(f'<polygon class="gnd" points="{pts}"/>')
    else:
        # No connection dot: LTspice does not draw one on a net label.
        size = font_px(FLAG_LABEL_SIZE)
        lx, ly, anchor, rotation = flag_label_anchor(flag)
        transform = (
            f' transform="rotate({_fmt(rotation)},{_fmt(lx)},{_fmt(ly)})"' if rotation else ""
        )
        out.append(
            f'<text class="t-flag" x="{_fmt(lx)}" y="{_fmt(ly)}" '
            f'text-anchor="{anchor}" font-size="{size}"{transform}>{escape(flag.text)}</text>'
        )
    return out


def render_svg(scene: Scene, margin: int = 16) -> str:
    """Return the scene as a self-contained SVG document string."""
    content = scene.content_bbox()
    if content is None:
        content = BBox(0, 0, 64, 64)
    view = content.expanded(margin)
    vw = max(view.width, 1)
    vh = max(view.height, 1)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{_fmt(view.x1)} {_fmt(view.y1)} {_fmt(vw)} {_fmt(vh)}" '
        f'width="{_fmt(vw)}" height="{_fmt(vh)}">'
    )
    parts.append(f"<title>{escape(scene.source.name)}</title>")
    parts.append(f"<style>{_STYLE}</style>")
    parts.append(
        f'<rect class="bg" x="{_fmt(view.x1)}" y="{_fmt(view.y1)}" '
        f'width="{_fmt(vw)}" height="{_fmt(vh)}"/>'
    )

    # Sheet-level graphics (drawn behind everything else).
    for g in scene.sheet_graphics:
        parts.append(_graphic_svg(g, "sym"))

    # Wires.
    for w in scene.wires:
        parts.append(
            f'<line class="wire" x1="{_fmt(w.x1)}" y1="{_fmt(w.y1)}" '
            f'x2="{_fmt(w.x2)}" y2="{_fmt(w.y2)}"/>'
        )

    # Symbol bodies and pins.
    for sym in scene.symbols:
        cls = "missing" if sym.missing else "sym"
        for g in sym.graphics:
            parts.append(_graphic_svg(g, cls))
        for pin in sym.pins:
            parts.append(f'<circle class="pin" cx="{_fmt(pin.x)}" cy="{_fmt(pin.y)}" r="1.5"/>')

    # Flags (ground glyphs and net labels).
    for flag in scene.flags:
        parts.extend(_flag_svg(flag))

    # Text on top: symbol attributes / placeholders, then sheet directives.
    for sym in scene.symbols:
        for t in sym.texts:
            px = font_px(t.size)
            cls = "t-placeholder" if t.role == "placeholder" else "t-attr"
            transform = (
                f' transform="rotate({_fmt(t.rotation)},{_fmt(t.x)},{_fmt(t.y)})"'
                if t.rotation
                else ""
            )
            parts.append(
                f'<text class="{cls}" x="{_fmt(t.x)}" y="{_fmt(t.y)}" '
                f'text-anchor="{t.anchor}" font-size="{px}"{transform}>{escape(t.text)}</text>'
            )
    for d in scene.directives:
        px = font_px(d.size)
        cls = "t-directive" if d.is_directive else "t-comment"
        # A multi-line TEXT record becomes one <text> with a <tspan> per line;
        # each tspan restates x so lines stack instead of running on.
        spans = "".join(
            f'<tspan x="{_fmt(d.x)}" dy="{_fmt(0 if k == 0 else px + LINE_GAP)}">'
            f"{escape(line)}</tspan>"
            for k, line in enumerate(decode_text_lines(d.text))
        )
        parts.append(
            f'<text class="{cls}" x="{_fmt(d.x)}" y="{_fmt(d.y)}" '
            f'text-anchor="start" font-size="{px}">{spans}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_to_file(scene: Scene, out_path: Path, margin: int = 16) -> Path:
    """Render ``scene`` to ``out_path`` (UTF-8). Returns the path written."""
    out_path = Path(out_path)
    out_path.write_text(render_svg(scene, margin=margin), encoding="utf-8")
    return out_path
