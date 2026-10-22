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
    UNCONNECTED_MARKER,
    DrawEllipse,
    DrawLine,
    DrawPolyline,
    DrawRect,
    Graphic,
    NetFlag,
    Scene,
    TextLayout,
    flag_label_layout,
    font_px,
    ground_polygon,
    line_height,
)

_STYLE = """
.bg { fill: #ffffff; }
.wire { stroke: #0a4bd8; stroke-width: 1.5; fill: none; }
.sym { stroke: #111111; stroke-width: 1.5; fill: none; }
.pin { stroke: #b00020; stroke-width: 1; fill: none; }
.gnd { stroke: #0a4bd8; stroke-width: 1.5; fill: none; }
.unconnected { stroke: #0a4bd8; stroke-width: 1; fill: none; }
text { font-family: Tahoma, 'DejaVu Sans', sans-serif; font-weight: bold; }
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
        out.append(
            _text_svg("t-flag", x, y, [flag.text], FLAG_LABEL_SIZE, flag_label_layout(flag))
        )
    return out


def _text_svg(cls: str, x: int, y: int, lines: list[str], size: int, layout: TextLayout) -> str:
    """One ``<text>`` laid out by ``layout`` from the point ``(x, y)``.

    A block of several lines gets a ``<tspan>`` per line, each restating x so
    the lines stack instead of running on; a turned block turns as a whole, so
    its lines stack along the glyphs' own "down".
    """
    bx, by = x + layout.dx, y + layout.dy
    transform = (
        f' transform="rotate({_fmt(layout.rotation)},{_fmt(bx)},{_fmt(by)})"'
        if layout.rotation
        else ""
    )
    if len(lines) == 1:
        body = escape(lines[0])
    else:
        step = line_height(size)
        body = "".join(
            f'<tspan x="{_fmt(bx)}" dy="{_fmt(0 if k == 0 else step)}">{escape(line)}</tspan>'
            for k, line in enumerate(lines)
        )
    return (
        f'<text class="{cls}" x="{_fmt(bx)}" y="{_fmt(by)}" text-anchor="{layout.anchor}" '
        f'font-size="{font_px(size)}"{transform}>{body}</text>'
    )


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
            cls = "t-placeholder" if t.role == "placeholder" else "t-attr"
            parts.append(_text_svg(cls, t.x, t.y, [t.text], t.size, t.layout))
    for d in scene.directives:
        if (layout := d.layout) is not None:
            cls = "t-directive" if d.is_directive else "t-comment"
            parts.append(_text_svg(cls, d.x, d.y, d.lines, d.size, layout))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_to_file(scene: Scene, out_path: Path, margin: int = 16) -> Path:
    """Render ``scene`` to ``out_path`` (UTF-8). Returns the path written."""
    out_path = Path(out_path)
    out_path.write_text(render_svg(scene, margin=margin), encoding="utf-8")
    return out_path
