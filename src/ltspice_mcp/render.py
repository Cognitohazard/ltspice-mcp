"""CLI: render an LTspice ``.asc`` schematic to SVG.

Usage::

    python -m ltspice_mcp.render <file.asc> [-o out.svg]

With no ``-o`` the SVG is written next to the source (same stem, ``.svg``).
Unresolved symbols are rendered as placeholders and reported on stderr, but do
not fail the render.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ltspice_mcp.lib.schematic_renderer import render_to_file
from ltspice_mcp.lib.schematic_scene import build_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ltspice_mcp.render",
        description="Render an LTspice .asc schematic to SVG.",
    )
    parser.add_argument("asc", type=Path, help="Path to the .asc schematic")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .svg path")
    parser.add_argument("--margin", type=int, default=16, help="Crop margin in schematic units")
    args = parser.parse_args(argv)

    asc_path: Path = args.asc
    if not asc_path.is_file():
        print(f"error: file not found: {asc_path}", file=sys.stderr)
        return 2

    out_path: Path = args.output or asc_path.with_suffix(".svg")

    try:
        scene = build_scene(asc_path)
        render_to_file(scene, out_path, margin=args.margin)
    except (OSError, UnicodeError, ValueError) as exc:
        # Unreadable/undecodable source or a malformed record that slipped past
        # the per-line guards: report cleanly instead of dumping a traceback.
        print(f"error: could not render {asc_path}: {exc}", file=sys.stderr)
        return 2

    for diag in scene.diagnostics:
        print(f"warning: {diag}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
