"""The schematic build benchmark's record as a table, or the score of any sheet.

    uv run python scripts/build_benchmark.py table
    uv run python scripts/build_benchmark.py score SHEET.asc NETLIST.cir

``table`` prints `tests/fixtures/build_benchmark_record.json`, one row per
circuit and build; reply size is as of the last recording. ``score`` scores a
sheet against a reference netlist on equivalence, the drawing and the sheet
rules, for a sheet a model built: the netlist's ``* names:`` line lists the
nodes a drawing must name. The sheet's symbols are found where the schematic
editor finds them on this machine, which is printed, since a sheet drawn with
LTspice's own symbols is not comparable with the record, drawn with the
suite's. The design is in `docs/design/schematic_engine.md`, section 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._build_benchmark import (
    drawing,
    equivalence,
    finding_counts,
    read_circuit,
)

from ltspice_mcp.lib.sheet import editor_symbol_resolver, read_sheet

RECORD = Path(__file__).resolve().parent.parent / "tests/fixtures/build_benchmark_record.json"

_COLUMNS = (
    ("calls", "cost", "calls"),
    ("ops", "cost", "ops"),
    ("refused", "cost", "refused"),
    ("coords", "cost", "coordinates"),
    ("bytes", "cost", "reply_bytes"),
    ("wires", "drawing", "wires"),
    ("length", "drawing", "wire_length"),
    ("bends", "drawing", "bends"),
    ("cross", "drawing", "crossings"),
    ("4-way", "drawing", "four_way_junctions"),
    ("by label", "drawing", "label_only_pins"),
    ("ground", "drawing", "ground_only_pins"),
    ("off grid", "drawing", "off_grid_points"),
)


def table() -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    width = max(len(name) for name in record)
    print(f"{'build':<{width}}  " + "  ".join(f"{title:>8}" for title, _, _ in _COLUMNS))
    for name, said in record.items():
        cells = "  ".join(f"{said[group][key]:>8}" for _, group, key in _COLUMNS)
        findings = ", ".join(f"{rule} {count}" for rule, count in said["findings"].items())
        print(f"{name:<{width}}  {cells}  {findings}")
    return 0


def score(sheet_path: Path, netlist: Path) -> int:
    resolver = editor_symbol_resolver(sheet_path)
    print("symbols found in:", ", ".join(sorted(resolver.active_set)) or "nowhere")
    sheet = read_sheet(sheet_path.read_bytes(), resolver)
    said = {
        "equivalence": equivalence(sheet, read_circuit(netlist)),
        "drawing": drawing(sheet),
        "findings": finding_counts(sheet, sheet_path),
    }
    print(json.dumps(said, indent=1))
    return 0 if said["equivalence"]["equivalent"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("table", help="print the record")
    scoring = commands.add_parser("score", help="score a sheet against a reference netlist")
    scoring.add_argument("sheet", type=Path)
    scoring.add_argument("netlist", type=Path)
    args = parser.parse_args(argv)
    if args.command == "table":
        return table()
    return score(args.sheet, args.netlist)


if __name__ == "__main__":
    raise SystemExit(main())
