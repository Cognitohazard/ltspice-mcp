"""Shared inline fixtures for the schematic scene/renderer tests.

Kept as a non-test helper module (leading underscore) so pytest does not try to
collect it. All ``.asy``/``.asc`` content is authored here and written to a
``tmp_path`` by the tests — nothing depends on a stock LTspice library install.
"""

from __future__ import annotations

from pathlib import Path

# A deliberately asymmetric symbol so every rotation/mirror is distinguishable.
# Rect 0,0..32,16; pin A at (0,8) order 1; pin B at (32,8) order 2.
BOX2_ASY = """\
Version 4
SymbolType CELL
RECTANGLE Normal 0 0 32 16
LINE Normal 0 8 32 8
PIN 0 8 NONE 0
PINATTR PinName A
PINATTR SpiceOrder 1
PIN 32 8 NONE 0
PINATTR PinName B
PINATTR SpiceOrder 2
WINDOW 0 16 -4 Center 2
WINDOW 3 16 24 Center 2
SYMATTR Prefix X
SYMATTR Value box2
"""

# Same symbol name, different body (one circle) — for collision precedence.
BOX2_ALT_ASY = """\
Version 4
SymbolType CELL
CIRCLE Normal 0 0 20 20
PIN 0 10 NONE 0
PINATTR PinName A
PINATTR SpiceOrder 1
SYMATTR Prefix X
"""


# A two-terminal symbol whose attribute windows are stacked in the symbol's own
# y, 16 units apart — the stock `cap` arrangement. That gap is narrower than a
# four-character value, so it is the spacing that exposes how a sideways
# placement lays the two strings out.
STACKED_ATTRS_ASY = """\
Version 4
SymbolType CELL
LINE Normal -16 -8 16 -8
LINE Normal -16 8 16 8
PIN 0 -32 NONE 0
PINATTR PinName 1
PINATTR SpiceOrder 1
PIN 0 32 NONE 0
PINATTR PinName 2
PINATTR SpiceOrder 2
WINDOW 0 24 -8 Left 2
WINDOW 3 24 8 Left 2
SYMATTR Prefix C
SYMATTR Value cap
"""


def asc_with_stacked_attrs(rotation: str, *, windows: str = "") -> str:
    """A ``stacked`` instance named C1 with value 100n, at the origin."""
    return (
        "Version 4\n"
        "SHEET 1 880 680\n"
        f"SYMBOL stacked 0 0 {rotation}\n"
        f"{windows}"
        "SYMATTR InstName C1\n"
        "SYMATTR Value 100n\n"
    )


def write_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def asc_with_box2(rotation: str, ox: int = 0, oy: int = 0) -> str:
    return (
        "Version 4\n"
        "SHEET 1 880 680\n"
        f"SYMBOL box2 {ox} {oy} {rotation}\n"
        "SYMATTR InstName X1\n"
        "SYMATTR Value 5\n"
    )
