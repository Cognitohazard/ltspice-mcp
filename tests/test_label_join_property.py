"""No label is placed where nets cross without a word of it, on any sheet.

``test_label_and_move_record.py`` holds this of every sheet in the suite: a
label put where wires of two or more nets cross either is refused (two nets
that each carry a name) or leaves the finished sheet saying what it joined
(``joined_at_crossing``, ``net_with_two_names``). Here it is held of sheets
made up for the purpose: orthogonal wires on a small grid, some of their ends
labelled, and a label asked for at each point where wires of two or more nets
cross.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.connectivity import crossing_points
from ltspice_mcp.lib.schematic_ops import OpAddNetLabel, apply_op_inplace, make_editor
from ltspice_mcp.lib.sheet import editor_symbol_resolver
from tests._proposal_record import sheet_says, what_is_there

Wire = tuple[int, int, int, int]

#: A point of a nine-by-nine grid, in sheet units.
_AT = st.integers(0, 8).map(lambda n: n * 16)


@st.composite
def _sheets(draw: st.DrawFn) -> tuple[list[Wire], list[tuple[int, int, str]]]:
    """Two to six wires, each level or upright, and labels on some of their ends."""
    wires: list[Wire] = []
    for _ in range(draw(st.integers(2, 6))):
        along, start = draw(_AT), draw(_AT)
        end = start + draw(st.integers(1, 6)) * 16
        wires.append(
            (start, along, end, along) if draw(st.booleans()) else (along, start, along, end)
        )
    ends = sorted({end for x1, y1, x2, y2 in wires for end in ((x1, y1), (x2, y2))})
    labelled = draw(st.lists(st.sampled_from(ends), max_size=3, unique=True))
    names = st.sampled_from(["A", "B", "0"])
    return wires, [(x, y, draw(names)) for x, y in labelled]


def _text(wires: list[Wire], labels: list[tuple[int, int, str]]) -> str:
    lines = ["Version 4", "SHEET 1 880 680"]
    lines += [f"WIRE {x1} {y1} {x2} {y2}" for x1, y1, x2, y2 in wires]
    lines += [f"FLAG {x} {y} {name}" for x, y, name in labels]
    return "\n".join(lines) + "\n"


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sheet=_sheets(), name=st.sampled_from(["A", "B", "0", "fresh"]))
def test_a_label_where_nets_cross_is_refused_or_said(
    sheet: tuple[list[Wire], list[tuple[int, int, str]]], name: str
) -> None:
    wires, labels = sheet
    text = _text(wires, labels)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "made.asc"
        target.write_text(text, encoding="utf-8")
        editor = make_editor(target)
        assert isinstance(editor, AscEditor)
        there = what_is_there(editor)
        resolver = editor_symbol_resolver(target)
        for x, y in crossing_points(((x1, y1), (x2, y2)) for x1, y1, x2, y2 in wires):
            _said, met, _named = there(name, (x, y))
            if met < 2:
                continue
            had = len(editor.labels)
            try:
                apply_op_inplace(
                    editor, OpAddNetLabel(op="add_net_label", net=name, x=x, y=y), target
                )
            except NetlistError:
                continue
            assert sheet_says(editor, {(x, y)}, resolver), (text, name, (x, y))
            del editor.labels[had:]
