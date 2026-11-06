"""Why a signal name reads no trace, on raw headers a test writes.

Each note is held against what LTspice wrote in
``test_recorded_ltspice_results.py``. What no recording holds is checked here,
on a header read by the server's own header reader: a name that two traces
miss alike by an instance's X, and a deck's run whose nodes are named as
LTspice names a sheet's unlabelled nets.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ltspice_mcp.lib.raw_header import RawPlotHeader, preflight_raw
from ltspice_mcp.lib.trace_names import from_schematic, read_in_plot, why_not
from tests.test_raw_header import LIMITS, synthetic_header, write_raw

#: The ``Command`` line LTspice 26 writes (recorded: every ``ltspice26`` raw).
LTSPICE_26 = "Linear Technology Corporation LTspice"


def operating_point(
    tmp_path: Path, title: str, names: tuple[str, ...], *, command: str = "ngspice-42"
) -> RawPlotHeader:
    """The header of a one-point raw titled ``title`` holding ``names``, as the
    simulator ``command`` names writes one: LTspice stores the first value as a
    double and the rest as floats, ngspice every value as a double."""
    variables = tuple((name, "device_current" if name[0] in "Ii" else "voltage") for name in names)
    text = synthetic_header(variables=variables, points=1, plot="Operating Point", command=command)
    data = text.replace("Title: synthetic", f"Title: {title}").encode("ascii")
    layout = "<d" + ("f" if command.startswith("Linear") else "d") * (len(names) - 1)
    path = write_raw(tmp_path, data + struct.pack(layout, *range(len(names))))
    (plot,) = preflight_raw(path, limits=LIMITS).plots
    return plot


def refusal(plot: RawPlotHeader, signal: str) -> str:
    read = read_in_plot(plot, signal)
    assert read.refused == "missing", read
    return why_not(read, signal, plot)


def test_a_name_two_traces_miss_alike_by_an_x_names_neither(tmp_path: Path):
    """``xa`` is ``a`` without its X and ``xxa`` with one more: a guess
    between them could name the wrong part's current, so neither is named."""
    plot = operating_point(tmp_path, "deck.cir", ("V(out)", "Id(a:M1)", "Id(xxa:M1)"))
    assert refusal(plot, "Id(xa:M1)") == "Signal 'Id(xa:M1)' not found."
    # With one of them gone, the other is named.
    plot = operating_point(tmp_path, "deck.cir", ("V(out)", "Id(xxa:M1)"))
    assert "Id(xxa:M1) is in this result, which names 'xa' as 'xxa'." in refusal(plot, "Id(xa:M1)")


def test_only_an_instance_on_the_path_is_read_without_its_x(tmp_path: Path):
    """The last name on a path is a node or a device, which no X marks:
    ``V(xa:xout)`` is not ``V(xa:out)``, nor is a top-level ``V(xout)``."""
    plot = operating_point(tmp_path, "deck.cir", ("V(xa:out)", "V(out)"))
    assert refusal(plot, "V(xa:xout)") == "Signal 'V(xa:xout)' not found."
    assert refusal(plot, "V(xout)") == "Signal 'V(xout)' not found."


def test_the_schematics_rule_is_the_reason_on_ltspices_run_of_a_schematic_only(tmp_path: Path):
    """A name with an X more than an instance's in the trace is a schematic
    part's netlist spelling on LTspice's run of a sheet. On a deck's run, or
    another simulator's, nothing puts the X there: the trace is named, and no
    rule is given as the reason."""
    names = ("V(out)", "Ix(x1:in)")
    for command, title, reason in (
        (LTSPICE_26, r"C:\work\sheet.net", True),
        (LTSPICE_26, r"C:\work\deck.cir", False),
        ("ngspice-42", r"* C:\work\sheet.asc", False),
    ):
        plot = operating_point(tmp_path, title, names, command=command)
        text = refusal(plot, "Ix(xx1:in)")
        assert text == "Signal 'Ix(xx1:in)' not found. Ix(x1:in) is in this result, " + (
            "which names 'xx1' as 'x1'."
            + (
                " LTspice names a part placed on a schematic by its reference, without "
                "the X its netlist line adds in front."
                if reason
                else ""
            )
        ), text


def test_a_decks_nodes_named_like_a_sheets_get_no_note_on_labels(tmp_path: Path):
    """A deck may name its own nodes N001; its run is no schematic's, so a
    missing name gets no word on net labels. The same traces under a sheet's
    title (XVII's ``* <sheet>.asc``, LTspice 26's exported ``.net``) do."""
    names = ("V(n001)", "V(n002)", "I(V1)")
    deck = operating_point(tmp_path, r"C:\work\deck.cir", names)
    assert not from_schematic(deck)
    assert refusal(deck, "V(inp)") == "Signal 'V(inp)' not found."
    for title in (r"* C:\work\sheet.asc", r"C:\work\sheet.net"):
        sheet = operating_point(tmp_path, title, names)
        assert from_schematic(sheet)
        assert "here V(n001), V(n002). Put a net label inp on the net" in refusal(sheet, "V(inp)")


def test_a_name_of_ltspices_own_form_is_not_told_to_take_a_label(tmp_path: Path):
    plot = operating_point(tmp_path, r"* C:\work\sheet.asc", ("V(n001)", "V(n002)"))
    text = refusal(plot, "V(n003)")
    assert "here V(n001), V(n002)." in text
    assert "net label" not in text.split("here", 1)[1]


def test_a_device_parameter_is_told_where_parameters_are_read_not_to_take_a_label(
    tmp_path: Path,
):
    """``V(@m1[vth])`` is ngspice's spelling of a saved parameter, not a node:
    a schematic's result that holds unlabelled nets still says where a
    device's parameters are read."""
    for title in (r"* C:\work\sheet.asc", r"C:\work\run_1.net"):
        plot = operating_point(tmp_path, title, ("V(n001)", "I(V1)"))
        for asked in ("V(@m1[vth])", "v(@m1[gm])"):
            text = refusal(plot, asked)
            assert "Per-device small-signal params" in text, text
            assert "net label" not in text, text
