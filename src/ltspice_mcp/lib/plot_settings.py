"""LTspice plot settings files (``.plt``): reading one, and writing panes into one.

A ``.plt`` beside a schematic is what LTspice's waveform window reads when the
sheet runs: which panes to draw, and which traces in each. Everything here
about how LTspice writes and reads the file is recorded, on LTspice 26 and on
LTspice XVII, by running a sheet in the window with the file beside it and
saving what the window then shows (the LTspice recordings among the test
fixtures, behaviours ``plot-settings`` and ``plot-settings-read``; the cases
are named below):

- The file is a list of sections, one per analysis, each headed by the plot
  name the analysis gives its raw file (``[Transient Analysis]``,
  ``[AC Analysis]``) and holding that analysis's panes. A build saving one
  section keeps the others as they were (``plot/read_two_sections``).
- A section lists its panes bottom first: the pane LTspice 26 adds below
  another is written before it (``plot/pane_below``), and the one either
  build adds above is written after it (``plot/pane_added``). Everything this
  module hands a caller is top first, the order the window shows.
- A trace is ``{id,axis,"expression"}``. Both builds work the id and the axis
  out for themselves when they read the file, so a file written with 0 for
  both comes back with each build's own values (``plot/read_two_panes``).
- A trace is read up to its first space: ``"V(in) - V(out)"`` comes back as
  ``V(in)`` (``plot/read_spaced``), so an expression with whitespace in it is
  refused rather than written.
- A pane's ``Log`` line is the scale of its X axis, its left Y axis and its
  right Y axis: 0 linear, 1 logarithmic, 2 decibels. A pane a build makes
  itself has 0 0 0 in a transient section (``plot/one_trace``) and 1 2 0 in an
  AC one (``plot/ac``); both builds keep the line they read
  (``plot/read_log_y``, ``plot/read_ac``). A pane written here always carries
  it, with the build's own default unless the caller names a scale.
- The axis ranges (``X:``, ``Y[0]:`` ...) are not written: a run of the sheet
  ranges every axis to its data (``plot/read_two_panes``).
- LTspice 26 writes the file in UTF-8 and LTspice XVII in UTF-16 LE, neither
  with a byte order mark, and both end lines with LF alone
  (``plot/one_trace``). Each reads the other's (``plot/read_two_panes``,
  ``plot/read_utf8``), but XVII saving over a UTF-8 file writes its own
  section and then the old file's bytes after it (``plot/read_utf8``), so a
  file in LTspice 26's form becomes a file in two encodings the first time
  XVII saves it. UTF-16 LE is the form neither build's own save damages, so it
  is the one written here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.encoding import decode_spice_bytes_strictly

PlotAnalysis = Literal["tran", "ac"]
XScale = Literal["linear", "log"]
YScale = Literal["linear", "log", "db"]

#: The section each analysis's panes are under: the plot name its raw file has.
SECTION_NAMES: dict[str, str] = {
    "tran": "Transient Analysis",
    "ac": "AC Analysis",
}

#: The ``Log`` line a build writes for a pane it made itself (``plot/one_trace``,
#: ``plot/ac``): what a pane gets when the caller names no scale.
DEFAULT_SCALES: dict[str, tuple[int, int, int]] = {
    "tran": (0, 0, 0),
    "ac": (1, 2, 0),
}

_X_SCALES: dict[str, int] = {"linear": 0, "log": 1}
_Y_SCALES: dict[str, int] = {"linear": 0, "log": 1, "db": 2}

#: How the file is written: the form LTspice XVII writes, which both builds read.
PLOT_ENCODING = "utf-16-le"

# Characters a trace expression cannot hold. The double quote ends the quoted
# expression in a trace entry and a brace opens or closes one; a line break
# ends the line the traces are on. Nothing recorded shows how a build reads any
# of them inside an expression, so none is written. Whitespace is refused on its
# own, with the recording that shows why.
_UNWRITABLE = re.compile(r'["{}\x00-\x1f\x7f]')

_TRACE = re.compile(r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]*)"\s*\}')
_LOG = re.compile(r"^\s*Log:\s*(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)
_TRACES = re.compile(r"^\s*traces:\s*\d+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class PlotPane:
    """One pane: its traces left to right, and the scales of its three axes.

    ``scales`` is the pane's ``Log`` line (X, left Y, right Y; 0 linear, 1
    logarithmic, 2 decibels), or None for a pane read from a file without one.
    """

    traces: tuple[str, ...]
    scales: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class PlotSection:
    """One analysis's entry: its name, its panes top first, and its text as read."""

    name: str
    panes: tuple[PlotPane, ...]
    body: str


@dataclass(frozen=True)
class PlotSettings:
    """A whole ``.plt``: its sections in file order."""

    sections: tuple[PlotSection, ...] = ()

    def section(self, name: str) -> PlotSection | None:
        return next((s for s in self.sections if s.name == name), None)


def plot_settings_path(sheet: Path) -> Path:
    """The ``.plt`` LTspice reads for ``sheet``: the sheet's name, beside it."""
    return sheet.with_suffix(".plt")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def decode_plot_settings(data: bytes) -> str:
    """The text of a ``.plt`` in either build's encoding (or with a byte order mark)."""
    try:
        text, _encoding = decode_spice_bytes_strictly(data)
    except UnicodeError as exc:
        raise NetlistError(f"the plot settings file is not text: {exc}") from exc
    return text


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _matching_brace(text: str, opening: int) -> int:
    """The index of the brace closing the one at ``opening``, quotes skipped."""
    depth = 0
    quoted = False
    for at in range(opening, len(text)):
        char = text[at]
        if char == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return at
    raise NetlistError("a brace in the plot settings file is never closed")


def _blocks(body: str) -> list[str]:
    """The text inside each top-level ``{...}`` of a section's body, in order."""
    blocks: list[str] = []
    at = 0
    while True:
        opening = body.find("{", at)
        if opening < 0:
            return blocks
        closing = _matching_brace(body, opening)
        blocks.append(body[opening + 1 : closing])
        at = closing + 1


def _pane(block: str) -> PlotPane:
    traces: list[str] = []
    for line in _TRACES.finditer(block):
        traces.extend(match.group(3) for match in _TRACE.finditer(line.group(1)))
    log = _LOG.search(block)
    scales = None if log is None else (int(log[1]), int(log[2]), int(log[3]))
    return PlotPane(traces=tuple(traces), scales=scales)


def parse_plot_settings(text: str) -> PlotSettings:
    """The sections of a ``.plt``, each with its panes top first.

    A section's panes are read from its own text; nothing else in it is
    interpreted, and ``body`` keeps that text so a section can be written back
    as it was.
    """
    sections: list[PlotSection] = []
    at = _skip_space(text, 0)
    while at < len(text):
        if text[at] != "[":
            raise NetlistError(
                "the plot settings file has text outside its sections at character "
                f"{at}; LTspice XVII leaves a file like that when it saves over one in "
                "LTspice 26's encoding"
            )
        end = text.find("]", at)
        if end < 0 or "\n" in text[at:end]:
            raise NetlistError("a section name in the plot settings file is not closed")
        name = text[at + 1 : end]
        opening = _skip_space(text, end + 1)
        if opening >= len(text) or text[opening] != "{":
            raise NetlistError(f"section [{name}] of the plot settings file has no body")
        closing = _matching_brace(text, opening)
        body = text[opening + 1 : closing]
        panes = tuple(_pane(block) for block in _blocks(body))
        sections.append(PlotSection(name=name, panes=panes[::-1], body=body))
        at = _skip_space(text, closing + 1)
    return PlotSettings(sections=tuple(sections))


def read_plot_settings(data: bytes) -> PlotSettings:
    """``parse_plot_settings`` of a ``.plt``'s bytes."""
    return parse_plot_settings(decode_plot_settings(data))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def check_trace(expression: str) -> str:
    """``expression`` if a ``.plt`` can hold it as a trace; raises otherwise."""
    if not expression.strip():
        raise NetlistError("a trace expression is empty")
    if any(char.isspace() for char in expression):
        raise NetlistError(
            f"trace {expression!r} has whitespace in it, and LTspice reads a trace "
            "from a plot settings file only up to its first space; write it without, "
            f"as {''.join(expression.split())!r}"
        )
    bad = _UNWRITABLE.search(expression)
    if bad is not None:
        raise NetlistError(
            f"trace {expression!r} holds {bad.group()!r}, which a plot settings file "
            "cannot carry inside a trace (a double quote, a brace or a control "
            "character)"
        )
    return expression


def scales_of(analysis: str, x_scale: str | None, y_scale: str | None) -> tuple[int, int, int]:
    """The ``Log`` line of a pane: the analysis's own default, with the named scales."""
    x, left, right = DEFAULT_SCALES[analysis]
    if x_scale is not None:
        x = _X_SCALES[x_scale]
    if y_scale is not None:
        left = _Y_SCALES[y_scale]
    return x, left, right


def scale_names(scales: tuple[int, int, int] | None) -> dict[str, str]:
    """``x_scale`` / ``y_scale`` for a pane's ``Log`` line, each one it has a name for."""
    if scales is None:
        return {}
    names: dict[str, str] = {}
    x = next((name for name, value in _X_SCALES.items() if value == scales[0]), None)
    y = next((name for name, value in _Y_SCALES.items() if value == scales[1]), None)
    if x is not None:
        names["x_scale"] = x
    if y is not None:
        names["y_scale"] = y
    return names


def render_section(name: str, panes: Sequence[PlotPane]) -> str:
    """The text of one section holding ``panes`` (top first), as LTspice lays it out."""
    lines = [f"[{name}]", "{", f"   Npanes: {len(panes)}"]
    written = list(panes)[::-1]
    for index, pane in enumerate(written):
        entries = " ".join(f'{{0,0,"{check_trace(trace)}"}}' for trace in pane.traces)
        scales = pane.scales if pane.scales is not None else (0, 0, 0)
        lines += [
            "   {",
            f"      traces: {len(pane.traces)} {entries}",
            "      Log: {} {} {}".format(*scales),
            "   }," if index < len(written) - 1 else "   }",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def with_panes(settings: PlotSettings, analysis: str, panes: Sequence[PlotPane]) -> PlotSettings:
    """``settings`` with ``analysis``'s section holding ``panes``; none removes it.

    The section keeps its place among the others; a new one goes last. Its
    ``body`` is the text written for it, so ``render_plot_settings`` needs
    nothing else.
    """
    name = SECTION_NAMES[analysis]
    if any(not pane.traces for pane in panes):
        raise NetlistError("every pane needs at least one trace")
    replacement: PlotSection | None = None
    if panes:
        text = render_section(name, panes)
        body = text[text.index("{") + 1 : text.rindex("}")]
        replacement = PlotSection(name=name, panes=tuple(panes), body=body)
    kept: list[PlotSection] = []
    placed = False
    for section in settings.sections:
        if section.name != name:
            kept.append(section)
        elif not placed:
            placed = True
            if replacement is not None:
                kept.append(replacement)
    if not placed and replacement is not None:
        kept.append(replacement)
    return PlotSettings(sections=tuple(kept))


def render_plot_settings(settings: PlotSettings) -> str:
    """The text of a whole ``.plt``: each section as read, or as written here."""
    return "".join(f"[{s.name}]\n{{{s.body}}}\n" for s in settings.sections)


def encode_plot_settings(text: str) -> bytes:
    """``text`` as the bytes of a ``.plt``: UTF-16 LE, no byte order mark, LF line ends."""
    return text.replace("\r\n", "\n").encode(PLOT_ENCODING)
