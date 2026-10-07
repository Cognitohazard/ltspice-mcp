"""The plot settings file LTspice reads when it opens a results file.

A results file opened in LTspice's waveform viewer shows an empty plot: the
traces are whatever the person adds. What decides otherwise is a plot settings
file, the results file's own name with ``.plt`` in place of its suffix, in the
same directory, which LTspice loads as it opens the results. This module
writes one, so that a run shown in an LTspice window opens with the traces
that were asked for already drawn.

The file is text in sections, one per analysis, each named as the results file
names its plot (``[Transient Analysis]``, ``[AC Analysis]``), holding a count of
panes and, per pane, its traces::

    [Transient Analysis]
    {
       Npanes: 1
       {
          traces: 2 {524290,0,"V(out)"} {524291,0,"I(R1)"}
       }
    }

That is all this module writes. LTspice writes more when a person saves the
settings (the axis ranges, the grid, which pane is active), and what it writes
is the shape of the files it ships with its examples. Checked on 26.1.1 by
opening results files in a window and looking: a file holding only the traces
is read, the axes scale themselves, an expression of traces is taken as a
name is, eighteen traces numbered in sequence all draw, a section named for
another analysis draws nothing, and a results file that is already open keeps
the traces it had. What a window draws cannot be asked of it, so those are
observations and not a recording; ``docs/TESTING.md`` says so.

A settings file a person saved is theirs. One is replaced only when it holds
nothing but traces, which is what this module writes and LTspice does not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ltspice_mcp.lib import atomic_write_bytes

SUFFIX = ".plt"
# The number LTspice gives the first trace of a pane in the files it writes;
# the ones after it count up from there, and it picks each one's colour.
_FIRST_TRACE = 524290
_LEFT_AXIS = 0
# How LTspice writes its own text files: cp1252, with Windows line ends. The
# section name of a noise run holds a character outside ASCII.
_CODEC = "cp1252"
# Every line of a file that holds nothing but panes and their traces.
_ONLY_TRACES = re.compile(r"\[[^\]]*\]|\{|\},?|Npanes: \d+|traces: \d+( \{\d+,\d+,\"[^\"]*\"\})*")


class PlotSettingsError(ValueError):
    """A set of traces that cannot be written as a plot settings file."""


def settings_text(analysis: str, panes: Sequence[Sequence[str]]) -> str:
    """A plot settings file drawing ``panes``, each a list of trace names.

    ``analysis`` is the plot's name as the results file gives it. A name is a
    trace or an expression of traces, as LTspice's own Add Trace box takes it.
    """
    if not panes or not all(panes):
        raise PlotSettingsError("a plot needs at least one trace in every pane")
    lines = [f"[{analysis}]", "{", f"   Npanes: {len(panes)}"]
    for index, names in enumerate(panes):
        for name in names:
            if '"' in name or "\n" in name or "\r" in name:
                raise PlotSettingsError(f"{name!r} cannot be written as a trace name")
        traces = " ".join(
            f'{{{_FIRST_TRACE + offset},{_LEFT_AXIS},"{name}"}}'
            for offset, name in enumerate(names)
        )
        lines += ["   {", f"      traces: {len(names)} {traces}"]
        lines.append("   }" if index == len(panes) - 1 else "   },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def holds_only_traces(data: bytes) -> bool:
    """Whether a plot settings file is one this module wrote.

    LTspice writes each pane's axes and grid with its traces, so a file with
    none of that was not saved from a window.
    """
    try:
        text = data.decode(_CODEC)
    except UnicodeDecodeError:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_ONLY_TRACES.fullmatch(line) for line in lines)


def settings_path(results: Path) -> Path:
    """Where LTspice looks for the settings of the results file ``results``."""
    return results.with_suffix(SUFFIX)


def write_beside(results: Path, analysis: str, panes: Sequence[Sequence[str]]) -> str | None:
    """Write the settings that draw ``panes`` beside ``results``.

    Returns None when it is written, and otherwise why it was left alone: a
    settings file saved from LTspice is already there. Raises
    ``PlotSettingsError`` for traces that cannot be written and ``OSError``
    when the directory cannot be written to.
    """
    target = settings_path(results)
    try:
        encoded = settings_text(analysis, panes).replace("\n", "\r\n").encode(_CODEC)
    except UnicodeEncodeError as error:
        raise PlotSettingsError(
            f"a trace name holds a character LTspice's settings file cannot ({error.object[error.start]!r})"
        ) from error
    if target.is_file() and not holds_only_traces(target.read_bytes()):
        return (
            f"{target.name} was saved from LTspice and is left as it is, so the "
            "window shows the traces saved in it"
        )
    atomic_write_bytes(target, encoded, durable=False)
    return None
