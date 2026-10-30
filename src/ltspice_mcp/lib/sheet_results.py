"""A run's results beside its sheet, where LTspice keeps a sheet's own.

LTspice writes a sheet's results beside it under its name (``<sheet>.raw``,
``<sheet>.log``, and ``<sheet>.op.raw`` or ``<sheet>.fra_1.raw`` where the run
wrote one) and opens them from there, so a click on a net in the sheet plots
it. The server's runs write theirs to the job folder instead, which LTspice
never looks in. This module copies one run's there on request: the results in
the job folder stay where they are, and remain what every read of the job
takes.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import atomic_write
from ltspice_mcp.lib.ltspice_window import results_beside
from ltspice_mcp.lib.raw_parser import other_raw_path
from ltspice_mcp.lib.simulator import dialect_for_simulator_name


def place_beside_sheet(
    sheet: Path,
    raw: Path,
    log: Path | None,
    other_raws: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Copy a run's results beside ``sheet`` under its name, as a run in
    LTspice leaves them; what is there is replaced, each file whole or not at
    all. Returns one ``{from, to, replaced}`` per file copied.

    A file of the sheet's that this run has no counterpart for (an
    ``.op.raw`` an earlier run left, say) is left where it is, and a file that
    already is the one beside the sheet is neither copied nor listed.
    """
    beside = results_beside(sheet)
    pairs = [(raw, beside), (log, sheet.with_suffix(".log"))]
    pairs += [(path, other_raw_path(beside, kind)) for kind, path in (other_raws or {}).items()]
    placed: list[dict[str, Any]] = []
    for source, target in pairs:
        if source is None or not source.is_file():
            continue
        replaced = target.exists()
        if replaced and os.path.samefile(source, target):
            continue
        with source.open("rb") as read, atomic_write(target, mode="wb", durable=False) as write:
            shutil.copyfileobj(read, write)
        placed.append({"from": str(source), "to": str(target), "replaced": replaced})
    return placed


def run_by_ltspice(simulator: str | None) -> bool:
    """Whether a run recorded as made by ``simulator`` can stand as a sheet's
    own in LTspice's window: an LTspice's, or one whose simulator is unknown.
    An ``.asc`` run on another simulator was only exported by LTspice."""
    return dialect_for_simulator_name(simulator) in (None, "ltspice")
