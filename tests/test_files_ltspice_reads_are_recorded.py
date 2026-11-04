"""Each file the server writes for LTspice to read is one LTspice was recorded reading.

Design principle "LTspice's own tools as peers" (``docs/DESIGN.md``): the server
writes the files LTspice reads (a sheet's netlist, its ``.plt`` plot settings,
a run's results beside its sheet, the sheet itself) rather than imitating
LTspice. Such a file is right only as LTspice reads it, so each is tied here
to the behaviour in ``tests/fixtures/ltspice_recorded/inputs/cases.toml``
whose recordings show LTspice reading it ("Evidence before assumption").

A row names its writer by function, so moving or renaming the writer fails
here and the row is revisited; a writer with no behaviour, or whose behaviours
have no recording, fails too. Add a row for every new file the server writes
beside a sheet for LTspice to open.
"""

from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests import _ltspice_recorded as rec

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WrittenForLTspice:
    #: ``module:qualified.name`` of the function that writes or renders the file.
    writer: str
    #: The ``cases.toml`` behaviours whose recordings show LTspice reading it.
    behaviours: tuple[str, ...]


WRITTEN_FOR_LTSPICE: dict[str, WrittenForLTspice] = {
    # edit_schematic's set_plot_panes writes <sheet>.plt; each build was given
    # files the server wrote and saved what it read from them (plot/read_*).
    "plot settings beside a sheet (<sheet>.plt)": WrittenForLTspice(
        "ltspice_mcp.lib.plot_settings:write_plot_settings", ("plot-settings-read",)
    ),
    # Every sheet edit_schematic commits; the benchmark sheets were built by
    # edit_schematic and each build exported them (benchmark/*).
    "a sheet edit_schematic saved (.asc)": WrittenForLTspice(
        "ltspice_mcp.lib.asc_document:AscDocument.to_bytes", ("build-benchmark-export",)
    ),
    # jobs(action="place_beside_sheet"); the bridge recording shows a sheet
    # opening results that were beside it, and finding ones put there while it
    # was open once it is opened again.
    "a case's results beside its sheet (<sheet>.raw, .log)": WrittenForLTspice(
        "ltspice_mcp.lib.sheet_results:place_beside_sheet",
        ("sheet-results-command", "sheet-closed-from-outside"),
    ),
    # verify_circuit(export_to="sidecar") has LTspice export <sheet>.net in
    # place: the netlist each build writes beside a sheet it runs
    # (export/hier_top_run keeps it; a run in the window leaves it there).
    "the netlist beside a sheet (<sheet>.net)": WrittenForLTspice(
        "ltspice_mcp.tools.verify:_run_export", ("export-hierarchical-block", "window-run")
    ),
}


def _recordings(key: str) -> list[str]:
    """Where a behaviour is recorded: the builds whose manifests hold a case of
    it, and the recording its ``evidence`` names, when that is a file here."""
    found = sorted({build for build, _ in rec.per_build(rec.cases_of(key))})
    evidence = rec.CASES.behaviours[key].evidence
    named = evidence.split(":", 1)[0].strip()
    if named.startswith("tests/") and (REPO / named).exists():
        found.append(named)
    return found


@pytest.mark.parametrize("written", list(WRITTEN_FOR_LTSPICE))
def test_a_file_for_ltspice_is_one_ltspice_was_recorded_reading(written: str):
    row = WRITTEN_FOR_LTSPICE[written]
    assert callable(pkgutil.resolve_name(row.writer)), row.writer
    assert row.behaviours, f"{written}: no behaviour records LTspice reading it"
    for key in row.behaviours:
        assert key in rec.CASES.behaviours, f"{written}: {key!r} is not in cases.toml"
        assert _recordings(key), (
            f"{written}: {key!r} has no recording, from either build or the bridge"
        )


def test_an_unrecorded_behaviour_is_caught():
    # A behaviour that is only described, with no recording, fails the check.
    unrecorded = [
        key
        for key, behaviour in rec.CASES.behaviours.items()
        if behaviour.unrecordable and not rec.CASES.of(key)
    ]
    for key in unrecorded:
        assert not _recordings(key), key
