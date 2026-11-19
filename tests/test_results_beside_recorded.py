"""Where a build's batch run writes a netlist's files, read from the
recordings under ``tests/fixtures/ltspice_recorded``, against
``sheet_results.files_beside``, which names the files a placement copies to."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.raw_parser import OTHER_RAW_KINDS
from ltspice_mcp.lib.sheet_results import files_beside
from tests import _ltspice_recorded as rec


@pytest.mark.parametrize("build", rec.BUILDS)
def test_files_beside_names_what_a_netlists_batch_run_wrote(build: str):
    """Every case under raw/ is a netlist run with -Run -b in a folder of its
    own; what it wrote beside itself is what files_beside names."""
    checked = 0
    for case_id, entry in rec.manifest(build)["cases"].items():
        if not case_id.startswith("raw/") or entry.get("kind") != "run":
            continue
        deck = Path(f"{case_id.rsplit('/', 1)[-1]}.cir")
        beside = files_beside(deck)
        named = {beside.raw.name, beside.log.name}
        named |= {beside.other_raw(kind).name for kind in OTHER_RAW_KINDS}
        # Not copied: the .db LTspice 26 keeps of a run's measurements, and the
        # .log.raw both builds write beside a stepped run, which no job keeps.
        uncopied = {".db", ".log.raw"}
        written = {deck.stem + suffix for suffix in entry["written"] if suffix not in uncopied}
        assert beside.log.name in written, (case_id, written)
        assert written <= named, (case_id, written - named)
        checked += 1
    assert checked > 10


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_netlist_with_two_dots_writes_under_its_whole_stem(build: str):
    behaviour = rec.CASES.behaviours["results-beside-a-netlist-two-dots"]
    if behaviour.pending:
        pytest.skip(behaviour.pending)
    entry = rec.entry(build, "beside/two.dots")
    beside = files_beside(Path("two.dots.cir"))
    written = {"two.dots" + suffix for suffix in entry["written"]}
    assert {beside.raw.name, beside.log.name} <= written, written
