"""Where a schematic's export goes on its way to a simulator.

LTspice's own ``<name>.net`` beside the schematic is its exporter's choice. What
the server keeps of it — the content-addressed snapshot a run claims, and the
ngspice-scrubbed twin — lives in the store, and the edit path keeps nothing.
The exporter here writes the sibling ``.net`` the way LTspice does; everything
after it is the real code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib.deck_prep import export_netlist_text, resolve_runnable_netlist
from ltspice_mcp.state import SessionState

_EXPORT = "* amp.asc\nV1 in 0 1\nR1 in 0 4.7µ\n.op\n.backanno\n.end\n"


def _exporting(state: SessionState) -> None:
    class _Exporter:
        @staticmethod
        def create_netlist(path: str, timeout: float | None = None) -> str:
            exported = Path(path).with_suffix(".net")
            exported.write_text(_EXPORT, encoding="utf-8")
            return str(exported)

    state.available_simulators["ltspice"] = _Exporter


def _schematic(work_dir: Path) -> Path:
    project = work_dir / "project"
    project.mkdir()
    sheet = project / "amp.asc"
    sheet.write_text("Version 4\n")
    return sheet


@pytest.mark.asyncio
async def test_the_ngspice_deck_goes_straight_into_the_store(
    state_no_sim: SessionState, work_dir: Path
):
    """No ``<name>.ngspice.net`` beside the schematic, and the scrub still happens."""
    _exporting(state_no_sim)
    sheet = _schematic(work_dir)

    deck = await resolve_runnable_netlist(str(sheet), state_no_sim, simulator=NGspiceSimulator)

    assert deck.parent == state_no_sim.store.exports_dir
    assert deck.name.startswith("amp.ngspice.run-")
    text = deck.read_text(encoding="utf-8")
    assert ".backanno" not in text
    assert "4.7u" in text
    assert sorted(p.name for p in sheet.parent.iterdir()) == ["amp.asc", "amp.net"]


@pytest.mark.asyncio
async def test_one_deck_content_is_one_snapshot(state_no_sim: SessionState, work_dir: Path):
    """Content-addressed: exporting an unchanged schematic again reuses the file."""
    _exporting(state_no_sim)
    sheet = _schematic(work_dir)

    first = await resolve_runnable_netlist(str(sheet), state_no_sim)
    second = await resolve_runnable_netlist(str(sheet), state_no_sim)

    assert first == second
    assert first.read_bytes() == _EXPORT.encode("utf-8")
    assert [p.name for p in state_no_sim.store.exports_dir.iterdir()] == [first.name]


@pytest.mark.asyncio
async def test_a_schematic_name_a_store_path_cannot_carry_is_respelled(
    state_no_sim: SessionState, work_dir: Path
):
    _exporting(state_no_sim)
    sheet = work_dir / "My Amp (v2).asc"
    sheet.write_text("Version 4\n")

    deck = await resolve_runnable_netlist(str(sheet), state_no_sim)

    assert deck.parent == state_no_sim.store.exports_dir
    assert deck.name.startswith("My_Amp_v2.run-")


@pytest.mark.asyncio
async def test_the_edit_path_export_keeps_no_snapshot(state_no_sim: SessionState, work_dir: Path):
    """``edit_schematic`` compares a throwaway copy's export; nothing names it later."""
    _exporting(state_no_sim)
    copy = state_no_sim.store.edit_export("build_1") / "committed.asc"
    copy.parent.mkdir(parents=True)
    copy.write_text("Version 4\n")

    text = await export_netlist_text(copy, state_no_sim)

    assert text == _EXPORT
    assert not state_no_sim.store.exports_dir.exists()
