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
from tests._fake_netlister import install_fixed_exporter

_EXPORT = "* amp.asc\nV1 in 0 1\nR1 in 0 4.7µ\n.op\n.backanno\n.end\n"


def _schematic(folder: Path, name: str = "amp.asc") -> Path:
    sheet = folder / name
    sheet.write_text("Version 4\n")
    return sheet


@pytest.mark.asyncio
async def test_the_ngspice_deck_goes_straight_into_the_store(
    state_no_sim: SessionState, project_dir: Path
):
    """No ``<name>.ngspice.net`` beside the schematic, and the scrub still happens."""
    install_fixed_exporter(state_no_sim, _EXPORT)
    sheet = _schematic(project_dir)

    deck = await resolve_runnable_netlist(str(sheet), state_no_sim, simulator=NGspiceSimulator)

    assert deck.parent == state_no_sim.store.exports_dir
    assert deck.name.startswith("amp-ngspice.run-")
    text = deck.read_text(encoding="utf-8")
    assert ".backanno" not in text
    # A value's micro sign is staging's to spell 'u', for every simulator.
    assert "R1 in 0 4.7µ\n" in text
    assert sorted(p.name for p in project_dir.iterdir()) == ["amp.asc", "amp.net"]  # noqa: ASYNC240


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ["utf-8", "cp1252"])
async def test_the_ngspice_scrub_changes_only_instance_names_and_backanno(
    state_no_sim: SessionState, project_dir: Path, codec: str
):
    """The '§' leaves the instance names LTspice put it in, wherever the deck
    names them, and nothing else: a comment, a quoted string and an include
    path keep every character, and the deck keeps its encoding."""
    export = (
        "* amp.asc § rev 2\n"
        '.include "models µ§/core.inc"\n'
        "V1 in 0 1\n"
        "R§Load in 0 4.7k ; §pnba A)B\n"
        ".meas op iload find I(R§Load)\n"
        '.param note="R§Load"\n'
        ".op\n"
        ".backanno\n"
        ".end\n"
    )
    install_fixed_exporter(state_no_sim, export, encoding=codec)
    sheet = _schematic(project_dir)

    deck = await resolve_runnable_netlist(str(sheet), state_no_sim, simulator=NGspiceSimulator)

    assert deck.read_bytes() == export.replace(".backanno\n", "").replace(
        "R§Load in 0", "RLoad in 0"
    ).replace("I(R§Load)", "I(RLoad)").encode(codec)


@pytest.mark.asyncio
async def test_one_deck_content_is_one_snapshot(state_no_sim: SessionState, project_dir: Path):
    """Content-addressed: exporting an unchanged schematic again reuses the file."""
    install_fixed_exporter(state_no_sim, _EXPORT)
    sheet = _schematic(project_dir)

    first = await resolve_runnable_netlist(str(sheet), state_no_sim)
    second = await resolve_runnable_netlist(str(sheet), state_no_sim)

    assert first == second
    assert first.read_bytes() == _EXPORT.encode("utf-8")
    assert [p.name for p in state_no_sim.store.exports_dir.iterdir()] == [first.name]


@pytest.mark.asyncio
async def test_a_schematic_name_a_store_path_cannot_carry_is_folded(
    state_no_sim: SessionState, project_dir: Path
):
    install_fixed_exporter(state_no_sim, _EXPORT)
    sheet = _schematic(project_dir, "My Amp (v2).asc")

    deck = await resolve_runnable_netlist(str(sheet), state_no_sim)

    assert deck.parent == state_no_sim.store.exports_dir
    assert deck.name.startswith("my-amp-v2.run-")


@pytest.mark.asyncio
async def test_the_edit_path_export_keeps_no_snapshot(state_no_sim: SessionState):
    """``edit_schematic`` compares a throwaway copy's export; nothing names it later."""
    install_fixed_exporter(state_no_sim, _EXPORT)
    copy = state_no_sim.store.edit_export("build_1") / "committed.asc"
    copy.parent.mkdir(parents=True)
    copy.write_text("Version 4\n")

    text = await export_netlist_text(copy, state_no_sim)

    assert text == _EXPORT
    assert not state_no_sim.store.exports_dir.exists()
