"""Getting a deck ready to run: resolve, export, snapshot, and place its output.

Everything here answers "which file does the simulator actually read, and where
does it write" — the questions that sit between a caller's path argument and
``runner_base.submit_netlist``:

* ``resolve_runnable_netlist`` — export an ``.asc`` schematic, at a path the
  caller already resolved, to a runnable ``.net`` when that is what was
  handed in (from a copy staged in the store, so nothing is written beside
  the sheet; sanitized for ngspice when that is the target simulator).
* ``_stage_deck_snapshot`` — a content-addressed copy of the exported deck in
  the store, so a parallel session re-exporting the same schematic cannot swap
  the bytes out from under a run that already claimed them.
* ``export_netlist_text`` — the same export for a schematic copy the server
  owns, as text, with no sandbox check and no snapshot.
* ``stage_sheet`` — a copy of a sheet to export away from its folder, with
  the symbols and model files of that folder beside it; ``stage_sheet_copy``
  takes it under the sheet's circuit lock.

Split out of ``tools/_base`` so the run path stops being part of the module
every tool imports; it lives in ``lib`` because nothing here is MCP-shaped.
``lib/deck_staging`` is the neighbouring but distinct concern: staging a whole
manifest of includes for an experiment.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from ltspice_mcp.config import (
    SIM_PATH_ENV as _SIM_PATH_ENV,
)
from ltspice_mcp.config import (
    SIM_PATH_KEY as _SIM_PATH_KEY,
)
from ltspice_mcp.config import (
    SIM_SECTION as _SIM_SECTION,
)
from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib.encoding import (
    decode_spice_bytes,
    decode_spice_bytes_with_encoding,
    encode_spice_text,
    rewrite_codec,
)
from ltspice_mcp.lib.filelock import circuit_file_lock, path_lock
from ltspice_mcp.lib.spice_lex import emit, lex
from ltspice_mcp.lib.spice_lex_ops import strip_instance_section_signs
from ltspice_mcp.lib.store import Store
from ltspice_mcp.lib.sweep_utils import sanitize_stem
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


# LTspice's ``create_netlist`` always writes the sidecar ``<name>.net`` next to
# the ``.asc``, so two concurrent exports of the same schematic would race on
# one output file (torn/partial reads of the deck). verify_circuit's exports
# serialize per resolved ``.asc`` path; distinct schematics still export in
# parallel. An export of a copy in a folder of the call's own needs no lock.
_asc_export_locks: dict[Path, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def asc_export_lock(asc_path: Path) -> AsyncIterator[None]:
    """Serialize LTspice netlist exports of one schematic.

    In-process: a per-``.asc`` asyncio lock. Cross-process: the shared
    circuit file locks on BOTH the schematic and the sidecar ``.net`` —
    LTspice reads the ``.asc`` and overwrites the ``.net``, and a parallel
    session may be editing the ``.net`` itself under its own file lock.
    Fixed acquisition order (``.asc`` then ``.net``); an edit takes its
    circuit's file lock before any other (``edit_guard``), so no cycle is
    possible.
    """
    async with (
        path_lock(_asc_export_locks, asc_path),
        circuit_file_lock(asc_path),
        circuit_file_lock(asc_path.with_suffix(".net")),
    ):
        yield


#: The families that run the netlist LTspice exports from a schematic: LTspice
#: as exported, ngspice once ``_read_export`` has scrubbed it. The capability
#: table (``capabilities``) declares ``reads_asc_export`` a gap for every other
#: family, and a test holds the two to each other.
EXPORT_READERS: frozenset[str] = frozenset({"ltspice", "ngspice"})


def _read_export(net_path: Path, *, for_ngspice: bool) -> tuple[str, bytes]:
    """The deck a run of this export reads: the stem it is named by, and its bytes.

    For ngspice two things are changed first. LTspice's exporter appends
    ``.backanno``, an LTspice-only dot command ngspice aborts on
    ("unimplemented dot command"), and names an instance whose name does not
    start with its element letter with LTspice's private ``§`` (``R§Load``),
    which ngspice's parser does not accept. The card is dropped and the ``§``
    leaves those instance names (``strip_instance_section_signs``); comments,
    quoted strings and include paths keep every character, and the deck keeps
    its encoding. A micro-sign value suffix is left to staging, which spells it
    ``u`` for every simulator without touching a path.

    The scrubbed deck goes straight into the snapshot, under a stem naming
    the simulator; the exported ``.net`` is left as LTspice wrote it.
    """
    data = net_path.read_bytes()
    if not for_ngspice:
        return net_path.stem, data
    text, encoding = decode_spice_bytes_with_encoding(data)
    cards = [card for card in lex(text).cards if card.body.strip().casefold() != ".backanno"]
    strip_instance_section_signs(cards)
    return f"{net_path.stem}.ngspice", encode_spice_text(emit(cards), rewrite_codec(encoding))


def _exporter(asc_path: Path, state: SessionState, simulator: type | None) -> type:
    """The LTspice that exports ``asc_path`` for a run on ``simulator``.

    When ``simulator`` is an LTspice, that build exports the schematic: a named
    executable (LTspice XVII beside a later default) netlists the sheet against
    its own symbols and model library, as running the sheet in it would.
    Otherwise the session's LTspice does, and a session without one is refused.
    """
    from ltspice_mcp.lib.simulator import is_ltspice

    ltspice_cls = simulator if is_ltspice(simulator) else state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        # Don't recommend export_netlist here — it ALSO needs LTspice, so that
        # advice dead-ends when only ngspice/etc. is available.
        raise SimulationError(
            f"{asc_path.name} is an .asc schematic, which only LTspice can "
            "convert to a netlist, and LTspice is not available "
            f"(simulators: {list(state.available_simulators.keys())}). Supply a "
            "hand-written .cir/.net to simulate with the current simulator, or "
            f"point the server at an LTspice executable ({_SIM_SECTION}.{_SIM_PATH_KEY} "
            f"in the config file or {_SIM_PATH_ENV}) and restart. (The .asc's "
            "embedded .model/.lib/analysis directives can be reused in a .cir.)",
            show_hint=False,
        )
    return ltspice_cls


async def _export_copy(
    asc_path: Path, ltspice_cls: type, state: SessionState, simulator: type | None
) -> tuple[str, bytes]:
    """Export a sheet the server copied into a folder of its own, and read the deck.

    LTspice writes the deck to ``<stem>.net`` beside the sheet. Every caller
    exports a copy in a folder no other call writes to, so no export lock is
    taken. Returns the stem the deck is named by and its bytes.

    ``simulator`` is the class the run will execute on (defaults to the
    session default): when it is ngspice, the LTspice export is sanitized
    for it (see ``_read_export``) — without that, every schematic run on
    ngspice dies on the exporter's ``.backanno``. The LTspice that exports
    it is the one ``_exporter`` chose.

    The export launches the LTspice binary and blocks until it exits — heavy
    work that would stall the shared event loop, so it is offloaded via
    ``asyncio.to_thread``. It touches no cached editors, so the offload is safe
    under the concurrency contract.
    """
    from ltspice_mcp.lib.simulator import is_ngspice

    for_ngspice = is_ngspice(simulator or state.default_simulator)
    try:
        # Bound the export: create_netlist launches LTspice, which can hang
        # indefinitely on a Windows-side modal dialog; cap it at the sim
        # timeout so it fails loudly.
        net_path = Path(
            await asyncio.to_thread(
                ltspice_cls.create_netlist,
                str(asc_path),
                timeout=state.config.default_timeout,
            )
        )
    except Exception as e:
        raise SimulationError(f"Auto-exporting {asc_path.name} to a netlist failed: {e}") from e
    try:
        return await asyncio.to_thread(_read_export, net_path, for_ngspice=for_ngspice)
    except FileNotFoundError as e:
        raise SimulationError(f"Auto-export of {asc_path.name} produced no .net file") from e


async def stage_sheet_copy(asc_path: Path, dest: Path) -> tuple[Path, str | None]:
    """``stage_sheet`` for a sheet that may be being edited: its bytes are read
    under its circuit lock, so an edit in progress is all in the copy or not at
    all, and its folder is staged after the lock is released."""
    async with circuit_file_lock(asc_path):
        data = await asyncio.to_thread(asc_path.read_bytes)
    return await asyncio.to_thread(stage_sheet, asc_path, dest, data)


async def _export_schematic(
    asc_path: Path, state: SessionState, simulator: type | None
) -> tuple[str, bytes]:
    """Export the caller's sheet at ``asc_path`` from a copy staged in the store.

    A run writes nothing beside the caller's files: LTspice writes the deck it
    exports to ``<stem>.net`` beside the sheet, where it would replace a
    netlist the author keeps under that name. The copy carries the files of
    the sheet's folder its export reads (``stage_sheet_copy``). The staging
    folder is this call's alone and is removed after.
    """
    ltspice_cls = _exporter(asc_path, state, simulator)
    staging = state.store.run_export(
        f"{sanitize_stem(asc_path.stem) or 'sheet'}.{uuid.uuid4().hex[:12]}"
    )
    try:
        staged, note = await stage_sheet_copy(asc_path, staging)
        if note:
            logger.warning(note)
        return await _export_copy(staged, ltspice_cls, state, simulator)
    finally:
        await asyncio.to_thread(shutil.rmtree, staging, True)


async def resolve_runnable_netlist(
    netlist_path: Path, state: SessionState, simulator: type | None = None
) -> Path:
    """The deck a run of ``netlist_path`` reads: the path itself, or for an
    ``.asc`` its export. The path is one the caller resolved and admitted.

    spicelib's ``SpiceEditor`` (used by the sweep / Monte Carlo runners)
    rejects ``.asc`` schematics — it expects the ``^*`` netlist comment
    header and otherwise fails with a cryptic ``Expected pattern "^\\*"
    not found``. This helper detects ``.asc`` and runs the LTspice
    ``create_netlist`` exporter to produce a netlist, so callers (sweep / MC
    config) can store the runnable path up front.

    What comes back for a schematic is a content-addressed snapshot of the
    export in the store (``Store.exports_dir``), not the shared ``<stem>.net``
    LTspice writes: a caller that stored the shared path (sweep/MC config, or a
    run staged moments later) would otherwise read a parallel session's
    re-export. Staging resolves its relative includes beside the schematic
    (``deck_staging.stage_deck``).

    A schematic is refused, before LTspice is launched, for a simulator that
    cannot read LTspice's netlist dialect (``simulator.asc_export_refusal``).
    """
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path
    from ltspice_mcp.lib.simulator import asc_export_refusal

    refusal = asc_export_refusal(simulator or state.default_simulator)
    if refusal is not None:
        raise SimulationError(
            f"{netlist_path.name} is an .asc schematic. {refusal}", show_hint=False
        )
    stem, data = await _export_schematic(netlist_path, state, simulator)
    return await asyncio.to_thread(_stage_deck_snapshot, state.store, stem, data)


#: What a sheet's folder holds that LTspice reads when it exports the sheet: the
#: symbols it finds beside the sheet, in its folder or in a folder within it
#: that the symbol's name says (the ``symbol-beside-sheet`` recordings), and
#: the model files they name. ``stage_sheet`` also stages each hierarchical
#: block's own sheet (``export-hierarchical-block``).
_STAGED_SUFFIXES = frozenset({".asy", ".lib", ".sub", ".inc", ".mod"})
#: The most of those files staged for one export.
_STAGE_FILE_CAP = 200


def stage_sheet(asc_path: Path, dest: Path, data: bytes | None = None) -> tuple[Path, str | None]:
    """A copy of the sheet at ``asc_path`` in the fresh folder ``dest``, to export there.

    ``data`` is the sheet's bytes when the copy is to hold other bytes than
    the file's (a sheet about to be written). A copy exported anywhere but its
    own folder finds its symbols only when they are staged beside it, since
    LTspice looks for a symbol beside the sheet before its libraries, so what
    the sheet's folder holds for its export is copied too: the folders within
    it keep their places, and dot-folders are passed over. A block's own
    sheet, the ``.asc`` beside the ``.asy`` of its name, goes with its symbol;
    the sheet itself is only ever the copy made here. Returns the copy and,
    when :data:`_STAGE_FILE_CAP` left files out, a note saying so. Blocking;
    the caller takes any lock the sheet needs.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    staged = dest / asc_path.name
    if data is None:
        shutil.copy2(asc_path, staged)
    else:
        from ltspice_mcp.lib import atomic_write_bytes

        atomic_write_bytes(staged, data, durable=False)
    origin = asc_path.parent
    found: list[Path] = []
    blocks: set[tuple[str, str]] = set()
    sheets: list[tuple[str, str, str]] = []
    for folder, folders, files in os.walk(origin):
        folders[:] = [name for name in folders if not name.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            stem, suffix = os.path.splitext(name)
            suffix = suffix.lower()
            if suffix in _STAGED_SUFFIXES:
                found.append(Path(folder, name))
                if suffix == ".asy":
                    blocks.add((folder, stem.casefold()))
            elif suffix == ".asc" and Path(folder, name) != asc_path:
                sheets.append((folder, stem.casefold(), name))
    found += (Path(folder, name) for folder, stem, name in sheets if (folder, stem) in blocks)
    found = sorted(path for path in found if path.is_file())
    for src in found[:_STAGE_FILE_CAP]:
        target = dest / src.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    if len(found) <= _STAGE_FILE_CAP:
        return staged, None
    return staged, (
        f"only the first {_STAGE_FILE_CAP} project-local symbol/library files were staged "
        f"for the export of {asc_path.name}, so a symbol beyond that may not resolve"
    )


async def export_netlist_text(asc_path: Path, state: SessionState) -> str:
    """The netlist LTspice exports from a schematic the server owns, as text.

    For a copy the server wrote into its own store: it has already been
    admitted, so no sandbox check runs (the store need not sit inside
    ``allowed_paths``), and no snapshot is kept, because nothing will name it.
    """
    _stem, data = await _export_copy(asc_path, _exporter(asc_path, state, None), state, None)
    return decode_spice_bytes(data)


def _stage_deck_snapshot(store: Store, stem: str, data: bytes) -> Path:
    """Write an exported deck to a content-addressed snapshot and return it.

    Named by a hash of its bytes so repeat exports of the same .asc reuse one
    file — the snapshots stay bounded to one per distinct deck content, not one
    per run (a plain per-call unique name accumulates unbounded). Written
    atomically so a concurrent reader sees a whole file, never a torn copy.

    It lands in the store rather than beside the schematic: an experiment's
    replay identity names this file, so it has to persist for as long as the
    receipt does, and one ``<name>.run-<hash>.net`` per distinct edit would
    accumulate in the author's tree forever. The schematic's name survives in
    the snapshot's, folded the way a job id folds it.
    """
    from ltspice_mcp.lib import atomic_write_bytes

    digest = hashlib.sha1(data).hexdigest()[:12]
    snapshot = store.export_snapshot(f"{sanitize_stem(stem) or 'deck'}.run-{digest}.net")
    if not snapshot.exists():
        store.ensure_root()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(snapshot, data, durable=False)
    return snapshot
