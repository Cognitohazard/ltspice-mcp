"""Getting a deck ready to run: resolve, export, snapshot, and place its output.

Everything here answers "which file does the simulator actually read, and where
does it write" — the questions that sit between a caller's path argument and
``runner_base.submit_netlist``:

* ``resolve_netlist_path`` / ``resolve_runnable_netlist`` — validate the path,
  and export an ``.asc`` schematic to a runnable ``.net`` when that is what was
  handed in (serialized per schematic by ``asc_export_lock``, sanitized for
  ngspice when that is the target simulator).
* ``_stage_deck_snapshot`` — a content-addressed copy of the exported deck in
  the store, so a parallel session re-exporting the same schematic cannot swap
  the bytes out from under a run that already claimed them.
* ``export_netlist_text`` — the same export for a schematic copy the server
  owns, as text, with no sandbox check and no snapshot.

Split out of ``tools/_base`` so the run path stops being part of the module
every tool imports; it lives in ``lib`` because nothing here is MCP-shaped.
``lib/deck_staging`` is the neighbouring but distinct concern: staging a whole
manifest of includes for an experiment.
"""

import asyncio
import contextlib
import hashlib
import logging
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
from ltspice_mcp.errors import PathSecurityError, SimulationError
from ltspice_mcp.lib.encoding import (
    decode_spice_bytes,
    decode_spice_bytes_with_encoding,
    encode_spice_text,
    rewrite_codec,
)
from ltspice_mcp.lib.filelock import circuit_file_lock, path_lock
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.spice_lex import emit, lex
from ltspice_mcp.lib.spice_lex_ops import strip_instance_section_signs
from ltspice_mcp.lib.store import Store
from ltspice_mcp.lib.sweep_utils import sanitize_stem
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def resolve_netlist_path(netlist_str: str, state: SessionState) -> Path:
    """Resolve and validate a netlist path. Raises SimulationError on failure.

    PathSecurityError propagates unchanged: the caller attaches the sandbox
    guidance (the allowed paths, and the config line that widens them, which is
    re-read on the next call) — re-wrapping it as SimulationError would replace
    that guidance with a misdirecting simulator hint.
    """
    try:
        # ``tools._base.safe_path`` is this same call bound to the session; the
        # tool layer sits above this module, so bind it here instead.
        netlist_path = resolve_safe_path(netlist_str, state.allowed_paths())
    except PathSecurityError:
        raise
    except Exception as e:
        raise SimulationError(f"Invalid netlist path: {e}") from e
    if not netlist_path.exists():
        raise SimulationError(f"Netlist file not found: {netlist_path}")
    return netlist_path


# LTspice's ``create_netlist`` always writes the sidecar ``<name>.net`` next to
# the ``.asc``, so two concurrent exports of the same schematic would race on
# one output file (torn/partial reads of the deck). Serialize per resolved
# ``.asc`` path; distinct schematics still export in parallel.
_asc_export_locks: dict[Path, asyncio.Lock] = {}


@contextlib.asynccontextmanager
async def asc_export_lock(asc_path: Path) -> AsyncIterator[None]:
    """Serialize LTspice netlist exports of one schematic.

    In-process: a per-``.asc`` asyncio lock. Cross-process: the shared
    circuit file locks on BOTH the schematic and the sidecar ``.net`` —
    LTspice reads the ``.asc`` and overwrites the ``.net``, and a parallel
    session may be editing the ``.net`` itself under its own file lock.
    Fixed acquisition order (``.asc`` then ``.net``); edit paths take exactly
    one file lock, so no cycle is possible.
    """
    async with (
        path_lock(_asc_export_locks, asc_path),
        circuit_file_lock(asc_path),
        circuit_file_lock(asc_path.with_suffix(".net")),
    ):
        yield


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

    The scrub never rewrites the shared ``.net`` in place: a concurrent
    LTspice-target run of the same schematic regenerates it after the export
    lock releases, and an in-place rewrite would hand one of the two runs the
    other simulator's deck. It goes straight into the snapshot, under a stem
    naming the simulator.
    """
    data = net_path.read_bytes()
    if not for_ngspice:
        return net_path.stem, data
    text, encoding = decode_spice_bytes_with_encoding(data)
    cards = [card for card in lex(text).cards if card.body.strip().casefold() != ".backanno"]
    strip_instance_section_signs(cards)
    return f"{net_path.stem}.ngspice", encode_spice_text(emit(cards), rewrite_codec(encoding))


async def _export_schematic(
    asc_path: Path, state: SessionState, simulator: type | None
) -> tuple[str, bytes]:
    """Export ``asc_path`` with LTspice and read the deck back inside the export lock.

    Returns the stem the deck is named by and its bytes. They are read before
    the lock is released because ``create_netlist`` writes a shared
    ``<stem>.net``: a parallel session re-exporting this schematic overwrites
    it the moment the lock is free.

    ``simulator`` is the class the run will execute on (defaults to the
    session default): when it is ngspice, the LTspice export is sanitized
    for it (see ``_read_export``) — without that, every schematic run on
    ngspice dies on the exporter's ``.backanno``.

    The export launches the LTspice binary and blocks until it exits — heavy
    work that would stall the shared event loop, so it is offloaded via
    ``asyncio.to_thread``. It touches no cached editors, so the offload is safe
    under the concurrency contract.
    """
    ltspice_cls = state.available_simulators.get("ltspice")
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
    from ltspice_mcp.lib.simulator import is_ngspice

    for_ngspice = is_ngspice(simulator or state.default_simulator)
    async with asc_export_lock(asc_path):
        try:
            # Bound the export: create_netlist launches LTspice, which can hang
            # indefinitely on a Windows-side modal dialog. The export lock is
            # held across this call, so an unbounded hang wedges every later run
            # of this schematic — cap it at the sim timeout so it fails loudly.
            net_path = Path(
                await asyncio.to_thread(
                    ltspice_cls.create_netlist,
                    str(asc_path),
                    timeout=state.config.default_timeout,
                )
            )
        except Exception as e:
            raise SimulationError(
                f"Auto-exporting {asc_path.name} to a netlist failed: {e}"
            ) from e
        try:
            return await asyncio.to_thread(_read_export, net_path, for_ngspice=for_ngspice)
        except FileNotFoundError as e:
            raise SimulationError(f"Auto-export of {asc_path.name} produced no .net file") from e


async def resolve_runnable_netlist(
    netlist_str: str, state: SessionState, simulator: type | None = None
) -> Path:
    """Resolve a path AND auto-export ``.asc`` → ``.net`` if needed.

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
    """
    netlist_path = resolve_netlist_path(netlist_str, state)
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path
    stem, data = await _export_schematic(netlist_path, state, simulator)
    return await asyncio.to_thread(_stage_deck_snapshot, state.store, stem, data)


async def export_netlist_text(asc_path: Path, state: SessionState) -> str:
    """The netlist LTspice exports from a schematic the server owns, as text.

    For a copy the server wrote into its own store: it has already been
    admitted, so no sandbox check runs (the store need not sit inside
    ``allowed_paths``), and no snapshot is kept, because nothing will name it.
    """
    _stem, data = await _export_schematic(asc_path, state, None)
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
