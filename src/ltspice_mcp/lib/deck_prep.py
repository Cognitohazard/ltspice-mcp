"""Getting a deck ready to run: resolve, export, snapshot, and place its output.

Everything here answers "which file does the simulator actually read, and where
does it write" — the questions that sit between a caller's path argument and
``runner_base.submit_netlist``:

* ``resolve_netlist_path`` / ``resolve_runnable_netlist`` — validate the path,
  and export an ``.asc`` schematic to a runnable ``.net`` when that is what was
  handed in (serialized per schematic by ``asc_export_lock``, sanitized for
  ngspice when that is the target simulator).
* ``_stage_deck_snapshot`` — a content-addressed copy of the exported deck, so
  a parallel session re-exporting the same schematic cannot swap the bytes out
  from under a run that already claimed them.

Split out of ``tools/_base`` so the run path stops being part of the module
every tool imports; it lives in ``lib`` because nothing here is MCP-shaped.
``lib/deck_staging`` is the neighbouring but distinct concern: staging a whole
manifest of includes for an experiment.
"""

import asyncio
import contextlib
import hashlib
import logging
import re
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
from ltspice_mcp.lib.filelock import circuit_file_lock, path_lock
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def resolve_netlist_path(netlist_str: str, state: SessionState) -> Path:
    """Resolve and validate a netlist path. Raises SimulationError on failure.

    PathSecurityError propagates unchanged: the dispatch layer has a dedicated
    branch that appends the sandbox-widening guidance (allowed paths, the TOML
    knob, the restart requirement) — re-wrapping it as SimulationError would
    replace that guidance with a misdirecting simulator hint.
    """
    try:
        # ``tools._base.safe_path`` is this same call bound to the session; the
        # tool layer sits above this module, so bind it here instead.
        netlist_path = resolve_safe_path(netlist_str, state.config.allowed_paths)
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


def _sanitize_export_for_ngspice(net_path: Path) -> Path:
    """Write an ngspice-runnable twin of an LTspice-exported netlist.

    LTspice's exporter appends ``.backanno`` — an LTspice-only dot command
    ngspice aborts on ("unimplemented dot command") — and can emit its
    private ``§`` name-prefix character and ``µ`` unit suffix, neither of
    which ngspice's parser accepts. The scrub goes to its own
    ``{stem}.ngspice.net`` sidecar rather than rewriting the shared ``.net``
    in place: a concurrent LTspice-target run of the same schematic
    regenerates ``.net`` after the export lock releases, and an in-place
    rewrite would hand one of the two runs the other simulator's deck.
    """
    from ltspice_mcp.lib import atomic_write_text
    from ltspice_mcp.lib.encoding import read_spice_text

    text = read_spice_text(net_path)
    lines = [ln for ln in text.splitlines() if ln.strip().lower() != ".backanno"]
    cleaned = "\n".join(lines).replace("§", "").replace("µ", "u").replace("μ", "u")
    out_path = net_path.with_name(net_path.stem + ".ngspice.net")
    atomic_write_text(out_path, cleaned + "\n", durable=False)
    return out_path


async def resolve_runnable_netlist(
    netlist_str: str, state: SessionState, simulator: type | None = None
) -> Path:
    """Resolve a path AND auto-export ``.asc`` → ``.net`` if needed.

    spicelib's ``SpiceEditor`` (used by the sweep / Monte Carlo runners)
    rejects ``.asc`` schematics — it expects the ``^*`` netlist comment
    header and otherwise fails with a cryptic ``Expected pattern "^\\*"
    not found``. This helper detects ``.asc`` and runs the LTspice
    ``create_netlist`` exporter to produce a sidecar ``.net``, so
    callers (sweep / MC config) can store the runnable path up front.

    ``simulator`` is the class the run will execute on (defaults to the
    session default): when it is ngspice, the LTspice export is sanitized
    for it (see ``_sanitize_export_for_ngspice``) — without that, every
    schematic run on ngspice dies on the exporter's ``.backanno``.

    The cheap safe_path/exists checks run inline, but the export launches the
    LTspice binary and blocks until it exits — heavy work that would stall the
    shared event loop, so it is offloaded via ``asyncio.to_thread``. It touches
    no cached editors, so the offload is safe under the concurrency contract.
    """
    netlist_path = resolve_netlist_path(netlist_str, state)
    if netlist_path.suffix.lower() != ".asc":
        return netlist_path

    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        # Don't recommend export_netlist here — it ALSO needs LTspice, so that
        # advice dead-ends when only ngspice/etc. is available.
        raise SimulationError(
            f"{netlist_path.name} is an .asc schematic, which only LTspice can "
            "convert to a netlist, and LTspice is not available "
            f"(simulators: {list(state.available_simulators.keys())}). Supply a "
            "hand-written .cir/.net to simulate with the current simulator, or "
            f"point the server at an LTspice executable ({_SIM_SECTION}.{_SIM_PATH_KEY} "
            f"in the config file or {_SIM_PATH_ENV}) and restart. (The .asc's "
            "embedded .model/.lib/analysis directives can be reused in a .cir.)",
            show_hint=False,
        )
    async with asc_export_lock(netlist_path):
        try:
            # Bound the export: create_netlist launches LTspice, which can hang
            # indefinitely on a Windows-side modal dialog. The export lock is
            # held across this call, so an unbounded hang wedges every later run
            # of this schematic — cap it at the sim timeout so it fails loudly.
            net_path = Path(
                await asyncio.to_thread(
                    ltspice_cls.create_netlist,
                    str(netlist_path),
                    timeout=state.config.default_timeout,
                )
            )
        except Exception as e:
            raise SimulationError(
                f"Auto-exporting {netlist_path.name} to a netlist failed: {e}"
            ) from e
        if not await asyncio.to_thread(net_path.exists):
            raise SimulationError(f"Auto-export of {netlist_path.name} produced no .net file")
        from ltspice_mcp.lib.simulator import is_ngspice

        if is_ngspice(simulator or state.default_simulator):
            net_path = await asyncio.to_thread(_sanitize_export_for_ngspice, net_path)

        # Snapshot the fresh deck INSIDE the lock and return THAT: create_netlist
        # writes a shared <stem>.net, so a parallel session re-exporting this .asc
        # overwrites it — and a caller that stored the shared path (sweep/MC
        # config, or a run staged moments later) would then read the peer's deck.
        # The snapshot stays in the same directory so a relative .include/.lib in
        # the deck still resolves against it.
        return await asyncio.to_thread(_stage_deck_snapshot, net_path)


def _stage_deck_snapshot(net_path: Path) -> Path:
    """Copy the exported deck to a content-addressed snapshot and return it.

    Named by a hash of its bytes so repeat exports of the same .asc reuse one
    file — the snapshots stay bounded to one per distinct deck content, not one
    per run (a plain per-call unique name accumulates unbounded). Written
    atomically so a concurrent reader sees a whole file, never a torn copy.

    It lands in the schematic's ``.ltspice-mcp/exports`` sidecar rather than
    beside the schematic: an experiment's replay identity names this file, so
    it has to persist for as long as the receipt does, and one visible
    ``<name>.run-<hash>.net`` per distinct edit accumulates in the author's
    tree forever. A deck carrying a RELATIVE include stays a sibling — the
    simulator resolves that include against the deck's own directory, so
    moving the deck breaks it.
    """
    from ltspice_mcp.lib import atomic_write_bytes

    data = net_path.read_bytes()
    digest = hashlib.sha1(data).hexdigest()[:12]
    name = f"{net_path.stem}.run-{digest}{net_path.suffix}"
    directory = net_path.parent
    if not _netlist_has_local_dependency(net_path):
        sidecar = net_path.parent / ".ltspice-mcp" / "exports"
        try:
            sidecar.mkdir(parents=True, exist_ok=True)
            directory = sidecar
        except OSError as exc:
            # A read-only or otherwise unusable sidecar must cost tidiness,
            # never the run: the sibling always works.
            logger.debug(
                f"Export sidecar {sidecar} unavailable, keeping snapshot beside deck: {exc}"
            )
    snapshot = directory / name
    if not snapshot.exists():
        atomic_write_bytes(snapshot, data, durable=False)
    return snapshot


# .include / .inc / .lib / .libfile <path> [extra]
_INCLUDE_DIRECTIVE_RE = re.compile(r"^\s*\.(?:include|inc|lib|libfile)\b\s+(.+)$", re.IGNORECASE)


def _first_path_token(rest: str) -> str:
    """First (possibly quoted) path token of an include/lib directive's args."""
    rest = rest.strip()
    if rest[:1] in ("'", '"'):
        end = rest.find(rest[0], 1)
        if end != -1:
            return rest[1:end]
    parts = rest.split()
    return parts[0] if parts else ""


def _netlist_has_local_dependency(netlist_path: Path) -> bool:
    """True if the netlist pulls in a sibling file via a *relative* .include/.lib.

    Such a netlist can't be relocated to the run sidecar: a simulator resolves a
    relative include against the (now-moved) netlist's own directory, so the
    dependency would no longer be found. Bare library NAMES resolved via the
    simulator's own lib path (no matching local file) and absolute paths both
    survive relocation and don't count.
    """
    from ltspice_mcp.lib.encoding import read_spice_text

    try:
        text = read_spice_text(netlist_path)
    except OSError:
        return True  # unreadable — be conservative, keep it in place
    base = netlist_path.parent
    for line in text.splitlines():
        m = _INCLUDE_DIRECTIVE_RE.match(line)
        if not m:
            continue
        tok = _first_path_token(m.group(1))
        if not tok:
            continue
        # Absolute (POSIX, Windows drive, or UNC) paths survive relocation.
        if Path(tok).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", tok) or tok.startswith("\\\\"):
            continue
        if (base / tok).exists():
            return True
    return False
