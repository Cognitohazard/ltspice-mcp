"""Scoped simulator-process termination by job-id token.

spicelib's ``kill_all_spice`` is name-global — it would terminate every
simulator process on the machine, including ones launched by a *different*
MCP server running in parallel in the same directory. (With the installed
spicelib it also matches nothing at all: it reads the base ``Simulator``
class's empty ``process_name``, so cancel/timeout silently failed to kill
the simulator everywhere the WSL taskkill path didn't apply.)

This module replaces it with a match that can only hit a job's own
process(es): every staged run netlist embeds the uuid-unique job id in its
filename (``run_filename=f"{job_id}{ext}"`` — see sim_runner and
``batch_run_filename``), so the id appears in the simulator's command line.
A process dies only when its command line carries the token AND it is
recognizably the simulator's executable.

The WSL + LTspice case is different — there the simulator is a *Windows*
process, invisible to the Linux psutil table; ``wsl.kill_windows_ltspice_by_token``
covers it with the same token idea via PowerShell + taskkill. Both kills are
attempted by the runners; on any given platform at most one finds a match.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from pathlib import PurePath

import psutil

logger = logging.getLogger(__name__)


def simulator_executable_names(simulator_class: type) -> frozenset[str]:
    """Candidate executable basenames for a spicelib simulator class.

    Draws from ``spice_exe`` (the launch argv — under Wine that is
    ``["wine", ".../LTspice.exe"]``, so both basenames are included) and
    ``process_name`` when the class sets one. Lower-cased for matching.
    """
    names: set[str] = set()
    for part in getattr(simulator_class, "spice_exe", None) or []:
        name = PurePath(str(part)).name.lower()
        if name:
            names.add(name)
    process_name = str(getattr(simulator_class, "process_name", "") or "")
    if process_name:
        names.add(process_name.lower())
    return frozenset(names)


def _token_in_arg(token: str, arg: str) -> bool:
    """True when ``token`` appears at a run-filename boundary in ``arg``.

    Staged run files are ``{job_id}.{ext}`` (single runs) or
    ``{job_id}_{n}.{ext}`` (batch sub-runs), so the id is always followed by
    ``.`` or ``_`` — or ends the argument. Requiring that boundary keeps a
    job id from matching a longer id it happens to prefix. (Generated ids
    are fixed-length per class, so a proper prefix can't occur today; the
    anchor makes the match safe rather than reliant on that invariant.)
    """
    return re.search(re.escape(token) + r"(?:[._]|$)", arg) is not None


def kill_simulator_by_token(token: str, executable_names: Collection[str]) -> int:
    """Kill local simulator processes whose command line carries ``token``.

    A process is killed only if BOTH hold:

    - some command-line argument contains ``token`` (the uuid-unique job id,
      present because the staged netlist filename embeds it) at a filename
      boundary (see ``_token_in_arg``), and
    - the process is the simulator: its name — or the basename of one of its
      first two argv entries (the Wine case: argv is ``wine …/LTspice.exe``)
      — is in ``executable_names``.

    The name gate is what keeps this safe against incidental token matches
    (e.g. the server's own WSL PowerShell interop helpers carry the token in
    their command line but are never named like a simulator).

    Best-effort: per-process psutil errors are skipped. Returns the number
    of processes killed.
    """
    wanted = {n.lower() for n in executable_names if n}
    if not token or not wanted:
        return 0
    killed = 0
    for proc in psutil.process_iter(("name", "cmdline")):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not any(_token_in_arg(token, arg) for arg in cmdline):
                continue
            candidates = {(proc.info.get("name") or "").lower()}
            candidates.update(PurePath(arg).name.lower() for arg in cmdline[:2])
            if not (candidates & wanted):
                continue
            proc.kill()
            killed += 1
            logger.info(
                "Killed simulator process %d (%s) carrying token %s",
                proc.pid,
                proc.info.get("name"),
                token,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed
