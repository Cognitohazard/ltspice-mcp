"""Which compatibility mode ngspice read a deck in, read off the run's log.

The server reads every ngspice deck in the mode it passes ngspice: the
configured ``[simulator] ngbehavior`` (``simulator.run_ngbehavior``), a
recoverable job's recorded mode, or a native setup's policy. Each run starts
with ``-n``, so no user ``.spiceinit`` can replace that mode, but an ordinary
run still reads the installation's ``spinit`` for ngspice's code models, and a
``set ngbehavior`` there replaces it.

ngspice names the mode it read a circuit in, in its log, just before the
circuit's title, as flags (ngspice-42)::

    Note: Compatibility modes selected: ps lt ki a
    Note: No compatibility mode selected!

So after each run, ``check_run`` compares the flags of the deck's note with
the flags the same ngspice gives the mode the server passed. It learns those
by running that ngspice once per mode on a deck of its own with no startup
file at all (``flags_for``), rather than by modelling how ngspice reads a mode
string. A log that names no mode for the deck (the ``mc`` mode prints none,
nor may an older build) or an ngspice that cannot be asked leaves the run
unchecked, and the run says so and why.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.simulator import NGSPICE_NO_USER_INIT

logger = logging.getLogger(__name__)

_NOTE = re.compile(
    r"^Note: (?:Compatibility modes selected:(?P<flags>[^\r\n]*)|No compatibility mode selected!)",
    re.MULTILINE,
)

_LOG_HEAD_BYTES = 1 << 20
"""How much of a log is searched for the deck's note. A PDK's model files can
print many warnings before it."""

_PROBE_DECK = b"* compatibility mode probe\nV1 1 0 1\nR1 1 0 1\n.op\n.end\n"
_PROBE_TIMEOUT_S = 10.0
_CREATE_NO_WINDOW = 0x08000000

_flags_by_mode: dict[tuple[tuple[str, ...], str], tuple[str, ...] | None] = {}
"""Each (command, mode) asked, with its answer, None when ngspice could not
be asked. A failure is kept too, so that no later run waits on it again."""
_flags_lock = threading.Lock()

Reason = Literal[
    "no_log", "log_unreadable", "no_mode_in_log", "deck_note_missing", "ngspice_not_asked"
]
"""Why a run's mode went unchecked, as its observation's evidence names it."""

_REASONS: dict[str, str] = {
    "no_log": "the run left no log",
    "log_unreadable": "its log could not be read",
    "no_mode_in_log": "ngspice's log names no compatibility mode",
    "deck_note_missing": (
        "ngspice's log names the driver's mode but none for the deck it sources, "
        "which it did not finish reading"
    ),
    "ngspice_not_asked": "the server could not ask ngspice which flags it gives that mode",
}


def mode_notes(log_text: str) -> list[tuple[str, ...]]:
    """The flags of every compatibility note in a log, in order; a note that
    says no mode was selected gives ``()``."""
    return [
        tuple(note["flags"].split()) if note["flags"] is not None else ()
        for note in _NOTE.finditer(log_text)
    ]


def deck_flags(log_text: str, *, driver: bool) -> tuple[str, ...] | None:
    """The flags of the note for the deck the run is about, or None when the
    log has none for it.

    In an ordinary run that is the first note: ngspice was started on the
    deck, and any later note is for a circuit the deck's own ``.control``
    read. A native setup or a seeded run starts ngspice on a driver that
    sources the deck, so the deck's note is the second, and is missing when
    ngspice could not read the deck.
    """
    notes = mode_notes(log_text)
    index = 1 if driver else 0
    return notes[index] if len(notes) > index else None


def flags_for(command: tuple[str, ...], mode: str) -> tuple[str, ...] | None:
    """The flags the ngspice that ``command`` starts gives ``mode``, or None
    when it cannot be asked.

    ngspice runs as a run does, on a deck file with an ``-o`` log, with
    ``-D ngbehavior=<mode>`` when ``mode`` is set, but in a folder of its own
    that ``SPICE_SCRIPTS`` names, which holds no ``spinit``, and with no other
    ``SPICE_`` variable, so no startup file can change the answer. The answer
    is kept for the life of the process, a failure to ask included.
    """
    key = (command, mode)
    with _flags_lock:
        if key in _flags_by_mode:
            return _flags_by_mode[key]
    flags = _ask(command, mode)
    with _flags_lock:
        _flags_by_mode[key] = flags
    return flags


def _ask(command: tuple[str, ...], mode: str) -> tuple[str, ...] | None:
    # Any failure: a check that cannot run must not fail the run it checks.
    try:
        log = run_batch(command, mode, _PROBE_DECK, clean_startup=True, timeout=_PROBE_TIMEOUT_S)
        return deck_flags(log, driver=False)
    except Exception as exc:
        logger.warning("Could not ask %s which flags it gives mode %r: %s", command, mode, exc)
        return None


def run_batch(
    command: tuple[str, ...],
    mode: str,
    deck: bytes,
    *,
    cwd: Path | None = None,
    clean_startup: bool,
    timeout: float,
) -> str:
    """Run ``deck`` through the ngspice ``command`` starts, in batch, and
    return its ``-o`` log; for the server's own questions to ngspice, not
    for runs.

    ngspice starts as a run does: ``-n``, ``-D ngbehavior=<mode>`` when
    ``mode`` is set, and a deck file and log in a folder of its own. With
    ``clean_startup`` it reads no startup file at all: ``SPICE_SCRIPTS``
    names that folder, which holds no ``spinit``, and no other ``SPICE_``
    variable is passed. Otherwise it reads the installation's ``spinit``, for
    its code models, as a run does. ``cwd`` is where it runs, the folder
    when None. Raises when ngspice cannot be started or writes no log.
    """
    environment = dict(os.environ)
    if clean_startup:
        environment = {
            name: value
            for name, value in environment.items()
            if not name.upper().startswith("SPICE_")
        }
    with tempfile.TemporaryDirectory(prefix="ngspice-ask-", ignore_cleanup_errors=True) as folder:
        path = Path(folder) / "probe.cir"
        path.write_bytes(deck)
        log = path.with_suffix(".log")
        argv = [*command, NGSPICE_NO_USER_INIT]
        if mode:
            argv += ["-D", f"ngbehavior={mode}"]
        argv += ["-b", "-o", log.as_posix(), path.as_posix()]
        if clean_startup:
            environment["SPICE_SCRIPTS"] = folder
        subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            cwd=cwd if cwd is not None else folder,
            env=environment,
            creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            check=False,
        )
        return decode_spice_bytes(log.read_bytes())


@dataclass(frozen=True)
class ModeCheck:
    """How a run's mode compares with the mode the server passed."""

    status: Literal["agrees", "differs", "unchecked"]
    mode: str
    expected: tuple[str, ...] | None = None
    logged: tuple[str, ...] | None = None
    reason: Reason | None = None

    def explanation(self) -> str:
        return _REASONS[self.reason] if self.reason else ""

    def evidence(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {"ngbehavior": self.mode}
        if self.status == "differs":
            evidence["expected_modes"] = list(self.expected or ())
        if self.reason is not None:
            evidence["reason"] = self.reason
        if self.logged is not None and self.status != "agrees":
            evidence["logged_modes"] = list(self.logged)
        return evidence


def check_run(log_file: Path | None, simulator: type, mode: str, *, driver: bool) -> ModeCheck:
    """Whether ngspice read the run's deck in ``mode``, the mode the server
    passed it and reads the deck in. ``driver`` says ngspice was started on
    a driver that sources the deck (``deck_flags``).

    Never raises: a log that cannot be read or names no mode for the deck,
    or an ngspice that cannot be asked, gives ``unchecked`` with the reason.
    """
    if log_file is None:
        return ModeCheck("unchecked", mode, reason="no_log")
    try:
        with log_file.open("rb") as handle:
            text = decode_spice_bytes(handle.read(_LOG_HEAD_BYTES))
    except Exception:
        return ModeCheck("unchecked", mode, reason="log_unreadable")
    notes = mode_notes(text)
    logged = deck_flags(text, driver=driver)
    if logged is None:
        # The driver's own note says nothing about the deck's mode.
        reason: Reason = "deck_note_missing" if notes else "no_mode_in_log"
        return ModeCheck("unchecked", mode, reason=reason)
    command = tuple(str(part) for part in getattr(simulator, "spice_exe", ()) or ())
    expected = flags_for(command, mode) if command else None
    if expected is None:
        return ModeCheck("unchecked", mode, logged=logged, reason="ngspice_not_asked")
    if logged == expected:
        return ModeCheck("agrees", mode, expected, logged)
    return ModeCheck("differs", mode, expected, logged)
