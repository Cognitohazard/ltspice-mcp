"""Which compatibility mode ngspice read a deck in, read off the run's log.

The server reads every ngspice deck in the mode it passes ngspice: the
configured ``[simulator] ngbehavior`` (``simulator.current_ngbehavior``), a
recoverable job's recorded mode, or a native setup's policy. Each run starts
with ``-n``, so no user ``.spiceinit`` can replace that mode, but an ordinary
run still reads the installation's ``spinit`` for ngspice's code models, and a
``set ngbehavior`` there replaces it.

ngspice names the mode it read a circuit in, in its log, just before the
circuit's title, as flags (ngspice-42)::

    Note: Compatibility modes selected: ps lt ki a
    Note: No compatibility mode selected!

So after each run, ``check_run`` compares the flags the log names with the
flags the same ngspice gives the mode the server passed. It learns those by
asking that ngspice once per mode, with no startup file at all
(``flags_for``), rather than by modelling how ngspice reads a mode string.
A log that names no mode (the ``mc`` mode prints none, nor may an older
build) or an ngspice that cannot be asked leaves the run unchecked, and the
run says so.
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
"""How much of a log is searched for the note. A deck read through a driver
(a native setup, a seeded run) is named after the driver's own note, and a
PDK's model files can print many warnings before it."""

_PROBE_DECK = b"* compatibility mode probe\nR1 1 0 1\n.end\n"
_PROBE_TIMEOUT_S = 10.0
_CREATE_NO_WINDOW = 0x08000000

_flags_by_mode: dict[tuple[tuple[str, ...], str], tuple[str, ...]] = {}
_flags_lock = threading.Lock()


def logged_flags(log_text: str) -> tuple[str, ...] | None:
    """The flags of the mode the log says ngspice read its last circuit in:
    ``()`` for no mode, None when the log names none.

    The last, because a driver deck is read first and then reads the deck it
    runs (``source``), each with its own note.
    """
    notes = list(_NOTE.finditer(log_text))
    if not notes:
        return None
    flags = notes[-1]["flags"]
    return tuple(flags.split()) if flags is not None else ()


def flags_for(command: tuple[str, ...], mode: str) -> tuple[str, ...] | None:
    """The flags the ngspice that ``command`` starts gives ``mode``, or None
    when it cannot be asked.

    It is started as a run is, with ``-D ngbehavior=<mode>`` when ``mode`` is
    set, but with ``SPICE_SCRIPTS`` naming an empty folder and no other
    ``SPICE_`` variable, so no startup file can change the answer. An answer
    is kept for the life of the process; a failure to ask is not.
    """
    key = (command, mode)
    with _flags_lock:
        known = _flags_by_mode.get(key)
    if known is not None:
        return known
    argv = [*command, NGSPICE_NO_USER_INIT]
    if mode:
        argv += ["-D", f"ngbehavior={mode}"]
    argv.append("-b")
    environment = {
        name: value for name, value in os.environ.items() if not name.upper().startswith("SPICE_")
    }
    try:
        with tempfile.TemporaryDirectory(prefix="ngspice-mode-") as empty:
            environment["SPICE_SCRIPTS"] = empty
            completed = subprocess.run(
                argv,
                input=_PROBE_DECK,
                capture_output=True,
                timeout=_PROBE_TIMEOUT_S,
                cwd=empty,
                env=environment,
                creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                check=False,
            )
    # Any failure: a check that cannot run must not fail the run it checks.
    except Exception as exc:
        logger.warning("Could not ask %s which flags it gives mode %r: %s", command, mode, exc)
        return None
    flags = logged_flags(decode_spice_bytes(completed.stdout + completed.stderr))
    if flags is not None:
        with _flags_lock:
            _flags_by_mode[key] = flags
    return flags


@dataclass(frozen=True)
class ModeCheck:
    """How a run's mode compares with the mode the server passed."""

    status: Literal["agrees", "differs", "unchecked"]
    mode: str
    expected: tuple[str, ...] | None = None
    logged: tuple[str, ...] | None = None
    reason: str | None = None

    def evidence(self) -> dict[str, Any]:
        if self.status != "differs":
            return {"ngbehavior": self.mode}
        return {
            "ngbehavior": self.mode,
            "expected_modes": list(self.expected or ()),
            "logged_modes": list(self.logged or ()),
        }


def check_run(log_file: Path | None, simulator: type, mode: str) -> ModeCheck:
    """Whether ngspice read the run's deck in ``mode``, the mode the server
    passed it and reads the deck in.

    Never raises: a log that cannot be read, names no mode, or an ngspice
    that cannot be asked gives ``unchecked`` with the reason.
    """
    if log_file is None:
        return ModeCheck("unchecked", mode, reason="the run left no log")
    try:
        with log_file.open("rb") as handle:
            head = handle.read(_LOG_HEAD_BYTES)
    except OSError as exc:
        return ModeCheck("unchecked", mode, reason=f"its log could not be read: {exc}")
    logged = logged_flags(decode_spice_bytes(head))
    if logged is None:
        return ModeCheck("unchecked", mode, reason="ngspice's log names no compatibility mode")
    command = tuple(str(part) for part in getattr(simulator, "spice_exe", ()) or ())
    expected = flags_for(command, mode) if command else None
    if expected is None:
        return ModeCheck(
            "unchecked",
            mode,
            logged=logged,
            reason="the server could not ask ngspice which flags it gives that mode",
        )
    if logged == expected:
        return ModeCheck("agrees", mode, expected, logged)
    return ModeCheck("differs", mode, expected, logged)
