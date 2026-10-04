"""Which simulator build ran: the executable a job launches, and what each run reports.

Two facts, taken at two moments.

**The executable** is identified when a job is submitted: the program the
simulator class launches, with its size, modification time and content digest.
Every case of the job launches that program, and a later submission replaying
the job's ``request_id`` is compared against the program it would launch now
(``same_executable``), so results from one build are never handed back as the
answer for another. The digest is the build's identity; the path is where it
was found, and the same file moved elsewhere is still the same build.

**The reported build** is what each run says about itself in its own output,
read after the run ends (``reported_build``). LTspice 24 and later name their
version on the first line of the log (``LTspice 26.0.2 for Windows``); Xyce
opens the log it is told to write with a banner naming its release
(``***** This is version Xyce Release 7.8.0-opensource``); ngspice prints a
banner to its console, which the runner captures in the run's ``.exe.log``
(``** ngspice-42 : Circuit level simulation program``); and the raw header's
``Command:`` field names the writer where none of those does (LTspice XVII,
which writes no log banner, and QSPICE). It is recorded per case because a case
is the unit that ran: an executable replaced while a job is in flight shows up
as two builds in one job.

Everything read here is simulator output or a file the configuration points at,
so every read is bounded: a fixed number of bytes from the head of each
artifact, and no digest of an executable past ``_DIGEST_CAP_BYTES``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.encoding import decode_spice_bytes
from ltspice_mcp.lib.raw_parser import raw_writer_command

logger = logging.getLogger(__name__)

#: Bytes read from the head of a log or console capture when looking for its
#: banner. Each simulator writes it before anything the deck can make it print.
_HEAD_BYTES = 8 * 1024
#: Longest reported build kept. A banner is one short line; anything past this
#: is not one.
_REPORTED_CHARS = 160
#: Largest executable whose content is digested. Simulator executables are tens
#: of megabytes; a configured path far past that is not one, and hashing it
#: would stall a submission for nothing.
_DIGEST_CAP_BYTES = 512 * 1024 * 1024
#: Digests by executable, recomputed when the file's (mtime, size) stamp moves.
#: Single-flight per path, so concurrent first identifications hash it once.
_DIGESTS: FileCache[str] = FileCache(maxsize=32)

# LTspice 24+: the log's first line, e.g. "LTspice 26.0.2 for Windows".
_LTSPICE_BANNER = re.compile(r"LTspice[ \t]+\S[^\r\n]*")
# Xyce's log banner, e.g. "***** This is version Xyce Release 7.8.0-opensource".
_XYCE_BANNER = re.compile(r"^\*+[ \t]*This is version[ \t]+(Xyce\b[^\r\n]*)", re.MULTILINE)
# ngspice's console banner, e.g. "** ngspice-42 : Circuit level simulation program".
_NGSPICE_BANNER = re.compile(r"^\*\*[ \t]*(ngspice-[^\s:]+)", re.MULTILINE)
_NGSPICE_CREATED = re.compile(r"^\*\*[ \t]*Creation Date:[ \t]*(\S[^\r\n]*)", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
# LTspice XVII and earlier decode a deck as cp1252. Their executables are
# XVIIx64.exe / XVIIx86.exe (XVII) and scad3.exe (IV), and the raw header names
# the writer as "Linear Technology Corporation LTspice XVII". LTspice 24 and
# later are LTspice.exe and name themselves "LTspice 24.0.12 ...".
_CP1252_LTSPICE_EXECUTABLE = re.compile(r"(?i)(?:^|[\\/])(?:XVIIx(?:64|86)|scad3)\.exe$")
_CP1252_LTSPICE_BUILD = re.compile(r"(?i)\bLTspice\s+(?:XVII|IV)\b")


def is_cp1252_ltspice_executable(program: str) -> bool:
    """Whether ``program`` is the executable of an LTspice that decodes decks as cp1252.

    Matched on the file name in either path spelling, so a Windows path read on
    Linux (under Wine or WSL) is recognized too.
    """
    return _CP1252_LTSPICE_EXECUTABLE.search(program) is not None


def is_cp1252_ltspice_build(reported: str) -> bool:
    """Whether a build a run reported (``reported_build``) is an LTspice that
    decodes decks as cp1252: XVII or earlier."""
    return _CP1252_LTSPICE_BUILD.search(reported) is not None


@dataclass(frozen=True)
class SimulatorExecutable:
    """The program a simulator class launches, as found on disk.

    ``path`` is always known when the class names a program. The other three
    are None when the file could not be read: absent, not a regular file, or
    (``sha256`` only) larger than a simulator executable is.
    """

    path: str
    sha256: str | None
    bytes: int | None
    modified: str | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, data: Any) -> SimulatorExecutable | None:
        """Rebuild a recorded identity; None for anything that is not one."""
        if not isinstance(data, Mapping):
            return None
        path = data.get("path")
        if not isinstance(path, str) or not path:
            return None
        sha256 = data.get("sha256")
        size = data.get("bytes")
        modified = data.get("modified")
        return cls(
            path=path,
            sha256=sha256 if isinstance(sha256, str) and sha256 else None,
            bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
            modified=modified if isinstance(modified, str) and modified else None,
        )


def same_executable(
    recorded: SimulatorExecutable | None,
    current: SimulatorExecutable | None,
) -> bool:
    """Whether two identities name the same build.

    Equal digests are the same build wherever the file lives. Without a digest
    on both sides, the path, size and modification time must all agree. An
    identity on one side only is a difference: a record that cannot say what
    it ran on cannot be shown to match what runs now.
    """
    if recorded is None or current is None:
        return recorded is None and current is None
    if recorded.sha256 and current.sha256:
        return recorded.sha256 == current.sha256
    return (recorded.path, recorded.bytes, recorded.modified) == (
        current.path,
        current.bytes,
        current.modified,
    )


def describe_executable(identity: SimulatorExecutable | None) -> str:
    """One phrase naming an executable, for a message a caller reads."""
    if identity is None:
        return "an unrecorded executable"
    if identity.sha256:
        return f"{identity.path} (sha256 {identity.sha256[:12]})"
    if identity.modified:
        return f"{identity.path} (modified {identity.modified})"
    return identity.path


def executable_path(simulator_class: type | None) -> str | None:
    """The program a simulator class launches, or None when it names none.

    spicelib's ``spice_exe`` is the launch command, and its last element is the
    simulator: ``["wine", ".../LTspice.exe"]`` under Wine, a single element
    everywhere else (``Simulator.create_from`` requires the executable last). A
    bare command name, which is how spicelib records an ngspice it found on
    ``PATH``, is resolved through ``PATH`` the way the launch resolves it.
    """
    command = getattr(simulator_class, "spice_exe", None) or []
    if not command:
        return None
    program = str(command[-1])
    if not program:
        return None
    if not Path(program).is_absolute():
        found = shutil.which(program)
        if found:
            return found
    return str(Path(program))


def executable_identity(simulator_class: type | None) -> SimulatorExecutable | None:
    """Identify the program ``simulator_class`` launches. Blocking: call off the loop.

    The digest is computed once per (path, size, mtime) in a process, so
    identifying the same executable again costs a ``stat``.
    """
    program = executable_path(simulator_class)
    if program is None:
        return None
    try:
        info = os.stat(program)
    except OSError:
        info = None
    if info is None or not stat.S_ISREG(info.st_mode):
        return SimulatorExecutable(path=program, sha256=None, bytes=None, modified=None)
    return SimulatorExecutable(
        path=program,
        sha256=_digest(Path(program), info.st_size),
        bytes=info.st_size,
        # In the zone every other record timestamp is written in.
        modified=datetime.fromtimestamp(info.st_mtime, tz=now().tzinfo).isoformat(),
    )


def _digest(program: Path, size: int) -> str | None:
    if size > _DIGEST_CAP_BYTES:
        return None
    try:
        # A read that fails raises through the cache, so it is not remembered.
        return _DIGESTS.get(program, sha256_file)
    except OSError as exc:
        logger.debug("Could not digest simulator executable %s: %s", program, exc)
        return None


def reported_build(log_file: Path | None, raw_file: Path | None = None) -> str | None:
    """The build a run named in its own output, or None when it named none.

    Sources, first match wins: the LTspice banner on the log's first line; the
    Xyce banner in the head of the log; the ngspice banner in the console
    capture beside the log (``.exe.log``), with its creation date when the
    banner has one; the raw header's ``Command:``. The logs come before the raw
    so that a run which failed without a raw reports the same string as one
    that produced it.

    Never raises: an artifact that is missing or unreadable answers None.
    """
    try:
        if log_file is not None:
            log_head = _head_text(log_file)
            first_line = next(
                (line.strip() for line in log_head.splitlines() if line.strip()),
                "",
            )
            if _LTSPICE_BANNER.fullmatch(first_line):
                return _clean(first_line)
            xyce = _XYCE_BANNER.search(log_head)
            if xyce is not None:
                return _clean(xyce[1])
            console = _head_text(log_file.with_suffix(".exe.log"))
            banner = _NGSPICE_BANNER.search(console)
            if banner is not None:
                created = _NGSPICE_CREATED.search(console)
                if created is not None:
                    return _clean(f"{banner[1]}, Creation Date: {created[1]}")
                return _clean(banner[1])
        if raw_file is not None:
            command = raw_writer_command(raw_file)
            if command:
                return _clean(command)
    except (OSError, ValueError) as exc:
        logger.debug("Could not read the reported build for %s: %s", log_file or raw_file, exc)
    return None


def _head_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            head = handle.read(_HEAD_BYTES)
    except OSError:
        return ""
    return decode_spice_bytes(head)


def _clean(text: str) -> str | None:
    cleaned = _WHITESPACE.sub(" ", text).strip()[:_REPORTED_CHARS].strip()
    return cleaned or None
