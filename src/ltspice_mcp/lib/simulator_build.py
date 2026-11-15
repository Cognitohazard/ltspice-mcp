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
import mmap
import os
import re
import shutil
import stat
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.encoding import decode_spice_bytes

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
# The file names LTspice XVII (XVIIx64.exe / XVIIx86.exe) and IV (scad3.exe)
# install under. LTspice 24 and later install as LTspice.exe, a name an XVII
# can be given too, so that name shows nothing.
_STOCK_BEFORE_24 = re.compile(r"(?i)(?:^|[\\/])(XVIIx(?:64|86)|scad3)\.exe$")
# How a run names an LTspice before 24: XVII's raw header names its writer
# "Linear Technology Corporation LTspice XVII"; LTspice 24 and later open their
# log with "LTspice 26.1.1 for Windows".
_REPORTED_BEFORE_24 = re.compile(r"(?i)\bLTspice\s+(XVII|IV)\b")
_REPORTED_NUMBERED = re.compile(r"(?i)\bLTspice\s+(\d+)\.")
#: The first release numbered after the year, which reads its decks as UTF-8.
_FIRST_CURRENT = 24
#: The signature that opens a version resource's VS_FIXEDFILEINFO.
_FIXED_FILE_INFO = struct.pack("<I", 0xFEEF04BD)
_RT_VERSION = 16
#: File versions by executable, reread when the file's (mtime, size) stamp moves.
_FILE_VERSIONS: FileCache[tuple[int, int, int, int] | None] = FileCache(maxsize=32)

LtspiceGeneration = Literal["iv", "xvii", "current"]


@dataclass(frozen=True)
class Generation:
    """Which LTspice an executable is, and what shows it.

    ``generation`` is ``"xvii"`` (LTspice XVII), ``"iv"`` (LTspice IV) or
    ``"current"`` (LTspice 24 and later). ``evidence`` names what shows it,
    for a finding or a note to cite.
    """

    generation: LtspiceGeneration
    evidence: str

    @property
    def before_24(self) -> bool:
        """An LTspice before 24 decodes a deck as cp1252, prints each
        semiconductor's operating point in the log of every ``.op`` run and
        ends the run with "unrecognized option" when a deck asks for it with
        ``.options logopinfo``, and writes each subcircuit instance's name
        without its X."""
        return self.generation != "current"


def ltspice_generation(program: str | None, reported: str | None) -> Generation | None:
    """Which LTspice ``program`` is, or None when nothing shows it.

    The one answer to that question, for everything that turns on it: the
    deck a run is handed, how a deck's and a result's names are spelled, which
    log wording is read, which encoding a deck is read in and which library a
    run stages against. Evidence, the first that answers:

    1. ``reported``, the build a run on this executable named in its own
       output (``reported_build``): what the program says it is.
    2. The version resource of the program file itself (its file version:
       17 for XVII, 24 and later for the current builds), read from the file
       without launching it, on any platform (``executable_file_version``).
    3. The file name LTspice XVII or IV installs under (``XVIIx64.exe``,
       ``XVIIx86.exe``, ``scad3.exe``).

    ``LTspice.exe`` alone shows nothing: an XVII can carry that name. A
    program none of these speak for (a wrapper script, a copy whose version
    resource is gone, a build with no run yet on a platform whose executables
    carry no version resource) answers None, and a caller says the generation
    is not established instead of assuming one. Call off the loop: it may read
    the program file.
    """
    if reported:
        generation = _reported_generation(reported)
        if generation is not None:
            evidence = f"{reported} ({program})" if program else reported
            return Generation(generation, evidence)
    if not program:
        return None
    version = executable_file_version(program)
    if version is not None:
        generation = _numbered_generation(version[0])
        if generation is not None:
            return Generation(
                generation, f"{program} (file version {'.'.join(map(str, version))})"
            )
    stock = _STOCK_BEFORE_24.search(program)
    if stock is not None:
        return Generation("iv" if stock[1].lower() == "scad3" else "xvii", program)
    return None


def is_cp1252_ltspice_build(reported: str) -> bool:
    """Whether a build a run reported (``reported_build``, or a raw header's
    ``Command``) is an LTspice before 24, which decodes decks as cp1252: the
    answer :func:`ltspice_generation` gives from that report alone."""
    found = ltspice_generation(None, reported)
    return found is not None and found.before_24


def _reported_generation(reported: str) -> LtspiceGeneration | None:
    named = _REPORTED_BEFORE_24.search(reported)
    if named is not None:
        return "iv" if named[1].upper() == "IV" else "xvii"
    # A number establishes only a current build here: XVII names itself in no
    # log banner, so a report of "LTspice 17.x" is not one this server has
    # seen, and is left unestablished.
    numbered = _REPORTED_NUMBERED.search(reported)
    if numbered is not None and int(numbered[1]) >= _FIRST_CURRENT:
        return "current"
    return None


def _numbered_generation(major: int) -> LtspiceGeneration | None:
    # Recorded: XVII's file version is 17.0.37.0, LTspice 26's 26.1.1.0
    # (tests/fixtures/ltspice_recorded/*/manifest.json, "executable").
    if major == 17:
        return "xvii"
    if major >= _FIRST_CURRENT:
        return "current"
    return None


def executable_file_version(program: str) -> tuple[int, int, int, int] | None:
    """The file version in ``program``'s own version resource, the four numbers
    Windows reports for it (``GetFileVersionInfo``, VS_FIXEDFILEINFO), or None
    for a file that is not a Windows executable or carries no version resource.

    Read from the file, so it answers on Linux, under Wine and on WSL as on
    Windows. Only :func:`ltspice_generation` reads it (``pyproject.toml`` bans
    it elsewhere): a version alone is not an answer to which build a program
    is. Blocking.
    """
    path = Path(program)
    try:
        return _FILE_VERSIONS.get(path, _read_file_version)
    except OSError:
        return None


def _read_file_version(path: Path) -> tuple[int, int, int, int] | None:
    from ltspice_mcp.lib.pe_menu import resources

    try:
        with (
            path.open("rb") as file,
            mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as image,
        ):
            blobs = resources(image, _RT_VERSION)
    except (ValueError, struct.error, IndexError):
        # Empty, not a Windows executable, or a resource table it cannot hold.
        return None
    for blob in blobs:
        at = blob.find(_FIXED_FILE_INFO)
        if at < 0 or len(blob) < at + 16:
            continue
        ms, ls = struct.unpack_from("<II", blob, at + 8)
        return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    return None


@dataclass(frozen=True)
class Producer:
    """Who reads a deck or wrote a result: the simulator family, the build a
    run named in its own output (``reported_build``), and the program that
    ran or will run.

    ``family`` is a dialect name (``ltspice``, ``ngspice``, ``qspice``,
    ``xyce``) or None when it is not known; ``build`` and ``program`` are None
    when not known. The log's rules (``log_conditions.rules_for``) and the
    names a build spells (``device_names.naming_for``) are read off it.
    """

    family: str | None
    build: str | None
    program: str | None


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
            # Here, not at the top: the raw parser names a result's Producer,
            # which this module defines.
            from ltspice_mcp.lib.raw_parser import raw_writer_command

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
