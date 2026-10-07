"""Record what installed LTspice builds do with a fixed set of minimal inputs.

The suite's model of LTspice (where a rotated symbol's pins land, which wires
connect, how a netlist export is spelled, how a deck's bytes are read, what a
raw file and a log look like) is checked against files LTspice itself wrote.
This module produces those files: it runs every case listed in
``fixtures/ltspice_recorded/inputs/cases.toml`` on a build and stores what the
build wrote under ``fixtures/ltspice_recorded/<build>/``, with a
``manifest.json`` saying which executable produced it and from which inputs.

Three rules shape it.

**The command lines are the server's.** A run is ``<exe> -Run -b <deck>`` and
an export is ``<exe> -netlist <sheet>``, as spicelib launches them. The one
addition is a trailing ``-ini <file>``: a recording must not depend on the
settings of the person recording, so each case runs against a copy of the
build's settings file with the keys that change a result removed
(``BEHAVIOUR_KEYS``), which leaves the build on its own defaults. Two things
about that switch were learned the hard way. It goes after the input: given
first, LTspice 26 exits 0 having run nothing, and LTspice XVII opens its
window instead of running the batch. And the copy is never an empty file: a
build that starts with one behaves as on first launch, and LTspice XVII then
runs its updater.

**Nothing from the recording machine is kept.** Logs, exports and raw headers
embed the working directory, the home directory, the date and how long the run
took. ``scrub`` rewrites those to fixed values, in the encoding the file is
in, and ``assert_private`` refuses to finish if a user name, a host name or a
local path survived. Because the rewrite is deterministic, recording the same
build twice gives the same bytes, so a re-record that changes a file is a
change in what LTspice did.

**A recording is compared, not trusted.** ``compare`` sets a fresh recording
against a committed one; the opt-in LTspice tier calls it, so a release that
changes behaviour fails there by name.

The recorder drives LTspice natively and is Windows-only. It starts each run
on a desktop of its own (``HiddenDesktop``), because LTspice otherwise takes
the keyboard focus for as long as a run lasts. Everything that reads a
recording (``load_manifest``, ``recorded``, ``compare``) works anywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.wintypes as wintypes
import fnmatch
import hashlib
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ltspice_recorded"
INPUTS = FIXTURES / "inputs"
CASES_FILE = "cases.toml"
MANIFEST = "manifest.json"
MANIFEST_SCHEMA = 1

#: What a recording shows in place of the directory a case ran in, and of the
#: recording user's home directory. Both pass the repository's privacy scan.
NEUTRAL_DIR = "C:\\recording"
NEUTRAL_HOME = "C:\\Users\\user"
#: The date every recorded ``Date:`` / ``Start Time:`` line carries. A two-digit
#: day, so the builds' different padding of a one-digit day does not arise.
NEUTRAL_DATE = "Thu Jan 15 00:00:00 2026"

#: Settings that change what a run or an export produces. They are removed from
#: the copy of the settings file a case runs against, so the build falls back
#: to its own default for each; a case sets one back with ``ini = {...}``.
BEHAVIOUR_KEYS = frozenset(
    key.casefold()
    for key in (
        "Solver",
        "GenerateExpandedListing",
        "DefaultTrtol",
        "DefaultTrapIntegration",
        "DefaultDoAntiTrapRinging",
        "TSKbypass",
        "Accept_3k4_Notation",
        "NoJFETtempAdjIsr",
        "AsciiRawFile",
        "DoRelinearization",
        "DoQuadraticWaves",
        "RelinIAbsTol",
        "RelinRelTol",
        "RelinVAbsTol",
        "RelinearWindowPnts",
        "CompressTranOnly",
        "SaveDeviceCurrents",
        "SaveSubcircuitNodeVoltages",
        "SaveSubcircuitDeviceCurrents",
        "saveOneCurrentPerDevice",
        "DirectCompPinShorts",
        "MinInductorDamping",
        "UseClocktoReseedMC",
        "EnableBetaOptimizations",
        "DefaultDeviceModels",
        "DefaultDeviceLibraries",
        "NoGreekMus",
        "RadianMeasure",
        "SymbolSearchPath",
        "LibrarySearchPath",
        "SymbolSearchPathDisabled",
        "LibrarySearchPathDisabled",
        "DisableSchSubdirPlaceComp",
        "UseRawTempDir",
        "RawTempDir",
        "WarnOnNoIndRser",
        "AutoDeleteRawFiles",
        "FastAccessRAM",
    )
)

#: Output suffixes kept per case kind when the case does not name its own.
DEFAULT_KEEP: Mapping[str, tuple[str, ...]] = {
    "run": ("log", "raw"),
    "netlist": ("net",),
    "kill": ("log", "raw"),
    "fastaccess": ("raw",),
}

_RAW_SUFFIXES = (".raw", ".op.raw")


class RecorderError(RuntimeError):
    """The recorder could not produce, or refused to keep, a recording."""


# --------------------------------------------------------------------------
# Builds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Build:
    """One installed LTspice executable.

    ``label`` names the directory its recordings live in: ``ltspice`` and the
    major version (``ltspice26``; XVII is ``ltspice17``). ``generation`` is the
    server's own split (``lib/simulator.py:generation_of``): ``xvii`` keeps its
    settings and library in other places than every later build does.
    """

    exe: Path
    label: str
    generation: str
    file_version: str | None

    @property
    def settings_file(self) -> Path | None:
        """The per-user settings file the build reads, if the user has one."""
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        name = "LTspiceXVII.ini" if self.generation == "xvii" else "LTspice.ini"
        path = Path(appdata) / name
        return path if path.is_file() else None

    @property
    def library_root(self) -> Path | None:
        """The per-user library directory the build unpacks and reads."""
        if self.generation == "xvii":
            root = Path.home() / "Documents" / "LTspiceXVII" / "lib"
        else:
            local = os.environ.get("LOCALAPPDATA")
            if not local:
                return None
            root = Path(local) / "LTspice" / "lib"
        return root if root.is_dir() else None


def file_version(exe: Path) -> str | None:
    """The executable's version resource (``26.1.1.0``), or None off Windows.

    LTspice XVII names its version in no file it writes, so the resource is the
    only record of which XVII a recording came from.
    """
    if sys.platform != "win32":
        return None

    class FixedFileInfo(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint32) for name in ("signature", "struct_version", "ms", "ls")
        ]

    api = ctypes.WinDLL("version", use_last_error=True)
    size = api.GetFileVersionInfoSizeW(str(exe), None)
    if not size:
        return None
    buffer = ctypes.create_string_buffer(size)
    if not api.GetFileVersionInfoW(str(exe), 0, size, buffer):
        return None
    info = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not api.VerQueryValueW(buffer, "\\", ctypes.byref(info), ctypes.byref(length)):
        return None
    if not info.value or length.value < ctypes.sizeof(FixedFileInfo):
        return None
    fixed = ctypes.cast(info.value, ctypes.POINTER(FixedFileInfo)).contents
    return f"{fixed.ms >> 16}.{fixed.ms & 0xFFFF}.{fixed.ls >> 16}.{fixed.ls & 0xFFFF}"


def identify_build(exe: Path, label: str | None = None) -> Build:
    """Describe the build at ``exe``; ``label`` overrides the derived one."""
    exe = Path(exe)
    if not exe.is_file():
        raise RecorderError(f"not an executable file: {exe.name}")
    generation = "xvii" if "xvii" in exe.name.casefold() else "current"
    version = file_version(exe)
    if label is None:
        if version is None:
            raise RecorderError(
                f"cannot read a version from {exe.name}; name the build with --label"
            )
        label = f"ltspice{version.split('.', 1)[0]}"
    if not re.fullmatch(r"[a-z0-9_]+", label):
        raise RecorderError(f"a build label is lowercase letters, digits and '_': {label!r}")
    return Build(exe=exe, label=label, generation=generation, file_version=version)


def _candidate_executables() -> list[Path]:
    """Where an LTspice install puts its executable, most recent layout first."""
    env = os.environ
    found: list[Path] = [
        Path(item) for item in env.get("LTSPICE_MCP_RECORDER_EXES", "").split(os.pathsep) if item
    ]
    if env.get("LOCALAPPDATA"):
        found.append(Path(env["LOCALAPPDATA"]) / "Programs" / "ADI" / "LTspice" / "LTspice.exe")
    for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root = env.get(variable)
        if not root:
            continue
        found.append(Path(root) / "ADI" / "LTspice" / "LTspice.exe")
        found.append(Path(root) / "LTC" / "LTspiceXVII" / "XVIIx64.exe")
    return found


def discover_builds() -> list[Build]:
    """Every LTspice build installed here, one per label, in a stable order.

    ``LTSPICE_MCP_RECORDER_EXES`` (paths joined by the platform's path
    separator) names builds installed anywhere else, and wins over the
    standard locations for a label both provide.
    """
    if sys.platform != "win32":
        return []
    builds: dict[str, Build] = {}
    seen: set[str] = set()
    for exe in _candidate_executables():
        key = os.path.normcase(str(exe))
        if key in seen or not exe.is_file():
            continue
        seen.add(key)
        try:
            build = identify_build(exe)
        except RecorderError:
            continue
        builds.setdefault(build.label, build)
    return [builds[label] for label in sorted(builds)]


def unavailable_reason(build: Build) -> str | None:
    """Why ``build`` cannot be recorded here, or None when it can."""
    if sys.platform != "win32":
        return "the recorder drives LTspice natively and runs only on Windows"
    if build.settings_file is None:
        return (
            f"{build.exe.name} has no settings file yet; start it once so it writes one "
            "(a build launched without one runs its first-start steps instead of the deck)"
        )
    return None


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One input and what to do with it.

    ``case_id`` is the recording's path under a build directory, without a
    suffix (``raw/tran_binary``). ``source`` is the input it runs, relative to
    the inputs directory, and ``extra`` the files copied beside it. The work
    copy is named for the case, not for the source, so two cases can run one
    input differently.
    """

    case_id: str
    behaviour: str
    kind: str
    source: str
    extra: tuple[str, ...] = ()
    switches: tuple[str, ...] = ()
    keep: tuple[str, ...] = ()
    ini: Mapping[str, str] = field(default_factory=dict)
    kill_after_bytes: int | None = None
    kill_after_s: float | None = None
    volatile: bool = False
    settings: bool = True
    builds: tuple[str, ...] = ()
    note: str = ""

    @property
    def stem(self) -> str:
        return self.case_id.rsplit("/", 1)[-1]

    @property
    def work_name(self) -> str:
        return self.stem + Path(self.source).suffix

    def kept(self) -> tuple[str, ...]:
        return self.keep or DEFAULT_KEEP[self.kind]

    def applies_to(self, build: Build) -> bool:
        return not self.builds or build.generation in self.builds or build.label in self.builds


@dataclass(frozen=True)
class Behaviour:
    """One LTspice behaviour the server models, and what records it.

    ``model`` names the code that encodes the behaviour. A behaviour with no
    case says why in ``evidence`` (the manifest records it another way) or in
    ``unrecordable`` (nothing can).
    """

    key: str
    summary: str
    model: tuple[str, ...]
    evidence: str = ""
    unrecordable: str = ""


@dataclass(frozen=True)
class CaseFile:
    behaviours: Mapping[str, Behaviour]
    cases: tuple[Case, ...]

    def case(self, case_id: str) -> Case:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def of(self, behaviour: str) -> tuple[Case, ...]:
        return tuple(case for case in self.cases if case.behaviour == behaviour)


def _kind_of(source: str) -> str:
    return "netlist" if source.endswith(".asc") else "run"


def load_cases(inputs: Path = INPUTS) -> CaseFile:
    """Read and validate the case list beside the inputs."""
    with (inputs / CASES_FILE).open("rb") as handle:
        data = tomllib.load(handle)
    behaviours: dict[str, Behaviour] = {}
    cases: list[Case] = []
    for key, entry in data.get("behaviour", {}).items():
        behaviours[key] = Behaviour(
            key=key,
            summary=entry["summary"],
            model=tuple(entry.get("model", ())),
            evidence=entry.get("evidence", ""),
            unrecordable=entry.get("unrecordable", ""),
        )
        cases.extend(
            Case(
                case_id=source.rsplit(".", 1)[0],
                behaviour=key,
                kind=_kind_of(source),
                source=source,
                extra=tuple(entry.get("extra", ())),
                keep=tuple(entry.get("keep", ())),
            )
            for source in entry.get("inputs", ())
        )
    for entry in data.get("case", []):
        cases.append(
            Case(
                case_id=entry["id"],
                behaviour=entry["behaviour"],
                kind=entry.get("kind", _kind_of(entry["source"])),
                source=entry["source"],
                extra=tuple(entry.get("extra", ())),
                switches=tuple(entry.get("switches", ())),
                keep=tuple(entry.get("keep", ())),
                ini={str(k): str(v) for k, v in entry.get("ini", {}).items()},
                kill_after_bytes=entry.get("kill_after_bytes"),
                kill_after_s=entry.get("kill_after_s"),
                volatile=bool(entry.get("volatile", False)),
                settings=bool(entry.get("settings", True)),
                builds=tuple(entry.get("builds", ())),
                note=entry.get("note", ""),
            )
        )
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise RecorderError(f"duplicate case id {case.case_id!r}")
        seen.add(case.case_id)
        if case.kind not in DEFAULT_KEEP:
            raise RecorderError(f"{case.case_id}: unknown kind {case.kind!r}")
        if case.behaviour not in behaviours:
            raise RecorderError(f"{case.case_id}: unknown behaviour {case.behaviour!r}")
        for name in (case.source, *case.extra):
            if not (inputs / name).is_file():
                raise RecorderError(f"{case.case_id}: missing input {name}")
    return CaseFile(behaviours=behaviours, cases=tuple(cases))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


#: How an 8-bit settings file is read and written back. Its real code page is
#: the recording machine's, which need not be cp1252; Latin-1 maps every byte
#: to one character and back, and the keys edited here are ASCII.
_ANSI = "latin-1"


def _settings_lines(text: str) -> list[str]:
    """The lines of a settings file, split at line feeds only.

    ``str.splitlines`` also breaks at characters such as U+0085, which a byte
    of a double-byte code page becomes when it is read as Latin-1.
    """
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    return lines[:-1] if lines and not lines[-1] else lines


def neutral_settings(source: bytes, overrides: Mapping[str, str]) -> bytes:
    """``source`` (a settings file's bytes) without the keys that change results.

    The file keeps its encoding: LTspice 26 writes it as UTF-16 with a byte
    order mark, earlier builds in the ANSI code page. The list of recently
    opened files goes too, since a batch run appends to it. ``overrides`` are
    written back into ``[Options]``.
    """
    utf16 = source.startswith(b"\xff\xfe")
    text = source[2:].decode("utf-16-le") if utf16 else source.decode(_ANSI)
    out: list[str] = []
    section = ""
    pending = dict(overrides)

    def flush_overrides() -> None:
        out.extend(f"{key}={value}" for key, value in pending.items())
        pending.clear()

    for line in _settings_lines(text):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == "options":
                flush_overrides()
            section = stripped[1:-1].casefold()
            if section != "recent file list":
                out.append(line)
            continue
        if section == "recent file list":
            continue
        if section == "options":
            key = stripped.split("=", 1)[0].strip().casefold()
            if key in BEHAVIOUR_KEYS or key in {k.casefold() for k in overrides}:
                continue
        out.append(line)
    if section == "options":
        flush_overrides()
    if pending:  # a file with no [Options] section at all
        out.append("[Options]")
        flush_overrides()
    rendered = "\r\n".join(out) + "\r\n"
    return b"\xff\xfe" + rendered.encode("utf-16-le") if utf16 else rendered.encode(_ANSI)


def read_settings(source: bytes, keys: Iterable[str]) -> dict[str, str]:
    """The ``[Options]`` values of ``keys`` in a settings file, by the file's spelling."""
    utf16 = source.startswith(b"\xff\xfe")
    text = source[2:].decode("utf-16-le") if utf16 else source.decode(_ANSI)
    wanted = {key.casefold() for key in keys}
    found: dict[str, str] = {}
    section = ""
    for line in _settings_lines(text):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].casefold()
        elif section == "options" and "=" in stripped:
            key, value = stripped.split("=", 1)
            if key.strip().casefold() in wanted:
                found[key.strip()] = value.strip()
    return found


# --------------------------------------------------------------------------
# Scrubbing
# --------------------------------------------------------------------------

_DATE = re.compile(
    r"(?m)^((?:Date|Start Time):[ \t]*)[A-Z][a-z]{2} [A-Z][a-z]{2} [ \d]\d \d\d:\d\d:\d\d \d{4}"
)
_ELAPSED = re.compile(r"(?m)^(Total elapsed time:[ \t]*)[\d.]+( seconds)")
_THREADS = re.compile(r"(?m)^(Maximum thread count:[ \t]*)\d+")
#: LTspice XVII times two matrix compilers and reports the one it kept
#: ("Matrix Compiler2: 175 bytes object code size  0.0/0.0/[0.0]", or "off").
#: Which one wins varies between runs of one deck, so the report is not kept.
_COMPILER_REPORT = re.compile(r"(?m)^(Matrix Compiler\d:)[^\r\n]*")
NEUTRAL_COMPILER_REPORT = " (timing-dependent)"
_RAW_MARKERS = tuple(f"\n{word}:\n" for word in ("Binary", "Values"))


def _path_pattern(path: str) -> re.Pattern[str]:
    """``path`` in any spelling a tool echoes it: either separator, any case."""
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    return re.compile(r"[\\/]+".join(re.escape(part) for part in parts), re.IGNORECASE)


def ansi_codec() -> str:
    """The codec this machine writes 8-bit text in: on Windows, its ANSI code page."""
    return "mbcs" if sys.platform == "win32" else locale.getpreferredencoding(False)


def _eight_bit_codecs(ansi: str | None) -> tuple[str, ...]:
    """The codecs a local path is looked for in when the file is 8-bit.

    LTspice 26 writes UTF-8. LTspice XVII writes cp1252, whatever the
    machine's code page is. The machine's own code page is searched as well:
    what this module removes and refuses should not rest on every release
    having behaved as the two recorded ones do.
    """
    return ("utf-8", "cp1252", ansi or ansi_codec())


def _spellings(path: str, ansi: str | None = None) -> set[str]:
    """``path`` as it reads in a file decoded as Latin-1, for a non-ASCII path."""
    spelled = {path}
    for codec in _eight_bit_codecs(ansi):
        with contextlib.suppress(UnicodeError, LookupError):
            spelled.add(path.encode(codec).decode("latin-1"))
    # LTspice XVII writes a question mark for each character cp1252 lacks, so
    # a folder named in Chinese reaches its export as question marks.
    spelled.add(path.encode("cp1252", "replace").decode("latin-1"))
    return spelled


@dataclass(frozen=True)
class Scrubber:
    """Rewrites what a file carries from the machine and moment it was made.

    ``paths`` maps a local directory to what stands in for it, longest first,
    so a working directory inside the home directory is replaced as a whole.
    ``ansi`` is the codec the recording machine writes 8-bit text in, this
    machine's when it is not given.
    """

    paths: tuple[tuple[str, str], ...]
    ansi: str | None = None

    @classmethod
    def for_run(
        cls, work_dir: Path, home: Path | None = None, ansi: str | None = None
    ) -> Scrubber:
        pairs = [(str(work_dir), NEUTRAL_DIR)]
        home = home if home is not None else Path.home()
        pairs.append((str(home), NEUTRAL_HOME))
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return cls(paths=tuple(pairs), ansi=ansi)

    def text(self, text: str) -> str:
        for local, neutral in self.paths:
            for spelled in sorted(_spellings(local, self.ansi), key=len, reverse=True):

                def replace(match: re.Match[str], neutral: str = neutral) -> str:
                    separator = "/" if "/" in match.group(0) else "\\"
                    return neutral.replace("\\", separator)

                text = _path_pattern(spelled).sub(replace, text)
        text = _DATE.sub(rf"\g<1>{NEUTRAL_DATE}", text)
        text = _ELAPSED.sub(r"\g<1>0.000\2", text)
        text = _THREADS.sub(r"\g<1>1", text)
        return _COMPILER_REPORT.sub(rf"\g<1>{NEUTRAL_COMPILER_REPORT}", text)

    def bytes(self, data: bytes, name: str) -> bytes:
        """``data`` scrubbed in its own encoding; a raw file's samples untouched."""
        if name.endswith(_RAW_SUFFIXES):
            head, payload = split_raw(data)
            if _looks_utf16(head):
                return self._utf16(head) + payload
            return self.text(head.decode("latin-1")).encode("latin-1") + payload
        if data.startswith(b"\xff\xfe"):
            return b"\xff\xfe" + self._utf16(data[2:])
        if _looks_utf16(data):
            return self._utf16(data)
        return self.text(data.decode("latin-1")).encode("latin-1")

    def _utf16(self, data: bytes) -> bytes:
        even = len(data) - len(data) % 2
        text = data[:even].decode("utf-16-le", errors="surrogatepass")
        return self.text(text).encode("utf-16-le", errors="surrogatepass") + data[even:]


def _looks_utf16(data: bytes) -> bool:
    probe = data[:256]
    probe = probe[: len(probe) - len(probe) % 2]
    return len(probe) >= 4 and probe[1::2].count(0) > 0.8 * (len(probe) // 2)


def split_raw(data: bytes) -> tuple[bytes, bytes]:
    """A raw file as (header text bytes, sample bytes).

    The header runs through the ``Binary:`` line. An ASCII raw (``Values:``) is
    text throughout and its values name no path, so it splits the same way; a
    file cut off inside its header is all header. The header is UTF-16 except
    in a text raw, which is 8-bit throughout.
    """
    codec = "utf-16-le" if _looks_utf16(data) else "latin-1"
    markers = [marker.encode(codec) for marker in _RAW_MARKERS]
    cuts = [data.find(marker) + len(marker) for marker in markers if marker in data]
    cut = min(cuts) if cuts else len(data)
    return data[:cut], data[cut:]


def private_strings(extra: Iterable[str] = ()) -> list[str]:
    """What must not appear in a recording made on this machine."""
    values = [str(Path.home()), *extra]
    for variable in ("USERNAME", "USER", "COMPUTERNAME", "USERDOMAIN"):
        value = os.environ.get(variable, "")
        # A two-letter name would match by chance; the home path still covers it.
        if len(value) >= 3:
            values.append(value)
    return [value for value in values if value]


def assert_private(
    files: Mapping[str, bytes], forbidden: Sequence[str], ansi: str | None = None
) -> None:
    """Refuse a recording that still names this machine or its user.

    Checked in the encodings LTspice writes text in, the machine's own 8-bit
    one (``ansi``) among them. A raw file's samples are left out: they are
    numbers, and four chance bytes can spell a short name. The message names
    the files and the kind of string, never the string.
    """
    leaks: list[str] = []
    for name, data in files.items():
        searched = split_raw(data)[0] if name.endswith(_RAW_SUFFIXES) else data
        folded = searched.lower()
        for index, value in enumerate(forbidden):
            needles: set[bytes] = set()
            for codec in ("utf-16-le", *_eight_bit_codecs(ansi)):
                with contextlib.suppress(UnicodeError, LookupError):
                    # Folded as bytes, as the file is, so a double-byte
                    # character whose second byte is a letter still matches.
                    needles.add(value.encode(codec).lower())
            if any(needle and needle in folded for needle in needles):
                leaks.append(f"{name} (private string #{index})")
    if leaks:
        raise RecorderError(
            "a recording still carries a local path, user name or host name: "
            + ", ".join(sorted(leaks))
        )


# --------------------------------------------------------------------------
# Launching off the user's desktop
# --------------------------------------------------------------------------


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


_GENERIC_ALL = 0x10000000
_WAIT_OBJECT_0 = 0
_INFINITE = 0xFFFFFFFF


class _Started:
    """A process started on a named desktop: the part of ``Popen`` a case uses.

    It is held in a job that ends with this object, so an LTspice no one can
    see cannot outlive a recorder that was itself stopped.
    """

    def __init__(self, handle: int, pid: int, command: Sequence[str]) -> None:
        from ltspice_mcp.lib.windows_job import WindowsJob

        self._handle: int | None = handle
        self.pid = pid
        self.args = list(command)
        self.returncode: int | None = None
        self._job: WindowsJob | None = None
        # A process that has already exited cannot join a job, and needs none.
        with contextlib.suppress(OSError):
            self._job = WindowsJob(pid, allow_breakaway=False)

    def _settle(self, milliseconds: int) -> int | None:
        if self.returncode is not None or self._handle is None:
            return self.returncode
        if sys.platform != "win32":
            return None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        if kernel.WaitForSingleObject(self._handle, milliseconds) != _WAIT_OBJECT_0:
            return None
        code = wintypes.DWORD(0)
        kernel.GetExitCodeProcess(self._handle, ctypes.byref(code))
        self.returncode = code.value
        return self.returncode

    def poll(self) -> int | None:
        return self._settle(0)

    def wait(self, timeout: float | None = None) -> int:
        code = self._settle(_INFINITE if timeout is None else max(0, int(timeout * 1000)))
        if code is None:
            raise subprocess.TimeoutExpired(self.args, timeout or 0.0)
        return code

    def kill(self) -> None:
        if sys.platform != "win32" or self._handle is None:
            return
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateProcess(self._handle, 1)

    def close(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None
        if sys.platform == "win32" and self._handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(self._handle)
        self._handle = None


class HiddenDesktop:
    """A desktop of its own for LTspice to open its window on.

    LTspice opens a window even for a batch run and holds the keyboard focus
    until the run ends, whatever show state it is started with: started
    minimised or hidden it was still the foreground window for most of a run.
    A process started on another desktop has its windows there and nowhere
    else, so recording a few hundred cases does not interrupt whoever is at
    the machine.

    It also keeps a person out of the recording. LTspice answers some inputs
    with a message box and waits for OK; on the desktop someone is working at,
    a stray key press answers it, and the run then looks as if it had ended by
    itself. Here nobody can, so ``dialog`` reads what the box says and the case
    records that the build stopped to ask.

    Where a desktop cannot be made, ``start`` launches the ordinary way and
    ``dialog`` sees nothing.
    """

    def __init__(self) -> None:
        self.name: str | None = None
        self._handle: int | None = None
        if sys.platform != "win32":
            return
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.CreateDesktopW.restype = wintypes.HANDLE
        user.CreateDesktopW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        name = f"ltspice-recorder-{os.getpid()}"
        handle = user.CreateDesktopW(name, None, None, 0, _GENERIC_ALL, None)
        if handle:
            self.name, self._handle = name, handle

    def __enter__(self) -> HiddenDesktop:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if sys.platform == "win32" and self._handle is not None:
            user = ctypes.WinDLL("user32", use_last_error=True)
            user.CloseDesktop.argtypes = [wintypes.HANDLE]
            user.CloseDesktop(self._handle)
        self.name = self._handle = None

    def dialog(self, pid: int) -> str | None:
        """What a message box ``pid`` has open here says, or None when it has none.

        The box's title, then each line of text in it. Its buttons are left
        out: what the build asked is the recording, not how it could be
        answered.
        """
        if sys.platform != "win32" or self._handle is None:
            return None
        user = ctypes.WinDLL("user32", use_last_error=True)
        callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.EnumDesktopWindows.argtypes = [wintypes.HANDLE, callback, wintypes.LPARAM]
        user.EnumChildWindows.argtypes = [wintypes.HWND, callback, wintypes.LPARAM]

        def text_of(window: int) -> str:
            buffer = ctypes.create_unicode_buffer(2048)
            user.GetWindowTextW(window, buffer, len(buffer))
            return buffer.value

        def class_of(window: int) -> str:
            buffer = ctypes.create_unicode_buffer(128)
            user.GetClassNameW(window, buffer, len(buffer))
            return buffer.value

        found: list[str] = []

        def child(window: int, _unused: int) -> bool:
            if class_of(window) == "Static" and text_of(window).strip():
                found.append(text_of(window))
            return True

        def top(window: int, _unused: int) -> bool:
            owner = wintypes.DWORD(0)
            user.GetWindowThreadProcessId(window, ctypes.byref(owner))
            # "#32770" is the window class of a dialog box.
            if (
                owner.value == pid
                and class_of(window) == "#32770"
                and user.IsWindowVisible(window)
            ):
                found.append(text_of(window))
                user.EnumChildWindows(window, callback(child), 0)
            return True

        user.EnumDesktopWindows(self._handle, callback(top), 0)
        return "\n".join(found) if found else None

    def start(self, command: Sequence[str], cwd: Path) -> _Started | subprocess.Popen[bytes]:
        """Start ``command`` in ``cwd`` with its windows on this desktop."""
        if sys.platform != "win32" or self.name is None:
            return subprocess.Popen(
                list(command), cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(_StartupInfo),
            ctypes.POINTER(_ProcessInformation),
        ]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        startup = _StartupInfo()
        startup.cb = ctypes.sizeof(startup)
        startup.lpDesktop = self.name
        created = _ProcessInformation()
        line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
        if not kernel.CreateProcessW(
            None, line, None, None, False, 0, None, str(cwd), ctypes.byref(startup), created
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(created.hThread)
        return _Started(created.hProcess, created.dwProcessId, command)


# --------------------------------------------------------------------------
# Running one case
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    """What one case produced: its manifest entry and its scrubbed files.

    ``defaults`` is what the build wrote back into the settings copy for the
    keys the recorder had removed: its own default for each.
    """

    entry: dict[str, Any]
    files: dict[str, bytes]
    defaults: dict[str, str] = field(default_factory=dict)


def _command(build: Build, case: Case, deck: Path, ini: Path) -> list[str]:
    mode = ["-netlist"] if case.kind == "netlist" else ["-Run", "-b"]
    settings = ["-ini", str(ini)] if case.settings else []
    return [str(build.exe), *mode, deck.as_posix(), *case.switches, *settings]


def _neutral_command(command: Sequence[str], build: Build, work: Path, ini: Path) -> list[str]:
    """``command`` as the manifest shows it: no install, run or settings directory."""
    shown: list[str] = []
    for token in command:
        if token == str(build.exe):
            shown.append(build.exe.name)
        elif token == str(ini):
            shown.append("<settings>")
        else:
            shown.append(token.replace(work.as_posix(), "<dir>").replace(str(work), "<dir>"))
    return shown


@dataclass(frozen=True)
class _Ended:
    """How one launch ended.

    ``stopped`` is True when the recorder ended it; ``dialog`` is then what the
    build was asking, when it was waiting on a message box.
    """

    stopped: bool
    exit_code: int | None
    dialog: str | None = None


def _wait(
    process: _Started | subprocess.Popen[bytes],
    case: Case,
    raw: Path,
    timeout: float,
    desktop: HiddenDesktop,
) -> _Ended:
    """Wait for ``process`` to end, stopping it when it will not end by itself.

    That is one of three things. A ``kill`` case is stopped once its raw file
    has grown to the size the case names (or the time it names has passed),
    which is how a run interrupted part way is recorded. A build that puts up a
    message box is stopped as soon as the box is seen on two looks running: it
    waits there for an answer nobody is going to give. Anything else is
    stopped at ``timeout``.
    """
    started = time.monotonic()
    asked: str | None = None
    while True:
        try:
            # timing: looks between waits at a file LTspice is writing and at the
            # windows it has open; neither signals a change
            process.wait(timeout=0.02 if case.kind == "kill" else 0.25)
        except subprocess.TimeoutExpired:
            pass
        else:
            return _Ended(stopped=False, exit_code=process.returncode)
        elapsed = time.monotonic() - started
        dialog = None
        if case.kind == "kill":
            try:
                size = raw.stat().st_size
            except OSError:
                size = 0
            grown = case.kill_after_bytes is not None and size >= case.kill_after_bytes
            waited = case.kill_after_s is not None and elapsed >= case.kill_after_s
            stop = grown or waited
        else:
            dialog = desktop.dialog(process.pid)
            stop = dialog is not None and dialog == asked
            asked = dialog
        if stop or elapsed >= timeout:
            process.kill()
            process.wait()
            exit_code = process.returncode if case.kind == "kill" else None
            return _Ended(stopped=True, exit_code=exit_code, dialog=dialog if stop else None)


def _launch(
    desktop: HiddenDesktop,
    command: Sequence[str],
    work: Path,
    case: Case,
    deck: Path,
    timeout: float,
) -> _Ended:
    """Run ``command`` to its end or to the point the recorder stops it."""
    process = desktop.start(command, work)
    try:
        return _wait(process, case, deck.with_suffix(".raw"), timeout, desktop)
    finally:
        if isinstance(process, _Started):
            process.close()


def run_case(
    build: Build,
    case: Case,
    inputs: Path,
    work_root: Path,
    *,
    timeout: float = 120.0,
    desktop: HiddenDesktop | None = None,
) -> CaseResult:
    """Run ``case`` on ``build`` in a fresh directory and return what it wrote."""
    if desktop is None:
        with HiddenDesktop() as own:
            return run_case(build, case, inputs, work_root, timeout=timeout, desktop=own)
    settings = build.settings_file
    if settings is None:
        raise RecorderError(unavailable_reason(build) or "no settings file")
    work = work_root / build.label / case.case_id.replace("/", "__")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    digests: dict[str, str] = {}
    for name in (case.source, *case.extra):
        data = (inputs / name).read_bytes()
        target = case.work_name if name == case.source else Path(name).name
        (work / target).write_bytes(data)
        digests[name] = sha256_bytes(data)
    # Outside the case directory, so nothing the build writes can pick it up.
    ini = work.parent / f"{work.name}.ini"
    ini.write_bytes(neutral_settings(settings.read_bytes(), case.ini))
    deck = work / case.work_name
    command = _command(build, case, deck, ini)
    ended = _launch(desktop, command, work, case, deck, timeout)
    if case.kind == "fastaccess" and not ended.stopped:
        raw = deck.with_suffix(".raw").as_posix()
        convert = [str(build.exe), "-FastAccess", raw, "-ini", str(ini)]
        ended = _launch(desktop, convert, work, case, deck, timeout)
        command = [*command, "&&", *convert]
    defaults = read_settings(ini.read_bytes(), BEHAVIOUR_KEYS) if ini.is_file() else {}
    ini.unlink(missing_ok=True)

    scrubber = Scrubber.for_run(work)
    files: dict[str, bytes] = {}
    for suffix in case.kept():
        produced = work / f"{case.stem}.{suffix}"
        if produced.is_file():
            files[f"{case.case_id}.{suffix}"] = scrubber.bytes(
                produced.read_bytes(), produced.name
            )
    written = sorted(
        path.name[len(case.stem) :]
        for path in work.iterdir()
        if path.is_file() and path.name.startswith(case.stem + ".") and path != deck
    )
    entry: dict[str, Any] = {
        "behaviour": case.behaviour,
        "kind": case.kind,
        "command": _neutral_command(command, build, work, ini),
        "inputs": digests,
        "exit_code": ended.exit_code,
        "stopped": ended.stopped,
        "written": written,
        "outputs": {
            name: {"sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in sorted(files.items())
        },
        "reported_build": _reported_build(work, case),
    }
    if case.ini:
        entry["settings"] = dict(case.ini)
    if not case.settings:
        entry["settings"] = "the recording user's own"
    if case.volatile:
        entry["volatile"] = True
    if ended.dialog is not None:
        entry["dialog"] = scrubber.text(ended.dialog)
    return CaseResult(entry=entry, files=files, defaults=defaults)


def _reported_build(work: Path, case: Case) -> str | None:
    """The build as the case's own output names it, read the way the server reads it."""
    from ltspice_mcp.lib.lint_rules import deck_generator
    from ltspice_mcp.lib.simulator_build import reported_build

    if case.kind == "netlist":
        net = work / f"{case.stem}.net"
        if not net.is_file():
            return None
        from ltspice_mcp.lib.encoding import read_spice_text

        return deck_generator(read_spice_text(net))
    log = work / f"{case.stem}.log"
    raw = work / f"{case.stem}.raw"
    return reported_build(log if log.is_file() else None, raw if raw.is_file() else None)


# --------------------------------------------------------------------------
# Recording a build
# --------------------------------------------------------------------------


def executable_record(build: Build) -> dict[str, Any]:
    """The build's identity as the server records one, without where it is installed.

    ``simulator_build.executable_identity`` gives the path, size, modification
    time and digest. The digest is the identity; the directory and the
    modification time say where and when this machine installed it, so only
    the file name is kept from them.
    """
    from ltspice_mcp.lib.simulator_build import executable_identity

    launcher = type("RecordedBuild", (), {"spice_exe": [str(build.exe)]})
    identity = executable_identity(launcher)
    if identity is None or identity.sha256 is None:
        raise RecorderError(f"cannot identify {build.exe.name}")
    return {
        "name": Path(identity.path).name,
        "sha256": identity.sha256,
        "bytes": identity.bytes,
        "file_version": build.file_version,
    }


def library_record(build: Build) -> dict[str, Any] | None:
    """Facts about the library the build reads: where it is, and how its files are encoded.

    The files themselves are the vendor's and are not copied. What the server
    models about them is recorded instead: the root (below the home directory,
    which is what differs per build), the encoding of each ``cmp/standard.*``
    model file, and the pins of the stock symbols the fixture symbols stand in
    for.
    """
    root = build.library_root
    if root is None:
        return None
    from ltspice_mcp.lib.encoding import read_spice_text_with_encoding
    from ltspice_mcp.lib.symbol_geometry import parse_asy_file

    def relative_to_home(path: Path) -> str:
        with contextlib.suppress(ValueError):
            return "~/" + path.relative_to(Path.home()).as_posix()
        return path.name

    models: dict[str, Any] = {}
    for path in sorted((root / "cmp").glob("standard.*")):
        if path.suffix == ".bak":
            continue
        data = path.read_bytes()
        _, encoding = read_spice_text_with_encoding(path)
        models[path.name] = {
            "encoding": encoding,
            "byte_order_mark": data[:2] in (b"\xff\xfe", b"\xfe\xff")
            or data[:3] == b"\xef\xbb\xbf",
        }
    symbols: dict[str, Any] = {}
    for name in STOCK_SYMBOLS:
        path = root / "sym" / f"{name}.asy"
        if not path.is_file():
            continue
        info = parse_asy_file(path)
        symbols[name] = {
            "prefix": info.prefix,
            "pins": [
                {"name": pin.name, "order": pin.order, "x": pin.x, "y": pin.y} for pin in info.pins
            ],
            "sha256": sha256_bytes(path.read_bytes()),
        }
    return {"root": relative_to_home(root), "models": models, "symbols": symbols}


#: Stock symbols whose pins the suite's hand-written stand-ins claim to match.
STOCK_SYMBOLS = (
    "res",
    "cap",
    "ind",
    "voltage",
    "current",
    "diode",
    "npn",
    "pnp",
    "nmos",
    "pmos",
    "nmos4",
    "pmos4",
    "njf",
    "e",
    "g",
    "bv",
)


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    (directory / MANIFEST).write_bytes(text.encode("utf-8"))


def record_build(
    build: Build,
    out: Path,
    *,
    inputs: Path = INPUTS,
    only: Sequence[str] = (),
    work_root: Path | None = None,
    timeout: float = 120.0,
    progress: Any = None,
) -> dict[str, Any]:
    """Record every applicable case on ``build`` into ``out / build.label``.

    With ``only`` (glob patterns over case ids) the named cases are re-recorded
    and the rest of an existing recording is kept. Without it the directory is
    rebuilt, so a case removed from the list leaves no file behind.
    """
    reason = unavailable_reason(build)
    if reason is not None:
        raise RecorderError(reason)
    case_file = load_cases(inputs)
    selected = [
        case
        for case in case_file.cases
        if case.applies_to(build)
        and (not only or any(fnmatch.fnmatch(case.case_id, pattern) for pattern in only))
    ]
    directory = out / build.label
    previous: dict[str, Any] = {}
    if only and (directory / MANIFEST).is_file():
        previous = load_manifest(directory).get("cases", {})
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    cleanup = work_root is None
    root = Path(tempfile.mkdtemp(prefix="ltspice-recording-")) if work_root is None else work_root
    cases: dict[str, Any] = dict(previous)
    defaults: dict[str, str] = {}
    if only and (directory / MANIFEST).is_file():
        defaults = dict(load_manifest(directory).get("settings", {}).get("defaults", {}))
    desktop = HiddenDesktop()
    try:
        forbidden = private_strings([str(root)])
        for case in selected:
            if progress is not None:
                progress(f"{build.label}: {case.case_id}")
            result = run_case(build, case, inputs, root, timeout=timeout, desktop=desktop)
            assert_private(result.files, forbidden)
            if case.settings and not case.ini and not defaults:
                defaults = _portable_defaults(result.defaults)
            for stale in cases.get(case.case_id, {}).get("outputs", {}):
                (directory / stale).unlink(missing_ok=True)
            for name, data in result.files.items():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            cases[case.case_id] = result.entry
        known = {case.case_id for case in case_file.cases if case.applies_to(build)}
        for case_id in set(cases) - known:
            # A case that left the list takes its recording with it.
            for stale in cases.pop(case_id).get("outputs", {}):
                (directory / stale).unlink(missing_ok=True)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "build": build.label,
            "generation": build.generation,
            "executable": executable_record(build),
            "reported_build": _build_banner(cases),
            "settings": {
                "source": "a copy of the build's per-user settings file, "
                "without the keys listed under defaults",
                "defaults": dict(sorted(defaults.items())),
            },
            "scrubbed": {
                "run_directory": NEUTRAL_DIR,
                "home_directory": NEUTRAL_HOME,
                "date": NEUTRAL_DATE,
                "elapsed_seconds": "0.000",
                "thread_count": "1",
                "matrix_compiler_report": NEUTRAL_COMPILER_REPORT.strip(),
            },
            "library": library_record(build),
            "cases": dict(sorted(cases.items())),
        }
        rendered = json.dumps(manifest).encode("utf-8")
        assert_private({MANIFEST: rendered}, forbidden)
        _write_manifest(directory, manifest)
        return manifest
    finally:
        desktop.close()
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)


def _portable_defaults(defaults: Mapping[str, str]) -> dict[str, str]:
    """The build's defaults without the ones that are local directories."""
    local = {"symbolsearchpath", "librarysearchpath", "rawtempdir"}
    return {key: value for key, value in defaults.items() if key.casefold() not in local}


def _build_banner(cases: Mapping[str, Any]) -> str | None:
    """The build's own name for itself: the string most of its runs reported."""
    counts: dict[str, int] = {}
    for entry in cases.values():
        if entry.get("kind") == "netlist":
            continue
        reported = entry.get("reported_build")
        if reported:
            counts[reported] = counts.get(reported, 0) + 1
    return max(counts, key=lambda name: counts[name]) if counts else None


# --------------------------------------------------------------------------
# Reading and comparing recordings
# --------------------------------------------------------------------------


def recorded_builds(root: Path = FIXTURES) -> list[str]:
    """The labels of the builds with a committed recording."""
    if not root.is_dir():
        return []
    return sorted(path.parent.name for path in root.glob(f"*/{MANIFEST}"))


def recorded(build: str, name: str, root: Path = FIXTURES) -> Path:
    """The path of one recorded file: ``recorded("ltspice26", "raw/tran.log")``."""
    return root / build / name


def _raw_samples(data: bytes) -> tuple[bytes, list[float] | None]:
    """A raw's header and its samples as numbers, or None when they cannot be read."""
    import numpy as np

    from ltspice_mcp.lib.raw_header import RawHeaderError, RawLimits, preflight_raw

    head, payload = split_raw(data)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "compare.raw"
        path.write_bytes(data)
        try:
            limits = RawLimits(64_000_000, 1_000_000, 64_000, 16, 4096, 10_000_000, 64_000_000)
            (plot,) = preflight_raw(path, limits=limits).plots
        except (RawHeaderError, ValueError):
            return head, None
    if plot.storage != "binary":
        return head, None
    formats = {4: "<f4", 8: "<f8", 16: "<c16"}
    record = np.dtype([(f"v{i}", formats[width]) for i, width in enumerate(plot.value_bytes)])
    usable = len(payload) - len(payload) % record.itemsize
    rows = np.frombuffer(payload[:usable], dtype=record)
    flat: list[float] = []
    for name in record.names or ():
        column = rows[name]
        if np.iscomplexobj(column):
            flat.extend(np.real(column).tolist())
            flat.extend(np.imag(column).tolist())
        else:
            flat.extend(column.tolist())
    return head, flat


def raw_header_text(data: bytes) -> str:
    """A raw file's header as text, whichever way the build encoded it."""
    head = split_raw(data)[0]
    if _looks_utf16(head):
        return head[: len(head) - len(head) % 2].decode("utf-16-le", errors="replace")
    return head.decode("latin-1")


def _header_without_count(data: bytes) -> str:
    """The header of a run stopped part way, less the one line that says how far it got."""
    return re.sub(r"(?m)^No\. Points:[^\n]*\n", "", raw_header_text(data))


def _same_raw(committed: bytes, fresh: bytes, rtol: float) -> str | None:
    """None when two raw files agree; otherwise what differs.

    The headers must be equal. The samples must be equal in count and agree to
    ``rtol``: a solver's last bits depend on the processor it ran on, and that
    is not a change in behaviour.
    """
    if committed == fresh:
        return None
    head_a, samples_a = _raw_samples(committed)
    head_b, samples_b = _raw_samples(fresh)
    if head_a != head_b:
        return "header differs"
    if samples_a is None or samples_b is None:
        return "samples differ"
    if len(samples_a) != len(samples_b):
        return f"sample count {len(samples_a)} became {len(samples_b)}"
    import numpy as np

    a, b = np.asarray(samples_a), np.asarray(samples_b)
    scale = max(float(np.max(np.abs(a))) if a.size else 0.0, 1e-300)
    if not np.allclose(a, b, rtol=rtol, atol=rtol * scale, equal_nan=True):
        return f"samples differ by more than {rtol:g}"
    return None


def compare(
    committed: Path,
    fresh: Path,
    *,
    only: Sequence[str] = (),
    rtol: float = 1e-6,
) -> list[str]:
    """Differences between a committed recording and a fresh one, by case.

    Both directories hold one build's recording. Text files are compared as
    bytes: the recorder already rewrote everything that legitimately varies.
    A ``volatile`` case (a run stopped part way) is compared on what it wrote
    and on its raw header up to the point count, never on its samples.
    """
    old, new = load_manifest(committed), load_manifest(fresh)
    differences: list[str] = []
    for case_id in sorted(set(old["cases"]) | set(new["cases"])):
        if only and not any(fnmatch.fnmatch(case_id, pattern) for pattern in only):
            continue
        a, b = old["cases"].get(case_id), new["cases"].get(case_id)
        if a is None:
            differences.append(f"{case_id}: not in the committed recording")
            continue
        if b is None:
            differences.append(f"{case_id}: not in the fresh recording")
            continue
        if a["inputs"] != b["inputs"]:
            differences.append(f"{case_id}: the input changed since it was recorded")
            continue
        volatile = bool(a.get("volatile"))
        for key in ("exit_code", "stopped", "dialog", "written", "reported_build"):
            if a.get(key) != b.get(key) and not (volatile and key == "exit_code"):
                differences.append(f"{case_id}: {key} was {a.get(key)!r}, is now {b.get(key)!r}")
        for name in sorted(set(a["outputs"]) | set(b["outputs"])):
            if name not in a["outputs"] or name not in b["outputs"]:
                side = "fresh" if name in b["outputs"] else "committed"
                differences.append(f"{name}: only in the {side} recording")
                continue
            data_a, data_b = (committed / name).read_bytes(), (fresh / name).read_bytes()
            if data_a == data_b:
                continue
            if name.endswith(_RAW_SUFFIXES):
                if volatile:
                    if _header_without_count(data_a) != _header_without_count(data_b):
                        differences.append(f"{name}: header differs")
                    continue
                problem = _same_raw(data_a, data_b, rtol)
                if problem is not None:
                    differences.append(f"{name}: {problem}")
            elif not volatile:
                differences.append(f"{name}: content differs")
    return differences


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record LTspice's output for the inputs under "
        "tests/fixtures/ltspice_recorded/inputs, one directory per build."
    )
    parser.add_argument(
        "executables",
        nargs="*",
        type=Path,
        help="LTspice executables to record (default: every build installed here)",
    )
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="directory of inputs")
    parser.add_argument("--out", type=Path, default=FIXTURES, help="where recordings go")
    parser.add_argument(
        "--only", action="append", default=[], help="glob over case ids; repeatable"
    )
    parser.add_argument("--work-dir", type=Path, help="run cases here instead of a temp dir")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per case")
    parser.add_argument(
        "--check",
        action="store_true",
        help="record into a temporary directory and compare with --out instead of writing it",
    )
    args = parser.parse_args(argv)

    builds = [identify_build(exe) for exe in args.executables] or discover_builds()
    if not builds:
        print("no LTspice build found; pass the executables to record", file=sys.stderr)
        return 2
    status = 0
    for build in builds:
        reason = unavailable_reason(build)
        if reason is not None:
            print(f"{build.label}: skipped: {reason}", file=sys.stderr)
            status = max(status, 2)
            continue
        if not args.check:
            manifest = record_build(
                build,
                args.out,
                inputs=args.inputs,
                only=args.only,
                work_root=args.work_dir,
                timeout=args.timeout,
                progress=lambda line: print(line, flush=True),
            )
            print(f"{build.label}: recorded {len(manifest['cases'])} cases")
            continue
        with tempfile.TemporaryDirectory(prefix="ltspice-check-") as scratch:
            record_build(
                build,
                Path(scratch),
                inputs=args.inputs,
                only=args.only,
                work_root=args.work_dir,
                timeout=args.timeout,
            )
            differences = compare(
                args.out / build.label, Path(scratch) / build.label, only=args.only
            )
        for line in differences:
            print(f"{build.label}: {line}")
        print(f"{build.label}: {len(differences)} difference(s)")
        status = max(status, 1 if differences else 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
