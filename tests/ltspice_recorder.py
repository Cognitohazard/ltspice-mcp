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
an export is ``<exe> -netlist <sheet>``, as spicelib launches them. A plot
case is the one exception, because what it records is what a person sees: a
sheet is run in the window (``<exe> -Run <sheet>``), the waveform window is
given its panes with its own menu commands, and its plot settings are saved
(``drive_plot``). The one addition is a trailing ``-ini <file>``: a recording must not depend on the
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
on a desktop of its own (``recording_desktop``), because LTspice otherwise takes
the keyboard focus for as long as a run lasts. Everything that reads a
recording (``load_manifest``, ``recorded``, ``compare``) works anywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fnmatch
import functools
import hashlib
import json
import locale
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.hidden_desktop import (
    BoxWatch,
    HiddenDesktop,
    StartedProcess,
    child_windows,
    window_class,
    window_text,
)

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
    "plot": ("plt",),
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

    A ``plot`` case may name a plot settings file (``plot``), copied beside
    the deck under the deck's name so the waveform window reads it, and the
    ``steps`` the window is driven through before it saves: each
    ``("trace", expression)`` or ``("command", menu label)``.
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
    plot: str = ""
    steps: tuple[tuple[str, str], ...] = ()

    @property
    def copies(self) -> dict[str, str]:
        """Each input file the case copies, and its name in the case's directory.

        The source takes the case's name and a plot settings file the deck's,
        which is the file the waveform window reads; an extra keeps its own.
        """
        copies = {self.source: self.work_name}
        copies.update((name, Path(name).name) for name in self.extra)
        if self.plot:
            copies[self.plot] = f"{self.stem}.plt"
        return copies

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
                plot=entry.get("plot", ""),
                steps=tuple(_step(entry["id"], step) for step in entry.get("steps", ())),
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
        if case.kind != "plot" and (case.plot or case.steps):
            raise RecorderError(f"{case.case_id}: only a plot case has plot settings or steps")
        if case.kind == "plot" and not (case.plot or case.steps):
            raise RecorderError(f"{case.case_id}: a plot case reads plot settings or makes them")
        for name in case.copies:
            if not (inputs / name).is_file():
                raise RecorderError(f"{case.case_id}: missing input {name}")
    return CaseFile(behaviours=behaviours, cases=tuple(cases))


def _step(case_id: str, step: Mapping[str, str]) -> tuple[str, str]:
    """One step of a plot case: ``{trace = "V(out)"}`` or ``{command = "<menu label>"}``."""
    if len(step) != 1 or next(iter(step)) not in ("trace", "command"):
        raise RecorderError(f"{case_id}: a step is one of trace = ... or command = ...")
    ((kind, value),) = step.items()
    return kind, str(value)


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


def recording_desktop() -> HiddenDesktop:
    """A desktop of its own for LTspice to open its window on.

    The launch is the server's (``lib/hidden_desktop.py``, which has the
    measurements): a process started on another desktop has its windows there
    and nowhere else, so recording a few hundred cases does not interrupt
    whoever is at the machine.

    It also keeps a person out of the recording. LTspice answers some inputs
    with a message box and waits for OK; on the desktop someone is working at,
    a stray key press answers it, and the run then looks as if it had ended by
    itself. Here nobody can, so the box is read and the case records that the
    build stopped to ask.

    Where a desktop cannot be made, a case is launched the ordinary way and
    its boxes are looked for among the windows of the desktop the recorder
    runs on (``HiddenDesktop.windows``). Under Wine one is made, but the
    windows on it cannot be listed, so neither a box nor the waveform window a
    plot case drives could be found there; the recorder launches the ordinary
    way instead, on the display Wine draws to, where both are.
    """
    desktop = HiddenDesktop(f"ltspice-recorder-{os.getpid()}")
    if recording_host() is not None:
        desktop.close()
    return desktop


# --------------------------------------------------------------------------
# Driving the waveform window (plot cases)
# --------------------------------------------------------------------------

_RT_MENU = 4
_WM_SETTEXT = 0x000C
_WM_COMMAND = 0x0111
_WM_MDIACTIVATE = 0x0222
_IDOK = 1
#: The dialog the waveform window's Add Trace command opens, in both builds.
_ADD_TRACES = "Add Traces to Plot"
#: The menu labels a plot case uses besides its own steps.
_ADD_TRACE = "Add trace"
_SAVE_PLOT = "Save Plot Settings"


def _pe_resources(image: bytes, kind: int) -> list[bytes]:
    """The data of every resource of type ``kind`` in a PE image, in directory order."""

    def u16(at: int) -> int:
        return struct.unpack_from("<H", image, at)[0]

    def u32(at: int) -> int:
        return struct.unpack_from("<I", image, at)[0]

    header = u32(0x3C)
    if image[header : header + 4] != b"PE\0\0":
        raise RecorderError("not a Windows executable")
    sections, optional_size = u16(header + 6), u16(header + 20)
    optional = header + 24
    directories = optional + (112 if u16(optional) == 0x20B else 96)
    resource_rva = u32(directories + 2 * 8)
    table = optional + optional_size
    spans = [
        (
            u32(table + 40 * i + 12),
            max(u32(table + 40 * i + 8), u32(table + 40 * i + 16)),
            u32(table + 40 * i + 20),
        )
        for i in range(sections)
    ]

    def offset(rva: int) -> int:
        for start, size, raw in spans:
            if start <= rva < start + size:
                return rva - start + raw
        raise RecorderError("a resource lies outside every section of the executable")

    root = offset(resource_rva)

    def entries(directory: int) -> list[tuple[int, int]]:
        count = u16(directory + 12) + u16(directory + 14)
        return [(u32(directory + 16 + 8 * i), u32(directory + 20 + 8 * i)) for i in range(count)]

    found: list[bytes] = []
    for type_id, type_target in entries(root):
        if type_id != kind or not type_target & 0x80000000:
            continue
        for _name, name_target in entries(root + (type_target & 0x7FFFFFFF)):
            for _language, data in entries(root + (name_target & 0x7FFFFFFF)):
                entry = root + data
                start = offset(u32(entry))
                found.append(image[start : start + u32(entry + 4)])
    return found


def _menu_items(template: bytes) -> list[tuple[int | None, str]]:
    """The (command id, text) of every item of a menu template; a submenu's id is None."""
    version, header = struct.unpack_from("<HH", template, 0)
    items: list[tuple[int | None, str]] = []

    def text_at(at: int) -> tuple[str, int]:
        end = at
        while template[end : end + 2] != b"\0\0":
            end += 2
        return template[at:end].decode("utf-16-le"), end + 2

    def extended(at: int) -> int:
        while True:
            at = (at + 3) & ~3
            _type, _state, command, flags = struct.unpack_from("<IIIH", template, at)
            text, at = text_at(at + 14)
            items.append((None if flags & 0x01 else command, text))
            if flags & 0x01:
                at = extended(((at + 3) & ~3) + 4)
            if flags & 0x80:
                return at

    def classic(at: int) -> int:
        while True:
            (flags,) = struct.unpack_from("<H", template, at)
            at += 2
            command = None
            if not flags & 0x10:
                (command,) = struct.unpack_from("<H", template, at)
                at += 2
            text, at = text_at(at)
            items.append((command, text))
            if flags & 0x10:
                at = classic(at)
            if flags & 0x80:
                return at

    if version == 1:
        extended(4 + header)
    else:
        classic(4)
    return items


def menu_label(text: str) -> str:
    """A menu item's text as a case names it: no accelerator, no ``&``, no trailing dots."""
    return text.split("\t", 1)[0].replace("&", "").rstrip(".").strip()


@functools.cache
def waveform_commands(exe: Path) -> dict[str, int]:
    """The command each waveform-window menu item sends, by ``menu_label``.

    Read from the build's own menu resource, the first menu holding Add
    trace, rather than from the running window, so the ids are the build's
    whatever window it shows them in. A label that appears twice keeps its
    first id: the File menu's Save Plot Settings, which writes the default
    file without asking where.
    """
    for template in _pe_resources(exe.read_bytes(), _RT_MENU):
        items = _menu_items(template)
        if not any(menu_label(text) == _ADD_TRACE for _command, text in items):
            continue
        commands: dict[str, int] = {}
        for command, text in items:
            if command is not None:
                commands.setdefault(menu_label(text), command)
        return commands
    raise RecorderError(f"{exe.name} has no waveform-window menu with an {_ADD_TRACE} item")


@functools.cache
def _user32() -> Any:
    """The user32 calls a plot case makes besides the ones ``hidden_desktop`` makes."""
    if sys.platform != "win32":
        raise RecorderError("a plot case drives LTspice's window, which needs Windows")
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    signatures: dict[str, tuple[list[Any], Any]] = {
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "IsWindow": ([wintypes.HWND], wintypes.BOOL),
        "GetWindowRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        "PostMessageW": (
            [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM],
            wintypes.BOOL,
        ),
        "SendMessageTimeoutW": (
            [
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
                wintypes.UINT,
                wintypes.UINT,
                ctypes.POINTER(ctypes.c_size_t),
            ],
            ctypes.c_ssize_t,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(user, name)
        function.argtypes = arguments
        function.restype = result
    return user


def _send(window: int, message: int, wparam: int, lparam: int) -> None:
    """Send a message a window must handle before the next step, without waiting on a hang.

    A plain SendMessage waits for as long as the window does not answer, and
    a window in a modal loop of its own can leave it waiting for good.
    """
    result = ctypes.c_size_t(0)
    abort_if_hung = 0x0002
    if not _user32().SendMessageTimeoutW(
        window, message, wparam, lparam, abort_if_hung, 5000, ctypes.byref(result)
    ):
        raise RecorderError(f"LTspice's window did not answer message {message:#06x}")


def _closed(window: int) -> bool:
    return not _user32().IsWindow(window)


def _top(window: int) -> int:
    from ctypes import wintypes

    rect = wintypes.RECT()
    _user32().GetWindowRect(window, ctypes.byref(rect))
    return int(rect.top)


class _WaveformWindow:
    """One LTspice process's windows, as a plot case drives them."""

    def __init__(self, desktop: HiddenDesktop, pid: int, stem: str) -> None:
        self.desktop = desktop
        self.pid = pid
        self.raw_title = f"{stem}.raw".casefold()

    def top_level(self) -> list[int]:
        visible = _user32().IsWindowVisible
        return [window for window in self.desktop.windows(self.pid) if visible(window)]

    def frame(self) -> int | None:
        return next(
            (
                w
                for w in self.top_level()
                if window_class(w).startswith("Afx:") and window_text(w).startswith("LTspice")
            ),
            None,
        )

    def waveform(self, frame: int) -> tuple[int, int] | None:
        """The waveform window of the case's raw file, and the MDI client it is in."""
        children = child_windows(frame)
        for child in children:
            if window_text(child).casefold() == self.raw_title and window_class(child).startswith(
                "Afx:"
            ):
                client = next((c for c in children if window_class(c) == "MDIClient"), frame)
                return child, client
        return None

    def dialog(self, title: str) -> int | None:
        return next(
            (
                w
                for w in self.top_level()
                if window_class(w) == "#32770" and window_text(w) == title
            ),
            None,
        )


class _PlotStop(Exception):
    """A plot case ended before it saved: a box was waiting, the build exited, or time ran out."""

    def __init__(self, ended: _Ended) -> None:
        super().__init__(repr(ended))
        self.ended = ended


def _plot_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def drive_plot(
    process: StartedProcess | subprocess.Popen[bytes],
    desktop: HiddenDesktop,
    build: Build,
    case: Case,
    work: Path,
    timeout: float,
) -> _Ended:
    """Run a plot case's steps in the waveform window, then save its plot settings.

    The sheet runs in the window (``-Run``); once its log is written and the
    waveform window of its raw file is open, the window has read any plot
    settings file beside the sheet. Each step then sends the menu command a
    person would choose, by the id the build's own menu gives it
    (``waveform_commands``); a trace is typed into the Add Traces dialog. The
    case ends with Save Plot Settings, which writes ``<sheet>.plt``; a window
    run never exits by itself, so ``_launch`` ends the build afterwards.

    A dialog the case did not open is a message box the build is waiting on;
    the case stops with what it says, as a run does, and keeps no plot
    settings: a file the build did not save is the case's own input. A build
    that does none of this in ``timeout`` fails the recording, naming the step
    it was waiting at.
    """
    deadline = time.monotonic() + timeout
    commands = waveform_commands(build.exe)
    windows = _WaveformWindow(desktop, process.pid, case.stem)
    boxes = BoxWatch(desktop, process.pid, ignore=_ADD_TRACES)
    user = _user32()

    def until(condition: Callable[[], Any], doing: str) -> Any:
        """What ``condition`` returns once it is true; stops the case if the build will not."""
        while True:
            found = condition()
            if found:
                return found
            exit_code = process.poll()
            if exit_code is not None:
                raise _PlotStop(_Ended(stopped=False, exit_code=exit_code))
            box = boxes.look()
            if box is not None:
                raise _PlotStop(_Ended(stopped=True, exit_code=None, dialog=box))
            if time.monotonic() >= deadline:
                raise RecorderError(
                    f"{case.case_id}: {build.exe.name} did not {doing} in {timeout:.0f}s"
                )
            # timing: looks at windows and files a GUI build is changing; neither signals
            time.sleep(0.25)

    def settle(doing: str) -> None:
        # timing: a posted command is handled when the window's message loop gets
        # to it, which nothing reports; a second keeps the next step behind it
        settled = time.monotonic() + 1.0
        until(lambda: time.monotonic() >= settled, doing)

    log = work / f"{case.stem}.log"
    plt = work / f"{case.stem}.plt"
    doing = "open the waveform window of the sheet's run"
    try:
        frame = until(windows.frame, doing)
        child, client = until(lambda: log.is_file() and windows.waveform(frame), doing)
        _send(client, _WM_MDIACTIVATE, child, 0)
        settle(doing)
        # Commands go to the waveform window's own frame, which hands them to
        # its view and document whichever window is active: after a run the
        # schematic can be, and its Save would save the sheet.
        for kind, value in case.steps:
            doing = f"take the step {kind} {value!r}"
            if kind == "command":
                user.PostMessageW(child, _WM_COMMAND, commands[value], 0)
                settle(doing)
                continue
            user.PostMessageW(child, _WM_COMMAND, commands[_ADD_TRACE], 0)
            dialog = until(lambda: windows.dialog(_ADD_TRACES), doing)
            _type_expression(dialog, value)
            user.PostMessageW(dialog, _WM_COMMAND, _IDOK, 0)
            until(functools.partial(_closed, dialog), doing)
            settle(doing)
        doing = "save its plot settings"
        before = _plot_stamp(plt)
        user.PostMessageW(child, _WM_COMMAND, commands[_SAVE_PLOT], 0)
        # Saved once the file has changed and then holds still for a settle.
        written = until(lambda: _plot_stamp(plt) not in (None, before) and _plot_stamp(plt), doing)
        while True:
            settle(doing)
            now = _plot_stamp(plt)
            if now == written:
                break
            written = now
    except _PlotStop as stop:
        plt.unlink(missing_ok=True)
        return stop.ended
    return _Ended(stopped=True, exit_code=None)


def _type_expression(dialog: int, expression: str) -> None:
    """Put ``expression`` in the Add Traces dialog's expression field.

    The field is the edit box below the "Expression(s) to add" label: the
    dialog's other edit box is the filter of the list above it.
    """
    children = child_windows(dialog)
    label = next(
        (
            c
            for c in children
            if window_class(c) == "Static" and window_text(c).startswith("Expression")
        ),
        None,
    )
    if label is None:
        raise RecorderError(f"the {_ADD_TRACES} dialog has no expression field")
    below = [c for c in children if window_class(c) == "Edit" and _top(c) >= _top(label)]
    if not below:
        raise RecorderError(f"the {_ADD_TRACES} dialog has no expression field")
    field = min(below, key=_top)
    text = ctypes.create_unicode_buffer(expression)
    _send(field, _WM_SETTEXT, 0, ctypes.addressof(text))


def recording_host() -> str | None:
    """``Wine <version>`` when the recorder runs under Wine; None on Windows itself."""
    if sys.platform != "win32":
        return None
    try:
        version = ctypes.WinDLL("ntdll").wine_get_version
    except (OSError, AttributeError):
        return None
    version.restype = ctypes.c_char_p
    return f"Wine {version().decode()}"


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
    # A plot case runs in the window, which a batch run (-b) never opens.
    if case.kind == "netlist":
        mode = ["-netlist"]
    elif case.kind == "plot":
        mode = ["-Run"]
    else:
        mode = ["-Run", "-b"]
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
    process: StartedProcess | subprocess.Popen[bytes],
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
    boxes = BoxWatch(desktop, process.pid)
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
            dialog = boxes.look()
            stop = dialog is not None
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
    build: Build,
) -> _Ended:
    """Run ``command`` to its end or to the point the recorder stops it.

    A build still running then, or when the recording fails on the way, is
    ended here: a window run never exits by itself, and leaving the ``with``
    of an ordinary launch waits for its process.
    """
    if case.kind == "plot":
        commands = waveform_commands(build.exe)
        for kind, value in case.steps:
            if kind == "command" and value not in commands:
                raise RecorderError(f"{case.case_id}: {build.exe.name} has no menu item {value!r}")
    started = (
        desktop.start(command, cwd=work)
        if desktop.available
        else subprocess.Popen(
            list(command), cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    )
    with started as process:
        try:
            if case.kind == "plot":
                return drive_plot(process, desktop, build, case, work, timeout)
            return _wait(process, case, deck.with_suffix(".raw"), timeout, desktop)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


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
        with recording_desktop() as own:
            return run_case(build, case, inputs, work_root, timeout=timeout, desktop=own)
    settings = build.settings_file
    if settings is None:
        raise RecorderError(unavailable_reason(build) or "no settings file")
    work = work_root / build.label / case.case_id.replace("/", "__")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    digests: dict[str, str] = {}
    for name, target in case.copies.items():
        data = (inputs / name).read_bytes()
        (work / target).write_bytes(data)
        digests[name] = sha256_bytes(data)
    # Outside the case directory, so nothing the build writes can pick it up.
    ini = work.parent / f"{work.name}.ini"
    ini.write_bytes(neutral_settings(settings.read_bytes(), case.ini))
    deck = work / case.work_name
    command = _command(build, case, deck, ini)
    ended = _launch(desktop, command, work, case, deck, timeout, build)
    if case.kind == "fastaccess" and not ended.stopped:
        raw = deck.with_suffix(".raw").as_posix()
        convert = [str(build.exe), "-FastAccess", raw, "-ini", str(ini)]
        ended = _launch(desktop, convert, work, case, deck, timeout, build)
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
    host = recording_host()
    if host is not None:
        entry["host"] = host
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
    and the rest of an existing recording is kept, with the settings defaults
    and the library facts it was made with; ``progress`` is told when this
    machine's library differs from those. Without ``only`` the directory is
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
    library = library_record(build)
    previous: dict[str, Any] = {}
    if only and (directory / MANIFEST).is_file():
        previous = load_manifest(directory)
    elif directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = dict(previous.get("cases", {}))
    defaults: dict[str, str] = dict(previous.get("settings", {}).get("defaults", {}))
    if previous:
        if previous.get("library") != library and progress is not None:
            progress(
                f"{build.label}: this machine's library differs from the one the "
                "recording was made with, whose facts are kept; record every case to "
                "replace them"
            )
        library = previous.get("library")

    cleanup = work_root is None
    root = Path(tempfile.mkdtemp(prefix="ltspice-recording-")) if work_root is None else work_root
    desktop = recording_desktop()
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
            "library": library,
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
