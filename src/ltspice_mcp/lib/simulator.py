"""Simulator detection and selection logic."""

import logging
import os
import platform
import re
import sys
from pathlib import Path, PureWindowsPath

from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import (
    SIM_ENABLED_KEY,
    SIM_EXECUTABLES_ENV,
    SIM_EXECUTABLES_KEY,
    SIM_PATH_ENV,
    SIM_PATH_KEY,
    SIM_SECTION,
    ServerConfig,
)
from ltspice_mcp.lib import hidden_desktop
from ltspice_mcp.lib.wsl import is_wsl

logger = logging.getLogger(__name__)


def _detection_disabled() -> bool:
    """Return True when env var ``LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION`` is truthy.

    Lets tests force the "degraded mode" code path on systems where a
    simulator binary happens to be on ``PATH`` (e.g. CI hosts with
    ngspice installed for unrelated reasons).
    """
    val = os.environ.get("LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _get_ltspice_class() -> type:
    """Return the appropriate LTspice class for the current platform.

    On WSL, returns LTspiceWSL which overrides run() to convert paths
    via wslpath instead of using Wine's Z: drive mapping. On Windows itself,
    the subclass that starts LTspice on a hidden desktop, so that its window
    does not take the keyboard focus on every run.
    """
    if is_wsl():
        from ltspice_mcp.lib.ltspice_wsl import LTspiceWSL

        return LTspiceWSL
    if sys.platform == "win32":
        from ltspice_mcp.lib.ltspice_windows import LTspice as LTspiceOffDesktop

        return LTspiceOffDesktop
    return LTspice


# Map simulator names to spicelib classes
SIMULATORS: dict[str, type] = {
    "ltspice": _get_ltspice_class(),
    "ngspice": NGspiceSimulator,
    "qspice": Qspice,
    "xyce": XyceSimulator,
}

#: How a person names each family, in messages and the server instructions.
SIMULATOR_DISPLAY: dict[str, str] = {
    "ltspice": "LTspice",
    "ngspice": "ngspice",
    "qspice": "QSPICE",
    "xyce": "Xyce",
}

# Each family's spicelib base class: the detected class, except that LTspice's
# platform subclass (LTspiceWSL) is widened to its base, so every LTspice class
# belongs to the family.
_FAMILY_BASES: dict[str, type] = {**SIMULATORS, "ltspice": LTspice}

# The family of each class name a job records (``job.simulator`` is the class's
# ``__name__``), so a persisted record resolves without the class at hand.
_FAMILY_BY_CLASS_NAME: dict[str, str] = {
    cls.__name__: family for family, cls in (*SIMULATORS.items(), *_FAMILY_BASES.items())
}

# Families that read the netlist LTspice exports from a schematic: LTspice as
# exported, ngspice once ``deck_prep`` has scrubbed it.
_READS_LTSPICE_EXPORT = frozenset({"ltspice", "ngspice"})


def simulator_family(simulator: type | str | None) -> str | None:
    """The supported family a simulator belongs to, or None for anything else.

    A class is judged by its spicelib lineage; a string is a recorded class
    name (``job.simulator``), judged by the classes the families are made of.
    """
    if isinstance(simulator, str):
        return _FAMILY_BY_CLASS_NAME.get(simulator)
    if not isinstance(simulator, type):
        return None
    return next(
        (family for family, base in _FAMILY_BASES.items() if issubclass(simulator, base)),
        None,
    )


def family_refusal(family: str | None) -> str | None:
    """Why runs on ``family`` cannot happen on this host, or None when they can.

    QSPICE is a Windows program, and spicelib starts it with this host's own
    file paths. Only LTspice has an adapter that translates them for a Windows
    simulator (``LTspiceWSL``, and spicelib's Wine prefix for LTspice), so off
    native Windows a QSPICE run is handed paths it cannot open. spicelib itself
    looks for QSPICE only on Windows and documents no Wine support for it.
    """
    host = _platform_key()
    if family == "qspice" and host != "windows":
        return (
            "QSPICE runs only when this server itself runs on Windows: it is a "
            "Windows program, spicelib starts it with this host's own file paths, "
            f"and under {'WSL' if host == 'wsl' else 'Wine'} it cannot open them "
            "(only LTspice has a path adapter here)."
        )
    return None


def asc_export_refusal(simulator_class: type | None) -> str | None:
    """Why a schematic cannot be run on ``simulator_class``, or None when it can.

    A schematic runs through the netlist LTspice exports from it, which is in
    LTspice's own dialect (``.backanno``, its ``§`` name prefix and ``µ`` unit
    suffix, ``.lib`` lines into LTspice's model library). LTspice reads it, and
    ngspice reads it once ``deck_prep`` has scrubbed those; any other family is
    refused rather than handed a dialect it was never checked against. A class
    outside the supported families is left to the export as before.
    """
    family = simulator_family(simulator_class)
    if family is None or family in _READS_LTSPICE_EXPORT:
        return None
    display = SIMULATOR_DISPLAY[family]
    return (
        "LTspice exports a schematic in its own netlist dialect, which is not "
        f"prepared for {display}, so a run on {display} takes a hand-written "
        ".cir/.net/.sp netlist."
    )


#: The families a run can be put on (``run_experiments``' ``execution.simulator``),
#: and so the families a named executable may belong to: every supported one.
#: Whether this host can run a family is asked per run (``family_refusal``).
_RUN_FAMILIES: tuple[str, ...] = tuple(SIMULATORS)

#: A named executable's own name: lower case, so a selector is spelled one way.
_EXECUTABLE_NAME_PATTERN = r"[a-z0-9][a-z0-9._-]{0,31}"
_EXECUTABLE_NAME_RE = re.compile(_EXECUTABLE_NAME_PATTERN)

#: What ``execution.simulator`` accepts: a family, or a family and the name of
#: one of its executables (``ltspice:xvii``).
SIMULATOR_SELECTOR_PATTERN = rf"^(?:{'|'.join(_RUN_FAMILIES)})(?::{_EXECUTABLE_NAME_PATTERN})?$"


def _exe_simulator_hint(exe_path: object) -> str | None:
    """Best-guess which simulator an executable path belongs to, by filename.

    Returns a simulator key (ltspice/ngspice/qspice/xyce) or None when the name
    matches no known pattern (in which case callers should trust the user).
    """
    name = Path(str(exe_path)).name.lower()
    if "ngspice" in name:
        return "ngspice"
    if "ltspice" in name or "xvii" in name or "scad3" in name:
        return "ltspice"
    if "qspice" in name:
        return "qspice"
    if "xyce" in name:
        return "xyce"
    return None


def _apply_simulator_exe(config: ServerConfig, diagnostics: list[str] | None = None) -> bool:
    """Apply config.simulator_exe to the appropriate spicelib simulator class.

    If the user has configured an explicit simulator executable path,
    use spicelib's create_from() to register it before auto-detection.
    This is essential for WSL where LTspice lives on the Windows side
    and spicelib's default search paths won't find it.

    Appends a human-readable note to ``diagnostics`` (if provided) when the
    configured path is missing or rejected, so ``server_status`` can surface
    the misconfiguration instead of leaving it buried in the server log.

    Returns:
        True if an explicit path was successfully applied (so callers can
        suppress auto-detection — a working hardcoded path wins). False when
        no path was configured, or it was missing / rejected.
    """
    if not config.simulator_exe:
        return False

    exe_path = config.simulator_exe
    target_name = config.simulator or "ltspice"
    if not exe_path.exists():
        msg = (
            f"Configured simulator path does not exist: {exe_path} "
            f"(requested '{target_name}'). Falling back to auto-detection."
        )
        logger.warning(msg)
        if diagnostics is not None:
            diagnostics.append(msg)
        return False

    # Guard against binding a path to the wrong simulator: if the exe filename
    # clearly belongs to a *different* known simulator than ``default``, skip it
    # (and fall back to auto-detection) rather than binding e.g. LTspice.exe to
    # ngspice — which silently mis-runs and hangs to timeout.
    guessed = _exe_simulator_hint(exe_path)
    if guessed is not None and guessed != target_name:
        msg = (
            f"Configured simulator path {exe_path} looks like a {guessed} "
            f"executable, but [simulator] default is '{target_name}'. Ignoring "
            f"the path to avoid binding the wrong binary — set the correct path "
            f"or change [simulator] default."
        )
        logger.warning(msg)
        if diagnostics is not None:
            diagnostics.append(msg)
        return False

    # Determine which simulator class to configure
    target_cls = SIMULATORS.get(target_name)
    if target_cls is None:
        msg = f"Unknown simulator '{target_name}' for exe override"
        logger.warning(msg)
        if diagnostics is not None:
            diagnostics.append(msg)
        return False

    try:
        target_cls.create_from(str(exe_path))
        logger.info(f"Applied simulator_exe override for {target_name}: {exe_path}")
        return True
    except Exception as e:
        msg = f"Failed to apply configured simulator path for {target_name}: {e}"
        logger.warning(msg)
        if diagnostics is not None:
            diagnostics.append(msg)
        return False


def bind_named_executable(base: type, selector: str, exe_path: Path) -> type:
    """A subclass of the family's ``base`` launching ``exe_path``.

    ``create_from`` writes the program onto the class it is called on
    (``docs/spicelib_bugs.md``, Bug 18), so it is called on a subclass: binding
    to ``base`` itself would retarget every runner holding it. The subclass
    keeps ``base.__name__``, which records, the raw dialect and the linter key
    on; a job's ``simulator_executable`` is what tells two builds apart.
    """
    cls = type(
        base.__name__,
        (base,),
        # Only for reading a traceback: the selector this class is bound to.
        {"__module__": __name__, "__qualname__": f"{base.__qualname__}[{selector}]"},
    )
    cls.create_from(exe_path)
    return cls


def _bind_entry(key: str, exe_path: Path, enabled: list[str]) -> tuple[str, type]:
    """The selector one configured entry binds, and its class. Raises
    ValueError saying why an entry binds none."""
    family, _, name = key.rpartition(":")
    if not _EXECUTABLE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"name {name!r} is not a valid executable name: lower-case letters, digits, "
            "'.', '_' and '-', starting with a letter or digit, at most 32 characters."
        )
    hint = _exe_simulator_hint(exe_path)
    family = family or hint
    if family is None:
        raise ValueError(
            f"its file name does not say which simulator it is; write the key as "
            f'"<family>:{name}" with a family from {list(_RUN_FAMILIES)}.'
        )
    if family not in _RUN_FAMILIES:
        raise ValueError(f"a run can be put on {list(_RUN_FAMILIES)} only, not {family!r}.")
    if hint is not None and hint != family:
        raise ValueError(
            f"{exe_path.name} looks like a {hint} executable, not {family}; it was not "
            "bound, to avoid running the wrong program."
        )
    if family not in enabled:
        raise ValueError(f"{family!r} is excluded by {SIM_SECTION}.{SIM_ENABLED_KEY}.")
    if not exe_path.is_file():
        wsl = (
            " On WSL, write a Windows path in its /mnt/<drive>/ form."
            if is_wsl() and PureWindowsPath(str(exe_path)).drive
            else ""
        )
        raise ValueError(f"{exe_path} does not exist or is not a file.{wsl}")
    selector = f"{family}:{name}"
    return selector, bind_named_executable(SIMULATORS[family], selector, exe_path)


def detect_named_simulators(
    config: ServerConfig | None,
    diagnostics: list[str] | None = None,
) -> dict[str, type]:
    """Bind each ``[simulator.executables]`` entry to a simulator class of its own,
    keyed by the selector a run names (``"ltspice:xvii"``). An entry that cannot
    be bound is left out with a diagnostic saying why."""
    if config is None or not config.simulator_executables or _detection_disabled():
        return {}
    enabled = _resolve_enabled_names(config)
    named: dict[str, type] = {}
    for key, exe_path in config.simulator_executables.items():
        try:
            selector, cls = _bind_entry(key, exe_path, enabled)
            if selector in named:
                raise ValueError(f"{selector} is named twice.")
        except Exception as exc:  # spicelib's own refusal to bind included
            _note(
                diagnostics,
                f"Named executable {SIM_SECTION}.{SIM_EXECUTABLES_KEY} "
                f"({SIM_EXECUTABLES_ENV}) entry {key!r} skipped: {exc}",
            )
            continue
        named[selector] = cls
        logger.info("Named executable %s: %s", selector, exe_path)
    return named


def _note(diagnostics: list[str] | None, message: str) -> None:
    logger.warning(message)
    if diagnostics is not None:
        diagnostics.append(message)


# spicelib's shipped ngspice compatibility default, captured before we ever
# override it, so an unset config can restore it. The attribute is process-wide,
# so a prior override in the same process (re-entrant config load, embedded use)
# would otherwise leak into a later unset config.
_SPICELIB_DEFAULT_NGBEHAVIOR: str = getattr(NGspiceSimulator, "_compatibility_mode", "kiltpsa")


def _apply_ngbehavior(config: ServerConfig | None) -> None:
    """Set ngspice's compatibility mode from ``config.ngbehavior`` (see that
    config field for why the shipped default breaks a sectioned ``.lib``).

    Writes spicelib's process-wide ``NGspiceSimulator._compatibility_mode`` (the
    ``-D ngbehavior=`` it injects on every run). Set once at startup, not per run.
    When unset it RESETS to spicelib's captured default rather than no-opping —
    otherwise a prior override would leak into a later re-entrant config load.
    """
    override = config.ngbehavior.strip().lower() if config and config.ngbehavior else ""
    mode = override or _SPICELIB_DEFAULT_NGBEHAVIOR
    # spicelib's public setter for the ``-D ngbehavior=`` it injects on every run.
    NGspiceSimulator.set_compatibility_mode(mode)
    if override and override != _SPICELIB_DEFAULT_NGBEHAVIOR:
        logger.info("Applied ngspice ngbehavior override: %s", override)


def current_ngbehavior() -> str | None:
    """Return the ngbehavior string ngspice runs with (spicelib class attribute).

    ``getattr`` (not attribute access) is deliberate: it reads the protected
    ``_compatibility_mode`` without tripping pyright's reportPrivateUsage, and
    spicelib exposes no public getter.
    """
    return getattr(NGspiceSimulator, "_compatibility_mode", None)


def generation_of(simulator_class: type | None) -> str | None:
    """Which LTspice a class launches, read off its program's file name.

    ``"xvii"`` for LTspice XVII (``XVIIx64.exe``), ``"current"`` for LTspice 24
    and later (``LTspice.exe``; the macOS app's ``LTspice`` too), None for a
    class that is not LTspice or a program whose name says neither (LTspice
    IV's ``scad3.exe``, a renamed copy, a wrapper script). The two keep their
    model libraries in different places, so a run's library roots are the
    ones of the build it runs on (``simulator_library_roots``).
    """
    from ltspice_mcp.lib.simulator_build import executable_path

    program = executable_path(simulator_class) if is_ltspice(simulator_class) else None
    if program is None:
        return None
    # PureWindowsPath splits on both separators, so a posix spice_exe and a
    # Windows one name the same file.
    name = PureWindowsPath(program).name.casefold()
    if "xvii" in name:
        return "xvii"
    if name.startswith("ltspice"):
        return "current"
    return None


def _in_generation(path: Path, generation: str | None) -> bool:
    """Whether one of spicelib's default library directories is ``generation``'s.

    spicelib lists every LTspice's library directory for every LTspice class
    and keeps whichever exist, so with XVII and a later build both installed a
    class would report both. XVII's are the ``LTspiceXVII`` folders.
    """
    if generation is None:
        return True
    xvii = any(part.casefold() == "ltspicexvii" for part in PureWindowsPath(str(path)).parts)
    return xvii if generation == "xvii" else not xvii


def simulator_library_roots(simulator_class: type | None) -> list[Path]:
    """Directories holding the detected simulator's own shipped model library.

    Same trust class as the ``.asy`` symbol paths resolved from the same
    install (see ``engine.configure_asc_editor``): both are read out of the
    simulator the server already runs, so a deck referencing a file inside one
    is naming the simulator, not the user's filesystem. Staging therefore
    accepts and snapshots them without the sandbox being widened — LTspice's
    ``.asc`` netlister appends ``.lib <install>/lib/cmp/standard.mos`` to every
    schematic carrying a MOSFET symbol, so under a default ``allowed_paths``
    no transistor schematic could otherwise be staged at all.

    Resolution mirrors the symbol path's: on WSL the install lives behind
    ``%LOCALAPPDATA%`` (``%USERPROFILE%\\Documents`` for XVII) and only the
    interop probe finds it, because spicelib's own derivation expands ``~``
    against the Linux home. The roots are the library of the build the class
    launches (``generation_of``), not of every LTspice on the machine: a
    server running XVII beside a later build, one per named executable, stages
    each run against its own simulator's library. Nonexistent directories and
    any root already contained in an earlier one are dropped, so the result is
    a minimal list of real directories.
    """
    if simulator_class is None:
        return []
    candidates: list[Path] = []
    generation = generation_of(simulator_class)
    if is_ltspice(simulator_class):
        from ltspice_mcp.lib.wsl import get_ltspice_lib_paths

        candidates += [Path(p) for p in get_ltspice_lib_paths(generation)]
    try:
        candidates += [
            Path(p)
            for p in simulator_class.get_default_library_paths()
            if _in_generation(Path(p), generation)
        ]
    except Exception as exc:
        # spicelib derives these from spice_exe and the platform; a simulator
        # class without one, or an install shape it does not know, must cost
        # the caller nothing beyond the roots already found.
        logger.debug(f"No default library paths for {simulator_class.__name__}: {exc}")

    found: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_dir() and resolved not in found:
            found.append(resolved)
    # Drop any root a broader one already covers, whichever order they arrived
    # in: the probe reports both ``lib`` and ``lib/sym``, and the nested one
    # adds no reach while giving the same files a second staging destination.
    return [
        root
        for root in found
        if not any(other != root and root.is_relative_to(other) for other in found)
    ]


def is_ltspice(simulator_class: type | None) -> bool:
    """True when the simulator is an LTspice: the family default, its WSL
    class, or a named executable of the family."""
    return simulator_family(simulator_class) == "ltspice"


def is_ngspice(simulator_class: type | None) -> bool:
    """True when the simulator is ngspice — whose compat-mode / sectioned-.lib
    quirks the ngbehavior diagnostic keys off. Checks class identity, not the
    RawRead dialect table (a header-parsing concern), so the two can't drift.
    """
    return simulator_family(simulator_class) == "ngspice"


def _autodetect_wsl_ltspice(diagnostics: list[str] | None = None) -> None:
    """Register LTspice on WSL by probing standard Windows install locations.

    spicelib's stock LTspice detection only searches Wine paths on Linux, so
    on WSL it never finds the Windows-side install under ``/mnt/<drive>/``.
    When no explicit ``simulator_exe`` has already populated ``spice_exe``,
    probe the common locations and register the first hit via ``create_from``.

    This is a no-op off WSL, when LTspice is already configured, or when no
    install is found. ``diagnostics`` (if provided) records a successful
    auto-detection so the user can see where it was found.
    """
    if not is_wsl():
        return

    ltspice_cls = SIMULATORS["ltspice"]
    # Already configured (explicit simulator_exe applied, or a prior call).
    if getattr(ltspice_cls, "spice_exe", None):
        return

    from ltspice_mcp.lib.wsl import find_windows_ltspice_exe

    exe = find_windows_ltspice_exe()
    if exe is None:
        return

    try:
        ltspice_cls.create_from(str(exe))
        logger.info(f"Auto-detected LTspice on WSL: {exe}")
        if diagnostics is not None:
            diagnostics.append(f"Auto-detected LTspice on WSL at {exe}.")
    except Exception as e:
        logger.warning(f"WSL LTspice auto-detection failed for {exe}: {e}")


# Recorded producer names identify the expected RAW dialect, including LTspice.
_DIALECT_MAP: dict[str, str] = {
    **_FAMILY_BY_CLASS_NAME,
    "LTspiceWSL": "ltspice",
}


def simulator_dialect(simulator_class: type | None) -> str | None:
    """Return the spicelib ``RawRead`` dialect for a simulator class.

    Known simulators, including LTspice and its WSL subclass, return explicit
    producing evidence so parser preflight can reject a contradictory writer.
    """
    return dialect_for_simulator_name(simulator_class.__name__) if simulator_class else None


def dialect_for_simulator_name(name: str | None) -> str | None:
    """``RawRead`` dialect for a simulator class *name* (e.g. ``job.simulator``).

    Same mapping as :func:`simulator_dialect` but keyed off the recorded name
    string, so a persisted job's dialect resolves even when that simulator is
    no longer configured (an ngspice sweep read back under an LTspice-only
    session still parses as ngspice). Unknown names return ``None``.
    """
    if not name:
        return None
    return _DIALECT_MAP.get(name)


def detect_simulators(
    config: ServerConfig | None = None,
    diagnostics: list[str] | None = None,
) -> dict[str, type]:
    """Detect available SPICE simulators on the system.

    If config is provided and has simulator_exe set, applies that override
    before running auto-detection. This allows WSL users to point to the
    Windows-side LTspice executable. On WSL, also probes standard Windows
    LTspice install locations (which spicelib's Wine-only search misses) so
    users don't have to hand-configure the path — but only when no explicit
    working path was pinned (a valid hardcoded path wins and suppresses
    auto-detection).

    ``config.enabled_simulators`` is an allowlist: when non-empty, only the
    listed simulators are probed/exposed. Empty (default) = probe all.

    Args:
        config: Optional server config with simulator_exe / enabled override.
        diagnostics: Optional list to collect human-readable notes about
            misconfiguration / fallback for surfacing via ``server_status``.

    Returns:
        Dictionary mapping simulator name to class for all available simulators.
        Returns empty dict if no simulators are detected (server can still start).
    """
    if _detection_disabled():
        logger.info("Simulator detection disabled via LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION")
        return {}

    # A valid explicit path takes control: when applied, skip auto-detection.
    applied = _apply_simulator_exe(config, diagnostics) if config is not None else False

    # Apply the ngspice compatibility-mode override (if configured) before any run.
    _apply_ngbehavior(config)
    hidden_desktop.configure(enabled=config is None or config.hidden_desktop)

    # Resolve the candidate set: empty allowlist = every supported simulator.
    names = _resolve_enabled_names(config, diagnostics)

    # On WSL, fill in LTspice from the Windows side — unless an explicit path
    # was already applied, or the user excluded ltspice from the allowlist.
    if not applied and "ltspice" in names:
        _autodetect_wsl_ltspice(diagnostics)

    available: dict[str, type] = {}

    for name in names:
        cls = SIMULATORS[name]
        try:
            if cls.is_available():
                logger.info(f"Detected simulator: {name}")
                available[name] = cls
            else:
                logger.debug(f"Simulator not available: {name}")
        except Exception as e:
            # spicelib may raise on import if platform incompatible
            logger.debug(f"Error checking {name} availability: {e}")

    if not available:
        logger.warning("No simulators detected - server will start in degraded mode")
    else:
        logger.info(f"Total simulators detected: {len(available)}")

    return available


def install_hint() -> str:
    """Platform-appropriate one-liner for getting a simulator onto this host.

    Backs the no-simulator startup instructions and the run-time "no simulator"
    error so a host with neither LTspice nor ngspice (a cloud sandbox, a fresh
    CI runner) gives the agent a concrete next step instead of a dead end.
    """
    # Every platform ends with the same clause: pointing the config at an
    # executable is the one route that exists everywhere, and the key it names
    # is what an agent needs to act on the message.
    configure = f"or set {SIM_PATH_ENV} ({SIM_SECTION}.{SIM_PATH_KEY} in ltspice-mcp.toml) to a "
    if is_wsl():
        return (
            "install ngspice (`sudo apt-get install -y ngspice`), "
            + configure
            + "Windows LTspice.exe path"
        )
    system = platform.system()
    if system == "Darwin":
        install = "install ngspice (`brew install ngspice`) or LTspice"
    elif system == "Windows":
        install = "install LTspice or ngspice and add it to PATH"
    else:
        install = "install ngspice (`sudo apt-get install -y ngspice`)"
    return f"{install}, {configure}simulator executable"


def _platform_key() -> str:
    if is_wsl():
        return "wsl"
    return {"Windows": "windows", "Darwin": "darwin"}.get(platform.system(), "linux")


# Realistic executable locations per simulator and platform, shown as the
# example value beside the config key that takes them. WSL reaches Windows
# binaries through /mnt/c — the LTspice example is the path this project's
# own development box uses. QSPICE has a Windows entry only: elsewhere it
# cannot run (``family_refusal``), so no path is suggested. The Windows Xyce
# path is the install location spicelib's own detection looks in.
_SIMULATOR_EXE_EXAMPLES: dict[str, dict[str, str]] = {
    "ltspice": {
        "wsl": "/mnt/c/Program Files/ADI/LTspice/LTspice.exe",
        "windows": "C:\\Program Files\\ADI\\LTspice\\LTspice.exe",
        "darwin": "/Applications/LTspice.app/Contents/MacOS/LTspice",
        "linux": "~/.wine/drive_c/Program Files/ADI/LTspice/LTspice.exe",
    },
    "ngspice": {
        "wsl": "/usr/bin/ngspice",
        "linux": "/usr/bin/ngspice",
        "darwin": "/opt/homebrew/bin/ngspice",
        "windows": "C:\\Spice64\\bin\\ngspice_con.exe",
    },
    "qspice": {
        "windows": "C:\\Program Files\\QSPICE\\QSPICE64.exe",
    },
    "xyce": {
        "linux": "/usr/local/bin/Xyce",
        "wsl": "/usr/local/bin/Xyce",
        "darwin": "/usr/local/bin/Xyce",
        "windows": "C:\\Program Files\\Xyce 7.9 NORAD\\bin\\xyce.exe",
    },
}


def simulator_remediation(name: str, config: ServerConfig) -> dict[str, object]:
    """How to make one undetected simulator available, as facts.

    Composed from the SAME constants the config loader reads
    (``SIM_SECTION``/``SIM_PATH_KEY``/``SIM_PATH_ENV`` — see config.py), so the
    key this tells a caller to set is the key the loader honors. When a
    non-empty allowlist is the reason the simulator is off, that is the first
    fact — pointing at an install would send the caller past the actual cause.
    """
    enabled = _resolve_enabled_names(config)
    excluded = name not in enabled
    key = f"{SIM_SECTION}.{SIM_PATH_KEY}"
    restart = "then restart this MCP server — detection runs at startup."
    refusal = family_refusal(name)
    if refusal is not None:
        # Installing it would not help: a run on it would still be refused, so
        # the platform is the fact to state, not a path to set.
        action = f"{refusal} Installing it on this host would not make it runnable."
    elif excluded:
        action = (
            f"'{name}' is excluded by {SIM_SECTION}.{SIM_ENABLED_KEY} = "
            f"{config.enabled_simulators} in {config.config_path}; add it there "
            f"(or clear the list), {restart}"
        )
    else:
        action = (
            f"Install {name}, or set {key} in {config.config_path} "
            f"(env {SIM_PATH_ENV}) to its executable; {restart}"
        )
    remediation: dict[str, object] = {
        "config_file": str(config.config_path),
        "config_key": key,
        "env_var": SIM_PATH_ENV,
        "excluded_by_allowlist": excluded,
        "action": action,
    }
    example = _SIMULATOR_EXE_EXAMPLES.get(name, {}).get(_platform_key())
    if example is not None:
        remediation["example_value"] = example
    return remediation


def no_simulator_message(short: bool = False) -> str:
    """Actionable 'no simulator detected' text shared by instructions and errors.

    Detection runs once at startup, so a simulator installed into a running
    sandbox is not picked up until the server is restarted — say so, or the
    agent installs ngspice and then loops on the same error.

    ``short`` is the compact form for the consolidated profile's instructions,
    which must fit a client-side truncation budget with the guide body intact.
    """
    if short:
        return (
            f"No SPICE simulator detected — {install_hint()}, then restart this "
            "MCP server (it detects at startup). Authoring and .asc editing "
            "still work."
        )
    return (
        f"No SPICE simulator detected. To run simulations, {install_hint()}, then "
        "restart (reconnect) this MCP server so it re-detects — detection happens "
        "at startup, so a just-installed simulator is invisible until then. If you "
        "cannot install one, tell the user. Netlist authoring/validation and .asc "
        "editing work without a simulator."
    )


def _resolve_enabled_names(
    config: ServerConfig | None,
    diagnostics: list[str] | None = None,
) -> list[str]:
    """Resolve which simulator names to probe from ``config.enabled_simulators``.

    Empty / unset allowlist → all supported simulators (preserving registry
    order). Unknown names are dropped with a diagnostic. If a non-empty
    allowlist contains no recognised names, returns an empty list (the user's
    explicit — if mistaken — intent; the diagnostics explain the degraded mode).
    """
    enabled = list(config.enabled_simulators) if config and config.enabled_simulators else []
    if not enabled:
        return list(SIMULATORS)

    names: list[str] = []
    for raw in enabled:
        name = raw.strip().lower()
        if name in SIMULATORS:
            if name not in names:
                names.append(name)
        else:
            msg = (
                f"Unknown simulator '{raw}' in [{SIM_SECTION}] {SIM_ENABLED_KEY} "
                f"(valid: {list(SIMULATORS)}); ignoring."
            )
            logger.warning(msg)
            if diagnostics is not None:
                diagnostics.append(msg)
    return names


def select_default_simulator(
    available: dict[str, type],
    config: ServerConfig,
    diagnostics: list[str] | None = None,
) -> type | None:
    """Select the default simulator based on config and availability.

    Selection logic:
    1. If config.simulator is set and available, use it
    2. If config.simulator is set but NOT available, log warning and fall back
    3. If no config preference, prefer LTSpice if available
    4. Otherwise use first available simulator
    5. If no simulators available, return None

    Args:
        available: Dictionary of available simulators from detect_simulators()
        config: Server configuration with simulator preference
        diagnostics: Optional list to collect a human-readable note when the
            requested simulator is unavailable and a fallback is chosen.

    Returns:
        Simulator class to use as default, or None if no simulators available
    """
    if not available:
        logger.warning("No simulators available - operations requiring simulation will fail")
        return None

    # Check user preference (case-insensitive, whitespace tolerant)
    if config.simulator:
        preferred = config.simulator.strip().lower()
        if preferred in available:
            logger.info(f"Using configured simulator: {preferred}")
            return available[preferred]
        else:
            # Requested simulator missing — pick a fallback and record WHY,
            # so the user isn't silently handed results from a simulator they
            # didn't ask for.
            fallback_name = "ltspice" if "ltspice" in available else next(iter(available))
            fallback = available[fallback_name]
            msg = (
                f"Requested simulator '{config.simulator}' is not available "
                f"(detected: {list(available.keys())}). Using '{fallback_name}' instead. "
                f"Results will come from {fallback_name}, not {config.simulator}. "
                f"To use '{config.simulator}', {install_hint()}, then restart; "
                "if it is installed, check it isn't excluded by [simulator] enabled."
            )
            logger.warning(msg)
            if diagnostics is not None:
                diagnostics.append(msg)
            return fallback

    # Prefer LTSpice if available (default when multiple simulators detected)
    if "ltspice" in available:
        logger.info("Defaulting to LTSpice (multiple simulators detected)")
        return available["ltspice"]

    # Use first available
    default_name = next(iter(available))
    logger.info(f"Using first available simulator: {default_name}")
    return available[default_name]
