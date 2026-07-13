"""Simulator detection and selection logic."""

import logging
import os
import platform
from pathlib import Path

from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import ServerConfig
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
    via wslpath instead of using Wine's Z: drive mapping.
    """
    if is_wsl():
        from ltspice_mcp.lib.ltspice_wsl import LTspiceWSL

        return LTspiceWSL
    return LTspice


# Map simulator names to spicelib classes
SIMULATORS: dict[str, type] = {
    "ltspice": _get_ltspice_class(),
    "ngspice": NGspiceSimulator,
    "qspice": Qspice,
    "xyce": XyceSimulator,
}


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


def is_ngspice(simulator_class: type | None) -> bool:
    """True when the simulator is ngspice — whose compat-mode / sectioned-.lib
    quirks the ngbehavior diagnostic keys off. Checks class identity, not the
    RawRead dialect table (a header-parsing concern), so the two can't drift.
    """
    return simulator_class is not None and issubclass(simulator_class, NGspiceSimulator)


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


_DIALECT_MAP: dict[str, str] = {
    "NGspiceSimulator": "ngspice",
    "Qspice": "qspice",
    "XyceSimulator": "xyce",
}


def simulator_dialect(simulator_class: type | None) -> str | None:
    """Return the spicelib ``RawRead`` dialect for a simulator class.

    LTspice (and its WSL subclass) return ``None`` — spicelib auto-detects
    the dialect from the ``Command:`` field. Other simulators need an
    explicit hint because older versions (e.g. ngspice < 44) omit that
    header.
    """
    if simulator_class is None:
        return None
    return _DIALECT_MAP.get(simulator_class.__name__)


def dialect_for_simulator_name(name: str | None) -> str | None:
    """``RawRead`` dialect for a simulator class *name* (e.g. ``job.simulator``).

    Same mapping as :func:`simulator_dialect` but keyed off the recorded name
    string, so a persisted job's dialect resolves even when that simulator is
    no longer configured (an ngspice sweep read back under an LTspice-only
    session still parses as ngspice). LTspice / unknown names → ``None``
    (spicelib auto-detects from the ``Command:`` header).
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
    if is_wsl():
        return (
            "install ngspice in this WSL distro (`sudo apt-get install -y ngspice`), "
            "or set LTSPICE_MCP_SIMULATOR_EXE to a Windows LTspice.exe path"
        )
    system = platform.system()
    if system == "Darwin":
        return "install ngspice (`brew install ngspice`) or LTspice"
    if system == "Windows":
        return "install LTspice (Analog Devices) or ngspice and add it to PATH"
    return "install ngspice (`sudo apt-get install -y ngspice`, or your distro's package manager)"


def no_simulator_message() -> str:
    """Actionable 'no simulator detected' text shared by instructions and errors.

    Detection runs once at startup, so a simulator installed into a running
    sandbox is not picked up until the server is restarted — say so, or the
    agent installs ngspice and then loops on the same error.
    """
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
                f"Unknown simulator '{raw}' in [simulator] enabled "
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
