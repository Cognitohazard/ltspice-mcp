"""Simulator detection and selection logic."""

import logging

from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.wsl import is_wsl

logger = logging.getLogger(__name__)


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


def _apply_simulator_exe(config: ServerConfig) -> None:
    """Apply config.simulator_exe to the appropriate spicelib simulator class.

    If the user has configured an explicit simulator executable path,
    use spicelib's create_from() to register it before auto-detection.
    This is essential for WSL where LTspice lives on the Windows side
    and spicelib's default search paths won't find it.
    """
    if not config.simulator_exe:
        return

    exe_path = config.simulator_exe
    if not exe_path.exists():
        logger.warning(f"Configured simulator_exe does not exist: {exe_path}")
        return

    # Determine which simulator class to configure
    target_name = config.simulator or "ltspice"
    target_cls = SIMULATORS.get(target_name)
    if target_cls is None:
        logger.warning(f"Unknown simulator '{target_name}' for exe override")
        return

    try:
        target_cls.create_from(str(exe_path))
        logger.info(f"Applied simulator_exe override for {target_name}: {exe_path}")
    except Exception as e:
        logger.warning(f"Failed to apply simulator_exe for {target_name}: {e}")


def detect_simulators(config: ServerConfig | None = None) -> dict[str, type]:
    """Detect available SPICE simulators on the system.

    If config is provided and has simulator_exe set, applies that override
    before running auto-detection. This allows WSL users to point to the
    Windows-side LTspice executable.

    Args:
        config: Optional server config with simulator_exe override.

    Returns:
        Dictionary mapping simulator name to class for all available simulators.
        Returns empty dict if no simulators are detected (server can still start).
    """
    if config is not None:
        _apply_simulator_exe(config)

    available: dict[str, type] = {}

    for name, cls in SIMULATORS.items():
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


def select_default_simulator(available: dict[str, type], config: ServerConfig) -> type | None:
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

    Returns:
        Simulator class to use as default, or None if no simulators available
    """
    if not available:
        logger.warning("No simulators available - operations requiring simulation will fail")
        return None

    # Check user preference
    if config.simulator:
        if config.simulator in available:
            logger.info(f"Using configured simulator: {config.simulator}")
            return available[config.simulator]
        else:
            logger.warning(
                f"Configured simulator '{config.simulator}' not available. "
                f"Some features may not work correctly. "
                f"Available simulators: {list(available.keys())}"
            )
            # Fall through to auto-select

    # Prefer LTSpice if available (default when multiple simulators detected)
    if "ltspice" in available:
        logger.info("Defaulting to LTSpice (multiple simulators detected)")
        return available["ltspice"]

    # Use first available
    default_name = next(iter(available))
    logger.info(f"Using first available simulator: {default_name}")
    return available[default_name]
