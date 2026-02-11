"""Simulator detection and selection logic."""

import logging
from typing import Type

from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.qspice_simulator import Qspice
from spicelib.simulators.xyce_simulator import XyceSimulator

from ltspice_mcp.config import ServerConfig

logger = logging.getLogger(__name__)

# Map simulator names to spicelib classes
SIMULATORS: dict[str, Type] = {
    "ltspice": LTspice,
    "ngspice": NGspiceSimulator,
    "qspice": Qspice,
    "xyce": XyceSimulator,
}


def detect_simulators() -> dict[str, Type]:
    """Detect available SPICE simulators on the system.

    Iterates through all known simulators and checks if they are available
    by calling their is_available() method. Handles platform-specific
    import errors gracefully.

    Returns:
        Dictionary mapping simulator name to class for all available simulators.
        Returns empty dict if no simulators are detected (server can still start).
    """
    available: dict[str, Type] = {}

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


def select_default_simulator(
    available: dict[str, Type], config: ServerConfig
) -> Type | None:
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
