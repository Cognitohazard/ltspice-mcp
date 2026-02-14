"""WSL detection and path conversion utilities."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache WSL detection result at module level (won't change during process lifetime)
_is_wsl_cached: bool | None = None


def is_wsl() -> bool:
    """Detect if running in WSL (Windows Subsystem for Linux).

    Uses environment variable check (fast path) and /proc/version fallback.
    Result is cached at module level for performance.

    Returns:
        True if running in WSL, False otherwise
    """
    global _is_wsl_cached

    if _is_wsl_cached is not None:
        return _is_wsl_cached

    # Fast path: check environment variable set by WSL
    if os.getenv("WSL_DISTRO_NAME"):
        logger.debug(f"WSL detected via WSL_DISTRO_NAME: {os.getenv('WSL_DISTRO_NAME')}")
        _is_wsl_cached = True
        return True

    # Fallback: check for "microsoft" in kernel version
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info:
                logger.debug("WSL detected via /proc/version")
                _is_wsl_cached = True
                return True
    except Exception as e:
        # File not found or read error - probably not Linux at all
        logger.debug(f"Could not read /proc/version: {e}")

    _is_wsl_cached = False
    return False


def to_windows_path(linux_path: Path) -> str:
    """Convert Linux path to Windows path for WSL interop.

    Uses the built-in wslpath utility when running in WSL.
    Falls back to returning the path as-is if not in WSL or if conversion fails.

    Args:
        linux_path: Linux-style path to convert

    Returns:
        Windows-style path (e.g., C:\\Users\\...) if in WSL, otherwise str(linux_path)
    """
    # Skip conversion if not in WSL
    if not is_wsl():
        return str(linux_path)

    try:
        result = subprocess.run(
            ["wslpath", "-w", str(linux_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        windows_path = result.stdout.strip()
        logger.debug(f"Converted path: {linux_path} -> {windows_path}")
        return windows_path
    except FileNotFoundError:
        # wslpath not found (should never happen in WSL, but be defensive)
        logger.warning(f"wslpath utility not found, using Linux path: {linux_path}")
        return str(linux_path)
    except subprocess.CalledProcessError as e:
        # Conversion failed (invalid path?)
        logger.warning(
            f"Path conversion failed for {linux_path}: {e.stderr.strip() if e.stderr else str(e)}"
        )
        return str(linux_path)
