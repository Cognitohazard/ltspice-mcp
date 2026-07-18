"""WSL detection and path conversion utilities."""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Process names the Windows LTspice binary may run as (current ADI vs legacy LTC).
_LTSPICE_PROCESS_NAMES = ("LTspice.exe", "XVIIx64.exe", "scad3.exe")

# One budget for every wslpath/cmd.exe/taskkill interop hop: a wedged Windows
# process must fail the request (or fall back), not hang it forever.
_WSL_INTEROP_TIMEOUT_S = 15

# A job_id / run_filename token is server-generated and safe, but validate it
# before splicing into a PowerShell command so a future caller can't inject.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")

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
        with open("/proc/version") as f:
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
            timeout=_WSL_INTEROP_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
        windows_path = result.stdout.strip()
        logger.debug(f"Converted path: {linux_path} -> {windows_path}")
        return windows_path
    except FileNotFoundError:
        # wslpath not found (should never happen in WSL, but be defensive)
        logger.warning(f"wslpath utility not found, using Linux path: {linux_path}")
        return str(linux_path)
    except subprocess.TimeoutExpired:
        # A hung wslpath must not wedge the run request forever — fall back
        # to the Linux path like any other conversion failure.
        logger.warning(f"wslpath timed out converting {linux_path}, using Linux path")
        return str(linux_path)
    except subprocess.CalledProcessError as e:
        # Conversion failed (invalid path?)
        logger.warning(
            f"Path conversion failed for {linux_path}: {e.stderr.strip() if e.stderr else str(e)}"
        )
        return str(linux_path)


# Cache for resolved Windows env-var paths (constant for the process lifetime)
_win_env_cache: dict[str, Path | None] = {}


def _resolve_win_env(var: str) -> Path | None:
    """Resolve a Windows environment variable to a WSL path.

    Runs cmd.exe to echo the variable, then wslpath to convert. Results are
    memoized for the process lifetime (these vars don't change), so repeated
    callers — e.g. find_windows_ltspice_exe and get_ltspice_lib_paths both
    resolving %LOCALAPPDATA% at startup — share a single pair of subprocess
    spawns instead of re-running cmd.exe each time. Returns None if resolution
    fails.
    """
    if var in _win_env_cache:
        return _win_env_cache[var]
    try:
        win_result = subprocess.run(
            ["cmd.exe", "/C", "echo", f"%{var}%"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_WSL_INTEROP_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
        wsl_result = subprocess.run(
            ["wslpath", "-u", win_result.stdout.strip()],
            capture_output=True,
            text=True,
            check=True,
            timeout=_WSL_INTEROP_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
        resolved = Path(wsl_result.stdout.strip())
    except Exception as e:
        # Broad by design: a hung cmd.exe/wslpath (TimeoutExpired) must not
        # wedge startup — treat any failure, timeout included, as unresolved.
        # Warn (not debug) to match to_windows_path. The consequence depends
        # on the variable (%LOCALAPPDATA% degrades LTspice detection and .asc
        # symbols; %TEMP% only drops the output dir to a fixed fallback), so
        # name the variable and let the caller's behavior speak.
        # Do NOT cache the failure — cache only successes below — so a later
        # call can retry after a transient timeout (callers hit this once at
        # startup, so an uncached failure costs nothing).
        logger.warning(
            f"Could not resolve Windows %{var}% via cmd.exe interop; features "
            f"that depend on it will degrade or fall back this session: {e}"
        )
        return None
    _win_env_cache[var] = resolved
    return resolved


# Cache for Windows temp dir
_win_temp_dir: Path | None = None


def get_windows_output_dir() -> Path | None:
    """Get a Windows-native temp directory for LTspice output on WSL.

    LTspice writes .db (SQLite) files alongside simulation output.
    SQLite fails on UNC paths (\\\\wsl.localhost\\...), which causes .MEAS
    results to be lost from .log files. Using a Windows-native output dir
    (under /mnt/c/) avoids this issue.

    Returns:
        Path under /mnt/c/ for simulation output, or None if not on WSL
        or if the Windows temp dir can't be resolved.
    """
    global _win_temp_dir

    if not is_wsl():
        return None

    if _win_temp_dir is not None:
        return _win_temp_dir

    temp_path = _resolve_win_env("TEMP")
    if temp_path is not None:
        win_temp = temp_path / "ltspice-mcp"
    else:
        win_temp = Path("/mnt/c/temp/ltspice-mcp")

    try:
        win_temp.mkdir(parents=True, exist_ok=True)
        _win_temp_dir = win_temp
        logger.info(f"WSL output directory: {win_temp}")
        return _win_temp_dir
    except OSError as e:
        logger.warning(f"Failed to create Windows output dir {win_temp}: {e}")
        return None


def is_windows_native_path(path: Path) -> bool:
    """Check if a path is on a WSL-mounted Windows drive (e.g. /mnt/c/...).

    WSL mounts each Windows drive under /mnt/<letter> where <letter> is a
    single character. Only such paths sit on the actual Windows filesystem;
    other /mnt/ entries (e.g. /mnt/cdrom, /mnt/extdata) are unrelated mounts.
    """
    try:
        parts = path.resolve().parts
    except OSError:
        return False
    return len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1


def get_ltspice_lib_paths() -> list[str]:
    """Discover LTspice symbol library paths on WSL.

    spicelib's AscEditor needs .asy symbol files to parse .asc schematics.
    On WSL, spicelib's default path expansion fails because it only handles
    Wine paths (/drive_c/), not WSL's /mnt/c/ mount. This function probes
    the standard LTspice library locations via the Windows user profile.

    Returns:
        List of existing library paths (may be empty if not on WSL or
        LTspice libs aren't found).
    """
    if not is_wsl():
        return []

    local_appdata = _resolve_win_env("LOCALAPPDATA")
    if local_appdata is None:
        return []

    # LTspice stores symbols under <LOCALAPPDATA>/LTspice/lib/sym
    candidates = [
        local_appdata / "LTspice" / "lib",
        local_appdata / "LTspice" / "lib" / "sym",
    ]

    found = []
    for candidate in candidates:
        if candidate.is_dir():
            found.append(str(candidate))
            logger.debug(f"Found LTspice library path: {candidate}")

    if not found:
        logger.debug("No LTspice library paths found on WSL")

    return found


def find_windows_ltspice_exe() -> Path | None:
    """Probe standard Windows LTspice install locations from WSL.

    spicelib's stock detection only searches Wine paths (``~/.wine/...``)
    on Linux, so on WSL — where the real install lives under ``/mnt/<drive>/``
    and there is usually no Wine — LTspice is undetectable unless an explicit
    path is configured. This probes the common Windows install locations via
    the resolved Windows env vars and returns the first existing executable.

    Order reflects current installers first:
      - ``%LOCALAPPDATA%\\Programs\\ADI\\LTspice\\LTspice.exe`` (current ADI)
      - ``%ProgramFiles%\\ADI\\LTspice\\LTspice.exe`` (older ADI machine-wide)
      - ``%ProgramFiles(x86)%\\LTC\\LTspiceXVII\\XVIIx64.exe`` and the
        LTspiceIV ``scad3.exe`` (legacy LTC builds)

    Returns:
        Path to the first existing LTspice executable, or None (not on WSL,
        Windows env vars unresolvable, or no install found).
    """
    if not is_wsl():
        return None

    candidates: list[Path] = []

    local_appdata = _resolve_win_env("LOCALAPPDATA")
    if local_appdata is not None:
        candidates.append(local_appdata / "Programs" / "ADI" / "LTspice" / "LTspice.exe")

    for var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = _resolve_win_env(var)
        if base is None:
            continue
        candidates.append(base / "ADI" / "LTspice" / "LTspice.exe")
        candidates.append(base / "LTC" / "LTspiceXVII" / "XVIIx64.exe")
        candidates.append(base / "LTC" / "LTspiceIV" / "scad3.exe")

    for candidate in candidates:
        if candidate.is_file():
            logger.debug(f"Found LTspice executable on WSL: {candidate}")
            return candidate

    logger.debug("No LTspice executable found in standard WSL/Windows locations")
    return None


def kill_windows_ltspice_by_token(token: str) -> int:
    """Terminate Windows LTspice processes whose command line contains ``token``.

    On WSL the simulator runs as a *Windows* process launched via interop; the
    Linux process table only shows the ``/init`` relay (named ``init``), and
    killing that relay does NOT terminate the Windows process (verified). So
    spicelib's ``kill_all_spice`` — which matches the Windows executable name
    against the *Linux* ``psutil`` table — finds nothing and is a no-op here.

    This locates the specific job's Windows process by the unique ``token`` (the
    job_id, which appears in the ``-Run -b ...\\<job_id>.cir`` command line) via
    PowerShell ``Get-CimInstance`` and terminates it with ``taskkill /F``.

    Returns the number of processes killed. No-op (returns 0) off WSL, on an
    unsafe/empty token, or if the Windows queries fail — degrading to the prior
    behaviour rather than raising.
    """
    if not is_wsl():
        return 0
    if not token or not _SAFE_TOKEN_RE.match(token):
        logger.warning("kill_windows_ltspice_by_token: refusing unsafe token %r", token)
        return 0

    name_filter = " or ".join(f"Name='{name}'" for name in _LTSPICE_PROCESS_NAMES)
    # Anchor the token at a run-filename boundary, mirroring the Linux twin
    # proc_kill._token_in_arg: the staged deck is ``{job_id}.{ext}`` (single
    # runs) or ``{job_id}_{n}.{ext}`` (batch sub-runs), so the id is always
    # followed by '.' or '_'. Requiring that boundary keeps a job id from
    # matching a longer id it happens to prefix. The token is validated above,
    # so it carries no PowerShell wildcard metacharacters.
    token_filter = f"$_.CommandLine -like '*{token}.*' -or $_.CommandLine -like '*{token}_*'"
    ps_script = (
        f'Get-CimInstance Win32_Process -Filter "{name_filter}" '
        f"| Where-Object {{ {token_filter} }} "
        "| ForEach-Object { $_.ProcessId }"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("kill_windows_ltspice_by_token: process query failed: %s", e)
        return 0

    pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    killed = 0
    for pid in pids:
        try:
            kill = subprocess.run(
                ["taskkill.exe", "/F", "/PID", pid],
                capture_output=True,
                text=True,
                timeout=_WSL_INTEROP_TIMEOUT_S,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("kill_windows_ltspice_by_token: taskkill PID %s raised: %s", pid, e)
            continue
        if kill.returncode == 0:
            killed += 1
        else:
            logger.warning(
                "kill_windows_ltspice_by_token: taskkill PID %s failed: %s",
                pid,
                (kill.stdout or kill.stderr).strip(),
            )
    if killed:
        logger.info("Killed %d Windows LTspice process(es) for token %s", killed, token)
    return killed
