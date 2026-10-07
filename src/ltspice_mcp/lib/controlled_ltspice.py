"""Native Windows LTspice recovery with established, independently copied settings.

The runner retains submission, permits, cancellation and completion. This
adapter only supplies the audited command and a private subprocess environment.
It never answers a dialog or changes the captured profile's effective values.

LTspice is started on the server's hidden desktop (``lib/hidden_desktop.py``)
where there is one, as every other LTspice launch on Windows is, so a recovery
run cannot take the keyboard focus. The launch is the audited one in every
other respect: the same command, working directory and environment, no stream
redirected and no handle inherited, and the timeout of ``subprocess.run``. A
message box there ends the attempt with what it said (``DialogError``) rather
than holding it to the timeout; none is ever answered. Where there is no
hidden desktop the launch is ``subprocess.run`` as audited.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.lib import atomic_write_bytes, hidden_desktop
from ltspice_mcp.lib.ltspice_windows import run_on_desktop
from ltspice_mcp.lib.pdk_native import ArtifactDigest
from ltspice_mcp.lib.recovery_records import ExecutionRecord, RecoveryError

STARTUP_VERSION = "ltspice-established-ini-v1"
# This build's startup decision and native profile getters were inspected.
# A new executable requires its own startup audit before adopting this policy.
AUDITED_EXECUTABLE_SHA256 = "a94eb1789084db9f46375cce05110e03578f9cdb931867a0faaca5b200793f06"
_REMINDER_SECONDS = 1_296_001
_MAX_PROFILE_BYTES = 1_048_576
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RecoveryError(
            "recovery_startup_unsupported", "Controlled LTspice requires native Windows"
        )


def _contained(path: Path, root: Path) -> None:
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not path.is_absolute()
        or path.resolve() != path
        or not path.is_relative_to(root)
    ):
        raise RecoveryError("recovery_path_escape", "LTspice startup path escaped its lineage")


def _template_path(template: Path, root: Path) -> None:
    _contained(template, root)
    if template != root / "startup" / "template.ini":
        raise RecoveryError("recovery_path_escape", "LTspice template path is incompatible")


def _attempt_path(path: Path, root: Path) -> None:
    _contained(path, root)
    if (
        path.name != "LTspice.ini"
        or path.parent.parent != root / "startup"
        or not _TOKEN.fullmatch(path.parent.name)
    ):
        raise RecoveryError("recovery_path_escape", "LTspice attempt path is incompatible")


def _profile_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            content = handle.read(_MAX_PROFILE_BYTES + 1)
    except OSError as exc:
        raise RecoveryError("recovery_startup_io", "LTspice settings are unavailable") from exc
    if not content or len(content) > _MAX_PROFILE_BYTES:
        raise RecoveryError("recovery_startup_unsupported", "LTspice settings size is unsupported")
    return content


def _read_profile(path: Path) -> dict[tuple[str, str], str]:
    """Read effective values through the same explicit-file Windows profile API.

    Whole-file decoding cannot recover the vendor's choices from a mixed ANSI
    and UTF-16 profile. Values stay private; errors never include their contents.
    """
    _require_windows()
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RecoveryError("recovery_startup_unsupported", "Windows profile API is unavailable")
    getter = loader("kernel32", use_last_error=True).GetPrivateProfileStringW
    getter.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_wchar_p,
    ]
    getter.restype = ctypes.c_uint32

    def query(section: str | None, key: str | None) -> str:
        buffer = ctypes.create_unicode_buffer(65536)
        count = getter(section, key, "", buffer, len(buffer), str(path))
        if count >= len(buffer) - 2:
            raise RecoveryError("recovery_startup_unsupported", "LTspice settings were truncated")
        return ctypes.wstring_at(ctypes.addressof(buffer), count)

    values = {}
    for section in query(None, None).split("\0"):
        if section:
            for key in query(section, None).split("\0"):
                if key:
                    values[(section.casefold(), key.casefold())] = query(section, key)
    return values


def _validate_profile(profile: dict[tuple[str, str], str], epoch_seconds: int) -> None:
    identifier = profile.get(("options", "uuid"), "")
    analytics = profile.get(("options", "captureanalytics"), "")
    query = profile.get(("options", "lastwebupdatequery"), "")
    # The audited vendor validates a fixed-width 20-digit string. Leading
    # zeros belong to that representation; an all-zero identifier is refused.
    if (
        not re.fullmatch(r"[0-9]{20}", identifier)
        or identifier == "0" * 20
        or analytics.casefold() not in {"true", "false"}
    ):
        raise RecoveryError(
            "recovery_startup_unsupported", "LTspice requires an established identifier and choice"
        )
    # The audited getter parses a signed 32-bit integer. Reject alternate
    # spellings, overflow, absent decisions and clocks preceding the decision.
    if not re.fullmatch(r"[1-9][0-9]{0,9}", query) or int(query) > 2_147_483_647:
        raise RecoveryError("recovery_startup_unsupported", "LTspice update decision is invalid")
    age = epoch_seconds - int(query)
    if not 0 <= age < _REMINDER_SECONDS:
        raise RecoveryError(
            "recovery_startup_stale",
            "LTspice update decision is stale or in the future; prepare a new established profile",
        )


def capture_ini_template(source: Path, destination: Path, lineage_root: Path) -> ArtifactDigest:
    """Retain exact source bytes and prove effective values survive the new path."""
    _require_windows()
    _template_path(destination, lineage_root)
    if not source.is_absolute():
        raise RecoveryError("recovery_startup_unsupported", "LTspice source must be explicit")
    content = _profile_bytes(source)
    profile = _read_profile(source)
    _validate_profile(profile, int(time.time()))
    if destination.exists() or destination.is_symlink():
        raise RecoveryError("recovery_startup_drift", "Refusing to overwrite an LTspice template")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, content, overwrite=False, durable=True)
    except OSError as exc:
        raise RecoveryError("recovery_startup_io", "Cannot capture LTspice settings") from exc
    if _profile_bytes(source) != content or _read_profile(destination) != profile:
        raise RecoveryError(
            "recovery_startup_drift", "LTspice effective settings changed in capture"
        )
    template = ArtifactDigest(destination, hashlib.sha256(content).hexdigest())
    verify_ini_template(template, lineage_root)
    return template


def verify_ini_template(template: ArtifactDigest, root: Path) -> None:
    """Verify immutable bytes and the actual decision's remaining startup window."""
    _require_windows()
    _template_path(template.path, root)
    if hashlib.sha256(_profile_bytes(template.path)).hexdigest() != template.sha256:
        raise RecoveryError("recovery_startup_drift", "LTspice template bytes changed")
    _validate_profile(_read_profile(template.path), int(time.time()))


def prepare_attempt_ini(template: ArtifactDigest, path: Path, root: Path) -> None:
    """Create one writable profile exclusively; previous attempts remain intact."""
    verify_ini_template(template, root)
    _attempt_path(path, root)
    if path.exists() or path.is_symlink():
        raise RecoveryError("recovery_startup_drift", "Refusing to overwrite an LTspice attempt")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, _profile_bytes(template.path), overwrite=False, durable=True)
    except OSError as exc:
        raise RecoveryError(
            "recovery_startup_io", "Cannot prepare LTspice attempt settings"
        ) from exc
    verify_attempt_ini(template, path, root)


def verify_attempt_ini(template: ArtifactDigest, path: Path, root: Path) -> None:
    """Check initial bytes immediately before spawn, never after vendor writeback."""
    verify_ini_template(template, root)
    _attempt_path(path, root)
    try:
        shared_links = path.stat().st_nlink != 1
    except OSError as exc:
        raise RecoveryError(
            "recovery_startup_io", "LTspice attempt settings are unavailable"
        ) from exc
    if shared_links:
        raise RecoveryError(
            "recovery_startup_drift", "LTspice writable settings have shared links"
        )
    if hashlib.sha256(_profile_bytes(path)).hexdigest() != template.sha256:
        raise RecoveryError("recovery_startup_drift", "LTspice attempt settings changed")
    if _read_profile(path) != _read_profile(template.path):
        raise RecoveryError(
            "recovery_startup_drift", "LTspice attempt effective settings differ from the template"
        )


def verify_ltspice_execution(execution: ExecutionRecord) -> None:
    """Refuse unverified builds and policies before recovery claims or launches."""
    _require_windows()
    startup = execution.startup
    if execution.executable.sha256 != AUDITED_EXECUTABLE_SHA256:
        raise RecoveryError(
            "recovery_startup_unsupported",
            "This LTspice executable has no controlled startup audit",
        )
    if (
        execution.platform != "win32"
        or execution.simulator_argv != (execution.executable.path,)
        or execution.simulator_seed is not None
        or execution.native_policy is not None
        or execution.ngbehavior is not None
        or startup.version != STARTUP_VERSION
        or startup.ini_template is None
        or startup.spinit is not None
        or startup.user_init_disabled
        or startup.environment
    ):
        raise RecoveryError(
            "recovery_startup_unsupported", "Controlled LTspice policy is incompatible"
        )


def controlled_ltspice(
    execution: ExecutionRecord, ini_path: Path, verify: Callable[[], None]
) -> type[LTspice]:
    """Bind one spicelib launch to its fresh profile without shared env changes."""
    verify_ltspice_execution(execution)
    template = execution.startup.ini_template
    assert template is not None
    root = template.path.parent.parent
    _attempt_path(ini_path, root)
    command = tuple(execution.simulator_argv)

    class ControlledLTspice(LTspice):
        spice_exe: ClassVar[list[str]] = list(command)
        process_name = Path(command[0]).name

        @classmethod
        def run(
            cls,
            netlist_file: str | Path,
            cmd_line_switches: list | None = None,
            timeout: float | None = None,
            stdout=None,
            stderr=None,
            cwd: str | Path | None = None,
            exe_log: bool = False,
        ) -> int:
            if cmd_line_switches:
                raise RecoveryError("startup_policy", "Unrecorded LTspice switches refused")
            if stdout is not None or stderr is not None:
                raise RecoveryError("startup_policy", "LTspice stream overrides refused")
            verify_ltspice_execution(execution)
            verify()
            verify_attempt_ini(template, ini_path, root)
            deck = Path(netlist_file)
            # Keep -ini and its operand distinct, bypassing spicelib's typo.
            argv = [*command, "-Run", "-b", str(deck), "-ini", str(ini_path)]
            env = dict(os.environ)
            env["APPDATA"] = str(ini_path.parent)
            # Keep the measured startup without stream redirection. Ignore
            # exe_log: LTspice's .log and .raw carry results.
            desktop = hidden_desktop.shared()
            if desktop is not None:
                # Naming no stream leaves STARTF_USESTDHANDLES unset and the
                # handle list empty, so nothing is inherited, as below.
                return run_on_desktop(desktop, argv, timeout=timeout, cwd=cwd, env=env)
            # Omitting every stream also avoids STARTF_USESTDHANDLES; closing
            # descriptors disables Windows process handle inheritance.
            return subprocess.run(
                argv,
                timeout=timeout,
                close_fds=True,
                cwd=cwd,
                env=env,
            ).returncode

    return ControlledLTspice
