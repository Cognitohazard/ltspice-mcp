"""Per-launch ngspice startup settings for frozen experiment inputs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib import atomic_write_bytes
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.ngspice_driver import seeded_commands, validate_seed
from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY, ArtifactDigest
from ltspice_mcp.lib.recovery_records import ExecutionRecord, RecoveryError
from ltspice_mcp.lib.simulator_build import (
    SimulatorExecutable,
    executable_identity,
    same_executable,
)
from ltspice_mcp.lib.store import Store

if TYPE_CHECKING:
    from ltspice_mcp.lib.experiment_types import ExperimentCase


def _seeded_driver_bytes(case: ExperimentCase, execution: ExecutionRecord, root: Path) -> bytes:
    seed = execution.simulator_seed
    if seed is None or case.native_statistics is not None:
        raise RecoveryError(
            "recovery_seed_unsupported", "Explicit seed requires ordinary ngspice inputs"
        )
    assert case.recovery is not None
    try:
        source = case.recovery.inputs.electrical.path.relative_to(root).as_posix()
        commands = seeded_commands(seed, source, case.run_token + ".raw")
    except ValueError as exc:
        raise RecoveryError("recovery_seed_unsupported", str(exc)) from exc
    return ("Seeded ngspice initialization\n.control\n" + commands + ".endc\n.end\n").encode(
        "utf-8"
    )


def prepare_seeded_driver(
    case: ExperimentCase,
    execution: ExecutionRecord,
    store: Store,
    root_job_id: str,
    simulator: type,
) -> None:
    """Retain a deterministic driver separately from the original electrical closure."""
    if execution.simulator_seed is None:
        return
    if case.recovery is None:
        raise RecoveryError("recovery_record_invalid", "Seeded case requires frozen inputs")
    path = store.native_driver(root_job_id, case.run_token, simulator)
    root = case.recovery.inputs.lineage_root
    if path.parent != root:
        raise RecoveryError("recovery_path_escape", "Seeded driver belongs to a different lineage")
    content = _seeded_driver_bytes(case, execution, root)
    if path.exists() or path.is_symlink():
        raise RecoveryError(
            "recovery_input_drift", "Refusing to overwrite a previous seeded driver"
        )
    atomic_write_bytes(path, content)
    case.recovery = replace(
        case.recovery,
        attempt=replace(
            case.recovery.attempt, seeded_driver=ArtifactDigest(path, sha256_file(path))
        ),
    )
    verify_seeded_driver(case, execution, root)


def verify_seeded_driver(case: ExperimentCase, execution: ExecutionRecord, root: Path) -> None:
    """Verify seed, driver identity and exact deterministic bytes before each launch."""
    if case.recovery is None:
        raise RecoveryError("recovery_record_invalid", "Seeded case requires frozen inputs")
    driver = case.recovery.attempt.seeded_driver
    if execution.simulator_seed is None:
        if driver is not None:
            raise RecoveryError("recovery_seed_unsupported", "Driver has no recorded seed")
        return
    expected = _seeded_driver_bytes(case, execution, root)
    if driver is None or driver.path != root / (case.run_token + ".setup.cir"):
        raise RecoveryError(
            "recovery_record_invalid", "Seeded driver identity is missing or changed"
        )
    try:
        if not driver.path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise RecoveryError("recovery_path_escape", "Seeded driver escaped its lineage")
        if driver.path.read_bytes() != expected or sha256_file(driver.path) != driver.sha256:
            raise RecoveryError("recovery_input_drift", "Seeded driver bytes have changed")
    except OSError as exc:
        raise RecoveryError("recovery_artifact_missing", "Seeded driver is unavailable") from exc


def simulator_command(simulator: type, executable: SimulatorExecutable) -> tuple[str, ...]:
    """Freeze executable and launcher PATH resolution for capture and verification."""
    command = [str(part) for part in getattr(simulator, "spice_exe", ())]
    if not command:
        raise RecoveryError("recovery_execution_unknown", "Simulator command is missing")
    command[-1] = executable.path
    if len(command) > 1:
        command[0] = shutil.which(command[0]) or command[0]
    return tuple(command)


def verify_execution_policy(execution: ExecutionRecord, simulator: type) -> None:
    """Refuse changed executable, command, platform or native launch policy."""
    from spicelib.simulators.ltspice_simulator import LTspice

    if issubclass(simulator, LTspice):
        from ltspice_mcp.lib.controlled_ltspice import verify_ltspice_execution

        verify_ltspice_execution(execution)
    elif execution.startup.ini_template is not None:
        raise RecoveryError("recovery_startup_drift", "LTspice settings require LTspice")
    elif not issubclass(simulator, NGspiceSimulator):
        raise RecoveryError("recovery_startup_unsupported", "Simulator startup is not verified")
    if execution.simulator_seed is not None:
        try:
            validate_seed(execution.simulator_seed)
        except ValueError as exc:
            raise RecoveryError("recovery_seed_unsupported", str(exc)) from exc
        if not issubclass(simulator, NGspiceSimulator) or execution.native_policy is not None:
            raise RecoveryError(
                "recovery_seed_unsupported", "Explicit seed requires ordinary ngspice"
            )
    current = executable_identity(simulator)
    if current is None:
        raise RecoveryError("recovery_execution_changed", "Simulator command is unavailable")
    try:
        command = simulator_command(simulator, current)
    except RecoveryError as exc:
        raise RecoveryError(
            "recovery_execution_changed", "Simulator command is unavailable"
        ) from exc
    if (
        execution.platform != sys.platform
        or execution.simulator_argv != command
        or not same_executable(execution.executable, current)
        or execution.native_policy not in (None, LAUNCH_POLICY)
    ):
        raise RecoveryError("recovery_execution_changed", "Recorded execution policy has changed")


def controlled_ngspice(
    execution: ExecutionRecord, verify: Callable[[], None]
) -> type[NGspiceSimulator]:
    """Bind a spicelib adapter to recorded options without changing shared state.

    spicelib's run method has no subprocess-environment argument. This adapter
    retains its batch/raw/log convention and returns the exit code to the same
    RunTask; the existing runner still owns submission, permits and cancellation.
    """
    startup = execution.startup
    if not startup.user_init_disabled or startup.spinit is None:
        raise RecoveryError("startup_policy", "Controlled ngspice requires frozen startup")
    environment = dict(startup.environment)
    if environment != {"SPICE_SCRIPTS": str(startup.spinit.path.parent)}:
        raise RecoveryError("startup_policy", "Controlled ngspice startup directory disagrees")
    command = tuple(execution.simulator_argv)
    mode = execution.native_policy.ngbehavior if execution.native_policy else execution.ngbehavior

    class ControlledNGspice(NGspiceSimulator):
        spice_exe: ClassVar[list[str]] = list(command)
        _compatibility_mode = mode

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
            if cmd_line_switches and any(switch != "-n" for switch in cmd_line_switches):
                raise RecoveryError("startup_policy", "Unrecorded ngspice switches refused")
            verify()
            deck = Path(netlist_file)
            argv = [*command, "-n"]
            if mode:
                argv.extend(("-D", f"ngbehavior={mode}"))
            argv.extend(
                (
                    "-b",
                    "-o",
                    str(deck.with_suffix(".log")),
                    "-r",
                    str(deck.with_suffix(".raw")),
                    str(deck),
                )
            )
            # Other SPICE_* variables can redirect startup or model lookups.
            # A captured include closure needs neither ambient search path.
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("SPICE_")
            }
            env.update(environment)
            if exe_log:
                with deck.with_suffix(".exe.log").open("wb") as output:
                    return subprocess.run(
                        argv,
                        timeout=timeout,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        cwd=cwd,
                        env=env,
                    ).returncode
            return subprocess.run(
                argv,
                timeout=timeout,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
            ).returncode

    return ControlledNGspice
