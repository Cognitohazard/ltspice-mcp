"""Prepare owned native setup artifacts before an experiment becomes durable."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import ExperimentJob, failure_row
from ltspice_mcp.lib.pdk_native import (
    NativeCaseError,
    NativePaths,
    SimulatorFacts,
    prepare_launch,
    validate_seed_collisions,
)
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import Store


def observe_simulator(simulator: type) -> SimulatorFacts:
    """Record the actual executable and its reported build once per job."""
    command = list(getattr(simulator, "spice_exe", ()))
    version = build = None
    # The same cached digest the job record carries, so one job never holds
    # two hashes of its simulator taken by two routes.
    identity = executable_identity(simulator)
    digest = identity.sha256 if identity is not None else None
    if command:
        try:
            completed = subprocess.run(
                [*command, "-n", "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if completed.returncode == 0:
                banner = completed.stdout + completed.stderr
                matched = re.search(r"(?i)\bngspice[- ]+(\d[^\s:]*)", banner)
                if matched:
                    version = "ngspice-" + matched[1]
                build_lines = [
                    line.strip("* \t")
                    for line in banner.splitlines()
                    if any(
                        word in line.casefold() for word in ("compiled", "creation date", "build")
                    )
                ]
                build = "\n".join(build_lines) or None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return SimulatorFacts(version, build, digest, sys.platform)


def prepare_native_cases(job: ExperimentJob, working_dir: Path, simulator: type) -> None:
    """Mutate only the unpublished candidate; normal case failures preserve peers."""
    native = [case for case in job.cases if case.native_statistics is not None]
    if not native:
        return
    validate_seed_collisions(
        [
            record.sample
            for case in native
            if (record := case.native_statistics) is not None and record.sample is not None
        ]
    )
    facts = (
        observe_simulator(simulator) if any(case.status == "queued" for case in native) else None
    )
    store = Store(working_dir)
    for case in native:
        record = case.native_statistics
        assert record is not None
        if case.status != "queued":
            record.unavailable_reason = case.error or case.status
            continue
        record.simulator = facts
        try:
            if record.sample is None:
                raise NativeCaseError("validation", "native sample has not been validated")
            input_path = store.native_input(job.job_id, case.run_token, simulator)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            driver = store.native_driver(job.job_id, case.run_token, simulator)
            paths = NativePaths(
                input_path.parent,
                input_path,
                driver,
                input_path.parent / f"{case.run_token}.cir",
                input_path.parent / f"{case.run_token}.raw",
                input_path.parent / f"{case.run_token}.log",
            )
            content = case.staged_deck.read_bytes()
            if hashlib.sha256(content).hexdigest() != case.deck_sha256:
                raise NativeCaseError("input_drift", "electrical case changed before preparation")
            record.prepared = prepare_launch(
                record.sample,
                paths=paths,
                token=case.run_token,
                electrical_bytes=content,
                dependencies=record.pending_dependencies,
            )
            record.pending_dependencies = ()
            record.unavailable_reason = "not submitted"
        except (NativeCaseError, OSError) as exc:
            case.status = "failed"
            case.failure_code = "pdk_native_preparation"
            case.error = str(exc)
            case.failure_evidence = {
                "reason": exc.code if isinstance(exc, NativeCaseError) else "io"
            }
            case.completed_at = now()
            record.unavailable_reason = str(exc)
            record.pending_dependencies = ()
    job.completeness.recount(job.cases)
    job.failures = [
        failure_row(case)
        for case in job.cases
        if case.status in {"failed", "skipped", "cancelled"}
    ]
