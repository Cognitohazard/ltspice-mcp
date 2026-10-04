"""Typed frozen-input and execution facts for recoverable experiment attempts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import ConfigDict, TypeAdapter, ValidationError

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib.instance_targeting import SourceLineage
from ltspice_mcp.lib.ngspice_driver import validate_seed
from ltspice_mcp.lib.pdk_native import ArtifactDigest, NativeLaunchPolicy
from ltspice_mcp.lib.simulator_build import SimulatorExecutable
from ltspice_mcp.lib.store import validate_job_id

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class RecoveryError(SimulationError):
    """A refused recovery contract with a stable reason and structured evidence."""

    code = "recovery_refused"

    def __init__(self, code: str, message: str, *, evidence: dict[str, Any] | None = None):
        super().__init__(message, show_hint=False)
        self.code = code
        self.evidence = evidence or {}


class _Record:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    def to_record(self) -> dict[str, Any]:
        return TypeAdapter(type(self)).dump_python(self, mode="json")

    @classmethod
    def from_record(cls, data: Any) -> Self:
        try:
            # JSON permits only the documented path/datetime/tuple spellings;
            # strict decoding rejects bools as numbers and string coercions.
            adapter = TypeAdapter(cls)
            schema = adapter.json_schema()
            _strict_fields(data, schema, schema.get("$defs", {}))
            return adapter.validate_json(json.dumps(data), strict=True)
        except (ValidationError, ValueError, TypeError) as exc:
            raise RecoveryError("recovery_record_invalid", "Invalid recovery record") from exc


def _strict_fields(data: Any, schema: dict[str, Any], definitions: dict[str, Any]) -> None:
    """Forbid extra fields in reused dataclasses without changing their config."""
    if "$ref" in schema:
        schema = definitions[schema["$ref"].rsplit("/", 1)[1]]
    if data is None:
        return
    for branch in schema.get("anyOf", []):
        if branch.get("type") != "null":
            _strict_fields(data, branch, definitions)
    if isinstance(data, dict) and "properties" in schema:
        properties = schema["properties"]
        if data.keys() - properties.keys():
            raise ValueError("Unknown nested recovery record field")
        for name, value in data.items():
            _strict_fields(value, properties[name], definitions)
    if isinstance(data, list):
        if "prefixItems" in schema:
            for item, branch in zip(data, schema["prefixItems"], strict=False):
                _strict_fields(item, branch, definitions)
        elif "items" in schema:
            for item in data:
                _strict_fields(item, schema["items"], definitions)


def _positive(value: float | int, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} is outside its supported range")


def _artifact(value: ArtifactDigest) -> None:
    if not value.path.is_absolute() or not _DIGEST.fullmatch(value.sha256):
        raise ValueError("Artifact requires an absolute path and SHA256 digest")


@dataclass(frozen=True)
class ProcessIdentity(_Record):
    pid: int
    start_marker: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("Owner PID must be a positive integer")
        if not isinstance(self.start_marker, str) or not self.start_marker:
            raise ValueError("Owner process-start identity must be recorded")


@dataclass(frozen=True)
class StartupPolicy(_Record):
    version: str
    user_init_disabled: bool
    spinit: ArtifactDigest | None = None
    environment: tuple[tuple[str, str], ...] = ()
    ini_template: ArtifactDigest | None = None

    def __post_init__(self) -> None:
        if not self.version or type(self.user_init_disabled) is not bool:
            raise ValueError("Startup policy must name its version and init behavior")
        if self.spinit is not None:
            _artifact(self.spinit)
        if self.ini_template is not None:
            _artifact(self.ini_template)
            if self.spinit is not None:
                raise ValueError("Simulator startup inputs must remain distinct")
        keys = [key for key, _ in self.environment]
        if len(keys) != len(set(keys)) or any(not k or not v for k, v in self.environment):
            raise ValueError("Startup environment must have distinct nonempty keys and values")


@dataclass(frozen=True)
class ExecutionRecord(_Record):
    run_timeout_s: float | None
    timeout_source: str
    max_parallel: int
    job_deadline_s: float | None
    kill_grace_s: float
    simulator_argv: tuple[str, ...]
    executable: SimulatorExecutable
    ngbehavior: str | None
    platform: str
    startup: StartupPolicy
    native_policy: NativeLaunchPolicy | None = None
    #: One explicit seed for every case and retry; never a sampling policy.
    simulator_seed: int | None = None

    def __post_init__(self) -> None:
        if self.simulator_seed is not None:
            validate_seed(self.simulator_seed)
            if self.native_policy is not None:
                raise ValueError("Explicit simulator seeds cannot mix with native statistics")
        for name in ("run_timeout_s", "job_deadline_s"):
            if (value := getattr(self, name)) is not None:
                _positive(value, name)
        _positive(self.kill_grace_s, "kill_grace_s", zero=True)
        if type(self.max_parallel) is not int or self.max_parallel <= 0:
            raise ValueError("max_parallel must be a positive integer")
        if not self.timeout_source or not self.platform or not self.simulator_argv:
            raise ValueError("Execution source, platform and argv must be recorded")
        if any(not arg for arg in self.simulator_argv):
            raise ValueError("Simulator argv must contain nonempty arguments")
        exe = self.executable
        if (
            not exe.path
            or exe.sha256 is None
            or not _DIGEST.fullmatch(exe.sha256)
            or type(exe.bytes) is not int
            or exe.bytes <= 0
            or not exe.modified
        ):
            raise ValueError("Recoverable execution requires a complete executable identity")


@dataclass(frozen=True)
class JobRecovery(_Record):
    root_job_id: str
    root_request_id: str
    parent_job_id: str | None
    attempt_index: int
    owner: ProcessIdentity
    execution: ExecutionRecord

    def __post_init__(self) -> None:
        validate_job_id(self.root_job_id)
        if self.parent_job_id is not None:
            validate_job_id(self.parent_job_id)
        if not self.root_request_id or type(self.attempt_index) is not int:
            raise ValueError("Recovery requires root request and integer attempt identity")
        if self.attempt_index < 0 or (self.parent_job_id is None) != (self.attempt_index == 0):
            raise ValueError("Recovery parent and attempt identity disagree")


@dataclass(frozen=True)
class FrozenInputs(_Record):
    lineage_root: Path
    electrical: ArtifactDigest
    files: tuple[ArtifactDigest, ...]
    source_lineage: tuple[SourceLineage, ...] = ()

    def __post_init__(self) -> None:
        if not self.lineage_root.is_absolute() or not self.files:
            raise ValueError("Frozen inputs require an absolute lineage root and files")
        for artifact in (self.electrical, *self.files):
            _artifact(artifact)
            if not artifact.path.is_relative_to(self.lineage_root):
                raise ValueError("Frozen input path must be inside its lineage root")
        by_path = {item.path: item.sha256 for item in self.files}
        if (
            len(by_path) != len(self.files)
            or by_path.get(self.electrical.path) != self.electrical.sha256
        ):
            raise ValueError(
                "Frozen files must be distinct and include the exact electrical input"
            )


@dataclass(frozen=True)
class LaunchIntent(_Record):
    recorded_at: datetime
    electrical_sha256: str
    executed: ArtifactDigest
    adaptation: Literal["identity", "logopinfo", "native_driver", "seeded_driver"]

    def __post_init__(self) -> None:
        _artifact(self.executed)
        if not _DIGEST.fullmatch(self.electrical_sha256):
            raise ValueError("Launch intent requires the source electrical digest")
        if self.adaptation not in {"identity", "logopinfo", "native_driver", "seeded_driver"}:
            raise ValueError("Unsupported execution adaptation")


@dataclass(frozen=True)
class ProducedArtifacts(_Record):
    raw: ArtifactDigest
    log: ArtifactDigest
    completed_at: datetime

    def __post_init__(self) -> None:
        _artifact(self.raw)
        _artifact(self.log)
        if self.raw.path == self.log.path:
            raise ValueError("Raw and log artifacts must be distinct")


@dataclass(frozen=True)
class CaseAttempt(_Record):
    execution_job_id: str
    attempt_index: int
    run_token: str
    reused: bool = False
    launch: LaunchIntent | None = None
    outputs: ProducedArtifacts | None = None
    seeded_driver: ArtifactDigest | None = None

    def __post_init__(self) -> None:
        if self.seeded_driver is not None:
            _artifact(self.seeded_driver)
        validate_job_id(self.execution_job_id)
        if type(self.attempt_index) is not int or self.attempt_index < 0:
            raise ValueError("Case attempt must be a nonnegative integer")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_token):
            raise ValueError("Case attempt requires a safe run token")
        if type(self.reused) is not bool:
            raise ValueError("Reused must be a boolean")


@dataclass(frozen=True)
class CaseRecovery(_Record):
    inputs: FrozenInputs
    attempt: CaseAttempt

    def __post_init__(self) -> None:
        driver = self.attempt.seeded_driver
        if driver is not None and not driver.path.is_relative_to(self.inputs.lineage_root):
            raise ValueError("Seeded driver must be inside its lineage root")
        if self.attempt.launch is not None:
            if self.attempt.launch.electrical_sha256 != self.inputs.electrical.sha256:
                raise ValueError("Launch intent and frozen electrical input disagree")
            if not self.attempt.launch.executed.path.is_relative_to(self.inputs.lineage_root):
                raise ValueError("Executed input must be inside its lineage root")
        if self.attempt.outputs is not None:
            for artifact in (self.attempt.outputs.raw, self.attempt.outputs.log):
                if not artifact.path.is_relative_to(self.inputs.lineage_root):
                    raise ValueError("Produced artifact must be inside its lineage root")
