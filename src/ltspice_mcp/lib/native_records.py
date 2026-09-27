"""Durable native statistical facts, shared by jobs and result identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.pdk_native import (
    ArtifactDigest,
    MosCoverage,
    NativePaths,
    NativeRequest,
    Occurrence,
    PreparedLaunch,
    SimulatorFacts,
    ValidatedSample,
    provenance,
)


@dataclass
class NativeCaseRecord:
    request: NativeRequest
    sample: ValidatedSample | None = None
    prepared: PreparedLaunch | None = None
    simulator: SimulatorFacts | None = None
    unavailable_reason: str = "not validated"
    # Candidate-only handoff to preparation before the first durable write.
    # Successful preparation carries these hashes in PreparedLaunch instead.
    pending_dependencies: tuple[ArtifactDigest, ...] = field(default=(), repr=False)

    def public(self, *, detailed: bool = False) -> dict[str, Any]:
        return provenance(
            self.request,
            sample=self.sample,
            prepared=self.prepared,
            simulator=self.simulator,
            unavailable_reason=self.unavailable_reason,
            detailed=detailed,
        )

    def to_record(self) -> dict[str, Any]:
        # The store's JSON writer already encodes Path values. Keep the typed
        # record separate from its public projection so reload loses no facts.
        record = asdict(self)
        record.pop("pending_dependencies")
        return record

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> NativeCaseRecord:
        request = NativeRequest(**data["request"])
        sample = None
        if raw := data.get("sample"):
            sample = ValidatedSample(
                request=request,
                sample_key=raw["sample_key"],
                effective_seed=raw["effective_seed"],
                input_digest=raw["input_digest"],
                model_digest=raw["model_digest"],
                population_digest=raw["population_digest"],
                coverage=tuple(_coverage(item) for item in raw["coverage"]),
                analysis=raw["analysis"],
                hierarchy_revision=raw["hierarchy_revision"],
                dependency_captures=tuple(raw["dependency_captures"]),
            )
        prepared = None
        if raw := data.get("prepared"):
            prepared = PreparedLaunch(
                paths=NativePaths(**{key: Path(value) for key, value in raw["paths"].items()}),
                input_sha256=raw["input_sha256"],
                driver_sha256=raw["driver_sha256"],
                dependencies=tuple(
                    ArtifactDigest(Path(item["path"]), item["sha256"], item["original_capture"])
                    for item in raw["dependencies"]
                ),
                sample_key=raw["sample_key"],
                effective_seed=raw["effective_seed"],
                analysis=raw["analysis"],
            )
        facts = data.get("simulator")
        result = cls(
            request=request,
            sample=sample,
            prepared=prepared,
            simulator=SimulatorFacts(**facts) if facts is not None else None,
            unavailable_reason=data.get("unavailable_reason", "not validated"),
        )
        result.public()  # Refuse inconsistent request/sample/preparation identities.
        return result


def _occurrence(data: dict[str, Any]) -> Occurrence:
    return Occurrence(data["capture"], data["line"], data.get("section"), tuple(data["scope"]))


def _coverage(data: dict[str, Any]) -> MosCoverage:
    return MosCoverage(
        instance=tuple(data["instance"]),
        source=_occurrence(data["source"]),
        wrapper=_occurrence(data["wrapper"]),
        models=tuple(_occurrence(item) for item in data["models"]),
        width_m=data["width_m"],
        length_m=data["length_m"],
        scale=data["scale"],
        m=data["m"],
        mult=data["mult"],
        nf=data["nf"],
    )
