"""Native statistical facts survive durable jobs and public run collection."""

from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pdk_native import (
    PROFILE,
    ArtifactDigest,
    NativePaths,
    NativeRequest,
    PreparedLaunch,
    SimulatorFacts,
    ValidatedSample,
)
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests.test_experiment_job import _job


def _record(folder: Path, *, prepared: bool) -> NativeCaseRecord:
    request = NativeRequest("deck", "native", PROFILE, "mismatch", 42, 7)
    if not prepared:
        return NativeCaseRecord(request, unavailable_reason="input could not be staged")
    sample = ValidatedSample(
        request,
        "a" * 64,
        123,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        (),
        ".op",
        "e" * 64,
        ("models/model.spice",),
    )
    paths = NativePaths(
        folder,
        folder / "sample.input.cir",
        folder / "sample.setup.cir",
        folder / "sample.cir",
        folder / "sample.raw",
        folder / "sample.log",
    )
    launch = PreparedLaunch(
        paths,
        "f" * 64,
        "0" * 64,
        (ArtifactDigest(folder / "model.spice", "1" * 64, "models/model.spice"),),
        sample.sample_key,
        sample.effective_seed,
    )
    return NativeCaseRecord(
        request, sample, launch, SimulatorFacts("ngspice-42", "test build", "2" * 64, "test")
    )


@pytest.mark.parametrize("prepared", [False, True])
async def test_failed_native_case_survives_reload_and_public_runs(
    state_no_sim, work_dir, prepared
):
    circuit = work_dir / "deck.cir"
    circuit.write_text("* deck\n.op\n.end\n", encoding="utf-8")
    job = _job(work_dir, circuit, status="failed")
    case = job.cases[0]
    case.status = "failed"
    case.native_statistics = _record(work_dir, prepared=prepared)
    job.completeness.recount(job.cases)
    experiment_store.save_job(job)
    loaded = experiment_store.load_job(job.job_id, work_dir, own_is_alive=True)
    assert loaded is not None
    restored = loaded.cases[0].native_statistics
    assert restored is not None
    assert restored == case.native_statistics
    state_no_sim.add_experiment_job(loaded)
    result = await handle_jobs(
        JobsInput.model_validate({"action": "runs", "job_id": job.job_id}), state_no_sim
    )
    assert result.structured_content is not None
    rows = result.structured_content["items"]
    assert rows[0]["native_statistics"] == restored.public()
    assert rows[0]["assignments"] == {"R1": "1k"}
    if not prepared:
        assert "validated" not in rows[0]["native_statistics"]
        assert "prepared" not in rows[0]["native_statistics"]
        assert "effective_seed" in rows[0]["native_statistics"]["unavailable"]
    else:
        assert "coverage" not in rows[0]["native_statistics"]["validated"]
        assert "dependencies" not in rows[0]["native_statistics"]["prepared"]
        detail = await handle_jobs(
            JobsInput.model_validate(
                {"action": "runs", "job_id": job.job_id, "run_fields": ["native_statistics"]}
            ),
            state_no_sim,
        )
        assert detail.structured_content is not None
        assert detail.structured_content["items"] == [
            {"native_statistics": restored.public(detailed=True)}
        ]


def test_native_projection_keeps_identity_checks_and_skips_unselected_evidence(
    work_dir, monkeypatch
):
    from dataclasses import replace

    from ltspice_mcp.lib import pdk_native
    from ltspice_mcp.lib.projection import keep_plan, project_row

    record = _record(work_dir, prepared=True)
    original_asdict = pdk_native.asdict
    calls = {"coverage": 0, "paths": 0, "digest": 0}
    original_digest = pdk_native._dependency_summary

    def counted_asdict(value):
        if isinstance(value, pdk_native.MosCoverage):
            calls["coverage"] += 1
        if isinstance(value, pdk_native.NativePaths):
            calls["paths"] += 1
        return original_asdict(value)

    def counted_digest(value, **kwargs):
        calls["digest"] += 1
        return original_digest(value, **kwargs)

    monkeypatch.setattr(pdk_native, "asdict", counted_asdict)
    monkeypatch.setattr(pdk_native, "_dependency_summary", counted_digest)
    for detailed in (False, True):
        full = record.public(detailed=detailed)
        plan = keep_plan(["validated.effective_seed"])
        calls.update(coverage=0, paths=0, digest=0)
        assert record.public(detailed=detailed, projection=plan) == project_row(full, plan)
        assert calls == {"coverage": 0, "paths": 0, "digest": 0}
        for fields in (
            ["requested.family_id", "prepared.ngbehavior"],
            ["validated.coverage" if detailed else "validated.coverage_count"],
            ["prepared.dependencies" if detailed else "prepared.dependency_digest"],
            ["prepared.paths" if detailed else "prepared.dependency_count"],
        ):
            plan = keep_plan(fields)
            assert record.public(detailed=detailed, projection=plan) == project_row(full, plan)
    assert record.prepared is not None
    record.prepared = replace(record.prepared, sample_key="different")
    for fields in (["validated.effective_seed"], ["requested.family_id"]):
        with pytest.raises(pdk_native.NativeRequestError, match="disagree"):
            record.public(projection=keep_plan(fields))
    with pytest.raises(pdk_native.NativeRequestError, match="disagree"):
        record.validate()


def test_unvalidated_native_projection_with_unavailable_index(work_dir):
    from ltspice_mcp.lib.projection import keep_plan, project_row

    record = NativeCaseRecord(NativeRequest("deck", "native", PROFILE, "nominal", 4, None))
    full = record.public()
    for fields in (["origin"], ["requested"], ["unavailable.sample_index"]):
        plan = keep_plan(fields)
        assert record.public(projection=plan) == project_row(full, plan)


def test_record_load_rejects_mismatched_nested_request(work_dir):
    from ltspice_mcp.lib.pdk_native import NativeRequestError

    raw = _record(work_dir, prepared=True).to_record()
    raw["sample"]["request"]["family_id"] = "other"
    with pytest.raises(NativeRequestError, match="disagree"):
        NativeCaseRecord.from_record(raw)


@pytest.mark.parametrize("route", ["public", "validate", "reload"])
def test_validated_record_requires_logical_sample_index(work_dir, route):
    from dataclasses import replace

    from ltspice_mcp.lib.pdk_native import NativeRequestError

    record = _record(work_dir, prepared=True)
    assert record.sample is not None
    record.request = replace(record.request, sample_index=None)
    record.sample = replace(record.sample, request=record.request)
    actions = {
        "public": lambda: record.public(projection={}),
        "validate": record.validate,
        "reload": lambda: NativeCaseRecord.from_record(record.to_record()),
    }
    with pytest.raises(NativeRequestError, match="sample index"):
        actions[route]()


@pytest.mark.parametrize("missing", [True, False])
def test_record_reload_requires_its_own_sample_request(work_dir, missing):
    raw = _record(work_dir, prepared=True).to_record()
    if missing:
        del raw["sample"]["request"]
    else:
        raw["sample"]["request"] = {}
    with pytest.raises((KeyError, TypeError)):
        NativeCaseRecord.from_record(raw)


@pytest.mark.parametrize("field", ["dependency_count", "dependency_digest_version"])
def test_compact_dependency_metadata_does_not_hash_unrequested_digest(
    work_dir, monkeypatch, field
):
    from ltspice_mcp.lib import pdk_native
    from ltspice_mcp.lib.projection import keep_plan, project_row

    record = _record(work_dir, prepared=True)
    plan = keep_plan(["prepared." + field])
    expected = project_row(record.public(), plan)
    calls = []
    original = pdk_native.canonical_json

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(pdk_native, "canonical_json", counted)
    assert record.public(projection=plan) == expected
    assert calls == []
