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
        sample.analysis,
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
