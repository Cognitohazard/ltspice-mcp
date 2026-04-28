"""Tests for per-circuit job persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib import job_store, now
from ltspice_mcp.state import (
    BatchJob,
    MonteCarloConfig,
    SimulationJob,
    SweepConfig,
    SweepDimension,
)

SimStatus = Literal[
    "queued", "running", "completed", "failed", "timeout", "cancelled", "interrupted"
]
BatchStatus = Literal["running", "completed", "failed", "cancelled", "interrupted"]


def _sim_job(
    netlist: Path,
    *,
    status: SimStatus = "completed",
    job_id: str = "sim_123_abcdef",
    raw_file: Path | None = None,
    log_file: Path | None = None,
    completed_at: datetime | None | str = "auto",
) -> SimulationJob:
    """Build a SimulationJob with test defaults."""
    resolved_completed = (
        (now() if status == "completed" else None) if completed_at == "auto" else completed_at
    )
    return SimulationJob(
        job_id=job_id,
        netlist=netlist,
        simulator="LTspice",
        status=status,
        started_at=now(),
        completed_at=resolved_completed,  # type: ignore[arg-type]
        raw_file=raw_file,
        log_file=log_file,
    )


def _batch_job(
    netlist: Path,
    *,
    status: BatchStatus = "completed",
    job_type: Literal["sweep", "montecarlo"] = "sweep",
    job_id: str | None = None,
    sweep_config: SweepConfig | None = None,
    mc_config: MonteCarloConfig | None = None,
    run_results: dict[int, dict[str, Any]] | None = None,
    completed_at: datetime | None | str = "auto",
) -> BatchJob:
    """Build a BatchJob with test defaults."""
    resolved_completed = (
        (now() if status == "completed" else None) if completed_at == "auto" else completed_at
    )
    return BatchJob(
        job_id=job_id or f"{job_type}_456_deadbeef",
        job_type=job_type,
        netlist=netlist,
        total_runs=3,
        completed_runs=3 if status == "completed" else 0,
        status=status,
        started_at=now(),
        completed_at=resolved_completed,  # type: ignore[arg-type]
        run_results=run_results or {},
        sweep_config=sweep_config,
        mc_config=mc_config,
    )


class TestSidecarDir:
    def test_sidecar_lives_next_to_circuit(self, tmp_path: Path) -> None:
        circuit = tmp_path / "amp" / "lna.asc"
        circuit.parent.mkdir(parents=True)
        circuit.write_text("")
        assert job_store.sidecar_dir(circuit) == tmp_path / "amp" / ".ltspice-mcp" / "jobs"


class TestSaveLoad:
    def test_save_creates_sidecar_directory(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _sim_job(circuit)
        path = job_store.save_job(job)
        assert path.exists()
        assert path.parent == tmp_path / ".ltspice-mcp" / "jobs"

    def test_save_writes_json(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _sim_job(
            circuit,
            raw_file=tmp_path / "rc.raw",
            log_file=tmp_path / "rc.log",
        )
        path = job_store.save_job(job)
        data = json.loads(path.read_text())
        assert data["schema"] == job_store.SCHEMA
        assert data["schema_version"] == job_store.SCHEMA_VERSION
        assert data["kind"] == "simulation"
        assert data["job_id"] == job.job_id
        assert data["status"] == "completed"
        assert data["raw_file"].endswith("rc.raw")

    def test_roundtrip_sim_job(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        original = _sim_job(circuit, raw_file=tmp_path / "rc.raw")
        job_store.save_job(original)

        sim_jobs, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert batch_jobs == []
        assert len(sim_jobs) == 1
        restored = sim_jobs[0]
        assert restored.job_id == original.job_id
        assert restored.netlist == original.netlist
        assert restored.status == "completed"
        assert restored.raw_file == original.raw_file
        assert restored.done_event.is_set()  # terminal → pre-set

    def test_roundtrip_batch_job(self, tmp_path: Path) -> None:
        circuit = tmp_path / "amp.cir"
        circuit.write_text("")
        sweep_cfg = SweepConfig(
            netlist=circuit,
            dimensions=[
                SweepDimension(type="component", name="R1", start=1.0, stop=10.0, points=3)
            ],
        )
        run_results = {
            0: {
                "raw_file": str(tmp_path / "run0.raw"),
                "log_file": str(tmp_path / "run0.log"),
                "params": {"R1": 1.0},
            },
            1: {
                "raw_file": str(tmp_path / "run1.raw"),
                "log_file": str(tmp_path / "run1.log"),
                "params": {"R1": 5.0},
            },
        }
        original = _batch_job(
            circuit,
            sweep_config=sweep_cfg,
            run_results=run_results,
        )
        job_store.save_job(original)

        sim_jobs, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []
        assert len(batch_jobs) == 1
        restored = batch_jobs[0]
        assert restored.job_id == original.job_id
        assert restored.total_runs == 3
        assert restored.completed_runs == 3
        assert restored.status == "completed"
        assert restored.sweep_config is not None
        assert restored.sweep_config.dimensions[0].name == "R1"
        # run_results keys become ints again
        assert set(restored.run_results.keys()) == {0, 1}
        assert restored.run_results[0]["params"] == {"R1": 1.0}

    def test_roundtrip_mc_job(self, tmp_path: Path) -> None:
        circuit = tmp_path / "mc.cir"
        circuit.write_text("")
        mc_cfg = MonteCarloConfig(
            netlist=circuit,
            type_tolerances={"R": (0.05, "uniform")},
            component_overrides={"R1": (0.01, "normal")},
            num_runs=50,
        )
        original = _batch_job(circuit, job_type="montecarlo", mc_config=mc_cfg)
        job_store.save_job(original)

        _, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert len(batch_jobs) == 1
        restored = batch_jobs[0]
        assert restored.job_type == "montecarlo"
        assert restored.mc_config is not None
        assert restored.mc_config.type_tolerances == {"R": (0.05, "uniform")}
        assert restored.mc_config.component_overrides == {"R1": (0.01, "normal")}
        assert restored.mc_config.num_runs == 50


class TestInterruptedRecovery:
    def test_running_job_loaded_as_interrupted(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        running = _sim_job(circuit, status="running", completed_at=None)
        job_store.save_job(running)

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert len(sim_jobs) == 1
        assert sim_jobs[0].status == "interrupted"
        assert sim_jobs[0].error and "restarted" in sim_jobs[0].error.lower()
        assert sim_jobs[0].done_event.is_set()

    def test_queued_job_loaded_as_interrupted(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        queued = _sim_job(circuit, status="queued", completed_at=None)
        job_store.save_job(queued)

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs[0].status == "interrupted"

    def test_running_batch_job_loaded_as_interrupted(self, tmp_path: Path) -> None:
        circuit = tmp_path / "amp.cir"
        circuit.write_text("")
        running = _batch_job(circuit, status="running", completed_at=None)
        job_store.save_job(running)

        _, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert batch_jobs[0].status == "interrupted"


class TestDeleteJob:
    def test_delete_removes_file(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _sim_job(circuit)
        path = job_store.save_job(job)
        assert path.exists()
        job_store.delete_job(job)
        assert not path.exists()

    def test_delete_missing_is_noop(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = _sim_job(circuit)
        # Never saved — delete should silently succeed.
        job_store.delete_job(job)


class TestSummarize:
    def test_summary_empty_when_no_sidecar(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 0
        assert summary["status_counts"] == {}

    def test_summary_counts_by_status(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job_store.save_job(_sim_job(circuit, status="completed", job_id="sim_a"))
        job_store.save_job(_sim_job(circuit, status="failed", job_id="sim_b"))
        job_store.save_job(_sim_job(circuit, status="running", completed_at=None, job_id="sim_c"))

        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 3
        assert summary["status_counts"]["completed"] == 1
        assert summary["status_counts"]["failed"] == 1
        # running persisted record means the prior server died — surface as interrupted
        assert summary["status_counts"]["interrupted"] == 1
        assert "sim_c" in summary["interrupted_job_ids"]

    def test_summary_skips_unreadable_files(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        (sidecar / "garbage.json").write_text("{not valid json")
        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 0


class TestLoadSkipsCorrupt:
    def test_unparseable_file_is_skipped(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        good = _sim_job(circuit)
        job_store.save_job(good)
        sidecar = job_store.sidecar_dir(circuit)
        (sidecar / "broken.json").write_text("{bad json")

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert len(sim_jobs) == 1
        assert sim_jobs[0].job_id == good.job_id


class TestSchemaVersion:
    def test_current_schema_version_persisted(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job_store.save_job(_sim_job(circuit))
        path = next((job_store.sidecar_dir(circuit)).glob("*.json"))
        data = json.loads(path.read_text())
        assert data["schema"] == job_store.SCHEMA
        assert data["schema_version"] == job_store.SCHEMA_VERSION

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        future = {
            "schema": job_store.SCHEMA,
            "schema_version": 999,
            "job_id": "sim_future",
            "kind": "simulation",
            "netlist": str(circuit),
            "simulator": "LTspice",
            "status": "completed",
            "started_at": now().isoformat(),
        }
        (sidecar / "sim_future.json").write_text(json.dumps(future))

        sim_jobs, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []
        assert batch_jobs == []

    def test_missing_schema_version_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        unversioned = {
            "schema": job_store.SCHEMA,
            "job_id": "sim_unversioned",
            "kind": "simulation",
            "netlist": str(circuit),
            "simulator": "LTspice",
            "status": "completed",
            "started_at": now().isoformat(),
        }
        (sidecar / "sim_unversioned.json").write_text(json.dumps(unversioned))

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []

    def test_unknown_schema_string_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        alien = {
            "schema": "different-project/job",
            "schema_version": 1,
            "job_id": "sim_alien",
            "kind": "simulation",
            "netlist": str(circuit),
            "simulator": "LTspice",
            "status": "completed",
            "started_at": now().isoformat(),
        }
        (sidecar / "sim_alien.json").write_text(json.dumps(alien))

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []

    def test_summarize_circuit_respects_schema(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        # One valid record.
        job_store.save_job(_sim_job(circuit, status="completed", job_id="sim_good"))
        # One record from a future version — ignored.
        sidecar = job_store.sidecar_dir(circuit)
        (sidecar / "sim_future.json").write_text(
            json.dumps(
                {
                    "schema": job_store.SCHEMA,
                    "schema_version": 999,
                    "job_id": "sim_future",
                    "kind": "simulation",
                    "status": "completed",
                }
            )
        )
        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 1
        assert summary["status_counts"] == {"completed": 1}


class TestSchemaMigration:
    def test_missing_schema_version_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        (sidecar / "sim_versionless.json").write_text(
            json.dumps(
                {
                    "schema": job_store.SCHEMA,
                    # no schema_version
                    "job_id": "sim_versionless",
                    "kind": "simulation",
                    "status": "completed",
                    "netlist": str(circuit),
                    "simulator": "LTspice",
                    "started_at": now().isoformat(),
                }
            )
        )
        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []

    def test_wrong_schema_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        (sidecar / "sim_alien.json").write_text(
            json.dumps(
                {
                    "schema": "something-else",
                    "schema_version": 1,
                    "job_id": "sim_alien",
                    "kind": "simulation",
                    "status": "completed",
                }
            )
        )
        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert sim_jobs == []

    def test_migration_chain_applies(self, tmp_path: Path, monkeypatch) -> None:
        """Forge a hypothetical v0 record + migration and verify it upgrades."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)

        # Pretend current schema is v2, v0 and v1 are readable.
        monkeypatch.setattr(job_store, "SCHEMA_VERSION", 2)
        monkeypatch.setattr(job_store, "SUPPORTED_VERSIONS", frozenset({0, 1, 2}))

        def v0_to_v1(data: dict) -> dict:
            # Fake migration: rename old_name -> netlist
            if "old_name" in data:
                data["netlist"] = data.pop("old_name")
            return data

        def v1_to_v2(data: dict) -> dict:
            # Fake migration: add a missing field with a default
            data.setdefault("error", None)
            return data

        monkeypatch.setitem(job_store._MIGRATIONS, 0, v0_to_v1)
        monkeypatch.setitem(job_store._MIGRATIONS, 1, v1_to_v2)

        (sidecar / "sim_legacy.json").write_text(
            json.dumps(
                {
                    "schema": job_store.SCHEMA,
                    "schema_version": 0,
                    "job_id": "sim_legacy",
                    "kind": "simulation",
                    "status": "completed",
                    "old_name": str(circuit),
                    "simulator": "LTspice",
                    "started_at": now().isoformat(),
                }
            )
        )
        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert len(sim_jobs) == 1
        assert sim_jobs[0].job_id == "sim_legacy"
        assert str(sim_jobs[0].netlist) == str(circuit)
