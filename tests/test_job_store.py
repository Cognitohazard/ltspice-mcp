"""Tests for per-circuit job persistence."""

from __future__ import annotations

import json
import subprocess
import sys
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

    def test_roundtrip_sim_job_output_alias_fields(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        original = _sim_job(circuit, raw_file=tmp_path / "rc.raw")
        original.output_basename = "myrun"
        original.output_alias_raw = tmp_path / "myrun.raw"
        original.output_alias_log = tmp_path / "myrun.log"
        original.output_alias_note = None
        job_store.save_job(original)

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        restored = sim_jobs[0]
        assert restored.output_basename == "myrun"
        assert restored.output_alias_raw == tmp_path / "myrun.raw"
        assert restored.output_alias_log == tmp_path / "myrun.log"
        assert restored.output_alias_note is None

    def test_roundtrip_sim_job_output_alias_skip_note(self, tmp_path: Path) -> None:
        # A skipped alias (collision, or a hardlink failure on a too-large
        # raw) must round-trip too — the "why" is as much a fact as the path.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        original = _sim_job(circuit, raw_file=tmp_path / "rc.raw")
        original.output_basename = "clash"
        original.output_alias_note = "raw: clash.raw already exists"
        job_store.save_job(original)

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        restored = sim_jobs[0]
        assert restored.output_alias_raw is None
        assert restored.output_alias_note == "raw: clash.raw already exists"

    def test_pre_alias_record_loads_with_no_basename(self, tmp_path: Path) -> None:
        # A record predating these fields is missing the keys entirely (they
        # were added additively within schema v2) — must load as "no alias
        # requested", not crash on a missing key.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job_store.save_job(_sim_job(circuit, raw_file=tmp_path / "rc.raw"))
        f = next(job_store.sidecar_dir(circuit).glob("*.json"))
        record = json.loads(f.read_text())
        for key in (
            "output_basename",
            "output_alias_raw",
            "output_alias_log",
            "output_alias_note",
        ):
            record.pop(key, None)
        f.write_text(json.dumps(record))

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        restored = sim_jobs[0]
        assert restored.output_basename is None
        assert restored.output_alias_raw is None
        assert restored.output_alias_log is None
        assert restored.output_alias_note is None

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

    def test_roundtrip_values_sweep(self, tmp_path: Path) -> None:
        # v2 shape: an explicit discrete-value sweep persists start/stop as null
        # and a populated ``values`` list, and round-trips intact.
        circuit = tmp_path / "esweep.cir"
        circuit.write_text("")
        sweep_cfg = SweepConfig(
            netlist=circuit,
            dimensions=[
                SweepDimension(type="component", name="R1", values=[1000.0, 2200.0, 4700.0])
            ],
        )
        job_store.save_job(_batch_job(circuit, sweep_config=sweep_cfg))

        record = json.loads(next(job_store.sidecar_dir(circuit).glob("*.json")).read_text())
        assert record["schema_version"] == 2
        dim0 = record["sweep_config"]["dimensions"][0]
        assert dim0["start"] is None
        assert dim0["values"] == [1000.0, 2200.0, 4700.0]

        _, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert len(batch_jobs) == 1
        assert batch_jobs[0].sweep_config is not None
        rdim = batch_jobs[0].sweep_config.dimensions[0]
        assert rdim.values == [1000.0, 2200.0, 4700.0]
        assert rdim.start is None and rdim.stop is None
        assert rdim.resolved_values() == [1000.0, 2200.0, 4700.0]

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
        # Written with THIS process's pid (the dataclass default): a running
        # record whose owner is this live server is a genuinely running job.
        job_store.save_job(_sim_job(circuit, status="running", completed_at=None, job_id="sim_c"))
        # A running record whose owning process is gone surfaces as interrupted.
        dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_proc.wait()
        orphan = _sim_job(circuit, status="running", completed_at=None, job_id="sim_d")
        orphan.owner_pid = dead_proc.pid
        job_store.save_job(orphan)

        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 4
        assert summary["status_counts"]["completed"] == 1
        assert summary["status_counts"]["failed"] == 1
        assert summary["status_counts"]["running"] == 1
        assert summary["status_counts"]["interrupted"] == 1
        assert "sim_d" in summary["interrupted_job_ids"]
        assert "sim_c" not in summary["interrupted_job_ids"]

    def test_summary_skips_unreadable_files(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        (sidecar / "garbage.json").write_text("{not valid json")
        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 0

    def test_summary_filters_by_netlist_path(self, tmp_path: Path) -> None:
        """the sidecar directory is shared across every circuit in
        the same parent dir. ``summarize_circuit`` must filter by netlist
        path or each circuit reports the directory's aggregate counts."""
        circuit_a = tmp_path / "a.cir"
        circuit_b = tmp_path / "b.cir"
        circuit_a.write_text("")
        circuit_b.write_text("")
        # 2 jobs against A, 3 against B — they all land in the same sidecar.
        for jid in ("sim_a1", "sim_a2"):
            job_store.save_job(_sim_job(circuit_a, job_id=jid))
        for jid in ("sim_b1", "sim_b2", "sim_b3"):
            job_store.save_job(_sim_job(circuit_b, job_id=jid))

        summary_a = job_store.summarize_circuit(circuit_a)
        summary_b = job_store.summarize_circuit(circuit_b)
        assert summary_a["total_jobs"] == 2
        assert summary_b["total_jobs"] == 3

    def test_summary_separates_batch_total_runs(self, tmp_path: Path) -> None:
        """a batch job is one ``total_jobs`` entry but counts its
        ``total_runs`` underlying simulations under ``total_runs`` so a
        100-run MC isn't mistaken for "circuit ran once"."""
        circuit = tmp_path / "rc_mc.cir"
        circuit.write_text("")
        job_store.save_job(_sim_job(circuit, job_id="sim_warmup"))
        batch = _batch_job(circuit, job_id="mc_run", job_type="montecarlo")
        # _batch_job sets total_runs=3 by default
        job_store.save_job(batch)

        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 2  # 1 sim + 1 batch
        assert summary["total_runs"] == 4  # 1 sim + 3 MC iterations


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

    def test_v1_sweep_record_still_loads(self, tmp_path: Path) -> None:
        # A genuine pre-v2 (v1) sweep record has no ``values`` and real
        # start/stop. After the v2 bump it must still load (migrated 1->2),
        # proving old persisted jobs survive the upgrade.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sweep_cfg = SweepConfig(
            netlist=circuit,
            dimensions=[
                SweepDimension(type="component", name="R1", start=1.0, stop=10.0, points=3)
            ],
        )
        job_store.save_job(_batch_job(circuit, sweep_config=sweep_cfg))
        # Rewrite the file as a real v1 record: drop ``values``, set version 1.
        f = next(job_store.sidecar_dir(circuit).glob("*.json"))
        record = json.loads(f.read_text())
        record["schema_version"] = 1
        for dim in record["sweep_config"]["dimensions"]:
            dim.pop("values", None)
        f.write_text(json.dumps(record))

        _, batch_jobs = job_store.load_jobs_for_circuit(circuit)
        assert len(batch_jobs) == 1
        assert batch_jobs[0].sweep_config is not None
        rdim = batch_jobs[0].sweep_config.dimensions[0]
        assert rdim.start == 1.0 and rdim.stop == 10.0
        assert rdim.values is None
