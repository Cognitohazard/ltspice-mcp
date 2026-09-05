"""Reading the job sidecars earlier releases left in a circuit's directory.

Nothing here writes a simulation or batch record — this version has no job type
that produces one. What it must do is read them: a directory of them cannot
break the registry or the startup preload, and a caller who asks about one has
to be told why it has no results rather than left waiting on a job that will
never finish.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import experiment_store, job_store, now, store_common
from ltspice_mcp.state import LegacyJobRecord


def _legacy_record(
    circuit: Path,
    *,
    job_id: str = "sim_123_abcdef",
    kind: str = "simulation",
    status: str = "completed",
    **extra: Any,
) -> dict[str, Any]:
    """One sidecar record in the shape a pre-0.6 release wrote."""
    record: dict[str, Any] = {
        "schema": job_store.SCHEMA,
        "schema_version": job_store.SCHEMA_VERSION,
        "job_id": job_id,
        "kind": kind,
        "netlist": str(circuit.resolve()),
        "simulator": "LTspice",
        "status": status,
        "started_at": now().isoformat(),
        "completed_at": now().isoformat() if status == "completed" else None,
        "raw_file": str(circuit.with_suffix(".raw")),
        "log_file": str(circuit.with_suffix(".log")),
        "pid": 0,
    }
    record.update(extra)
    return record


def _write(circuit: Path, record: dict[str, Any]) -> Path:
    sidecar = job_store.sidecar_dir(circuit)
    sidecar.mkdir(parents=True, exist_ok=True)
    path = sidecar / f"{record['job_id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


class TestSidecarDir:
    def test_sidecar_dir_is_beside_the_circuit(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        assert job_store.sidecar_dir(circuit) == tmp_path / ".ltspice-mcp" / "jobs"


class TestLegacyRecordsLoad:
    """A pre-0.6 record parses into the inert shape, keeping only what a caller
    can still be told: which job, which circuit, what it claimed to be, and the
    status that release last wrote."""

    def test_simulation_record_loads_inert(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_a"))

        (record,) = job_store.load_jobs_for_circuit(circuit)
        assert isinstance(record, LegacyJobRecord)
        assert record.job_id == "sim_a"
        assert record.netlist == circuit.resolve()
        assert record.kind == "sim"
        assert record.status == "completed"

    def test_batch_record_reports_its_own_kind(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(
            circuit,
            _legacy_record(
                circuit, job_id="batch_a", kind="batch", job_type="sweep", total_runs=3
            ),
        )

        (record,) = job_store.load_jobs_for_circuit(circuit)
        assert record.kind == "batch"

    def test_running_record_with_a_dead_owner_reads_as_interrupted(self, tmp_path: Path) -> None:
        # The owning server is long gone: a status only a live runner could
        # justify must not come back claiming the job is still going.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_r", status="running", pid=999_999))

        (record,) = job_store.load_jobs_for_circuit(circuit)
        assert record.status == "interrupted"

    def test_a_directory_of_records_loads_without_raising(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        for index in range(5):
            _write(circuit, _legacy_record(circuit, job_id=f"sim_{index}"))

        records = job_store.load_jobs_for_circuit(circuit)
        assert {record.job_id for record in records} == {f"sim_{i}" for i in range(5)}

    def test_load_job_by_id(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_one"))

        loaded = job_store.load_job("sim_one", circuit)
        assert isinstance(loaded, LegacyJobRecord)
        assert loaded.job_id == "sim_one"
        assert job_store.load_job("sim_missing", circuit) is None


class TestLoadSkipsCorrupt:
    def test_corrupt_file_is_skipped_not_fatal(self, tmp_path: Path, caplog) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_good"))
        (job_store.sidecar_dir(circuit) / "broken.json").write_text("{not json")

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.job_store"):
            records = job_store.load_jobs_for_circuit(circuit)

        assert [record.job_id for record in records] == ["sim_good"]
        assert any("unreadable" in item.message for item in caplog.records)


class TestSummarize:
    def test_counts_only_this_circuit(self, tmp_path: Path) -> None:
        # One sidecar directory serves every circuit beside it, so a summary
        # that ignored the recorded netlist would report the directory's totals
        # for each circuit in it.
        one = tmp_path / "one.cir"
        two = tmp_path / "two.cir"
        one.write_text("")
        two.write_text("")
        _write(one, _legacy_record(one, job_id="sim_one"))
        _write(two, _legacy_record(two, job_id="sim_two"))

        assert job_store.summarize_circuit(one)["total_jobs"] == 1
        assert job_store.summarize_circuit(two)["total_jobs"] == 1

    def test_batch_runs_counted_separately_from_jobs(self, tmp_path: Path) -> None:
        # A 100-run Monte Carlo is ONE job record but a hundred simulations;
        # collapsing the two would read as "this circuit ran once".
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(
            circuit,
            _legacy_record(circuit, job_id="batch_a", kind="batch", total_runs=100),
        )

        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 1
        assert summary["total_runs"] == 100

    def test_interrupted_ids_reported(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_r", status="running", pid=999_999))

        summary = job_store.summarize_circuit(circuit)
        assert summary["interrupted_job_ids"] == ["sim_r"]
        assert summary["status_counts"] == {"interrupted": 1}

    def test_missing_sidecar_is_empty_not_an_error(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        summary = job_store.summarize_circuit(circuit)
        assert summary["total_jobs"] == 0
        assert summary["exists"] is True


class TestSchemaVersion:
    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_future", schema_version=999))

        assert job_store.load_jobs_for_circuit(circuit) == []

    def test_missing_schema_version_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        record = _legacy_record(circuit, job_id="sim_unversioned")
        del record["schema_version"]
        _write(circuit, record)

        assert job_store.load_jobs_for_circuit(circuit) == []

    def test_unknown_schema_string_rejected(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(
            circuit, _legacy_record(circuit, job_id="sim_alien", schema="different-project/job")
        )

        assert job_store.load_jobs_for_circuit(circuit) == []

    def test_v1_record_still_loads(self, tmp_path: Path) -> None:
        # v1 differed only in fields this build no longer reads, so the
        # migration re-stamps the version and the record still parses.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_v1", schema_version=1))

        (record,) = job_store.load_jobs_for_circuit(circuit)
        assert record.job_id == "sim_v1"


class TestSiblingSchemasAreSilent:
    """Both stores write into ``.ltspice-mcp/jobs/``; meeting the other's
    records is the layout, not corruption, so a scan must not warn about it —
    while a genuinely unknown schema still must."""

    @staticmethod
    def _experiment_record(circuit: Path, schema: str, job_id: str) -> dict[str, Any]:
        return {
            "schema": schema,
            "schema_version": 2,
            "job_id": job_id,
            "kind": "experiment",
            "netlist": str(circuit),
            "status": "completed",
        }

    def test_experiment_records_scan_without_warnings(self, tmp_path: Path, caplog) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_good"))
        sidecar = job_store.sidecar_dir(circuit)
        for index in range(3):
            record = self._experiment_record(circuit, experiment_store.SCHEMA, f"exp_{index}")
            (sidecar / f"exp_{index}.json").write_text(json.dumps(record))

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.store_common"):
            summary = job_store.summarize_circuit(circuit)
            records = job_store.load_jobs_for_circuit(circuit)

        assert summary["total_jobs"] == 1
        assert [record.job_id for record in records] == ["sim_good"]
        assert caplog.records == []

    def test_alien_schema_still_warns(self, tmp_path: Path, caplog) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        sidecar = job_store.sidecar_dir(circuit)
        sidecar.mkdir(parents=True)
        record = self._experiment_record(circuit, "different-project/job", "sim_alien")
        (sidecar / "sim_alien.json").write_text(json.dumps(record))

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.store_common"):
            summary = job_store.summarize_circuit(circuit)

        assert summary["total_jobs"] == 0
        assert any("unexpected schema" in item.message for item in caplog.records)

    def test_experiment_store_is_silent_about_legacy_job_records(
        self, tmp_path: Path, caplog
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        legacy = _write(circuit, _legacy_record(circuit, job_id="sim_good"))

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.store_common"):
            loaded = experiment_store.load_job_from_path(legacy, tmp_path)

        assert loaded is None
        assert caplog.records == []


class TestForgedMigrationChain:
    def test_migration_chain_applies(self, tmp_path: Path, monkeypatch: Any) -> None:
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
        records = job_store.load_jobs_for_circuit(circuit)
        assert [record.job_id for record in records] == ["sim_legacy"]
        assert str(records[0].netlist) == str(circuit)


_FOREIGN_PID = 999_999_999


class TestOwnerLivenessUnknown:
    """A probe that could not reach an answer must not read as "owner dead".

    Sessions share a working directory, and "the owner is gone" is exactly the
    reading that licenses one session to rewrite another's running job as
    interrupted. A psutil call that raises is not evidence of anything, so the
    record stands as the owning server wrote it. The experiment records this
    probe now guards are exercised in tests/test_experiment_job.py; what is
    pinned here is the probe's own three answers.
    """

    @staticmethod
    def _break_the_probe(monkeypatch: Any) -> None:
        def boom(pid: int) -> bool:
            raise OSError("process table unavailable")

        monkeypatch.setattr(store_common.psutil, "pid_exists", boom)

    def test_probe_reports_unknown_rather_than_dead(self, monkeypatch: Any) -> None:
        self._break_the_probe(monkeypatch)
        liveness = store_common.owner_liveness(_FOREIGN_PID)
        assert liveness is store_common.OwnerLiveness.UNKNOWN
        assert liveness.is_dead is False

    def test_probe_still_answers_dead_and_alive(self, monkeypatch: Any) -> None:
        """The two real answers must survive the third one being added."""
        monkeypatch.setattr(store_common.psutil, "pid_exists", lambda pid: False)
        assert store_common.owner_liveness(_FOREIGN_PID) is store_common.OwnerLiveness.DEAD
        monkeypatch.setattr(store_common.psutil, "pid_exists", lambda pid: True)
        assert store_common.owner_liveness(_FOREIGN_PID) is store_common.OwnerLiveness.ALIVE
        # A record with no pid predates pid tracking; recovering those jobs is
        # the behaviour this probe was added to, not something it takes away.
        assert store_common.owner_liveness(None) is store_common.OwnerLiveness.DEAD
