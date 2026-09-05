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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from ltspice_mcp.lib import job_store, now, store
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
        "schema_version": max(job_store.SUPPORTED_VERSIONS),
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

    def test_both_written_versions_load(self, tmp_path: Path) -> None:
        # The two versions differ only in fields this build no longer reads,
        # so both parse into the same inert shape. Nothing will ever write a
        # third, which is why there is no migration machinery left here.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        for version in sorted(job_store.SUPPORTED_VERSIONS):
            _write(
                circuit,
                _legacy_record(circuit, job_id=f"sim_v{version}", schema_version=version),
            )

        loaded = {record.job_id for record in job_store.load_jobs_for_circuit(circuit)}
        assert loaded == {f"sim_v{v}" for v in job_store.SUPPORTED_VERSIONS}


class TestTheLegacyDirectoryHoldsOnlyLegacyRecords:
    """This build writes its records to the working-directory store instead,
    so a file here that is not one of these is genuinely unexpected — including
    one carrying the current store's schema, which nothing puts in this folder.
    """

    def test_a_store_record_here_is_reported_not_ignored(self, tmp_path: Path, caplog) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_good"))
        sidecar = job_store.sidecar_dir(circuit)
        (sidecar / "exp_stray.json").write_text(
            json.dumps(store.envelope(store.KIND_EXPERIMENT, job_id="exp_stray"))
        )

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.job_store"):
            summary = job_store.summarize_circuit(circuit)
            records = job_store.load_jobs_for_circuit(circuit)

        assert summary["total_jobs"] == 1
        assert [record.job_id for record in records] == ["sim_good"]
        assert any("unexpected schema" in item.message for item in caplog.records)

    def test_alien_schema_warns(self, tmp_path: Path, caplog) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        _write(circuit, _legacy_record(circuit, job_id="sim_alien"))
        sidecar = job_store.sidecar_dir(circuit)
        (sidecar / "sim_alien.json").write_text(
            json.dumps({"schema": "different-project/job", "schema_version": 1})
        )

        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.lib.job_store"):
            summary = job_store.summarize_circuit(circuit)

        assert summary["total_jobs"] == 0
        assert any("unexpected schema" in item.message for item in caplog.records)


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

        monkeypatch.setattr(store.psutil, "pid_exists", boom)

    def test_probe_reports_unknown_rather_than_dead(self, monkeypatch: Any) -> None:
        self._break_the_probe(monkeypatch)
        liveness = store.owner_liveness(_FOREIGN_PID)
        assert liveness is store.OwnerLiveness.UNKNOWN
        assert liveness.is_dead is False

    def test_probe_still_answers_dead_and_alive(self, monkeypatch: Any) -> None:
        """The two real answers must survive the third one being added."""
        monkeypatch.setattr(store.psutil, "pid_exists", lambda pid: False)
        assert store.owner_liveness(_FOREIGN_PID) is store.OwnerLiveness.DEAD
        monkeypatch.undo()
        # A real live process for the ALIVE answer: the probe now also asks
        # what the process is doing, and a pid that exists only in a stub has
        # nothing to answer with.
        assert store.owner_liveness(os.getppid()) is store.OwnerLiveness.ALIVE
        # A record with no pid predates pid tracking; recovering those jobs is
        # the behaviour this probe was added to, not something it takes away.
        assert store.owner_liveness(None) is store.OwnerLiveness.DEAD


class TestOwnerLivenessExitedProcess:
    """A process that has exited is not an owner, collected or not.

    An exited child keeps its pid in the process table until the process that
    started it collects it, so the pid alone still reads as "there". A job
    whose owner stopped there has nobody supervising it, and reading that as
    running is what keeps the restart reconciliation from ever running.
    """

    def test_probe_reports_an_exited_uncollected_process_as_dead(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", ""])
        try:
            deadline = time.monotonic() + 30
            # Deliberately never poll() or wait() here: either would collect
            # the child and remove the state under test.
            while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
                if time.monotonic() >= deadline:
                    pytest.fail("the child process never exited")
                time.sleep(0.02)
            assert psutil.pid_exists(child.pid)
            assert store.owner_liveness(child.pid) is store.OwnerLiveness.DEAD
        finally:
            child.wait(timeout=30)
