"""Persistence, discovery, lifecycle, and compatibility tests for experiments."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from ltspice_mcp.errors import BatchJobError, ResultError, SimulationError
from ltspice_mcp.lib import experiment_store, job_store, now, recent, services
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    ExperimentRunner,
    ExperimentRunRequest,
    IdempotencyConflictError,
    canonical_fingerprint,
)
from ltspice_mcp.lib.experiment_types import (
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.job_lifecycle import InvalidTransitionError, transition
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.simulation import (
    CancelJobInput,
    CheckJobInput,
    handle_cancel_job,
    handle_check_job,
)


def _source(circuit: Path, staged: Path | None = None) -> SourceRecord:
    staged = staged or circuit
    return SourceRecord(
        circuit=circuit.stem,
        path=circuit,
        sha256="source-sha",
        staged_deck=staged,
        manifest=[
            ManifestEntry(
                path=circuit,
                sha256="source-sha",
                staged=True,
                live=False,
                staged_path=staged,
            )
        ],
        linter_version="1",
        simulator="FakeSim",
    )


def _case(circuit: Path, staged: Path | None = None, *, index: int = 0) -> ExperimentCase:
    staged = staged or circuit
    return ExperimentCase(
        case_id=f"case_{index:04d}",
        run_index=index,
        circuit=circuit.stem,
        circuit_path=circuit,
        staged_deck=staged,
        deck_sha256="deck-sha",
        assignments={"R1": f"{index + 1}k"},
    )


def _job(
    working_dir: Path,
    circuit: Path,
    *,
    job_id: str = "exp_test_0001",
    request_id: str = "request-1",
    status: str = "queued",
) -> ExperimentJob:
    case = _case(circuit)
    return ExperimentJob(
        job_id=job_id,
        request_id=request_id,
        fingerprint="f" * 64,
        canonicalizer_version=CANONICALIZER_VERSION,
        control_token="control-secret",
        store_path=experiment_store.record_path(job_id, working_dir),
        cases=[case],
        sources=[_source(circuit)],
        simulator="FakeSim",
        completeness=Completeness(declared=1, expanded=1),
        status=status,  # type: ignore[arg-type]
    )


def _barrier_process(
    working_dir: str,
    circuit_path: str,
    request_id: str,
    fingerprint: str,
    job_id: str,
    start: Any,
    result: Any,
) -> None:
    """Process worker exercising the real request lock and durable barrier."""
    working = Path(working_dir)
    circuit = Path(circuit_path)
    state = SimpleNamespace(working_dir=working)
    request = SimpleNamespace(
        state=state,
        request_id=request_id,
        fingerprint=fingerprint,
        canonicalizer_version=CANONICALIZER_VERSION,
    )
    candidate = _job(
        working,
        circuit,
        job_id=job_id,
        request_id=request_id,
    )
    start.wait(10)
    try:
        barrier = ExperimentRunner._durable_barrier(
            cast("ExperimentRunRequest", request), candidate
        )
        result.put((barrier.job.job_id, barrier.replayed, None))
    except Exception as exc:
        result.put((None, None, f"{type(exc).__name__}: {exc}"))


class FingerprintInput(BaseModel):
    request_id: str
    optional_value: int = 7
    nested: dict[str, int]


class TestExperimentTypesAndStore:
    def test_experiment_has_no_legacy_netlist_attribute(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        assert not hasattr(job, "netlist")
        assert job.cases[0].staged_deck == circuit

    def test_completeness_terminal_invariant(self):
        completeness = Completeness(expanded=3, produced=1, failed=1, skipped=1)
        completeness.validate_terminal()
        completeness.cancelled = 1
        with pytest.raises(ValueError, match="does not reconcile"):
            completeness.validate_terminal()

    def test_completeness_recount_derives_all_case_counters(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        statuses = [
            "queued",
            "submitted",
            "running",
            "produced",
            "failed",
            "cancelled",
            "skipped",
        ]
        cases = [_case(circuit, index=index) for index in range(len(statuses))]
        for case, status in zip(cases, statuses, strict=True):
            case.status = status  # type: ignore[assignment]
        cases[4].submitted_at = now()

        completeness = Completeness()
        completeness.recount(cases)

        assert completeness.submitted == 4
        assert completeness.produced == 1
        assert completeness.failed == 1
        assert completeness.cancelled == 1
        assert completeness.skipped == 1

    def test_round_trip_preserves_cases_sources_token_and_analysis(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        job.analysis = AnalysisStage(status="pending", request={"recipes": [{"kind": "summary"}]})
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir, own_is_alive=True)
        assert loaded is not None
        assert loaded.control_token == "control-secret"
        assert loaded.cases[0].assignments == {"R1": "1k"}
        assert loaded.sources[0].manifest[0].staged_path == circuit
        assert loaded.analysis.status == "pending"
        assert loaded.store_path == job.store_path.resolve()

    def test_job_store_delegates_experiment_kind_without_misparsing(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        experiment_store.save_job(job)
        loaded = job_store._load_job_file(job.store_path)
        assert isinstance(loaded, ExperimentJob)

    def test_pointer_written_in_dedicated_subdirectory(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        experiment_store.save_job(job)
        pointers = experiment_store.save_pointers(job)
        assert pointers == [
            circuit.parent / ".ltspice-mcp" / "jobs" / "experiments" / f"{job.job_id}.json"
        ]
        loaded, observations = experiment_store.load_pointer_jobs(circuit, work_dir)
        assert [item.job_id for item in loaded] == [job.job_id]
        assert observations == []

    @pytest.mark.parametrize(
        "job_id",
        [
            "../escape",
            "nested/job",
            r"nested\job",
            ".hidden",
            "x" * 65,
            "",
        ],
    )
    def test_direct_lookup_rejects_non_server_job_ids_before_path_build(
        self,
        work_dir: Path,
        job_id: str,
    ):
        with pytest.raises(ValueError, match="Invalid job id"):
            experiment_store.load_job(job_id, work_dir)

    def test_direct_lookup_rejects_symlinked_record_outside_store(self, work_dir: Path):
        root = experiment_store.working_store_root(work_dir)
        root.mkdir(parents=True)
        outside = work_dir / "outside.json"
        outside.write_text("{}")
        link = root / "exp_symlink.json"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="escapes the working store"):
            experiment_store.load_job("exp_symlink", work_dir)

    def test_canonical_fingerprint_is_sorted_and_includes_defaults(self):
        first = FingerprintInput(request_id="r", nested={"b": 2, "a": 1})
        second = FingerprintInput.model_validate(
            {"nested": {"a": 1, "b": 2}, "optional_value": 7, "request_id": "r"}
        )
        without_default = {
            "request_id": "r",
            "nested": {"a": 1, "b": 2},
        }
        assert canonical_fingerprint(first) == canonical_fingerprint(second)
        assert canonical_fingerprint(first) != canonical_fingerprint(without_default)


class TestExperimentLifecycle:
    def test_runtime_transition_cannot_enter_restart_only_interrupted(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)

        with pytest.raises(InvalidTransitionError, match="no event mapping"):
            transition(job, "interrupted")

        assert job.status == "queued"
        assert not job.done_event.is_set()

    def test_analysis_stage_transitions_and_terminal_event(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        transition(job, "running")
        transition(job, "analyzing")
        transition(job, "completed_with_failures")
        assert job.done_event.is_set()
        assert job.completed_at is not None
        with pytest.raises(InvalidTransitionError):
            transition(job, "running")

    def test_restart_fails_running_analysis_but_preserves_produced_case(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        raw = work_dir / "case.raw"
        circuit.write_text(".op\n.end\n")
        raw.write_bytes(b"Title: result")
        job = _job(work_dir, circuit, status="analyzing")
        job.owner_pid = 999_999_999
        job.cases[0].status = "produced"
        job.cases[0].raw_file = raw
        job.analysis = AnalysisStage(status="running", request={"recipes": []})
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)
        assert loaded is not None
        assert loaded.status == "completed_with_failures"
        assert loaded.analysis.status == "failed"
        assert loaded.cases[0].raw_file == raw
        assert loaded.completeness.submitted == 1
        assert loaded.completeness.produced == 1
        assert loaded.runs_done_event.is_set()
        assert loaded.done_event.is_set()

    def test_restart_recovers_completed_analysis_write_gap(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        raw = work_dir / "case.raw"
        circuit.write_text(".op\n.end\n")
        raw.write_bytes(b"Title: result")
        job = _job(work_dir, circuit, status="analyzing")
        job.owner_pid = 999_999_999
        job.cases[0].status = "produced"
        job.cases[0].raw_file = raw
        job.completeness.produced = 1
        job.analysis = AnalysisStage(status="completed", result={"summary": "done"})
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)
        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.analysis.status == "completed"
        assert loaded.analysis.result == {"summary": "done"}
        assert loaded.done_event.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_without_live_runner_reconciles_all_cases(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="running")
        job.cases.append(_case(circuit, index=1))
        job.completeness.expanded = 2
        job.analysis = AnalysisStage(status="pending", request={"recipes": []})
        registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
        registry.add_experiment_job(job)
        runners = SimpleNamespace(get_experiment_runner_for=lambda _job: None)

        await registry.cancel_running(runners, None)

        assert job.status == "cancelled"
        assert job.completeness.cancelled == 2
        assert job.completeness.terminal == job.completeness.expanded
        assert job.analysis.status == "cancelled"
        assert job.runs_done_event.is_set()
        assert job.done_event.is_set()


class TestExperimentDiscovery:
    def test_experiment_jobs_have_an_independent_finished_job_cap(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
        started = now()
        for index in range(205):
            job = _job(
                work_dir,
                circuit,
                job_id=f"exp_cap_{index:03d}",
                status="completed",
            )
            job.started_at = started + timedelta(seconds=index)
            registry.add_experiment_job(job)

        assert len(registry.experiment_jobs) == 200
        assert all(f"exp_cap_{index:03d}" not in registry.jobs for index in range(5))

    def test_preload_recent_discovers_pointer_only_experiment(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        state_home = work_dir / "state"
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(state_home))
        circuit = work_dir / "subdir" / "deck.cir"
        circuit.parent.mkdir()
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        job.cases[0].status = "produced"
        job.completeness.produced = 1
        experiment_store.save_job(job)
        experiment_store.save_pointers(job)
        recent.touch(circuit)

        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        assert registry.preload_recent() == 1
        assert registry.experiment_jobs[job.job_id].store_path == job.store_path

    def test_tampered_pointer_outside_working_store_is_skipped_with_observation(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "subdir" / "deck.cir"
        circuit.parent.mkdir()
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        experiment_store.save_job(job)
        pointer = experiment_store.save_pointers(job)[0]
        payload = json.loads(pointer.read_text())
        payload["target"] = str(work_dir / "outside.json")
        pointer.write_text(json.dumps(payload))

        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        registry.ensure_loaded_for(circuit)
        assert job.job_id not in registry.experiment_jobs
        assert any(item["code"] == "experiment_pointer_invalid" for item in registry.observations)

    @pytest.mark.asyncio
    async def test_direct_store_lookup_finds_non_recent_experiment(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "not-recent" / "deck.cir"
        circuit.parent.mkdir()
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        experiment_store.save_job(job)

        loaded = await services.resolve_job_async(job.job_id, state_no_sim)
        assert isinstance(loaded, ExperimentJob)
        assert loaded.job_id in state_no_sim.experiment_jobs

    @pytest.mark.asyncio
    async def test_foreign_refresh_uses_coordinator_store_path(
        self,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        stale = _job(work_dir, circuit, status="running")
        stale.owner_pid = -1
        completed = _job(work_dir, circuit, status="completed")
        completed.owner_pid = -1
        completed.cases[0].status = "produced"
        completed.completeness.produced = 1
        experiment_store.save_job(completed)
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        registry.jobs[stale.job_id] = stale

        refreshed = await registry.refresh_foreign_job_async(stale)

        assert isinstance(refreshed, ExperimentJob)
        assert refreshed.status == "completed"
        assert registry.jobs[stale.job_id] is refreshed


class TestRequestBarrier:
    def test_two_processes_same_request_create_one_coordinator(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        result = context.Queue()
        processes = [
            context.Process(
                target=_barrier_process,
                args=(
                    str(work_dir),
                    str(circuit),
                    "shared-request",
                    "a" * 64,
                    f"exp_process_{index}",
                    start,
                    result,
                ),
            )
            for index in range(2)
        ]
        for process in processes:
            process.start()
        start.set()
        outcomes = [result.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        assert all(error is None for _job_id, _replayed, error in outcomes)
        assert len({job_id for job_id, _replayed, _error in outcomes}) == 1
        assert sorted(replayed for _job_id, replayed, _error in outcomes) == [False, True]

    def test_canonicalizer_version_mismatch_is_an_honest_conflict(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        experiment_store.save_request_index(
            request_id="versioned-request",
            fingerprint="a" * 64,
            canonicalizer_version=CANONICALIZER_VERSION + 1,
            job_id="exp_old_version",
            working_dir=work_dir,
        )
        request = ExperimentRunRequest(
            state=state_no_sim,
            request_id="versioned-request",
            fingerprint="a" * 64,
            cases=[_case(circuit)],
            sources=[_source(circuit)],
            simulator="FakeSim",
        )
        candidate = _job(
            work_dir,
            circuit,
            job_id="exp_new_version",
            request_id=request.request_id,
        )
        with pytest.raises(IdempotencyConflictError, match="canonicalizer version"):
            ExperimentRunner._durable_barrier(request, candidate)

    def test_dangling_request_index_is_replaced_and_observed(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        request_id = "dangling-request"
        experiment_store.save_request_index(
            request_id=request_id,
            fingerprint="a" * 64,
            canonicalizer_version=CANONICALIZER_VERSION,
            job_id="exp_missing",
            working_dir=work_dir,
        )
        request = ExperimentRunRequest(
            state=state_no_sim,
            request_id=request_id,
            fingerprint="a" * 64,
            cases=[_case(circuit)],
            sources=[_source(circuit)],
            simulator="FakeSim",
        )
        candidate = _job(
            work_dir,
            circuit,
            job_id="exp_recreated",
            request_id=request_id,
        )
        result = ExperimentRunner._durable_barrier(request, candidate)
        assert not result.replayed
        assert any(
            item["code"] == "dangling_request_index_replaced" for item in result.job.observations
        )

    def test_request_index_rejects_an_inconsistent_coordinator(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        existing = _job(
            work_dir,
            circuit,
            job_id="exp_inconsistent",
            request_id="different-request",
        )
        experiment_store.save_job(existing)
        experiment_store.save_request_index(
            request_id="indexed-request",
            fingerprint=existing.fingerprint,
            canonicalizer_version=CANONICALIZER_VERSION,
            job_id=existing.job_id,
            working_dir=work_dir,
        )
        request = ExperimentRunRequest(
            state=state_no_sim,
            request_id="indexed-request",
            fingerprint=existing.fingerprint,
            cases=[_case(circuit)],
            sources=[_source(circuit)],
            simulator="FakeSim",
        )
        candidate = _job(
            work_dir,
            circuit,
            job_id="exp_candidate",
            request_id=request.request_id,
        )

        with pytest.raises(IdempotencyConflictError, match="inconsistent coordinator"):
            ExperimentRunner._durable_barrier(request, candidate)


@pytest.mark.asyncio
class TestLegacyCompatibility:
    async def test_batch_resolver_rejects_persisted_experiment(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        experiment_store.save_job(job)
        with pytest.raises(BatchJobError, match="experiment job"):
            await services.resolve_batch_job_async(job.job_id, state_no_sim)

    async def test_check_and_cancel_job_reject_experiment_without_netlist_deref(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        experiment_store.save_job(job)

        with pytest.raises(SimulationError, match=r"experiment job.*completed"):
            await handle_check_job(CheckJobInput(job_id=job.job_id), state_no_sim)
        with pytest.raises(SimulationError, match=r"experiment job.*completed"):
            await handle_cancel_job(CancelJobInput(job_id=job.job_id), state_no_sim)

    async def test_cancelled_experiment_produced_case_remains_analyzable(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        raw = work_dir / "case.raw"
        log = work_dir / "case.log"
        circuit.write_text(".op\n.end\n")
        raw.write_bytes(b"Title: result")
        log.write_text("ok")
        job = _job(work_dir, circuit, status="cancelled")
        job.cases[0].status = "produced"
        job.cases[0].raw_file = raw
        job.cases[0].log_file = log
        job.completeness.produced = 1
        job.done_event.set()
        job.runs_done_event.set()
        state_no_sim.add_experiment_job(job)

        context = services.resolve_experiment_run(job.job_id, state_no_sim)
        assert context.raw == raw
        assert context.netlist == circuit
        assert context.identity["case_id"] == "case_0000"
        source = services.resolve_analysis_source(None, state_no_sim, injected=context)
        assert source.trusted_job_artifact
        assert source.identity == context.identity

    async def test_experiment_case_without_raw_is_not_analysis_source(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="cancelled")
        job.cases[0].status = "cancelled"
        job.completeness.cancelled = 1
        state_no_sim.experiment_jobs[job.job_id] = job
        with pytest.raises(ResultError, match="did not produce a raw"):
            services.resolve_experiment_run(job.job_id, state_no_sim)

    async def test_failed_experiment_case_with_partial_raw_is_not_analysis_source(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        raw = work_dir / "partial.raw"
        circuit.write_text(".tran 1u 1m\n.end\n")
        raw.write_bytes(b"partial")
        job = _job(work_dir, circuit, status="completed_with_failures")
        job.cases[0].status = "failed"
        job.cases[0].raw_file = raw
        job.completeness.failed = 1
        state_no_sim.experiment_jobs[job.job_id] = job
        with pytest.raises(ResultError, match="did not produce a raw"):
            services.resolve_experiment_run(job.job_id, state_no_sim)


def test_persist_jobs_false_submission_fails_clearly(
    state_no_sim: SessionState,
    work_dir: Path,
):
    state_no_sim.config.persist_jobs = False
    circuit = work_dir / "deck.cir"
    circuit.write_text(".op\n.end\n")

    async def exercise() -> None:
        runner = ExperimentRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=object,
            output_folder=work_dir,
        )
        request = ExperimentRunRequest(
            state=state_no_sim,
            request_id="no-persistence",
            fingerprint="a" * 64,
            cases=[_case(circuit)],
            sources=[_source(circuit)],
            simulator="FakeSim",
        )
        with pytest.raises(SimulationError, match=r"persist_jobs = true"):
            await asyncio.shield(runner.submit(request))

    asyncio.run(exercise())
