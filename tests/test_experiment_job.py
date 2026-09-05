"""Persistence, discovery, lifecycle, and compatibility tests for experiments."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from ltspice_mcp.errors import BatchJobError, ResultError, SimulationError
from ltspice_mcp.lib import (
    analysis_snapshot,
    experiment_store,
    job_registry,
    job_store,
    now,
    recent,
    services,
    store_common,
)
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    ExperimentCancellationError,
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
    handle_cancel_job,
)
from tests.conftest import make_batch_job


def _source(circuit: Path, staged: Path | None = None) -> SourceRecord:
    """Provenance for an already-written circuit.

    The digest is the file's real one: a manifest whose sha256 does not
    describe the file it names is a record no replay could accept.
    """
    staged = staged or circuit
    digest = sha256_file(circuit)
    return SourceRecord(
        circuit=circuit.stem,
        path=circuit,
        sha256=digest,
        staged_deck=staged,
        manifest=[
            ManifestEntry(
                path=circuit,
                sha256=digest,
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

    def test_completeness_fell_short_counts_uncounted_runs(self):
        assert not Completeness(expanded=3, produced=3).fell_short
        assert Completeness(expanded=3, produced=1, failed=2).fell_short
        # A run that reached no terminal counter at all: summing the shortfall
        # counters would call this complete, which is the loss going unreported.
        assert Completeness(expanded=3, produced=2).fell_short
        # The same loss masked by a double-counted failure — terminal
        # reconciles to expanded, but a run that was promised never landed.
        assert Completeness(expanded=3, produced=2, failed=1, cancelled=1).fell_short
        # Over-counted the other way: every promised run produced, yet a counter
        # also logged a failure. Only the terminal-side half of the predicate
        # sees this — produced alone reads "3 of 3, complete" while the job's
        # own accounting says four runs reached a terminal state out of three.
        assert Completeness(expanded=3, produced=3, failed=1).fell_short

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

    def test_v2_store_writes_a_self_describing_analysis_snapshot(self, work_dir: Path):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        job.analysis = AnalysisStage(
            status="completed",
            result=analysis_snapshot.envelope({"top": {}, "results": {}}),
        )

        data = experiment_store.serialize_job(job)

        assert data["schema_version"] == 2
        assert data["analysis"]["result"]["schema"] == analysis_snapshot.SCHEMA
        assert data["analysis"]["result"]["schema_version"] == analysis_snapshot.SNAPSHOT_VERSION

    def test_v1_analysis_snapshot_migrates_to_the_single_assembly_shape(self):
        stored = {
            "kind": analysis_snapshot.SCHEMA,
            "snapshot_version": 1,
            "top": {},
            "answer_top": {},
            "answer_coverage_cursor_base": "old-answer-cursor",
            "results": {
                "summary": {
                    "facts": {},
                    "answer_facts": {},
                    "answer_rows": [],
                    "projection_presence": {
                        "present": True,
                        "children": {
                            "value": {
                                "present": True,
                                "children": {"nested": {"present": True, "children": {}}},
                            }
                        },
                    },
                }
            },
        }

        assert analysis_snapshot.classify(stored) == "snapshot"
        assert stored["schema"] == analysis_snapshot.SCHEMA
        assert stored["schema_version"] == analysis_snapshot.SNAPSHOT_VERSION
        assert "answer_top" not in stored
        assert "answer_coverage_cursor_base" not in stored
        block = stored["results"]["summary"]
        assert "answer_facts" not in block
        assert block["projection_presence"] == {"value": {"nested": {}}}

    def test_new_reader_admits_v1_public_analysis_results_without_rewriting_them(
        self, work_dir: Path
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        legacy_result = {"outcome": "complete", "results": {"summary": {"values": []}}}
        job.analysis = AnalysisStage(status="completed", result=legacy_result)
        experiment_store.save_job(job)
        data = json.loads(job.store_path.read_text())
        data["schema_version"] = 1
        job.store_path.write_text(json.dumps(data))

        loaded = experiment_store.load_job(job.job_id, work_dir, own_is_alive=True)

        assert loaded is not None
        assert loaded.analysis.result == legacy_result
        assert json.loads(job.store_path.read_text())["schema_version"] == 1

    def test_old_reader_rejects_a_v2_snapshot_before_deserialization(
        self,
        work_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        job.analysis = AnalysisStage(
            status="completed",
            result=analysis_snapshot.envelope({"top": {}, "results": {}}),
        )
        data = experiment_store.serialize_job(job)

        with caplog.at_level(logging.WARNING, logger="old-experiment-reader"):
            accepted = store_common.accept_schema(
                data,
                job.store_path,
                schema=experiment_store.SCHEMA,
                current_version=1,
                supported_versions=frozenset({1}),
                migrations={},
                logger=logging.getLogger("old-experiment-reader"),
            )

        assert accepted is False
        assert "unsupported schema_version 2" in caplog.text
        assert data["analysis"]["result"]["schema"] == analysis_snapshot.SCHEMA

    @pytest.mark.asyncio
    async def test_pre_stem_job_id_still_loads_and_resolves(
        self, work_dir: Path, state_no_sim: SessionState
    ):
        # Ids gained a deck-name segment; the records already on disk kept the
        # old prefix_timestamp_random form. Nothing on the read path parses an
        # id, so such a record must still load and address exactly as before.
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        legacy_id = "exp_1707916800_a3f7b2c4"
        job = _job(work_dir, circuit, job_id=legacy_id, status="completed")
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(legacy_id, work_dir, own_is_alive=True)
        assert loaded is not None
        assert loaded.job_id == legacy_id
        assert loaded.cases[0].assignments == {"R1": "1k"}

        resolved = await services.resolve_job_async(legacy_id, state_no_sim)
        assert isinstance(resolved, ExperimentJob)
        assert resolved.job_id == legacy_id

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

    def test_restart_promotes_a_case_whose_results_outlived_its_checkpoint(
        self,
        work_dir: Path,
    ):
        """A produced run must not be reported as a failure because a crash beat
        its checkpoint.

        Case progress is persisted every ``total // 20``-th event, so a job over
        ~20 cases can lose the terminal mark of a case that already wrote its
        raw. Counting that as a shortfall is data loss dressed as accounting.
        """
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        runs = work_dir / "runs"
        runs.mkdir()
        job = _job(work_dir, circuit, status="running")
        job.owner_pid = 999_999_999
        job.output_folder = runs
        # On disk from the run that finished; never recorded on the case.
        job.cases[0].run_token = f"{job.job_id}_case_0"
        job.cases[0].status = "running"
        (runs / f"{job.cases[0].run_token}.raw").write_bytes(b"Title: result")
        (runs / f"{job.cases[0].run_token}.log").write_text("ok\n")
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)

        assert loaded is not None
        assert loaded.cases[0].status == "produced"
        assert loaded.cases[0].raw_file == runs / f"{job.cases[0].run_token}.raw"
        assert loaded.completeness.produced == 1
        assert loaded.failures == []
        assert any(item["code"] == "unpersisted_runs_recovered" for item in loaded.observations)

    def test_restart_does_not_promote_a_case_with_no_readable_raw(self, work_dir: Path):
        """The promotion is gated on the raw's header magic, not on a filename.

        An empty or truncated file at the expected path is what a run killed
        mid-write leaves behind; promoting it would report data that is not
        there.
        """
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        runs = work_dir / "runs"
        runs.mkdir()
        job = _job(work_dir, circuit, status="running")
        job.owner_pid = 999_999_999
        job.output_folder = runs
        job.cases[0].run_token = f"{job.job_id}_case_0"
        job.cases[0].status = "running"
        (runs / f"{job.cases[0].run_token}.raw").write_bytes(b"\x00\x00truncated")
        (runs / f"{job.cases[0].run_token}.log").write_text("ok\n")
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)

        assert loaded is not None
        assert loaded.cases[0].status == "failed"
        assert loaded.cases[0].failure_code == "server_restarted"
        assert [row["code"] for row in loaded.failures] == ["server_restarted"]
        # The message names the mechanism the caller can act on, not a door.
        assert "owning process exited" in (loaded.cases[0].error or "")

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

    def _shutdown_pair(
        self, work_dir: Path, runner: Any, *, live_count: int = 1
    ) -> tuple[JobRegistry, Any, list[ExperimentJob], ExperimentJob]:
        """A registry holding ``live_count`` jobs served by ``runner``, then one
        with no runner of its own.

        Both halves are returned: the DELEGATED jobs (whose cancel is the part
        that can hang or refuse) are the ones shutdown is least certain to
        reconcile, so a test that only checked the undelegated one would pass
        while the risky half stayed running.
        """
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        live = [
            _job(work_dir, circuit, job_id=f"exp_live_{index:04d}", status="running")
            for index in range(live_count)
        ]
        following = _job(work_dir, circuit, job_id="exp_next_0001", status="running")
        registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
        for job in live:
            registry.add_experiment_job(job)
        registry.add_experiment_job(following)
        live_ids = {job.job_id for job in live}
        runners = SimpleNamespace(
            get_experiment_runner_for=lambda job: runner if job.job_id in live_ids else None
        )
        return registry, runners, live, following

    @pytest.mark.asyncio
    async def test_shutdown_bounds_a_runner_cancel_that_never_returns(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        entered: list[str] = []

        class HungRunner:
            async def cancel(self, job: Any) -> list[dict[str, Any]]:
                # A wedged simulator: the coordinator's done_event never fires.
                entered.append(job.job_id)
                await asyncio.Event().wait()
                return []

        registry, runners, live, following = self._shutdown_pair(
            work_dir, HungRunner(), live_count=3
        )
        monkeypatch.setattr(job_registry, "_SHUTDOWN_CANCEL_TIMEOUT_S", 0.05)

        await asyncio.wait_for(registry.cancel_running(runners, None), timeout=5)

        assert len(entered) == 3
        assert following.status == "cancelled"
        assert following.done_event.is_set()
        # The DELEGATED jobs — the wedged ones. A cancel that never returned
        # reconciled nothing, so shutdown's own bookkeeping owes them a terminal
        # status; without it they persist as "running" under a dying pid.
        for job in live:
            assert job.status == "cancelled", f"{job.job_id} left non-terminal by a hung cancel"
            assert job.runs_done_event.is_set()
            assert job.done_event.is_set()
            assert [case.failure_code for case in job.cases] == ["server_shutdown"]

    @pytest.mark.asyncio
    async def test_shutdown_persists_a_terminal_status_for_a_wedged_cancel(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The sidecar a wedged cancel leaves behind must not read ``running``.

        This process is exiting; a record still claiming ``running`` under a pid
        that is about to disappear reloads as an interrupted job, mislabelling a
        deliberate shutdown as a crash.
        """

        class HungRunner:
            async def cancel(self, job: Any) -> list[dict[str, Any]]:
                await asyncio.Event().wait()
                return []

        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, job_id="exp_wedged_0001", status="running")
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        registry.add_experiment_job(job)
        # Delegated: a runner owns this job, and its cancel is the one that hangs.
        runners = SimpleNamespace(get_experiment_runner_for=lambda _job: HungRunner())
        monkeypatch.setattr(job_registry, "_SHUTDOWN_CANCEL_TIMEOUT_S", 0.05)

        # The real shutdown sequence: cancel the live work, then flush.
        await asyncio.wait_for(registry.cancel_running(runners, None), timeout=5)
        await registry.drain_pending()

        reloaded = experiment_store.load_job(job.job_id, work_dir)
        assert reloaded is not None
        assert reloaded.status == "cancelled"
        assert reloaded.failures and reloaded.failures[0]["code"] == "server_shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_cancels_live_experiments_concurrently(self, work_dir: Path):
        """The cancel timeout bounds the stage, not each job in it.

        Shutdown flushes job persistence only after this returns, so a bound
        that multiplied by the number of live experiments would let a client's
        shutdown grace kill the process before the sidecars it protects land.
        """
        released = asyncio.Event()
        entered: list[str] = []

        class BlockingRunner:
            async def cancel(self, job: Any) -> list[dict[str, Any]]:
                entered.append(job.job_id)
                # Only the last job to start releases the first, so this
                # returns at all only if the cancels overlap.
                if len(entered) == 3:
                    released.set()
                await released.wait()
                return []

        registry, runners, _live, following = self._shutdown_pair(
            work_dir, BlockingRunner(), live_count=3
        )

        # No monkeypatched timeout: under a per-job bound the first cancel waits
        # out the real one and this outer wait expires first.
        await asyncio.wait_for(registry.cancel_running(runners, None), timeout=3)

        assert len(entered) == 3
        assert released.is_set()
        assert following.status == "cancelled"

    @pytest.mark.asyncio
    async def test_shutdown_bounds_a_task_that_swallows_its_cancellation(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Cancelling a job's task is a request, not a guarantee.

        A task that catches ``CancelledError`` — or sits inside a shielded
        section — keeps its await open for as long as it likes, and shutdown
        awaited it unbounded. Both task passes (batch, then experiment) sit
        ahead of the persistence flush, so either one stalling loses every
        sidecar the flush had left to write.
        """
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        release = asyncio.Event()
        swallowed: list[str] = []

        async def _refuses_to_stop(name: str) -> None:
            while True:
                try:
                    await release.wait()
                    return
                except asyncio.CancelledError:
                    swallowed.append(name)

        batch = make_batch_job("batch_stubborn", status="running", netlist=circuit)
        experiment = _job(work_dir, circuit, job_id="exp_stubborn", status="running")
        batch.task = asyncio.create_task(_refuses_to_stop("batch"))
        experiment.task = asyncio.create_task(_refuses_to_stop("experiment"))
        await asyncio.sleep(0)  # let both reach their first await

        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        registry.add_batch_job(batch)
        registry.add_experiment_job(experiment)
        runners = SimpleNamespace(
            get_batch_runner_for=lambda _job: None,
            get_experiment_runner_for=lambda _job: None,
        )
        monkeypatch.setattr(job_registry, "_SHUTDOWN_CANCEL_TIMEOUT_S", 0.05)

        # The real shutdown sequence: cancel the live work, then flush.
        await asyncio.wait_for(registry.cancel_running(runners, None), timeout=5)
        await registry.drain_pending()

        # Both tasks really did refuse — otherwise this passes for the wrong reason.
        assert set(swallowed) == {"batch", "experiment"}
        reloaded = experiment_store.load_job(experiment.job_id, work_dir)
        assert reloaded is not None and reloaded.status == "cancelled", "the flush never ran"
        _, batches = job_store.load_jobs_for_circuit(circuit)
        assert [(bj.job_id, bj.status) for bj in batches] == [("batch_stubborn", "cancelled")]

        release.set()
        await asyncio.gather(batch.task, experiment.task)

    @pytest.mark.asyncio
    async def test_shutdown_survives_a_batch_cancel_that_raises(self, work_dir: Path):
        """Experiments are reconciled LAST, so every earlier collection's cancel
        stands between them and a terminal status.

        A raise from a batch runner used to propagate straight out of
        ``cancel_running``, leaving the experiment loop below it unrun and the
        persistence flush after it unreached — the experiment records lost to a
        failure in an unrelated job type.
        """
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        batch = make_batch_job("batch_refused", status="running", netlist=circuit)
        experiment = _job(work_dir, circuit, job_id="exp_after_batch", status="running")
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        registry.add_batch_job(batch)
        registry.add_experiment_job(experiment)

        class RefusingBatchRunner:
            async def cancel(self, job: Any, state: Any = None) -> None:
                raise BatchJobError(f"batch {job.job_id} could not be stopped")

        runners = SimpleNamespace(
            get_batch_runner_for=lambda _job: RefusingBatchRunner(),
            get_experiment_runner_for=lambda _job: None,
        )

        # The real shutdown sequence: cancel the live work, then flush.
        await registry.cancel_running(runners, None)
        await registry.drain_pending()

        assert experiment.status == "cancelled", "the experiment loop never ran"
        reloaded = experiment_store.load_job(experiment.job_id, work_dir)
        assert reloaded is not None and reloaded.status == "cancelled", "the flush never ran"

    @pytest.mark.asyncio
    async def test_shutdown_survives_a_runner_cancel_that_raises(self, work_dir: Path):
        class RefusingRunner:
            async def cancel(self, job: Any) -> list[dict[str, Any]]:
                raise ExperimentCancellationError(
                    f"Experiment job {job.job_id} is not owned by a live coordinator"
                )

        registry, runners, live, following = self._shutdown_pair(work_dir, RefusingRunner())

        await registry.cancel_running(runners, None)

        assert following.status == "cancelled"
        assert following.done_event.is_set()
        # A refused cancel is a cancel that did not happen: the DELEGATED job it
        # refused still needs shutdown's bookkeeping, not an exemption for having
        # been asked.
        for job in live:
            assert job.status == "cancelled", f"{job.job_id} left non-terminal by a refused cancel"
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

    def test_a_skipped_pointer_does_not_narrate_itself_at_startup(
        self,
        work_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """The pointer index is global, so a brand-new working directory
        reaches other projects' stale records: at warning level a library's
        first call opened with a dozen lines about someone else's tempdirs.
        The observation channel is what carries the fact to whoever asked."""
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
        with caplog.at_level(logging.DEBUG, logger="ltspice_mcp.lib.experiment_store"):
            registry.ensure_loaded_for(circuit)

        skipped = [
            record for record in caplog.records if "Skipped experiment pointer" in record.message
        ]
        assert skipped, "the fact must still be logged, just not shouted"
        assert [record.levelno for record in skipped] == [logging.DEBUG] * len(skipped)

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

    def test_unknown_drift_code_still_refuses_the_replay(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A drift code the reason table does not know must refuse, not crash.

        The replay guard renders each drift observation through a fixed
        code→reason table. Today staging emits exactly the two codes the table
        knows, so the lookup cannot miss — but a third observation code added
        in deck_staging would turn a clean refusal into a KeyError on the
        replay path. The guard must fail closed: refuse the replay and name
        the unknown code verbatim.
        """
        from ltspice_mcp.lib import experiment_runner

        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit)
        monkeypatch.setattr(
            experiment_runner,
            "verify_staged_manifest",
            lambda manifest: [
                {"code": "source_relocated_after_staging", "evidence": {"path": str(circuit)}}
            ],
        )
        with pytest.raises(IdempotencyConflictError, match="source_relocated_after_staging"):
            experiment_runner.verify_replay_sources(job, "request-1")


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

    async def test_cancel_job_rejects_experiment_without_netlist_deref(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        job = _job(work_dir, circuit, status="completed")
        experiment_store.save_job(job)

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
