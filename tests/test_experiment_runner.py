"""Coordinator execution tests using callback-controlled mock simulations."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_runner import (
    ExperimentCancellationError,
    ExperimentRunner,
    ExperimentRunRequest,
    IdempotencyConflictError,
)
from ltspice_mcp.lib.experiment_types import (
    ExperimentCase,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.runner_base import RunnerBase, RunOutcome
from ltspice_mcp.state import SessionState


class MockSimulator:
    """Simulator identity used only for runner construction and scoped kills."""


async def _wait_for(condition, *, timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            pytest.fail(f"condition not met within {timeout_s}s")
        await asyncio.sleep(0.005)


def _cases(work_dir: Path, count: int) -> tuple[list[ExperimentCase], list[SourceRecord]]:
    circuit = work_dir / "deck.cir"
    circuit.write_text(".op\n.end\n")
    # The recorded digest is the file's real one: a replay checks the manifest
    # against the source on disk, so a fictional digest reads as an edited deck.
    digest = sha256_file(circuit)
    source = SourceRecord(
        circuit="deck",
        path=circuit,
        sha256=digest,
        staged_deck=circuit,
        manifest=[
            ManifestEntry(
                path=circuit,
                sha256=digest,
                staged=True,
                live=False,
                staged_path=circuit,
            )
        ],
        simulator="MockSimulator",
    )
    cases = [
        ExperimentCase(
            case_id=f"case_{index:04d}",
            run_index=index,
            circuit="deck",
            circuit_path=circuit,
            staged_deck=circuit,
            deck_sha256=f"deck-sha-{index}",
            assignments={"R1": f"{index + 1}k"},
        )
        for index in range(count)
    ]
    return cases, [source]


def _request(
    state: SessionState,
    work_dir: Path,
    *,
    request_id: str,
    count: int = 1,
    fingerprint: str = "a" * 64,
    max_parallel: int = 1,
    run_timeout_s: float | None = None,
    job_deadline_s: float | None = None,
    kill_grace_s: float = 0.05,
    analysis_callback=None,
) -> ExperimentRunRequest:
    cases, sources = _cases(work_dir, count)
    return ExperimentRunRequest(
        state=state,
        request_id=request_id,
        fingerprint=fingerprint,
        cases=cases,
        sources=sources,
        simulator="MockSimulator",
        max_parallel=max_parallel,
        run_timeout_s=run_timeout_s,
        job_deadline_s=job_deadline_s,
        kill_grace_s=kill_grace_s,
        analysis_request={"recipes": []} if analysis_callback is not None else None,
        analysis_callback=analysis_callback,
    )


def _controlled_submit(
    monkeypatch: pytest.MonkeyPatch,
    runner: ExperimentRunner,
) -> tuple[dict[str, Any], list[str]]:
    callbacks: dict[str, Any] = {}
    submissions: list[str] = []

    def submit(_netlist: Path, run_filename: str, callback):
        token = Path(run_filename).stem
        submissions.append(token)
        callbacks[token] = callback
        return object()

    monkeypatch.setattr(runner, "submit_netlist", submit)
    return callbacks, submissions


def _success(work_dir: Path, token: str) -> RunOutcome:
    raw = work_dir / f"{token}.raw"
    log = work_dir / f"{token}.log"
    raw.write_bytes(b"Title: mock result")
    log.write_text("ok")
    return RunOutcome(str(raw), str(log), raw.stat().st_size, None)


async def _cancel_during_launch(
    state: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[str], str]:
    """Cancel a case while its worker thread is inside the simulator launch.

    This is the window the coordinator has to get right: the case is handed to
    a thread, the event loop is free, and the cancel arrives before the loop
    resumes. Blocking the fake submit reproduces that interleaving exactly
    rather than racing for it. A cancel landing here must still see a case that
    launched -- otherwise it reports a run that never started, skips the
    token-scoped kill, and the simulator finishes into an unclaimed artifact.
    """
    runner = ExperimentRunner(
        asyncio.get_running_loop(),
        MockSimulator,
        work_dir,
        max_parallel=1,
    )
    launching = threading.Event()
    release = threading.Event()
    callbacks: dict[str, Any] = {}
    submissions: list[str] = []

    def submit(_netlist: Path, run_filename: str, callback):
        token = Path(run_filename).stem
        launching.set()
        release.wait(5)
        submissions.append(token)
        callbacks[token] = callback
        return object()

    monkeypatch.setattr(runner, "submit_netlist", submit)
    killed: list[str] = []

    async def record_kill(token: str) -> None:
        killed.append(token)

    monkeypatch.setattr(runner, "_kill_case", record_kill)
    receipt = await asyncio.shield(
        runner.submit(_request(state, work_dir, request_id="cancel-mid-launch", kill_grace_s=0.2))
    )
    await _wait_for(launching.is_set)
    cancel_task = asyncio.create_task(
        runner.cancel(receipt.job, control_token=receipt.control_token)
    )
    # Ordering has to be exact, so wait on the stop flag itself rather than on
    # a public symptom of it: the point of the test is what the coordinator
    # does with a cancel that arrives DURING the launch.
    execution = runner._executions[receipt.job.job_id]
    await _wait_for(execution.cancel_event.is_set)
    release.set()
    await _wait_for(lambda: bool(submissions))
    await _wait_for(lambda: bool(killed) or cancel_task.done())
    token = submissions[0]
    # The launched process reports exit either way; a coordinator that disowned
    # it simply has nowhere to put the news.
    callbacks[token](RunOutcome("", str(work_dir / f"{token}.fail"), 0, "killed"))
    await asyncio.wait_for(cancel_task, 2)
    assert await runner.wait(receipt.job, 2)
    return receipt.job, killed, token


@pytest.mark.asyncio
class TestSubmitPrimitive:
    async def test_job_agnostic_submit_bridges_outcome_to_event_loop(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        loop = asyncio.get_running_loop()
        base = RunnerBase(loop, MockSimulator, work_dir, max_parallel=1)
        netlist = work_dir / "primitive.cir"
        raw = work_dir / "primitive.raw"
        log = work_dir / "primitive.log"
        netlist.write_text(".op\n.end\n")
        raw.write_bytes(b"Title: mock result")
        log.write_text("ok")
        calls: list[dict[str, Any]] = []

        class FakeHandle:
            def run(self, _netlist: str, **kwargs):
                calls.append(kwargs)
                kwargs["callback"](raw, log)

        monkeypatch.setattr(base, "_build_sim_runner", FakeHandle)
        received: asyncio.Future[RunOutcome] = loop.create_future()
        handle = await asyncio.to_thread(
            base.submit_netlist,
            netlist,
            "case-token.cir",
            received.set_result,
        )
        outcome = await asyncio.wait_for(received, 1)
        assert isinstance(handle, FakeHandle)
        assert outcome.raw_file == str(raw)
        assert calls[0]["run_filename"] == "case-token.cir"
        assert calls[0]["callback_on_error"] is True
        assert calls[0]["exe_log"] is True

    async def test_submitted_runner_outlives_a_caller_that_discards_it(
        self,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """spicelib's SimRunner.__del__ calls wait_completion().

        So a caller that drops the returned runner has its thread pinned in
        the destructor for the whole simulation — the coroutine that submitted
        never resumes, never reaches the code watching for a cancellation, and
        the run cannot be stopped. Submission has to survive being called for
        its side effect, which is how the experiment coordinator calls it.
        """
        base = RunnerBase(asyncio.get_running_loop(), MockSimulator, work_dir, max_parallel=1)
        netlist = work_dir / "discarded.cir"
        netlist.write_text(".op\n.end\n")
        raw, log = work_dir / "discarded.raw", work_dir / "discarded.log"
        raw.write_bytes(b"Title: mock result")
        log.write_text("ok")
        simulating = threading.Event()
        finish = threading.Event()
        destroyed = threading.Event()
        threads: list[threading.Thread] = []

        class FakeSimRunner:
            def __init__(self) -> None:
                self.active_tasks: list[threading.Thread] = []

            def run(self, _netlist: str, **kwargs):
                callback = kwargs["callback"]

                # A closure, not a bound method: spicelib's RunTask does not
                # reference the SimRunner that started it, so a fake that does
                # would keep itself alive and pass on its own.
                def simulate() -> None:
                    simulating.set()
                    finish.wait(5)
                    callback(raw, log)

                task = threading.Thread(target=simulate)
                self.active_tasks.append(task)
                threads.append(task)
                task.start()

            def __del__(self) -> None:
                destroyed.set()

        monkeypatch.setattr(base, "_build_sim_runner", FakeSimRunner)

        def submit_and_discard(token: str) -> None:
            base.submit_netlist(netlist, token, lambda _outcome: None)

        await asyncio.to_thread(submit_and_discard, "kept.cir")
        await _wait_for(simulating.is_set)
        assert not destroyed.is_set()

        # Released once its thread is done, so a long session does not hoard
        # one runner per run.
        finish.set()
        await _wait_for(lambda: not threads[0].is_alive())
        await asyncio.to_thread(submit_and_discard, "next.cir")
        assert destroyed.is_set()


def _capturing_submit(
    monkeypatch: pytest.MonkeyPatch,
    runner: ExperimentRunner,
) -> tuple[dict[str, Any], list[tuple[Path, bytes]]]:
    """Like ``_controlled_submit``, but also snapshots the deck AS SUBMITTED.

    The bytes are read inside the fake submit because a generated runnable copy
    is deleted the moment submit returns — reading it afterwards would find
    nothing and prove only that the file is gone.
    """
    callbacks: dict[str, Any] = {}
    submitted: list[tuple[Path, bytes]] = []

    def submit(netlist: Path, run_filename: str, callback):
        submitted.append((netlist, netlist.read_bytes()))
        callbacks[Path(run_filename).stem] = callback
        return object()

    monkeypatch.setattr(runner, "submit_netlist", submit)
    return callbacks, submitted


@pytest.mark.asyncio
class TestLogopinfoInjection:
    """An LTspice ``.op`` case must reach the simulator with ``.options
    logopinfo``, or its log carries no per-device operating-point block and
    gm/vth/vdsat read back empty. The experiments path is the consolidated
    profile's only execution path, so the injection the single-run path already
    does has to happen here too — at submit time, on a copy, because the staged
    deck and its recorded digest are what replay and provenance compare against.
    """

    async def test_ltspice_op_case_submits_augmented_copy_leaving_staged_deck_intact(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            LTspice,
            work_dir,
            max_parallel=1,
        )
        callbacks, submitted = _capturing_submit(monkeypatch, runner)
        request = _request(state_no_sim, work_dir, request_id="logopinfo-op")
        case = request.cases[0]
        staged = case.staged_deck
        staged_bytes = staged.read_bytes()
        case.deck_sha256 = sha256_file(staged)

        receipt = await asyncio.shield(runner.submit(request))
        await _wait_for(lambda: len(submitted) == 1)
        run_deck, run_bytes = submitted[0]

        assert run_deck != staged
        assert b".options logopinfo" in run_bytes
        # The staged deck is byte-identical and still hashes to what the record
        # pins — an idempotent replay compares against exactly these.
        assert staged.read_bytes() == staged_bytes
        assert sha256_file(staged) == case.deck_sha256
        # The augmented copy is per-case scratch, gone once spicelib staged it.
        assert not run_deck.exists()

        token = next(iter(callbacks))
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.completeness.produced == 1

    async def test_non_ltspice_case_is_submitted_unmodified(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # ngspice reads device params off @dev[param] raw traces and has no
        # logopinfo option, so its decks must reach the simulator untouched.
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submitted = _capturing_submit(monkeypatch, runner)
        request = _request(state_no_sim, work_dir, request_id="logopinfo-ngspice")
        staged = request.cases[0].staged_deck

        receipt = await asyncio.shield(runner.submit(request))
        await _wait_for(lambda: len(submitted) == 1)
        run_deck, run_bytes = submitted[0]

        assert run_deck == staged
        assert b"logopinfo" not in run_bytes.lower()
        token = next(iter(callbacks))
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)


class TestArtifactCleanup:
    def test_case_cleanup_unlinks_exact_heavy_artifacts_only(self, work_dir: Path):
        runner = ExperimentRunner(
            asyncio.new_event_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        try:
            case = _cases(work_dir, 1)[0][0]
            case.run_token = "exp_case_0"
            # The job names its own artifact directory; cleanup reconstructs
            # the paths inside it, not beside the runner's output folder.
            job = SimpleNamespace(job_id="exp_cleanup", output_folder=work_dir)
            case.log_file = work_dir / f"{case.run_token}.fail"
            run_netlist = work_dir / f"{case.run_token}.cir"
            raw = work_dir / f"{case.run_token}.raw"
            unrelated = work_dir / f"{case.run_token}.tmp"
            run_netlist.write_text(".op\n.end\n")
            raw.write_bytes(b"partial")
            case.log_file.write_text("killed")
            unrelated.write_text("keep")

            runner._remove_case_artifacts(job, case)  # type: ignore[arg-type]

            assert not run_netlist.exists()
            assert not raw.exists()
            assert case.log_file.exists()
            assert unrelated.exists()
        finally:
            runner.loop.close()


@pytest.mark.asyncio
class TestExperimentSubmission:
    async def test_receipt_is_durable_and_cases_submit_after_registration(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        request = _request(state_no_sim, work_dir, request_id="durable-receipt")

        receipt = await asyncio.shield(runner.submit(request))
        assert receipt.job.job_id in state_no_sim.experiment_jobs
        assert receipt.job.store_path.is_file()
        assert submissions == []
        index = experiment_store.load_request_index(request.request_id, work_dir)
        assert index is not None
        assert index["job_id"] == receipt.job.job_id
        await _wait_for(lambda: len(submissions) == 1)
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.status == "completed"
        assert receipt.job.completeness.produced == 1

    async def test_matching_replay_returns_same_job_and_control_token_without_resubmit(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        first_request = _request(state_no_sim, work_dir, request_id="replay-request")
        first = await asyncio.shield(runner.submit(first_request))
        await _wait_for(lambda: len(submissions) == 1)

        replay_request = _request(state_no_sim, work_dir, request_id="replay-request")
        replay = await asyncio.shield(runner.submit(replay_request))
        assert replay.replayed
        assert replay.job is first.job
        assert replay.control_token == first.control_token
        assert any(item["code"] == "idempotent_replay" for item in replay.job.observations)
        assert len(submissions) == 1

        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(first.job, 1)

    async def test_same_request_id_different_fingerprint_conflicts(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        first = await asyncio.shield(
            runner.submit(_request(state_no_sim, work_dir, request_id="conflict"))
        )
        await _wait_for(lambda: len(submissions) == 1)
        with pytest.raises(IdempotencyConflictError, match="different request payload"):
            await asyncio.shield(
                runner.submit(
                    _request(
                        state_no_sim,
                        work_dir,
                        request_id="conflict",
                        fingerprint="b" * 64,
                    )
                )
            )
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(first.job, 1)

    async def test_transport_cancel_during_barrier_does_not_cancel_durable_job(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        request = _request(state_no_sim, work_dir, request_id="cancelled-dwell")
        entered = threading.Event()
        release = threading.Event()
        original = runner._durable_barrier

        def blocked_barrier(run_request, candidate):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test barrier was not released")
            return original(run_request, candidate)

        monkeypatch.setattr(runner, "_durable_barrier", blocked_barrier)

        async def handler_dwell():
            return await asyncio.shield(runner.submit(request))

        handler = asyncio.create_task(handler_dwell())
        assert await asyncio.to_thread(entered.wait, 2)
        handler.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await handler

        await _wait_for(
            lambda: experiment_store.load_request_index(request.request_id, work_dir) is not None
        )
        index = experiment_store.load_request_index(request.request_id, work_dir)
        assert index is not None
        await _wait_for(lambda: len(submissions) == 1)
        persisted = experiment_store.load_job(
            str(index["job_id"]),
            work_dir,
            own_is_alive=True,
        )
        assert persisted is not None
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        local_job = state_no_sim.experiment_jobs[str(index["job_id"])]
        assert await runner.wait(local_job, 1)


@pytest.mark.asyncio
class TestCaseConcurrencyAndTimeouts:
    async def test_case_progress_persistence_uses_sparse_checkpoints(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=40,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        persisted_statuses: list[str] = []
        monkeypatch.setattr(
            state_no_sim,
            "persist_job",
            lambda job: persisted_statuses.append(job.status),
        )

        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="sparse-case-persistence",
                    count=40,
                    max_parallel=40,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 40)
        for token in submissions:
            callbacks[token](_success(work_dir, token))

        assert await runner.wait(receipt.job, 1)
        assert len(persisted_statuses) == 63
        assert persisted_statuses[-1] == "completed"

    async def test_one_runner_caps_cases_across_concurrent_jobs(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The concurrency cap belongs to the runner, not to one job's share of it.

        Every experiment used to get a private semaphore and nothing else
        bounded a launch, so a server capped at one simulator process launched
        one process PER JOB: three concurrent run_experiments calls meant three
        simulators. The per-job semaphore still divides a job's own share; the
        runner's is what the machine is actually protected by.
        """
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)

        first = await asyncio.shield(
            runner.submit(_request(state_no_sim, work_dir, request_id="shared-cap-first"))
        )
        second = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="shared-cap-second",
                    fingerprint="b" * 64,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        # The second job's case is queued on the runner's permit, not launched.
        await asyncio.sleep(0.05)
        assert len(submissions) == 1

        callbacks[submissions[0]](_success(work_dir, submissions[0]))
        await _wait_for(lambda: len(submissions) == 2)
        callbacks[submissions[1]](_success(work_dir, submissions[1]))

        assert await runner.wait(first.job, 1)
        assert await runner.wait(second.job, 1)
        assert first.job.completeness.produced == 1
        assert second.job.completeness.produced == 1

    async def test_a_job_cannot_raise_its_share_above_the_runner_cap(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """max_parallel on a request lowers a job's share; it never raises the cap."""
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="cap-override",
                    count=3,
                    max_parallel=3,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        await asyncio.sleep(0.05)
        assert len(submissions) == 1
        assert runner._executions[receipt.job.job_id].capacity == 1

        for index in range(3):
            await _wait_for(lambda wanted=index + 1: len(submissions) == wanted)
            token = submissions[index]
            callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.completeness.produced == 3

    async def test_semaphore_is_held_from_submission_until_callback(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="semaphore-hold",
                    count=2,
                    max_parallel=1,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        execution = runner._executions[receipt.job.job_id]
        assert execution.semaphore._value == 0
        assert len(execution.slots_held) == 1
        assert receipt.job.completeness.submitted == 1

        first = submissions[0]
        callbacks[first](_success(work_dir, first))
        await _wait_for(lambda: len(submissions) == 2)
        assert receipt.job.completeness.submitted == 2
        second = submissions[1]
        callbacks[second](_success(work_dir, second))
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.completeness.produced == 2

    async def test_run_timeout_retains_permit_until_late_callback(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        kill_started = asyncio.Event()
        kills: list[str] = []

        async def no_kill(token: str) -> None:
            kills.append(token)
            kill_started.set()

        monkeypatch.setattr(runner, "_kill_case", no_kill)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="run-timeout",
                    run_timeout_s=0.01,
                    kill_grace_s=0.01,
                )
            )
        )
        assert await runner.wait(receipt.job, 1)
        execution = runner._executions[receipt.job.job_id]
        case = receipt.job.cases[0]
        assert case.failure_code == "kill_unconfirmed"
        assert case.case_id in execution.retained_slots
        assert execution.semaphore._value == 0
        assert receipt.job.status == "completed_with_failures"

        token = submissions[0]
        assert kills == [token]
        assert await runner.cancel(receipt.job, control_token=receipt.control_token) == []
        assert kills == [token, token]
        raw = work_dir / f"{token}.raw"
        raw.write_bytes(b"partial")
        callbacks[token](RunOutcome(str(raw), str(work_dir / f"{token}.fail"), 0, "killed"))
        await _wait_for(lambda: case.case_id not in execution.retained_slots)
        await _wait_for(lambda: not raw.exists())
        assert execution.semaphore._value == 1
        assert any(item["code"] == "late_simulator_exit" for item in case.observations)
        assert receipt.job.completeness.failed == 1

    async def test_run_timeout_callback_within_grace_releases_capacity(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        kill_started = asyncio.Event()

        async def record_kill(_token: str) -> None:
            kill_started.set()

        monkeypatch.setattr(runner, "_kill_case", record_kill)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="run-timeout-grace",
                    run_timeout_s=0.01,
                    kill_grace_s=0.2,
                )
            )
        )
        await asyncio.wait_for(kill_started.wait(), 1)
        token = submissions[0]
        raw = work_dir / f"{token}.raw"
        raw.write_bytes(b"partial")
        callbacks[token](
            RunOutcome(
                str(raw),
                str(work_dir / f"{token}.fail"),
                raw.stat().st_size,
                "killed",
            )
        )

        assert await runner.wait(receipt.job, 1)
        assert receipt.job.cases[0].failure_code == "run_timeout"
        assert receipt.job.cases[0].raw_file is None
        # Execution cleanup runs after done_event, past the watcher-task
        # cancellation awaits — poll instead of asserting synchronously.
        await _wait_for(lambda: runner._executions.get(receipt.job.job_id) is None)
        assert not await asyncio.to_thread(raw.exists)

    async def test_all_zombie_capacity_fails_queued_cases_without_overlaunch(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        _callbacks, submissions = _controlled_submit(monkeypatch, runner)

        async def no_kill(_token: str) -> None:
            return None

        monkeypatch.setattr(runner, "_kill_case", no_kill)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="capacity-backstop",
                    count=3,
                    max_parallel=1,
                    run_timeout_s=0.01,
                    kill_grace_s=0.01,
                )
            )
        )
        assert await runner.wait(receipt.job, 1)
        assert len(submissions) == 1
        assert receipt.job.completeness.submitted == 1
        assert receipt.job.completeness.failed == 3
        assert {case.failure_code for case in receipt.job.cases} == {
            "kill_unconfirmed",
            "kill_unconfirmed_capacity",
        }
        assert receipt.job.completeness.terminal == receipt.job.completeness.expanded

    async def test_job_deadline_stops_active_and_queued_cases(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        _callbacks, submissions = _controlled_submit(monkeypatch, runner)

        async def no_kill(_token: str) -> None:
            return None

        analysis_started = False

        async def analyze(_job):
            nonlocal analysis_started
            analysis_started = True
            return {}

        monkeypatch.setattr(runner, "_kill_case", no_kill)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="job-deadline",
                    count=2,
                    max_parallel=1,
                    job_deadline_s=0.01,
                    kill_grace_s=0.01,
                    analysis_callback=analyze,
                )
            )
        )
        assert await runner.wait(receipt.job, 1)
        assert len(submissions) == 1
        assert receipt.job.status == "completed_with_failures"
        assert receipt.job.completeness.failed == 2
        assert receipt.job.analysis.status == "cancelled"
        assert not analysis_started
        assert any(item["code"] == "job_deadline" for item in receipt.job.observations)
        assert {case.failure_code for case in receipt.job.cases} == {
            "kill_unconfirmed",
            "job_deadline",
        }


@pytest.mark.asyncio
class TestCancellationAndAnalysis:
    async def test_cancel_barrier_prevents_queued_submission_and_kills_active_case(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        killed: list[str] = []

        async def record_kill(token: str) -> None:
            killed.append(token)

        monkeypatch.setattr(runner, "_kill_case", record_kill)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="explicit-cancel",
                    count=2,
                    max_parallel=1,
                    kill_grace_s=0.2,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        cancel_task = asyncio.create_task(
            runner.cancel(receipt.job, control_token=receipt.control_token)
        )
        await _wait_for(lambda: bool(killed))
        token = submissions[0]
        callbacks[token](RunOutcome("", str(work_dir / f"{token}.fail"), 0, "killed"))
        cancel_receipts = await asyncio.wait_for(cancel_task, 1)
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.status == "cancelled"
        assert len(submissions) == 1
        assert receipt.job.completeness.cancelled == 2
        assert {item["case_id"] for item in cancel_receipts} == {
            "case_0000",
            "case_0001",
        }

    async def test_cancel_mid_launch_counts_the_case_as_submitted(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        job, _killed, _token = await _cancel_during_launch(state_no_sim, work_dir, monkeypatch)
        case = job.cases[0]
        assert case.submitted_at is not None
        assert job.completeness.submitted == 1
        assert "before submission" not in (case.error or "")

    async def test_cancel_mid_launch_kills_the_simulator_it_started(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _job, killed, token = await _cancel_during_launch(state_no_sim, work_dir, monkeypatch)
        assert killed == [token]

    async def test_wrong_control_token_cannot_cancel_foreign_owned_job(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(_request(state_no_sim, work_dir, request_id="token-auth"))
        )
        await _wait_for(lambda: len(submissions) == 1)
        receipt.job.owner_pid = -1
        with pytest.raises(ExperimentCancellationError, match="not authorized"):
            await runner.cancel(receipt.job, control_token="wrong-token")
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)

    async def test_matching_control_token_can_cancel_foreign_owned_job(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        kill_started = asyncio.Event()

        async def no_kill(_token: str) -> None:
            kill_started.set()

        monkeypatch.setattr(runner, "_kill_case", no_kill)
        receipt = await asyncio.shield(
            runner.submit(_request(state_no_sim, work_dir, request_id="token-owner"))
        )
        await _wait_for(lambda: len(submissions) == 1)
        receipt.job.owner_pid = -1
        cancel_task = asyncio.create_task(
            runner.cancel(receipt.job, control_token=receipt.control_token)
        )
        await asyncio.wait_for(kill_started.wait(), 1)
        token = submissions[0]
        callbacks[token](RunOutcome("", str(work_dir / f"{token}.fail"), 0, "killed"))
        await asyncio.wait_for(cancel_task, 1)
        assert receipt.job.status == "cancelled"

    async def test_runs_done_event_precedes_attached_analysis_terminality(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        analysis_started = asyncio.Event()
        release_analysis = asyncio.Event()

        async def analyze(_job):
            analysis_started.set()
            await release_analysis.wait()
            return {"summary": "done"}

        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="analysis-stage",
                    analysis_callback=analyze,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        await asyncio.wait_for(analysis_started.wait(), 1)
        assert receipt.job.runs_done_event.is_set()
        assert not receipt.job.done_event.is_set()
        assert receipt.job.status == "analyzing"
        assert receipt.job.analysis.status == "running"
        assert await runner.wait(receipt.job, 0.01, wait_for="runs")
        assert not await runner.wait(receipt.job, 0.01, wait_for="all")

        release_analysis.set()
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.status == "completed"
        assert receipt.job.analysis.result == {"summary": "done"}

    async def test_analysis_failure_keeps_runs_and_completes_with_failures(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        async def analyze(_job):
            raise RuntimeError("recipe failed")

        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="analysis-failure",
                    analysis_callback=analyze,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        token = submissions[0]
        callbacks[token](_success(work_dir, token))
        assert await runner.wait(receipt.job, 1)
        assert receipt.job.status == "completed_with_failures"
        assert receipt.job.analysis.status == "failed"
        assert receipt.job.completeness.produced == 1
        assert receipt.job.cases[0].raw_file is not None

    async def test_cancel_during_analysis_preserves_produced_run(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        analysis_started = asyncio.Event()
        analysis_stopped = asyncio.Event()

        async def analyze(_job):
            analysis_started.set()
            try:
                await asyncio.Future()
            finally:
                analysis_stopped.set()

        runner = ExperimentRunner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks, submissions = _controlled_submit(monkeypatch, runner)
        receipt = await asyncio.shield(
            runner.submit(
                _request(
                    state_no_sim,
                    work_dir,
                    request_id="cancel-analysis",
                    analysis_callback=analyze,
                )
            )
        )
        await _wait_for(lambda: len(submissions) == 1)
        token = submissions[0]
        outcome = _success(work_dir, token)
        callbacks[token](outcome)
        await asyncio.wait_for(analysis_started.wait(), 1)

        await asyncio.wait_for(
            runner.cancel(receipt.job, control_token=receipt.control_token),
            1,
        )
        assert analysis_stopped.is_set()
        assert receipt.job.status == "cancelled"
        assert receipt.job.analysis.status == "cancelled"
        assert receipt.job.cases[0].status == "produced"
        assert receipt.job.cases[0].raw_file == Path(outcome.raw_file)
        assert await asyncio.to_thread(Path(outcome.raw_file).exists)
