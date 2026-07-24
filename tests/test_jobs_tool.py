"""Consolidated jobs control-plane contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jsonschema
import pytest

from ltspice_mcp.lib import experiment_store, now, recent
from ltspice_mcp.lib.experiment_runner import ExperimentRunner, ExperimentRunRequest
from ltspice_mcp.lib.experiment_types import (
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import (
    JOBS_OUTPUT_SCHEMA,
    JobsInput,
    RunExperimentsInput,
    handle_jobs,
    handle_run_experiments,
)
from tests.conftest import make_batch_job, make_sim_job


class MockSimulator:
    """Simulator identity for coordinator-only tests."""


# Positive so the stored record round-trips it (pid_of drops pid <= 0);
# liveness is monkeypatched per test.
_FOREIGN_PID = 999_999_999


def _circuit(work_dir: Path, name: str = "deck.cir") -> Path:
    path = work_dir / name
    path.write_text("V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    return path


def _experiment(
    work_dir: Path,
    circuit: Path,
    *,
    job_id: str = "exp_jobs_0001",
    request_id: str = "jobs-request",
    count: int = 1,
    status: str = "queued",
    case_status: str = "queued",
) -> ExperimentJob:
    cases = [
        ExperimentCase(
            case_id=f"case_{index:04d}",
            run_index=index,
            circuit=circuit.stem,
            circuit_path=circuit,
            staged_deck=circuit,
            deck_sha256=f"sha-{index}",
            assignments={"R1": f"{index + 1}k"},
            status=case_status,  # type: ignore[arg-type]
            raw_file=(work_dir / f"{job_id}-{index}.raw" if case_status == "produced" else None),
            log_file=(work_dir / f"{job_id}-{index}.log" if case_status == "produced" else None),
        )
        for index in range(count)
    ]
    for case in cases:
        if case.raw_file is not None:
            case.raw_file.write_bytes(b"Title: mock")
        if case.log_file is not None:
            case.log_file.write_text("ok")
    source = SourceRecord(
        circuit=circuit.stem,
        path=circuit,
        sha256="source-sha",
        staged_deck=circuit,
        manifest=[
            ManifestEntry(
                path=circuit,
                sha256="source-sha",
                staged=True,
                live=False,
                staged_path=circuit,
            )
        ],
        linter_version="test",
        simulator="MockSimulator",
    )
    completeness = Completeness(declared=1, expanded=count)
    completeness.recount(cases)
    job = ExperimentJob(
        job_id=job_id,
        request_id=request_id,
        fingerprint="f" * 64,
        canonicalizer_version=experiment_store.CANONICALIZER_VERSION,
        control_token="control-secret",
        store_path=experiment_store.record_path(job_id, work_dir),
        cases=cases,
        sources=[source],
        simulator="MockSimulator",
        completeness=completeness,
        status=status,  # type: ignore[arg-type]
    )
    if all(case.status in {"produced", "failed", "cancelled", "skipped"} for case in cases):
        job.runs_done_event.set()
    if status in {"completed", "completed_with_failures", "failed", "cancelled", "interrupted"}:
        job.completed_at = now()
        job.done_event.set()
    return job


def _persist_experiment(job: ExperimentJob, work_dir: Path) -> None:
    experiment_store.save_job(job)
    experiment_store.save_pointers(job)
    experiment_store.save_request_index(
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        canonicalizer_version=job.canonicalizer_version,
        job_id=job.job_id,
        working_dir=work_dir,
    )


def _args(action: str, **values) -> JobsInput:
    return JobsInput.model_validate({"action": action, **values})


def _assert_jobs_schema(result) -> dict:
    data = result.structuredContent
    assert data is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(data)
    return data


def _assert_no_control_token(value) -> None:
    if isinstance(value, dict):
        assert "control_token" not in value
        for item in value.values():
            _assert_no_control_token(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_control_token(item)


def test_jobs_output_schema_is_discriminated_by_action():
    assert JOBS_OUTPUT_SCHEMA["discriminator"]["propertyName"] == "action"
    actions = {branch["properties"]["action"]["const"] for branch in JOBS_OUTPUT_SCHEMA["oneOf"]}
    assert actions == {"status", "wait", "cancel", "list", "runs"}


async def _wait_for(condition, *, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            pytest.fail("condition was not met before the test deadline")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
class TestActionShapesAndTokenSecrecy:
    async def test_every_action_is_schema_valid_and_never_echoes_token(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            status="completed",
            case_status="produced",
        )
        job.analysis = AnalysisStage(
            status="completed",
            request={"control_token": "nested-secret"},
            result={"control_token": "nested-result-secret"},
        )
        _persist_experiment(job, work_dir)
        state_no_sim.all_jobs[job.job_id] = job
        await asyncio.to_thread(recent.touch, circuit)

        results = [
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim),
            await handle_jobs(
                _args("wait", request_id=job.request_id, timeout_s=0),
                state_no_sim,
            ),
            await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim),
            await handle_jobs(_args("list"), state_no_sim),
            await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim),
        ]

        for result in results:
            data = _assert_jobs_schema(result)
            _assert_no_control_token(data)

    async def test_request_id_resolves_to_receipt(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, status="running")
        _persist_experiment(job, work_dir)

        data = _assert_jobs_schema(
            await handle_jobs(
                _args("status", request_id=job.request_id),
                state_no_sim,
            )
        )

        assert data["job_id"] == job.job_id
        assert data["request_id"] == job.request_id
        assert data["analysis_status"] == "not_requested"

    async def test_terminal_receipt_with_paged_runs_emits_working_cursor(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        """A terminal experiment with more cases than one inline page must
        return a receipt (not crash) whose runs page carries a next_cursor
        that jobs(action="runs") actually accepts."""
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            count=60,
            status="completed",
            case_status="produced",
        )
        _persist_experiment(job, work_dir)
        state_no_sim.all_jobs[job.job_id] = job

        for action in ("status", "wait"):
            kwargs = {"timeout_s": 0} if action == "wait" else {}
            data = _assert_jobs_schema(
                await handle_jobs(_args(action, job_id=job.job_id, **kwargs), state_no_sim)
            )
            runs = data["runs"]
            assert runs["truncated"] is True
            assert runs["returned"] == 50
            assert runs["total"] == 60
            assert runs["next_cursor"] == "o:50"
            assert runs["next_cursor"] in data["hint"]

        follow = _assert_jobs_schema(
            await handle_jobs(
                _args("runs", job_id=job.job_id, cursor="o:50"),
                state_no_sim,
            )
        )
        assert follow["returned"] == 10
        assert follow["truncated"] is False

    async def test_completed_with_failures_status_reports_partial_outcome(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            status="completed_with_failures",
            case_status="failed",
        )
        state_no_sim.all_jobs[job.job_id] = job

        data = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )

        assert data["outcome"] == "partial"


@pytest.mark.asyncio
class TestWait:
    async def test_timeout_and_runs_only_wait_flags(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        running = _experiment(work_dir, circuit, status="running")
        state_no_sim.all_jobs[running.job_id] = running

        timed_out = _assert_jobs_schema(
            await handle_jobs(
                _args("wait", job_id=running.job_id, timeout_s=0),
                state_no_sim,
            )
        )
        assert timed_out["timed_out"] is True

        analyzing = _experiment(
            work_dir,
            circuit,
            job_id="exp_jobs_analysis",
            request_id="analysis-request",
            status="analyzing",
            case_status="produced",
        )
        analyzing.analysis = AnalysisStage(status="running", request={"recipes": []})
        state_no_sim.all_jobs[analyzing.job_id] = analyzing

        runs_done = _assert_jobs_schema(
            await handle_jobs(
                _args(
                    "wait",
                    job_id=analyzing.job_id,
                    timeout_s=0,
                    wait_for="runs",
                ),
                state_no_sim,
            )
        )
        assert runs_done["timed_out"] is False
        assert runs_done["status"] == "analyzing"
        assert runs_done["analysis_status"] == "running"

    async def test_foreign_wait_refreshes_a_second_registry_from_sidecar(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        circuit = _circuit(work_dir)
        owner_job = _experiment(work_dir, circuit, status="running")
        owner_job.owner_pid = _FOREIGN_PID
        experiment_store.save_job(owner_job)
        foreign_state = SessionState.create(state_no_sim.config, available={})
        # Patch liveness BEFORE the first load: an owner presumed dead would
        # reconcile the record to interrupted (terminal), and a terminal job
        # correctly returns from wait immediately.
        monkeypatch.setattr(experiment_store, "owner_alive", lambda *_args, **_kwargs: True)
        stale = experiment_store.load_job(owner_job.job_id, work_dir, own_is_alive=True)
        assert stale is not None
        foreign_state.all_jobs[stale.job_id] = stale
        monkeypatch.setattr(
            "ltspice_mcp.tools.experiments._FOREIGN_WAIT_POLL_S",
            0.01,
        )

        async def finish_owner() -> None:
            await asyncio.sleep(0.02)
            completed = _experiment(
                work_dir,
                circuit,
                status="completed",
                case_status="produced",
            )
            completed.owner_pid = _FOREIGN_PID
            await asyncio.to_thread(experiment_store.save_job, completed)

        writer = asyncio.create_task(finish_owner())
        data = _assert_jobs_schema(
            await handle_jobs(
                _args("wait", job_id=owner_job.job_id, timeout_s=1),
                foreign_state,
            )
        )
        await writer

        assert data["timed_out"] is False
        assert data["status"] == "completed"


@pytest.mark.asyncio
class TestCancellationAuthority:
    async def test_owner_and_token_route_through_coordinator_cancel(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, status="running")
        state_no_sim.all_jobs[job.job_id] = job

        async def cancel(candidate, *, control_token=None):
            assert candidate is job
            candidate.cases[0].status = "cancelled"
            candidate.completeness.recount(candidate.cases)
            candidate.status = "cancelled"
            candidate.runs_done_event.set()
            candidate.done_event.set()
            return [
                {
                    "case_id": candidate.cases[0].case_id,
                    "prior_status": "running",
                    "status": "cancelled",
                }
            ]

        runner = SimpleNamespace(cancel=AsyncMock(side_effect=cancel))
        monkeypatch.setattr(
            state_no_sim.runners,
            "get_experiment_runner_for",
            lambda _job: runner,
        )

        owner = _assert_jobs_schema(
            await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim)
        )
        assert owner["status"] == "cancelled"
        runner.cancel.assert_awaited_once()

        token_job = _experiment(
            work_dir,
            circuit,
            job_id="exp_jobs_token",
            request_id="token-request",
            status="running",
        )
        token_job.owner_pid = -1
        state_no_sim.all_jobs[token_job.job_id] = token_job
        job = token_job
        token = _assert_jobs_schema(
            await handle_jobs(
                _args(
                    "cancel",
                    job_id=token_job.job_id,
                    control_token=token_job.control_token,
                ),
                state_no_sim,
            )
        )
        assert token["status"] == "cancelled"

    async def test_foreign_token_sets_durable_submission_barrier(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        circuit = _circuit(work_dir)
        template = _experiment(work_dir, circuit, count=2, status="queued")
        runner = state_no_sim.runners.get_experiment_runner(
            asyncio.get_running_loop(),
            MockSimulator,
            work_dir,
            max_parallel=1,
        )
        callbacks = {}

        def submit(_netlist: Path, run_filename: str, callback):
            callbacks[Path(run_filename).stem] = callback
            return object()

        monkeypatch.setattr(runner, "submit_netlist", submit)
        # The real kill path is safe here: the token names no live process, so
        # the kill is a no-op and the grace window resolves the case as
        # kill-unconfirmed — mocking _kill_case away would leave the case
        # future unresolved forever.
        receipt = await asyncio.shield(
            runner.submit(
                ExperimentRunRequest(
                    state=state_no_sim,
                    request_id="foreign-token-request",
                    fingerprint="a" * 64,
                    cases=template.cases,
                    sources=template.sources,
                    simulator="MockSimulator",
                    kill_grace_s=0.01,
                )
            )
        )
        await _wait_for(lambda: bool(callbacks))
        receipt.job.owner_pid = _FOREIGN_PID
        await asyncio.to_thread(experiment_store.save_job, receipt.job)
        foreign_state = SessionState.create(state_no_sim.config, available={})
        monkeypatch.setattr(experiment_store, "owner_alive", lambda *_args, **_kwargs: True)

        data = _assert_jobs_schema(
            await asyncio.wait_for(
                handle_jobs(
                    _args(
                        "cancel",
                        job_id=receipt.job.job_id,
                        control_token=receipt.control_token,
                    ),
                    foreign_state,
                ),
                2,
            )
        )

        assert data["status"] == "cancelled"
        assert experiment_store.cancellation_requested(receipt.job.job_id, work_dir)
        assert receipt.job.completeness.submitted == 1
        callback = next(iter(callbacks.values()))
        callback(RunOutcome("", str(work_dir / "cancelled.fail"), 0, "killed"))
        await _wait_for(lambda: not runner.has_active_work())

    async def test_neither_token_nor_ownership_is_rejected(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, status="running")
        job.owner_pid = -1
        state_no_sim.all_jobs[job.job_id] = job

        result = await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim)
        data = _assert_jobs_schema(result)

        assert result.isError
        assert data["error"]["code"] == "cancel_not_authorized"
        assert job.status == "running"

    async def test_foreign_legacy_cancel_has_honest_authority_error(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        job = make_sim_job(
            "legacy_foreign",
            status="running",
            netlist=_circuit(work_dir),
            owner_pid=-1,
        )
        state_no_sim.all_jobs[job.job_id] = job

        result = await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim)
        data = _assert_jobs_schema(result)

        assert result.isError
        assert data["error"]["code"] == "cancel_not_authorized"
        assert "no transferable control token" in data["error"]["message"]


@pytest.mark.asyncio
class TestListAndRunsPagination:
    async def test_list_pages_recent_groups_and_surfaces_malformed_pointer(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
        first = _circuit(work_dir, "first.cir")
        second = _circuit(work_dir, "second.cir")
        await asyncio.to_thread(recent.touch, first)
        await asyncio.to_thread(recent.touch, second)
        pointer_dir = experiment_store.pointer_dir(first)
        await asyncio.to_thread(pointer_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            (pointer_dir / "broken.json").write_text,
            "{not-json",
        )

        page_one = _assert_jobs_schema(await handle_jobs(_args("list", limit=1), state_no_sim))
        page_two = _assert_jobs_schema(
            await handle_jobs(
                _args("list", limit=1, cursor=page_one["next_cursor"]),
                state_no_sim,
            )
        )

        assert page_one["total"] == 2
        assert page_one["returned"] == page_two["returned"] == 1
        observations = [*page_one["observations"], *page_two["observations"]]
        assert any(item["code"] == "experiment_pointer_invalid" for item in observations)

    async def test_submission_appears_in_unfiltered_recent_view(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
        circuit = _circuit(work_dir, "submitted.cir")

        def submit(self, _netlist: Path, run_filename: str, callback):
            raw = self.output_folder / f"{Path(run_filename).stem}.raw"
            log = self.output_folder / f"{Path(run_filename).stem}.log"
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            self.loop.call_soon_threadsafe(
                callback,
                RunOutcome(str(raw), str(log), raw.stat().st_size, None),
            )
            return object()

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
        await handle_run_experiments(
            RunExperimentsInput.model_validate(
                {
                    "request_id": "recent-submission",
                    "circuits": [{"path": str(circuit), "id": "submitted"}],
                    "execution": {"wait_s": 1},
                }
            ),
            state_with_sim,
        )

        data = _assert_jobs_schema(await handle_jobs(_args("list"), state_with_sim))

        assert any(item["path"] == str(circuit.resolve()) for item in data["items"])
        group = next(item for item in data["items"] if item["path"] == str(circuit.resolve()))
        assert group["status_counts"]["completed"] == 1

    async def test_runs_cursor_resumes_after_first_page(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            count=51,
            status="completed",
            case_status="produced",
        )
        state_no_sim.all_jobs[job.job_id] = job

        first = _assert_jobs_schema(
            await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim)
        )
        second = _assert_jobs_schema(
            await handle_jobs(
                _args("runs", job_id=job.job_id, cursor=first["next_cursor"]),
                state_no_sim,
            )
        )

        assert first["returned"] == 50
        assert second["returned"] == 1
        assert second["items"][0]["run_index"] == 50


@pytest.mark.asyncio
class TestLegacyPassthrough:
    async def test_owned_legacy_cancel_delegates_to_legacy_handler(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import ltspice_mcp.tools.simulation as simulation_tools

        job = make_sim_job(
            "legacy_owned",
            status="running",
            netlist=_circuit(work_dir),
        )
        state_no_sim.all_jobs[job.job_id] = job

        async def cancel(args, _state):
            assert args.job_id == job.job_id
            job.status = "cancelled"
            job.done_event.set()
            return SimpleNamespace()

        legacy_cancel = AsyncMock(side_effect=cancel)
        monkeypatch.setattr(simulation_tools, "handle_cancel_job", legacy_cancel)

        data = _assert_jobs_schema(
            await handle_jobs(_args("cancel", job_id=job.job_id), state_no_sim)
        )

        legacy_cancel.assert_awaited_once()
        assert data["status"] == "cancelled"

    async def test_single_and_batch_status_wait_runs_follow_job_dialect(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        raw = work_dir / "legacy.raw"
        log = work_dir / "legacy.log"
        raw.write_bytes(b"Title: mock")
        log.write_text("ok")
        sim = make_sim_job(
            "legacy_sim",
            netlist=circuit,
            simulator="NGspiceSimulator",
            raw_file=raw,
            log_file=log,
        )
        sim.done_event.set()
        batch = make_batch_job(
            "legacy_batch",
            netlist=circuit,
            simulator="NGspiceSimulator",
            total_runs=1,
            completed_runs=1,
            run_results={
                0: {
                    "raw_file": str(raw),
                    "log_file": str(log),
                    "params": {"R1": "1k"},
                }
            },
        )
        batch.done_event.set()
        state_no_sim.all_jobs[sim.job_id] = sim
        state_no_sim.all_jobs[batch.job_id] = batch

        responses = [
            await handle_jobs(_args("status", job_id=sim.job_id), state_no_sim),
            await handle_jobs(
                _args("wait", job_id=batch.job_id, timeout_s=0),
                state_no_sim,
            ),
            await handle_jobs(_args("runs", job_id=sim.job_id), state_no_sim),
            await handle_jobs(_args("runs", job_id=batch.job_id), state_no_sim),
        ]
        payloads = [_assert_jobs_schema(result) for result in responses]

        assert all(data["dialect"] == "ngspice" for data in payloads)
        assert payloads[1]["timed_out"] is False
        assert payloads[2]["items"][0]["raw"] == str(raw)
        assert payloads[3]["items"][0]["assignments"] == {"R1": "1k"}

    async def test_existing_legacy_job_never_reports_job_not_found(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        job = make_sim_job(
            "legacy_failed",
            status="failed",
            netlist=_circuit(work_dir),
            error="simulator failed",
        )
        state_no_sim.all_jobs[job.job_id] = job

        data = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )

        assert data["job_id"] == job.job_id
        assert data["status"] == "failed"
        assert data.get("error", {}).get("code") != "job_not_found"
