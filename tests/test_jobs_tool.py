"""Consolidated jobs control-plane contracts."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jsonschema
import pytest
from pydantic import ValidationError

from ltspice_mcp.lib import analysis_snapshot, experiment_store, now, recent, store
from ltspice_mcp.lib.experiment_runner import ExperimentRunRequest
from ltspice_mcp.lib.experiment_types import (
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._schema import build_input_schema
from ltspice_mcp.tools.experiments import (
    RunExperimentsInput,
    handle_run_experiments,
)
from ltspice_mcp.tools.jobs import (
    _RECENT_JOBS_CAP,
    JOBS_ACTIONS,
    JOBS_OUTPUT_SCHEMA,
    JobsInput,
    handle_jobs,
)
from ltspice_mcp.tools.receipts import (
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    progress_from_completeness,
    project_receipt_runs,
    render_receipt_snapshot,
    snapshot_receipt,
)
from tests.conftest import fake_simulator


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
        store_path=Store(work_dir).job_record(job_id),
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
    experiment_store.register_circuits(job, work_dir)
    experiment_store.save_request_index(
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        canonicalizer_version=job.canonicalizer_version,
        job_id=job.job_id,
        working_dir=work_dir,
    )


def _args(action: str, **values) -> JobsInput:
    return JobsInput.model_validate({"action": action, **values})


def _jobs_branch_ref(schema: dict, action: str) -> str:
    """The ``$ref`` the jobs schema selects for one action.

    Read off the ``if``/``then`` pairs, which are what the server SDK validates
    against; the discriminator carries only ``propertyName``, because a branch
    table would repeat these pairs.
    """
    return next(
        entry["then"]["$ref"]
        for entry in schema["allOf"]
        if entry["if"]["properties"]["action"]["const"] == action
    )


# ---------------------------------------------------------------------------
# Accepted argument spellings
# ---------------------------------------------------------------------------

# Every spelling a client may send, with the values the accepted call carries.
# The same table ran against the previous model — one 'action' string plus nine
# optionals policed after validation by a hand-written allowed-field table —
# and against the discriminated union that replaced it, so the union is proved
# to accept and reject exactly what that table did. Values are asserted by
# projection, not whole-dump equality: an action model only carries its own
# fields, which is the point of the change.
_ACCEPTED_JOBS_ARGUMENTS: tuple[tuple[str, dict, dict], ...] = (
    (
        "status-by-job-id",
        {"action": "status", "job_id": "exp-1"},
        {"action": "status", "job_id": "exp-1", "request_id": None, "budget": None},
    ),
    (
        "status-by-request-id",
        {"action": "status", "request_id": "req-1"},
        {"action": "status", "job_id": None, "request_id": "req-1"},
    ),
    (
        "status-with-explicit-null-alternative",
        {"action": "status", "job_id": "exp-1", "request_id": None},
        {"action": "status", "job_id": "exp-1", "request_id": None},
    ),
    (
        "status-under-a-budget",
        {"action": "status", "job_id": "exp-1", "budget": 900},
        {"action": "status", "job_id": "exp-1", "budget": 900},
    ),
    (
        "status-strips-surrounding-whitespace",
        {"action": "status", "job_id": "  exp-1  "},
        {"action": "status", "job_id": "exp-1"},
    ),
    (
        "wait-defaults",
        {"action": "wait", "job_id": "exp-1"},
        {"action": "wait", "job_id": "exp-1", "timeout_s": 60.0, "wait_for": "all"},
    ),
    (
        "wait-with-dwell-and-mode",
        {"action": "wait", "request_id": "req-1", "timeout_s": 0, "wait_for": "runs"},
        {"action": "wait", "request_id": "req-1", "timeout_s": 0.0, "wait_for": "runs"},
    ),
    (
        "wait-at-the-cap",
        {"action": "wait", "job_id": "exp-1", "timeout_s": 300},
        {"action": "wait", "job_id": "exp-1", "timeout_s": 300.0},
    ),
    (
        "wait-under-a-budget",
        {"action": "wait", "job_id": "exp-1", "timeout_s": 0, "budget": 800},
        {"action": "wait", "job_id": "exp-1", "timeout_s": 0.0, "budget": 800},
    ),
    (
        "cancel-by-owner",
        {"action": "cancel", "job_id": "exp-1"},
        {"action": "cancel", "job_id": "exp-1", "control_token": None},
    ),
    (
        "cancel-with-control-token",
        {"action": "cancel", "request_id": "req-1", "control_token": "tok"},
        {"action": "cancel", "request_id": "req-1", "control_token": "tok"},
    ),
    (
        "list-everything-recent",
        {"action": "list"},
        {"action": "list", "circuit": None, "limit": 50, "cursor": None, "budget": None},
    ),
    (
        "list-one-circuit",
        {"action": "list", "circuit": "dut.cir"},
        {"action": "list", "circuit": "dut.cir", "limit": 50},
    ),
    (
        "list-paged",
        {"action": "list", "limit": 1, "cursor": "o:1"},
        {"action": "list", "limit": 1, "cursor": "o:1"},
    ),
    (
        "list-under-a-budget",
        {"action": "list", "budget": 700},
        {"action": "list", "budget": 700},
    ),
    (
        "runs-first-page",
        {"action": "runs", "job_id": "exp-1"},
        {"action": "runs", "job_id": "exp-1", "cursor": None},
    ),
    (
        "runs-continued",
        {"action": "runs", "request_id": "req-1", "cursor": "o:50"},
        {"action": "runs", "request_id": "req-1", "cursor": "o:50"},
    ),
)

# The spellings that are refused, and why they were refused before the union.
_REJECTED_JOBS_ARGUMENTS: tuple[tuple[str, dict], ...] = (
    ("unknown-action", {"action": "frobnicate", "job_id": "exp-1"}),
    ("no-action", {"job_id": "exp-1"}),
    ("status-without-a-selector", {"action": "status"}),
    ("wait-without-a-selector", {"action": "wait", "timeout_s": 0}),
    ("cancel-without-a-selector", {"action": "cancel", "control_token": "tok"}),
    ("runs-without-a-selector", {"action": "runs"}),
    ("status-with-both-selectors", {"action": "status", "job_id": "exp-1", "request_id": "r"}),
    ("list-with-a-job-id", {"action": "list", "job_id": "exp-1"}),
    ("list-with-a-request-id", {"action": "list", "request_id": "req-1"}),
    ("list-with-a-null-job-id", {"action": "list", "job_id": None}),
    ("status-with-a-dwell", {"action": "status", "job_id": "exp-1", "timeout_s": 5}),
    ("status-with-the-default-dwell", {"action": "status", "job_id": "exp-1", "timeout_s": 60}),
    ("status-with-a-cursor", {"action": "status", "job_id": "exp-1", "cursor": "o:0"}),
    ("status-with-a-control-token", {"action": "status", "job_id": "exp-1", "control_token": "t"}),
    ("wait-with-a-control-token", {"action": "wait", "job_id": "exp-1", "control_token": "t"}),
    ("cancel-with-a-dwell", {"action": "cancel", "job_id": "exp-1", "timeout_s": 5}),
    ("cancel-with-a-wait-mode", {"action": "cancel", "job_id": "exp-1", "wait_for": "runs"}),
    ("runs-with-a-limit", {"action": "runs", "job_id": "exp-1", "limit": 5}),
    ("runs-with-a-dwell", {"action": "runs", "job_id": "exp-1", "timeout_s": 1}),
    ("list-with-a-control-token", {"action": "list", "control_token": "tok"}),
    ("list-with-a-wait-mode", {"action": "list", "wait_for": "runs"}),
    ("unknown-field", {"action": "status", "job_id": "exp-1", "verbose": True}),
    ("empty-job-id", {"action": "status", "job_id": ""}),
    ("empty-request-id", {"action": "status", "request_id": ""}),
    ("dwell-past-the-cap", {"action": "wait", "job_id": "exp-1", "timeout_s": 301}),
    ("negative-dwell", {"action": "wait", "job_id": "exp-1", "timeout_s": -1}),
    ("unknown-wait-mode", {"action": "wait", "job_id": "exp-1", "wait_for": "cases"}),
    ("limit-below-one", {"action": "list", "limit": 0}),
    ("limit-past-the-page-cap", {"action": "list", "limit": 51}),
    ("budget-below-the-floor", {"action": "status", "job_id": "exp-1", "budget": 1}),
)


class TestAcceptedArgumentSpellings:
    """The advertised argument grammar, spelling by spelling."""

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [(payload, expected) for _name, payload, expected in _ACCEPTED_JOBS_ARGUMENTS],
        ids=[name for name, _payload, _expected in _ACCEPTED_JOBS_ARGUMENTS],
    )
    def test_accepted_call_keeps_its_values(self, payload: dict, expected: dict):
        dumped = JobsInput.model_validate(payload).model_dump()
        assert {key: dumped[key] for key in expected} == expected

    @pytest.mark.parametrize(
        "payload",
        [payload for _name, payload in _REJECTED_JOBS_ARGUMENTS],
        ids=[name for name, _payload in _REJECTED_JOBS_ARGUMENTS],
    )
    def test_rejected_call_stays_rejected(self, payload: dict):
        with pytest.raises(ValidationError):
            JobsInput.model_validate(payload)

    def test_a_missing_selector_names_both_ways_to_address_a_job(self):
        """The message a client acts on: the e2e surface asserts this wording."""
        with pytest.raises(ValidationError) as excinfo:
            JobsInput.model_validate({"action": "status"})
        assert "requires exactly one of job_id or request_id" in str(excinfo.value)

    def test_an_unknown_action_names_every_action(self):
        with pytest.raises(ValidationError) as excinfo:
            JobsInput.model_validate({"action": "frobnicate"})
        message = str(excinfo.value)
        for action in ("status", "wait", "cancel", "list", "runs"):
            assert action in message


class TestAdvertisedActionBranches:
    """What the published schema says each action takes.

    The argument shape used to be one flat list of nine optionals with the
    per-action rules enforced only after validation, so a client reading the
    schema could not tell which action took which field — and the SDK, which
    validates arguments against this schema before dispatch, could not either.
    """

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ("status", {"action", "job_id", "request_id", "budget"}),
            ("wait", {"action", "job_id", "request_id", "timeout_s", "wait_for", "budget"}),
            ("cancel", {"action", "job_id", "request_id", "control_token", "budget"}),
            ("list", {"action", "circuit", "limit", "cursor", "budget"}),
            ("runs", {"action", "job_id", "request_id", "cursor", "budget"}),
        ],
    )
    def test_each_action_advertises_its_own_fields(self, action: str, expected: set[str]):
        schema = build_input_schema(JobsInput)
        branch = schema["$defs"][_jobs_branch_ref(schema, action).split("/")[-1]]
        assert set(branch["properties"]) == expected
        assert branch["additionalProperties"] is False
        assert branch["properties"]["action"]["const"] == action

    def test_shared_arguments_stay_at_the_top_level(self):
        """A client that reads `properties` and stops there still sees them."""
        schema = build_input_schema(JobsInput)
        assert set(schema["properties"]) == {"action", "budget"}
        assert schema["properties"]["action"]["enum"] == list(JOBS_ACTIONS)
        assert schema["required"] == ["action"]

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [(payload, expected) for _name, payload, expected in _ACCEPTED_JOBS_ARGUMENTS],
        ids=[name for name, _payload, _expected in _ACCEPTED_JOBS_ARGUMENTS],
    )
    def test_the_schema_accepts_everything_the_validator_accepts(
        self,
        payload: dict,
        expected: dict,
    ):
        """The advertised schema may not be stricter than the tool: the SDK
        validates against it first, so a call it rejects never reaches the
        handler at all."""
        del expected
        jsonschema.Draft202012Validator(build_input_schema(JobsInput)).validate(payload)

    def test_a_field_the_action_does_not_take_is_named_by_the_schema(self):
        with pytest.raises(jsonschema.ValidationError) as excinfo:
            jsonschema.validate(
                instance={"action": "list", "job_id": "exp-1"},
                schema=build_input_schema(JobsInput),
            )
        assert "job_id" in excinfo.value.message

    def test_an_unknown_action_is_told_the_five_by_the_schema(self):
        """Branches applied by if/then rather than oneOf, so the error the SDK
        reports is this one instead of 'not valid under any of the given
        schemas' — which names neither the actions nor the offending field."""
        with pytest.raises(jsonschema.ValidationError) as excinfo:
            jsonschema.validate(
                instance={"action": "frobnicate"},
                schema=build_input_schema(JobsInput),
            )
        for action in JOBS_ACTIONS:
            assert action in excinfo.value.message


def _assert_jobs_schema(result) -> dict:
    data = result.structured_content
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


def test_receipt_schemas_require_progress():
    assert "progress" in RUN_EXPERIMENTS_OUTPUT_SCHEMA["required"]
    assert "progress" in RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]
    for action in ("status", "wait"):
        branch = next(
            item
            for item in JOBS_OUTPUT_SCHEMA["oneOf"]
            if item["properties"]["action"]["const"] == action
        )
        assert "progress" in branch["required"]
        assert "progress" in branch["properties"]


def test_top_level_properties_describe_what_every_action_shares():
    """A client that reads `properties` and stops there must not be told this
    tool returns one key. Registration injects `warnings` into any schema that
    declares no properties, so a bare oneOf advertises exactly that — while the
    shape all five actions share is sitting one level down."""
    shared = JOBS_OUTPUT_SCHEMA["properties"]

    assert set(shared) > {"warnings"}
    for branch in JOBS_OUTPUT_SCHEMA["oneOf"]:
        # Hoisting may not constrain anything: every branch declares these keys
        # itself, and closes with additionalProperties false.
        for name, schema in shared.items():
            if name == "action":
                continue
            assert branch["properties"][name] == schema


async def _wait_for(condition, *, timeout_s: float = 15.0) -> None:
    # The bound exists to catch a hang, not to assert latency: the cancel
    # paths under test run a real process-table scan, whose duration scales
    # with system load (parallel test workers, live simulators on the box).
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
            # The counts the hint must carry, read off the structured payload
            # it summarises — not the sentence it wraps them in.
            progress = data["progress"]
            assert f"{progress['terminal']}/{progress['expanded']}" in data["hint"]
            assert f"{progress['remaining']} remaining" in data["hint"]

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

    async def test_cancelled_status_reports_partial_even_when_runs_reconcile(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        """A cancel landing after the last run still leaves the counters fully
        reconciled, so only the status can tell a cancelled experiment from a
        finished one. Reading the counters here reports a cancelled experiment
        as a success."""
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            status="cancelled",
            case_status="produced",
        )
        assert job.completeness.produced == job.completeness.expanded
        state_no_sim.all_jobs[job.job_id] = job

        data = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )

        assert data["outcome"] == "partial"

    async def test_uncounted_run_reports_partial_outcome(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        """A run that vanishes without landing in any shortfall counter is the
        silent-data-loss case: the shortfall is read off ``produced`` against
        ``expanded``, so an unaccounted run reads as one."""
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            status="completed_with_failures",
            case_status="produced",
        )
        # Two runs promised, one produced, and nothing recorded the other's fate.
        job.completeness.expanded = 2
        assert job.completeness.failed == 0
        assert job.completeness.cancelled == 0
        assert job.completeness.skipped == 0
        state_no_sim.all_jobs[job.job_id] = job

        data = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )

        assert data["outcome"] == "partial"


@pytest.mark.asyncio
class TestReceiptSnapshotCoherence:
    async def test_mutation_after_snapshot_cannot_mix_receipt_generations(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, count=2, status="running")
        job.analysis = AnalysisStage(status="pending", request={"recipes": []})

        before = snapshot_receipt(job, state_no_sim)

        produced, failed = job.cases
        produced.status = "produced"
        produced.raw_file = work_dir / "coherent.raw"
        produced.log_file = work_dir / "coherent.log"
        failed.status = "failed"
        failed.log_file = work_dir / "failed.log"
        job.completeness.recount(job.cases)
        job.failures.append(
            {
                "case_id": failed.case_id,
                "code": "execution_failed",
                "message": "simulator failed",
            }
        )
        job.observations.append(
            {
                "code": "job_finished",
                "kind": "execution",
                "detail": "The terminal generation was recorded.",
            }
        )
        job.artifacts.append(
            {
                "path": str(work_dir / "summary.json"),
                "content_type": "application/json",
                "sha256": "a" * 64,
                "bytes": 1,
            }
        )
        job.analysis = AnalysisStage(
            status="completed",
            request={"recipes": []},
            result=analysis_snapshot.envelope(
                {
                    "top": {
                        "rows": [
                            {"case_id": produced.case_id, "status": produced.status},
                            {"case_id": failed.case_id, "status": failed.status},
                        ],
                        "next": None,
                    },
                    "results": {},
                    "natural_cursor_base": None,
                    "natural_has_next": False,
                    "natural_deferred": False,
                }
            ),
            observations=[
                {
                    "code": "analysis_finished",
                    "kind": "analysis",
                    "detail": "The attached result describes the terminal rows.",
                }
            ],
        )
        job.status = "completed_with_failures"

        old_receipt = render_receipt_snapshot(before)
        after = snapshot_receipt(job, state_no_sim)
        job.analysis.status = "failed"
        job.analysis.result = None
        job.analysis.error = "later analysis failure"
        job.analysis.observations.append(
            {
                "code": "later_analysis_failure",
                "kind": "analysis",
                "detail": "This belongs to the next job generation.",
            }
        )
        new_receipt = render_receipt_snapshot(after)

        old_rows = old_receipt["runs"]["items"]
        assert all(row["status"] == "queued" for row in old_rows)
        assert before.completeness.produced == before.completeness.failed == 0
        assert old_receipt["outcome"] == "in_progress"
        assert old_receipt["failures"] == []
        assert old_receipt["observations"] == []
        assert old_receipt["artifacts"] == []
        assert old_receipt["analysis"] == {
            "status": "pending",
            "result": None,
            "error": None,
            "observations": [],
        }
        assert "jobs(wait)" in old_receipt["hint"]

        new_rows = new_receipt["runs"]["items"]
        assert sum(row["status"] == "produced" for row in new_rows) == 1
        assert sum(row["status"] == "failed" for row in new_rows) == 1
        assert after.completeness.produced == 1
        assert after.completeness.failed == 1
        assert new_receipt["outcome"] == "partial"
        assert new_receipt["failures"][0]["case_id"] == failed.case_id
        assert new_receipt["observations"][0]["code"] == "job_finished"
        assert new_receipt["artifacts"][0]["path"].endswith("summary.json")
        assert new_receipt["analysis"]["status"] == "completed"
        assert new_receipt["analysis"]["result"]["rows"] == [
            {"case_id": row["case_id"], "status": row["status"]} for row in new_rows
        ]
        assert any(
            item["code"] == "analysis_finished" for item in new_receipt["analysis"]["observations"]
        )
        assert all(
            item["code"] != "later_analysis_failure"
            for item in new_receipt["analysis"]["observations"]
        )
        assert new_receipt["analysis"]["error"] is None
        assert "jobs(wait)" not in new_receipt["hint"]

    async def test_restart_hint_carries_both_recovery_routes(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        """A restart sets failures AND a failed analysis; both routes must show.

        The exclusive ladder let the failures branch win, so the receipt's first
        line of guidance never mentioned that the produced runs are analyzable
        by job_id — pointing the caller at a re-run of results already on disk.
        """
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, count=2, status="running")
        job.cases[0].status = "produced"
        job.analysis = AnalysisStage(status="running", request={"recipes": []})
        job.owner_pid = 999_999_999
        experiment_store.save_job(job)

        restarted = experiment_store.load_job(job.job_id, work_dir)
        assert restarted is not None
        assert restarted.failures and restarted.analysis.status == "failed"

        hint = render_receipt_snapshot(snapshot_receipt(restarted, state_no_sim))["hint"]

        assert "analyze_results" in hint
        assert restarted.job_id in hint
        assert "Inspect failures" in hint

    async def test_scheduled_mutation_cannot_run_inside_snapshot(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, status="running")

        async def finish_job() -> None:
            job.cases[0].status = "produced"
            job.completeness.recount(job.cases)
            job.status = "completed"

        mutator = asyncio.create_task(finish_job())
        snapshot = snapshot_receipt(job, state_no_sim)
        await mutator

        rendered = render_receipt_snapshot(snapshot)
        assert job.status == "completed"
        assert snapshot.status == "running"
        assert snapshot.completeness.produced == 0
        assert rendered["runs"]["items"][0]["status"] == "queued"
        assert rendered["outcome"] == "in_progress"
        assert "jobs(wait)" in rendered["hint"]

    async def test_jobs_runs_and_receipt_share_one_canonical_inventory(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            count=2,
            status="completed",
            case_status="produced",
        )
        state_no_sim.all_jobs[job.job_id] = job

        receipt = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )
        runs = _assert_jobs_schema(
            await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim)
        )

        receipt_rows = receipt["runs"]["items"]
        assert [
            {key: row[key] for key in receipt_row}
            for row, receipt_row in zip(runs["items"], receipt_rows, strict=True)
        ] == receipt_rows
        assert runs["status"] == receipt["status"]
        assert runs["outcome"] == receipt["outcome"]
        assert runs["total"] == receipt["runs"]["total"]
        assert runs["total"] == receipt["completeness"]["expanded"]

    async def test_projection_is_pure_over_copied_rows(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(
            work_dir,
            circuit,
            status="completed",
            case_status="produced",
        )
        snapshot = snapshot_receipt(job, state_no_sim)
        original = copy.deepcopy(snapshot.runs_by_key)

        projected = project_receipt_runs(
            snapshot,
            ["case_id", "assignments"],
            lean_default=True,
        )
        lean = project_receipt_runs(snapshot, None, lean_default=True)

        assert set(projected["items"][0]) == {"case_id", "assignments"}
        assert "raw" not in lean["items"][0] and "log" not in lean["items"][0]
        assert snapshot.runs_by_key == original
        assert "raw" in next(iter(snapshot.runs_by_key.values()))


@pytest.mark.asyncio
class TestDurableProgress:
    async def test_midflight_poll_is_monotonic_and_keeps_the_wait_route(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, count=3, status="running")
        state_no_sim.all_jobs[job.job_id] = job

        first = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )
        job.cases[0].status = "produced"
        job.completeness.recount(job.cases)
        middle = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )
        for case in job.cases[1:]:
            case.status = "produced"
        job.completeness.recount(job.cases)
        job.status = "completed"
        final = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=job.job_id), state_no_sim)
        )

        terminals = [item["progress"]["terminal"] for item in (first, middle, final)]
        assert terminals == sorted(terminals) == [0, 1, 3]
        assert middle["progress"]["terminal"] < middle["progress"]["expanded"]
        assert final["progress"]["terminal"] == final["progress"]["expanded"]
        assert "cases_failed" not in middle["progress"]
        assert "jobs(action='wait'" in middle["hint"]
        counts = middle["progress"]
        assert f"{counts['terminal']}/{counts['expanded']}" in middle["hint"]
        assert f"{counts['remaining']} remaining" in middle["hint"]

    async def test_foreign_status_uses_the_persisted_completeness_snapshot(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        circuit = _circuit(work_dir)
        persisted = _experiment(work_dir, circuit, count=3, status="running")
        persisted.owner_pid = _FOREIGN_PID
        persisted.cases[0].status = "produced"
        persisted.completeness.recount(persisted.cases)
        experiment_store.save_job(persisted)

        stale = _experiment(work_dir, circuit, count=3, status="running")
        stale.owner_pid = _FOREIGN_PID
        state_no_sim.all_jobs[stale.job_id] = stale
        monkeypatch.setattr(
            experiment_store,
            "owner_liveness",
            lambda *_args, **_kwargs: store.OwnerLiveness.ALIVE,
        )

        data = _assert_jobs_schema(
            await handle_jobs(_args("status", job_id=stale.job_id), state_no_sim)
        )

        assert data["completeness"] == asdict(persisted.completeness)
        assert data["progress"] == progress_from_completeness(persisted.completeness)
        assert data["progress"]["terminal"] == 1
        assert data["progress"]["remaining"] == 2
        assert all(isinstance(value, int) for value in data["progress"].values())


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
        monkeypatch.setattr(
            experiment_store,
            "owner_liveness",
            lambda *_args, **_kwargs: store.OwnerLiveness.ALIVE,
        )
        stale = experiment_store.load_job(owner_job.job_id, work_dir, own_is_alive=True)
        assert stale is not None
        foreign_state.all_jobs[stale.job_id] = stale
        monkeypatch.setattr(
            "ltspice_mcp.tools.jobs._FOREIGN_WAIT_POLL_S",
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
        # Wait for the case to reach 'running': its submitted/running
        # checkpoints each re-persist the record under THIS pid, so a foreign
        # pid written before them is overwritten and the job reads as locally
        # owned with no live coordinator.
        await _wait_for(lambda: bool(callbacks) and receipt.job.cases[0].status == "running")
        receipt.job.owner_pid = _FOREIGN_PID
        await asyncio.to_thread(experiment_store.save_job, receipt.job)
        foreign_state = SessionState.create(state_no_sim.config, available={})
        monkeypatch.setattr(
            experiment_store,
            "owner_liveness",
            lambda *_args, **_kwargs: store.OwnerLiveness.ALIVE,
        )

        cancel_args = _args(
            "cancel",
            job_id=receipt.job.job_id,
            control_token=receipt.control_token,
        )
        data = _assert_jobs_schema(
            await asyncio.wait_for(handle_jobs(cancel_args, foreign_state), 30)
        )

        # The durable barrier is what the contract acknowledges: no further case
        # enters submission. The receipt's own status is whatever the owner had
        # reached inside a bounded best-effort poll, so pinning a terminal one
        # here would pin a latency instead — the wait scales with a real
        # process-table scan, which is why this asserts state and then waits on
        # the coordinator's own completion event rather than on a clock.
        assert data["job_id"] == receipt.job.job_id
        assert experiment_store.cancellation_requested(receipt.job.job_id, work_dir)

        await asyncio.wait_for(receipt.job.done_event.wait(), 30)
        assert receipt.job.status == "cancelled"
        assert receipt.job.completeness.submitted == 1

        # And a foreign caller asking again, now that the owner has finished,
        # gets the terminal status on the receipt.
        settled = _assert_jobs_schema(
            await asyncio.wait_for(handle_jobs(cancel_args, foreign_state), 30)
        )
        assert settled["status"] == "cancelled"
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

        assert result.is_error
        assert data["error"]["code"] == "cancel_not_authorized"
        assert job.status == "running"


@pytest.mark.asyncio
class TestListAndRunsPagination:
    async def test_list_pages_recent_groups_and_surfaces_malformed_index_entry(
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
        index_dir = Store(work_dir).circuit_index_dir(first)
        await asyncio.to_thread(index_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            (index_dir / "broken.json").write_text,
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
        assert any(item["code"] == "experiment_index_invalid" for item in observations)

    async def test_submission_appears_in_unfiltered_recent_view(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
        circuit = _circuit(work_dir, "submitted.cir")

        fake_simulator(monkeypatch)
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

    async def test_list_names_recent_job_ids_and_the_count_it_capped(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A completed job has to be addressable from discovery, not just counted.

        Without an id on the group, "find the Monte Carlo I ran earlier" has no
        answer inside the product — the store holds the identity and the read
        path declines to show it.
        """
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
        circuit = _circuit(work_dir, "addressable.cir")
        await asyncio.to_thread(recent.touch, circuit)
        overflow = _RECENT_JOBS_CAP + 2
        for index in range(overflow):
            job = _experiment(
                work_dir,
                circuit,
                job_id=f"exp_listed_{index:04d}",
                request_id=f"listed-request-{index}",
                status="completed",
                case_status="produced",
            )
            await asyncio.to_thread(_persist_experiment, job, work_dir)

        data = _assert_jobs_schema(await handle_jobs(_args("list"), state_no_sim))
        group = next(item for item in data["items"] if item["path"] == str(circuit.resolve()))

        assert group["recent_jobs_total"] == overflow
        assert len(group["recent_jobs"]) == _RECENT_JOBS_CAP
        assert {record["job_id"] for record in group["recent_jobs"]} <= {
            f"exp_listed_{index:04d}" for index in range(overflow)
        }
        newest = group["recent_jobs"][0]
        assert newest["status"] == "completed"
        assert newest["request_id"].startswith("listed-request-")
        assert newest["finished_at"] is not None
        assert "recent_jobs" in data["hint"]

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

    async def test_runs_cursor_applies_to_a_fresh_invocation_snapshot(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
    ):
        circuit = _circuit(work_dir)
        job = _experiment(work_dir, circuit, count=51, status="running")
        state_no_sim.all_jobs[job.job_id] = job

        first = _assert_jobs_schema(
            await handle_jobs(_args("runs", job_id=job.job_id), state_no_sim)
        )
        job.cases[50].status = "produced"
        job.completeness.recount(job.cases)
        second = _assert_jobs_schema(
            await handle_jobs(
                _args("runs", job_id=job.job_id, cursor=first["next_cursor"]),
                state_no_sim,
            )
        )

        assert first["items"][-1]["status"] == "queued"
        assert second["items"][0]["run_index"] == 50
        assert second["items"][0]["status"] == "produced"
