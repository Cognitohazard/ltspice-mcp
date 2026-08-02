"""run_experiments receipt, lint, idempotency, and accounting contract."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock

import jsonschema
import pytest
from pydantic import ValidationError

from ltspice_mcp.lib import experiment_store, recent, response_budget, result_store, wsl
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
from ltspice_mcp.lib.runner_base import RunOutcome, collect_run_outcome
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import experiments as experiments_mod
from ltspice_mcp.tools._base import _build_input_schema
from ltspice_mcp.tools.analyze import AnalyzeResultsInput
from ltspice_mcp.tools.experiments import (
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    AnalysisPerRun,
    JobsInput,
    RunExperimentsInput,
    handle_jobs,
    handle_run_experiments,
)
from tests.conftest import (
    fake_simulator,
    make_sim_job,
    recorded_fixture_simulator,
    resolve_local_ref,
)


def test_attached_per_run_limit_shares_the_analyze_page_cap():
    """One cap, both surfaces.

    The attached block is handed straight to analyze_results, so a per_run
    limit run_experiments advertises but that engine rejects would be a lever
    that cannot work. Both bounds must come from the same constant, and the
    over-cap request must be refused at submission, not at the analysis stage.
    """
    advertised = AnalysisPerRun.model_json_schema()["properties"]["limit"]["maximum"]
    engine = analyze_mod.PerRunInclude.model_json_schema()["properties"]["limit"]["maximum"]

    assert advertised == engine == analyze_mod.MAX_PAGE_SIZE

    with pytest.raises(ValidationError):
        RunExperimentsInput.model_validate(
            {
                "request_id": "over-cap",
                "circuits": [{"path": "dut.cir"}],
                "analyze": {
                    "recipes": [{"key": "vout", "metric": "summary"}],
                    "include": {"per_run": {"limit": analyze_mod.MAX_PAGE_SIZE + 1}},
                },
            }
        )


def test_wait_caps_keep_the_submission_and_control_plane_contracts():
    run_schema = _build_input_schema(RunExperimentsInput)
    execution_schema = resolve_local_ref(
        run_schema,
        run_schema["properties"]["execution"],
    )
    jobs_schema = _build_input_schema(JobsInput)

    assert execution_schema["properties"]["wait_s"]["maximum"] == 120
    assert jobs_schema["properties"]["timeout_s"]["maximum"] == 300

    with pytest.raises(ValidationError) as excinfo:
        experiments_mod.ExperimentExecution.model_validate({"wait_s": 121})
    message = str(excinfo.value)
    assert 'jobs(action="wait"' in message
    assert "timeout_s<=300" in message


def test_variation_schema_keeps_discriminated_union_through_defs():
    """Schemas keep $defs (followups item 30): the assign/random discriminated
    union must stay fully resolvable through local refs, so a client sees the
    same composition contract inlining used to spell out."""
    schema = _build_input_schema(RunExperimentsInput)
    variations = resolve_local_ref(schema, schema["properties"]["variations"]["items"])

    assert variations["discriminator"]["propertyName"] == "kind"
    assert len(variations["oneOf"]) == 2
    branches = {}
    for ref in variations["oneOf"]:
        branch = resolve_local_ref(schema, ref)
        branches[branch["properties"]["kind"]["const"]] = branch
    assert branches["assign"]["additionalProperties"] is False
    assert branches["assign"]["properties"]["combine"]["default"] == "grid"
    random_rules = resolve_local_ref(schema, branches["random"]["properties"]["rules"]["items"])
    assert random_rules["discriminator"]["propertyName"] == "rule"
    assert all(
        resolve_local_ref(schema, ref)["additionalProperties"] is False
        for ref in random_rules["oneOf"]
    )


def _deck(path: Path, body: str | None = None) -> Path:
    path.write_text(body or "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    return path


def _schematic(path: Path, resistance: str) -> Path:
    """A schematic carrying one editable value, in .asc's own syntax."""
    path.write_text(
        f"Version 4\nSYMBOL res 0 0 R0\nSYMATTR InstName R1\nSYMATTR Value {resistance}\n"
    )
    return path


def _asc_exporter(state: SessionState) -> None:
    """Stand in for the LTspice binary that turns an .asc into a netlist.

    The real exporter is a Windows process; what the replay path depends on is
    only that a schematic reaches the simulator as a netlist written from it,
    leaving the .asc itself read by nothing downstream.
    """

    class _Exporter:
        @staticmethod
        def create_netlist(path: str, timeout: float | None = None) -> str:
            schematic = Path(path)
            value = next(
                line.split()[-1]
                for line in schematic.read_text().splitlines()
                if line.startswith("SYMATTR Value")
            )
            netlist = schematic.with_suffix(".net")
            netlist.write_text(f"V1 in 0 1\nR1 in 0 {value}\n.op\n.end\n")
            return str(netlist)

    state.available_simulators["ltspice"] = _Exporter


def _args(
    path: Path,
    request_id: str,
    *,
    wait_s: float = 1.0,
    lint: str = "block",
    **overrides,
) -> RunExperimentsInput:
    payload = {
        "request_id": request_id,
        "circuits": [{"path": str(path), "id": "dut"}],
        "execution": {"wait_s": wait_s},
        "lint": lint,
    }
    payload.update(overrides)
    return RunExperimentsInput.model_validate(payload)


def _assert_schema(result) -> dict:
    data = result.structuredContent
    assert data is not None
    jsonschema.Draft202012Validator(RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(data)
    return data


async def _wait_for(condition, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        if loop.time() >= deadline:
            pytest.fail("condition was not met before the test deadline")
        await asyncio.sleep(0.005)


# One assign variation plus one real recipe grouped by the assigned target, so
# a group_by that never matched would collapse to a single group and be seen.
_VARIED_ANALYSIS: dict[str, Any] = {
    "variations": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
    "analyze": {
        "recipes": [
            {
                "key": "vout",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "reduce": ["mean"],
            }
        ],
        "group_by": ["R1"],
    },
}


async def _jobs_wait(
    state: SessionState,
    job_id: str,
    wait_for: Literal["all", "runs"],
    timeout_s: float,
) -> dict[str, Any]:
    result = await handle_jobs(
        JobsInput.model_validate(
            {
                "action": "wait",
                "job_id": job_id,
                "wait_for": wait_for,
                "timeout_s": timeout_s,
            }
        ),
        state,
    )
    assert result.structuredContent is not None
    return result.structuredContent


@pytest.mark.asyncio
class TestReceiptThenDwell:
    async def test_quick_completion_returns_inline(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "quick.cir")

        result = await handle_run_experiments(
            _args(deck, "quick-complete"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert data["outcome"] == "complete"
        assert data["status"] == "completed"
        # No cancel authority on a terminal receipt: there is nothing left to
        # stop, and jobs(cancel) refuses a finished job anyway.
        assert "control_token" not in data
        assert data["completeness"]["produced"] == 1
        assert data["progress"]["terminal"] == data["progress"]["expanded"] == 1
        assert data["progress"]["remaining"] == 0
        # progress is the DERIVED view; the raw counters live in completeness and
        # are not restated here (they arrived twice in one receipt before).
        assert set(data["progress"]) == {"expanded", "terminal", "remaining"}
        assert len(submissions) == 1

    async def test_zero_dwell_returns_receipt_then_job_finishes(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        callbacks = {}

        def submit(self, _netlist: Path, run_filename: str, callback):
            callbacks[run_filename] = callback
            return object()

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
        deck = _deck(work_dir / "slow.cir")

        result = await handle_run_experiments(
            _args(deck, "slow-receipt", wait_s=0),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert data["outcome"] == "in_progress"
        assert data["job_id"] in state_with_sim.experiment_jobs
        assert data["progress"]["expanded"] == 1
        assert data["progress"]["terminal"] == 0
        assert data["progress"]["remaining"] == 1
        assert "jobs(wait)" in data["hint"]
        assert "Progress: 0/1 terminal; 1 remaining." in data["hint"]

        await _wait_for(lambda: bool(callbacks))
        for run_filename, callback in callbacks.items():
            raw = work_dir / f"{Path(run_filename).stem}.raw"
            log = work_dir / f"{Path(run_filename).stem}.log"
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            callback(RunOutcome(str(raw), str(log), raw.stat().st_size, None))
        job = state_with_sim.experiment_jobs[data["job_id"]]
        await asyncio.wait_for(job.done_event.wait(), 1)

    async def test_failure_after_submit_reports_committed_with_handles(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Submission is irreversible, so a later failure cannot say nothing started.

        The fleet is running by then and the job_id plus control_token are the
        only handles that reach it; reporting not_started with a null job_id
        leaves the caller no way to poll or cancel real simulator work.
        """
        callbacks = {}

        def submit(self, _netlist: Path, run_filename: str, callback):
            callbacks[run_filename] = callback
            return object()

        async def failing_wait(self, job, timeout_s, *, wait_for="all"):
            raise OSError("dwell exploded")

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
        monkeypatch.setattr(ExperimentRunner, "wait", failing_wait)
        deck = _deck(work_dir / "post-submit.cir")

        result = await handle_run_experiments(
            _args(deck, "post-submit-failure", wait_s=1.0),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert data["error"]["commit_state"] == "committed"
        assert data["error"]["message"] == "dwell exploded"
        assert data["job_id"] in state_with_sim.experiment_jobs
        assert data["control_token"]
        assert data["outcome"] == "in_progress"
        assert data["job_id"] in data["hint"]

        # Let the still-live job finish so teardown is not racing it.
        await _wait_for(lambda: bool(callbacks))
        for run_filename, callback in callbacks.items():
            raw = work_dir / f"{Path(run_filename).stem}.raw"
            log = work_dir / f"{Path(run_filename).stem}.log"
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            callback(RunOutcome(str(raw), str(log), raw.stat().st_size, None))
        job = state_with_sim.experiment_jobs[data["job_id"]]
        await asyncio.wait_for(job.done_event.wait(), 1)

    async def test_receipt_builder_failure_still_returns_handles(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The last-resort path: even the payload builder failing keeps the handles."""
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)

        def exploding_payload(*_args, **_kwargs):
            raise ValueError("payload exploded")

        monkeypatch.setattr(experiments_mod, "render_receipt_snapshot", exploding_payload)
        deck = _deck(work_dir / "payload-fail.cir")

        result = await handle_run_experiments(
            _args(deck, "payload-failure"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert data["error"]["commit_state"] == "committed"
        assert data["error"]["message"] == "payload exploded"
        assert data["job_id"] in state_with_sim.experiment_jobs
        assert data["control_token"]

    async def test_budget_renderer_failure_after_submit_keeps_minimal_handles(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)

        async def exploding_renderer(*_args, **_kwargs):
            raise RuntimeError("budget renderer exploded")

        monkeypatch.setattr(experiments_mod, "_render_run_receipt", exploding_renderer)
        deck = _deck(work_dir / "budget-render-fail.cir")

        result = await handle_run_experiments(
            _args(deck, "budget-render-fail", budget=500),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert result.isError
        assert data["error"]["commit_state"] == "committed"
        assert "budget renderer exploded" in data["error"]["message"]
        assert data["job_id"] in state_with_sim.experiment_jobs
        assert data["control_token"]


@pytest.mark.asyncio
class TestIdempotency:
    async def test_matching_replay_returns_token_and_observation(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        # A job still in flight, so the receipt carries the cancel handle whose
        # stability across a replay is what this test is about.
        fake_simulator(monkeypatch, submissions, delay_s=None)
        deck = _deck(work_dir / "replay.cir")
        args = _args(deck, "same-payload", wait_s=0)

        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        replay = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert replay["job_id"] == first["job_id"]
        assert replay["control_token"] == first["control_token"]
        assert any(item["code"] == "idempotent_replay" for item in replay["observations"])
        # A zero dwell returns before the coordinator has necessarily reached the
        # simulator, so wait for the one submission rather than racing it.
        await _wait_for(lambda: len(submissions) == 1)
        assert len(submissions) == 1

    async def test_different_payload_replay_conflicts(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "conflict.cir")
        await handle_run_experiments(
            _args(deck, "conflicting-payload"),
            state_with_sim,
        )

        result = await handle_run_experiments(
            _args(deck, "conflicting-payload", lint="off", budget=500),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert result.isError
        assert set(data["error"]) == {
            "code",
            "message",
            "stage",
            "retryable",
            "commit_state",
        }
        assert data["error"] == {
            "code": "idempotency_conflict",
            "message": (
                "request_id 'conflicting-payload' was already used for a different "
                "request payload or canonicalizer version"
            ),
            "stage": "submission",
            "retryable": False,
            "commit_state": "not_started",
        }
        assert "control_token" not in data
        assert len(submissions) == 1


@pytest.mark.asyncio
class TestLeanReceipt:
    """The default receipt is the answer channel: completed rows drop their
    artifact paths (reachable via jobs(runs) or run_fields), non-completed
    rows keep them (the failed row's log is its diagnostic), and the caller's
    own attached-analysis request is echoed only under provenance."""

    async def test_completed_rows_drop_artifact_paths(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "lean_rows.cir")

        data = _assert_schema(
            await handle_run_experiments(_args(deck, "lean-rows"), state_with_sim)
        )

        (row,) = data["runs"]["items"]
        assert row["status"] == "produced"
        assert "raw" not in row and "log" not in row
        assert row["assignments"] == {}

    async def test_run_fields_still_fetch_artifact_paths(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "lean_fetch.cir")

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "lean-fetch", run_fields=["case_id", "raw", "log"]),
                state_with_sim,
            )
        )

        (row,) = data["runs"]["items"]
        assert row["raw"] and row["log"]

    @pytest.mark.parametrize(
        ("request_id", "run_fields"),
        [
            ("snapshot-projected", ["case_id", "assignments"]),
            ("snapshot-lean", None),
        ],
    )
    async def test_handler_projection_matches_the_same_neutral_snapshot(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        request_id: str,
        run_fields: list[str] | None,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / f"{request_id}.cir")
        captured: list[experiments_mod.ReceiptSnapshot] = []
        snapshot_receipt = experiments_mod.snapshot_receipt

        def capture_snapshot(*args: Any, **kwargs: Any) -> experiments_mod.ReceiptSnapshot:
            snapshot = snapshot_receipt(*args, **kwargs)
            captured.append(snapshot)
            return snapshot

        monkeypatch.setattr(experiments_mod, "snapshot_receipt", capture_snapshot)
        request = (
            _args(deck, request_id, run_fields=run_fields)
            if run_fields is not None
            else _args(deck, request_id)
        )

        data = _assert_schema(
            await handle_run_experiments(
                request,
                state_with_sim,
            )
        )

        (snapshot,) = captured
        expected = experiments_mod.project_receipt_runs(
            snapshot,
            run_fields,
            lean_default=True,
        )
        assert data["runs"] == expected

    async def test_non_completed_rows_keep_artifact_path_keys(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "lean_blocked.cir", body="V1 in 0 1\nR1 out 1k\n.op\n.end\n")

        data = _assert_schema(
            await handle_run_experiments(_args(deck, "lean-blocked"), state_with_sim)
        )

        (row,) = data["runs"]["items"]
        assert row["status"] != "produced"
        assert "raw" in row and "log" in row

    async def test_analysis_request_echo_is_provenance(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "lean_echo.cir")

        lean = _assert_schema(
            await handle_run_experiments(
                _args(deck, "lean-echo", **_VARIED_ANALYSIS),
                state_with_sim,
            )
        )
        assert lean["analysis"]["status"] == "completed"
        assert "request" not in lean["analysis"]

        loud = _assert_schema(
            await handle_run_experiments(
                _args(deck, "lean-echo", provenance=True, **_VARIED_ANALYSIS),
                state_with_sim,
            )
        )
        assert loud["analysis"]["request"] is not None


@pytest.mark.asyncio
class TestAttachedBlockPreflight:
    """A malformed attached analyze block is refused BEFORE anything runs.

    Validated only at the analysis stage, a typo'd recipe burns the whole
    simulation cycle — and the corrected block changes the fingerprint, so
    the retry re-runs every case."""

    async def test_malformed_attached_block_refused_before_any_simulation(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "attached_bad.cir")
        bad = {"recipes": [{"key": "x", "metric": "no_such_metric"}]}

        result = await handle_run_experiments(
            _args(deck, "attached-bad", analyze=bad), state_with_sim
        )

        assert result.isError
        assert "attached analyze block" in json.dumps(result.structuredContent)
        assert submissions == []

        # The refusal left no durable record: the same id retries the same
        # way instead of replaying a failure or conflicting.
        again = await handle_run_experiments(
            _args(deck, "attached-bad", analyze=bad), state_with_sim
        )
        assert again.isError
        assert submissions == []

    async def test_model_level_analyze_fault_also_refused_at_the_door(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "attached_dup.cir")

        result = await handle_run_experiments(
            _args(
                deck,
                "attached-dup-group",
                analyze={
                    "recipes": [
                        {"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}
                    ],
                    "group_by": ["R1", "R1"],
                },
            ),
            state_with_sim,
        )

        assert result.isError
        assert "attached analyze block" in json.dumps(result.structuredContent)
        assert submissions == []


class TestOptionalRequestId:
    """request_id may be omitted: a fresh id is generated per call, so a
    one-off run pays no idempotency ceremony, while an explicit id keeps the
    durable replay/conflict semantics on the same code path."""

    def test_omitted_request_id_autogenerates(self, work_dir: Path):
        args = RunExperimentsInput.model_validate(
            {"circuits": [{"path": str(work_dir / "a.cir"), "id": "dut"}]}
        )
        assert args.request_id.startswith("req_")

    def test_autogenerated_ids_never_collide_or_replay(self, work_dir: Path):
        from ltspice_mcp.lib.experiment_runner import canonical_fingerprint

        payload = {"circuits": [{"path": str(work_dir / "a.cir"), "id": "dut"}]}
        first = RunExperimentsInput.model_validate(payload)
        second = RunExperimentsInput.model_validate(payload)
        assert first.request_id != second.request_id
        assert canonical_fingerprint(first) != canonical_fingerprint(second)

    def test_explicit_request_id_is_preserved(self, work_dir: Path):
        assert _args(work_dir / "a.cir", "chosen-id").request_id == "chosen-id"

    @pytest.mark.asyncio
    async def test_omitted_id_runs_echoes_and_stays_durable(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "auto_id.cir")
        args = RunExperimentsInput.model_validate(
            {
                "circuits": [{"path": str(deck), "id": "dut"}],
                "execution": {"wait_s": 1.0},
            }
        )

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))
        assert data["request_id"] == args.request_id
        assert data["status"] == "completed"

        replay = _assert_schema(
            await handle_run_experiments(_args(deck, args.request_id), state_with_sim)
        )
        assert replay["job_id"] == data["job_id"]
        assert len(submissions) == 1


@pytest.mark.asyncio
class TestReplayRejectsChangedSources:
    """A reused request_id over an edited circuit must not return the old numbers.

    The canonical fingerprint covers the request arguments only, so nothing in
    it moves when a deck is edited: an identical payload over a changed circuit
    used to replay the earlier receipt as a confirmed success. That is the one
    failure that returns wrong data rather than a wrong field, so it fails
    closed on the digests the coordinator already recorded.
    """

    async def test_edited_deck_conflicts_instead_of_replaying(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "edited.cir")
        args = _args(deck, "edited-deck")
        await handle_run_experiments(args, state_with_sim)
        _deck(deck, "V1 in 0 1\nR1 in 0 2k\n.op\n.end\n")

        result = await handle_run_experiments(args, state_with_sim)
        data = _assert_schema(result)

        assert result.isError
        assert data["error"]["code"] == "idempotency_conflict"
        assert str(deck) in data["error"]["message"]
        assert "control_token" not in data
        assert len(submissions) == 1

    async def test_edited_include_conflicts_with_the_root_deck_untouched(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)
        args = _args(deck, "edited-include", lint="off")
        await handle_run_experiments(args, state_with_sim)
        root_bytes = deck.read_bytes()
        core = work_dir / "core.inc"
        core.write_text(".subckt core in out\nR1 in mid 2k\nC1 mid out 1n\n.ends\n")

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert deck.read_bytes() == root_bytes
        assert data["error"]["code"] == "idempotency_conflict"
        assert str(core) in data["error"]["message"]
        assert len(submitted) == 1

    async def test_unchanged_deck_and_include_still_replay(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)
        args = _args(deck, "unchanged-include", lint="off")

        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        replay = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert replay["job_id"] == first["job_id"]
        assert any(item["code"] == "idempotent_replay" for item in replay["observations"])
        assert len(submitted) == 1

    async def test_deleted_source_conflicts_rather_than_replaying(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "removed.cir")
        args = _args(deck, "removed-deck")
        await handle_run_experiments(args, state_with_sim)
        deck.unlink()

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert data["error"]["code"] == "idempotency_conflict"
        assert "no longer readable" in data["error"]["message"]
        assert str(deck) in data["error"]["message"]
        assert len(submissions) == 1

    async def test_edited_schematic_conflicts_though_its_export_is_untouched(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The .asc is the file the author edits, and nothing else moves with it.

        A schematic runs through an exported netlist, and a replay skips the
        export — so the netlist the manifest recorded is still on disk, still
        byte-identical, and still describes the circuit as it was. Checking only
        that export is checking the one file an edit cannot reach.
        """
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        schematic = _schematic(work_dir / "amp.asc", "1k")
        _asc_exporter(state_with_sim)
        args = _args(schematic, "edited-schematic", provenance=True)
        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        exported = work_dir / "amp.net"
        exported_bytes = exported.read_bytes()
        _schematic(schematic, "2k")

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert exported.read_bytes() == exported_bytes
        assert first["source"][0]["sha256"] != sha256_file(exported)
        assert data["error"]["code"] == "idempotency_conflict"
        assert str(schematic) in data["error"]["message"]
        assert len(submissions) == 1

    async def test_schematic_source_digest_describes_the_schematic(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A record that pairs the .asc's path with the .net's digest describes
        neither file, and no later reader can tell which one it meant."""
        fake_simulator(monkeypatch)
        schematic = _schematic(work_dir / "paired.asc", "3k")
        _asc_exporter(state_with_sim)

        data = _assert_schema(
            await handle_run_experiments(
                _args(schematic, "paired-digest", provenance=True), state_with_sim
            )
        )

        source = data["source"][0]
        assert source["path"] == str(schematic)
        assert source["sha256"] == sha256_file(schematic)

    async def test_unchanged_schematic_still_replays(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Adding the schematic to the manifest must not make every .asc replay a
        conflict — the export is regenerated per submission and never matches."""
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        schematic = _schematic(work_dir / "stable.asc", "4k7")
        _asc_exporter(state_with_sim)
        args = _args(schematic, "stable-schematic")

        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        replay = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert replay["job_id"] == first["job_id"]
        assert any(item["code"] == "idempotent_replay" for item in replay["observations"])
        assert len(submissions) == 1

    async def test_live_include_conflicts_rather_than_replaying(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A file that was never digested cannot be shown to be unchanged.

        allow_live_includes already says the job cannot prove what those files
        held; a replay would make that claim a second time, later. It fails the
        way every other unprovable case here does — a re-run, not stale numbers.
        """
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        outside = work_dir.parent / f"{work_dir.name}-shared.inc"
        outside.write_text(".param supply=5\n")
        deck = _deck(
            work_dir / "live.cir",
            f'.include "{outside}"\nV1 in 0 {{supply}}\nR1 in 0 1k\n.op\n.end\n',
        )
        args = _args(deck, "live-include", lint="off", allow_live_includes=True)
        first = _assert_schema(await handle_run_experiments(args, state_with_sim))

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert first["outcome"] == "complete"
        assert data["error"]["code"] == "idempotency_conflict"
        assert str(outside) in data["error"]["message"]
        assert len(submissions) == 1

    async def test_record_without_source_digests_conflicts(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A record predating recorded digests must not replay as a match.

        Such a record carries no manifest to compare against at all, so there is
        no drift to find — the guard has to refuse it on the absence itself, or
        the fix is inert for exactly the jobs already sitting on disk. The deck
        is named so that no assertion here can pass on the filename instead of
        the message.
        """
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "older.cir")
        args = _args(deck, "older-record")
        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        await state_with_sim.job_registry.drain_pending()

        record = experiment_store.record_path(first["job_id"], work_dir)
        stored = json.loads(record.read_text())
        for source in stored["sources"]:
            source["sha256"] = ""
            source["manifest"] = []
        record.write_text(json.dumps(stored))
        del state_with_sim.experiment_jobs[first["job_id"]]

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        message = data["error"]["message"]
        assert data["error"]["code"] == "idempotency_conflict"
        assert "carries no source digest" in message
        # Not the drift path: nothing on disk changed, and a record with no
        # manifest cannot report that it did.
        assert "changed since" not in message
        assert len(submissions) == 1


@pytest.mark.asyncio
class TestLintModes:
    async def test_block_prevents_simulator_submission(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(
            work_dir / "blocked.cir",
            "V1 in 0 1\nR1 out 1k\n.op\n.end\n",
        )

        result = await handle_run_experiments(
            _args(
                deck,
                "lint-block",
                variations=[
                    {
                        "kind": "assign",
                        "assign": {"R1": ["1k", "2k"]},
                    }
                ],
            ),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert submissions == []
        assert data["completeness"]["skipped"] == 2
        assert data["failures"][0]["code"] == "lint_blocked"
        assert [item["assignments"]["R1"] for item in data["runs"]["items"]] == [
            "1k",
            "2k",
        ]

    async def test_warn_proceeds_and_preserves_findings(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(
            work_dir / "warn.cir",
            "V1 in 0 1\nR1 in 0 1M\n.op\n.end\n",
        )

        result = await handle_run_experiments(
            _args(deck, "lint-warn", lint="warn"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert len(submissions) == 1
        assert any(
            finding["rule_id"] == "suffix-mega-milli" for finding in data["lint"][0]["findings"]
        )

    async def test_block_resolves_models_through_staged_includes(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Windows-native staging spells the root deck's include references in
        Windows form, which nothing on the Linux side can re-read from disk.
        The staged include closure itself must satisfy the model lookup: the
        include here defines the only subckt the deck instantiates, and a
        model-missing false positive would refuse a perfectly runnable deck.
        """
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        (work_dir / "amp.inc").write_text(".subckt AMP a b\nRA a b 1k\n.ends AMP\n")
        deck = _deck(
            work_dir / "with_include.cir",
            '.include "amp.inc"\nX1 in 0 AMP\nV1 in 0 1\n.op\n.end\n',
        )
        route = experiments_mod.resolve_experiment_paths

        def windows_native(working_dir, job_id, circuit_id, simulator):
            return dataclasses.replace(
                route(working_dir, job_id, circuit_id, simulator),
                windows_native=True,
            )

        monkeypatch.setattr(experiments_mod, "resolve_experiment_paths", windows_native)
        monkeypatch.setattr(
            wsl, "to_windows_path", lambda path: "Z:" + str(path).replace("/", "\\")
        )

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "staged-include-models"),
                state_with_sim,
            )
        )

        assert len(submissions) == 1
        assert data["outcome"] == "complete"
        assert data["failures"] == []

    async def test_off_skips_linter(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(
            work_dir / "off.cir",
            "V1 in 0 1\nR1 in 0 1M\n.op\n.end\n",
        )

        result = await handle_run_experiments(
            _args(deck, "lint-off", lint="off"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert len(submissions) == 1
        # No findings (linting was off) -> no lint entry at all; an empty
        # per-circuit row is ceremony the lean receipt no longer carries.
        assert data["lint"] == []


@pytest.mark.asyncio
class TestPerCircuitFailuresAndAccounting:
    async def test_multi_circuit_failure_is_isolated_and_ordered(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        valid = _deck(work_dir / "valid.cir")
        missing = work_dir / "missing.cir"
        args = RunExperimentsInput.model_validate(
            {
                "request_id": "mixed-circuits",
                "circuits": [
                    {"path": str(valid), "id": "valid"},
                    {"path": str(missing), "id": "missing"},
                ],
                "execution": {"wait_s": 1},
            }
        )

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert len(submissions) == 1
        assert data["outcome"] == "partial"
        assert data["completeness"]["expanded"] == 2
        assert [item["circuit"] for item in data["runs"]["items"]] == [
            "valid",
            "missing",
        ]
        assert [item["run_index"] for item in data["runs"]["items"]] == [0, 1]
        assert data["failures"][0]["case_id"] == "missing-case-0000"

    async def test_case_deck_keeps_staged_relative_includes_reachable(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []

        def submit(self, netlist: Path, run_filename: str, callback):
            submitted.append(netlist)
            raw = self.output_folder / f"{Path(run_filename).stem}.raw"
            log = self.output_folder / f"{Path(run_filename).stem}.log"
            raw.write_bytes(b"Title: mock")
            log.write_text("ok")
            outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
            self.loop.call_soon_threadsafe(callback, outcome)
            return object()

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
        (work_dir / "support.inc").write_text(".param supply=1\n")
        deck = _deck(
            work_dir / "relative.cir",
            '.include "support.inc"\nV1 in 0 {supply}\nR1 in 0 1k\n.op\n.end\n',
        )

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "relative-include"),
                state_with_sim,
            )
        )

        assert data["outcome"] == "complete"
        assert (submitted[0].parent / "support.inc").is_file()

    async def test_asc_without_exporter_is_accounted(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        schematic = _deck(work_dir / "missing-exporter.asc", "Version 4\n")

        result = await handle_run_experiments(
            _args(schematic, "asc-unavailable"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert submissions == []
        assert data["failures"][0]["code"] == "asc_export_unavailable"
        assert data["completeness"]["failed"] == 1

    async def test_terminal_counters_reconcile(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "grid.cir")
        args = _args(
            deck,
            "counter-grid",
            variations=[
                {
                    "kind": "assign",
                    "assign": {"R1": ["1k", "2k", "3k"]},
                }
            ],
        )

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))
        counts = data["completeness"]

        assert counts["expanded"] == 3
        assert (
            counts["produced"] + counts["failed"] + counts["cancelled"] + counts["skipped"]
            == counts["expanded"]
        )

    async def test_recent_circuit_is_noted_explicitly(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "recent.cir")
        note = AsyncMock()
        monkeypatch.setattr(state_with_sim, "note_recent_circuit", note)

        result = await handle_run_experiments(
            _args(deck, "recent-note"),
            state_with_sim,
        )
        _assert_schema(result)

        note.assert_awaited_once_with(deck.resolve())


def _failing_simulator(monkeypatch: pytest.MonkeyPatch, log_text: str) -> None:
    """Every case aborts the way the simulator aborts: non-zero exit, .fail log."""

    def submit(self, _netlist: Path, run_filename: str, callback):
        log = self.output_folder / f"{Path(run_filename).stem}.fail"
        log.write_text(log_text)
        self.loop.call_soon_threadsafe(callback, collect_run_outcome(".", str(log)))
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)


@pytest.mark.asyncio
class TestFailureChannel:
    """What a caller learns from a batch that failed the same way N times."""

    async def test_identical_failures_collapse_into_one_counted_row(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The channel the budget ladder may never trim must bound itself.

        Twelve cases failing for one reason is one fact and twelve copies of a
        20-line log excerpt; the row names its cases and its true count so
        nothing is rounded away by the collapse.
        """
        _failing_simulator(
            monkeypatch,
            "Direct Newton iteration failed to find operating point.\n"
            "Time step too small; time = 1.2e-06, timestep = 1e-18\n",
        )
        deck = _deck(work_dir / "stiff.cir")
        args = _args(
            deck,
            "collapsing-failures",
            variations=[{"kind": "assign", "assign": {"R1": [f"{n}k" for n in range(1, 13)]}}],
        )

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert data["completeness"]["expanded"] == 12
        assert data["completeness"]["failed"] == 12
        assert len(data["failures"]) == 1
        row = data["failures"][0]
        assert row["count"] == 12
        assert len(row["case_ids"]) == 10
        assert row["case_id"] == row["case_ids"][0]
        assert json.dumps(data["failures"]).count("Time step too small") == 1

    async def test_a_classified_failure_carries_its_code_and_recovery_route(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _failing_simulator(
            monkeypatch,
            "Time step too small; time = 1.2e-06, timestep = 1e-18\n",
        )
        deck = _deck(work_dir / "stiff-one.cir")

        data = _assert_schema(
            await handle_run_experiments(_args(deck, "classified-failure"), state_with_sim)
        )

        row = data["failures"][0]
        assert row["code"] == "convergence_failed"
        assert "reltol" in row["hint"]

    async def test_missing_model_failure_names_the_unresolved_reference(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _failing_simulator(
            monkeypatch,
            'Error on line 2 : q1 c b e mystery Unable to find definition of model "mystery"\n',
        )
        deck = _deck(work_dir / "unresolved.cir")

        data = _assert_schema(
            await handle_run_experiments(_args(deck, "missing-model-failure"), state_with_sim)
        )

        row = data["failures"][0]
        assert row["code"] == "missing_model"
        assert row["evidence"] == {"missing_refs": ["mystery"]}


@pytest.mark.asyncio
class TestAttachedAnalysis:
    """The analyze block runs on the real recipe engine, not a stub."""

    async def test_recipes_and_group_by_run_end_to_end(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached.cir")

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "attached-analysis", **_VARIED_ANALYSIS),
                state_with_sim,
            )
        )

        analysis = data["analysis"]
        assert analysis["status"] == "completed", analysis["error"]
        result = analysis["result"]
        assert result is not None, analysis
        assert result["coverage"]["missing_cases"]["items"] == []
        # Real recipe output, addressed by the recipe key the caller chose.
        groups = result["results"]["vout"]["groups"]
        assert sorted(group["by"]["R1"] for group in groups) == ["1k", "2k"]
        assert all(group["count"] == 1 for group in groups)
        assert all(
            entry["stat"] == "mean" and entry["value"] is not None
            for group in groups
            for entry in group["reduced"]
        )
        assert result["coverage"]["runs_analyzed"] == 2

    async def test_include_block_reaches_the_analysis_engine(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The include block is the caller's only lever over what comes back.

        A recipe with 'reduce' returns reductions alone unless per_run is asked
        for, so the rows are proof the block was forwarded rather than accepted
        and dropped — a silent drop leaves the response looking exactly like a
        caller who never asked.
        """
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-include.cir")
        analysis = {
            **_VARIED_ANALYSIS["analyze"],
            "include": {"per_run": {"limit": 5}, "signals_available": True},
        }

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-include",
                    variations=_VARIED_ANALYSIS["variations"],
                    analyze=analysis,
                ),
                state_with_sim,
            )
        )

        result = data["analysis"]["result"]
        assert result is not None, data["analysis"]
        entry = result["results"]["vout"]
        assert entry["per_run"]["returned"] == 2
        assert {row["run_index"] for row in entry["per_run"]["items"]} == {0, 1}
        assert result["signals_available"]

    async def test_replay_projects_nested_content_from_the_neutral_snapshot(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-neutral.cir")
        base = {
            "recipes": [{"key": "summary", "metric": "summary"}],
        }

        lean = _assert_schema(
            await handle_run_experiments(
                _args(deck, "attached-neutral", analyze=base),
                state_with_sim,
            )
        )
        wide = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-neutral",
                    analyze={**base, "include": {"fields": ["value"]}},
                ),
                state_with_sim,
            )
        )

        assert wide["request_id"] == lean["request_id"]
        assert any(item["code"] == "idempotent_replay" for item in wide["observations"])

        lean_value = lean["analysis"]["result"]["results"]["summary"]["values"][0]["value"]
        wide_value = wide["analysis"]["result"]["results"]["summary"]["values"][0]["value"]
        assert not any(isinstance(value, (dict, list)) for value in lean_value.values())
        assert any(isinstance(value, (dict, list)) for value in wide_value.values())
        missing = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-neutral",
                    analyze={
                        **base,
                        "include": {"fields": ["value.not_recorded"]},
                    },
                ),
                state_with_sim,
            )
        )
        warning = missing["analysis"]["result"]["results"]["summary"]["warnings"][-1]
        assert "value.not_recorded" in warning
        assert "absent from every row" in warning
        assert "keys present" in warning
        job = state_with_sim.experiment_jobs[lean["job_id"]]
        assert job.analysis.result is not None
        assert job.analysis.result["schema"] == "ltspice-mcp/attached-analysis-snapshot"
        assert job.analysis.request is not None
        assert job.analysis.request["include"] is None

    async def test_projection_survives_the_attached_continuation_and_replay_view_change(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-pages.cir")
        analyze_request = {
            "recipes": [{"key": "summary", "metric": "summary"}],
            "include": {"per_run": {"limit": 1}, "fields": ["value"]},
        }
        first = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-pages",
                    variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
                    analyze=analyze_request,
                ),
                state_with_sim,
            )
        )
        result = first["analysis"]["result"]
        page = result["results"]["summary"]["per_run"]
        assert page["returned"] == 1 and page["next_cursor"] is not None
        assert set(page["items"][0]) == {"value"}
        assert any(isinstance(value, (dict, list)) for value in page["items"][0]["value"].values())

        continuation = AnalyzeResultsInput.model_validate(
            {
                "continue": {
                    "result_set_id": result["result_set_id"],
                    "cursor": page["next_cursor"],
                }
            }
        )
        continued = await analyze_mod.handle_analyze_results(continuation, state_with_sim)
        assert continued.structuredContent is not None
        continued_page = continued.structuredContent["results"]["summary"]["per_run"]
        assert set(continued_page["items"][0]) == {"value"}
        assert continued_page["items"][0] == page["items"][0]
        assert any(
            isinstance(value, (dict, list))
            for value in continued_page["items"][0]["value"].values()
        )

        changed = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-pages",
                    variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
                    analyze={
                        **analyze_request,
                        "include": {"per_run": {"limit": 1}, "fields": ["case_id"]},
                    },
                ),
                state_with_sim,
            )
        )
        changed_cursor = changed["analysis"]["result"]["results"]["summary"]["per_run"][
            "next_cursor"
        ]
        assert result_store.cursor_view(changed_cursor) == (True, ["case_id"])

    async def test_budget_answer_reconstructs_values_independent_of_requested_page(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        original_trace_names = OffsetAwareRawRead.get_trace_names

        def wide_trace_names(raw: OffsetAwareRawRead) -> list[str]:
            return [
                *original_trace_names(raw),
                *(f"budget_trace_{index:03d}" for index in range(500)),
            ]

        monkeypatch.setattr(OffsetAwareRawRead, "get_trace_names", wide_trace_names)
        deck = _deck(work_dir / "attached-answer.cir")
        variations = [{"kind": "assign", "assign": {"R1": ["1k", "2k", "3k"]}}]
        recipe = [{"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}]
        expected_result = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "attached-answer-expected",
                    variations=variations,
                    analyze={"recipes": recipe},
                ),
                state_with_sim,
            )
        )["analysis"]["result"]
        expected = expected_result["results"]["vout"]["values"]

        request = _args(
            deck,
            "attached-answer-budget",
            variations=variations,
            analyze={
                "recipes": recipe,
                "include": {
                    "per_run": {"limit": 1},
                    "signals_available": True,
                },
            },
        )
        full = _assert_schema(await handle_run_experiments(request, state_with_sim))
        replay = _assert_schema(await handle_run_experiments(request, state_with_sim))
        assert any(item["code"] == "idempotent_replay" for item in replay["observations"])
        await state_with_sim.job_registry.drain_pending()
        job = state_with_sim.experiment_jobs[full["job_id"]]
        assert job.analysis.result is not None
        assert "answer_top" not in job.analysis.result
        assert all(
            "answer_facts" not in block for block in job.analysis.result["results"].values()
        )
        snapshot_identity = job.analysis.result
        pristine_snapshot = copy.deepcopy(job.analysis.result)
        pristine_job = copy.deepcopy(experiment_store.serialize_job(job))
        trim_rung = response_budget.Rung(
            response_budget.RUNG_TRIM,
            budget=10_000,
            measured=0,
            reserve=experiments_mod._RUN_BUDGET_NOTES.reserve,
        )
        answer_rung = dataclasses.replace(trim_rung, level=response_budget.RUNG_ANSWER)
        manual_snapshot = experiments_mod.snapshot_receipt(
            job,
            None,
            control_token=job.control_token,
        )
        trim_view = experiments_mod.finalize_receipt(
            experiments_mod.render_receipt_snapshot(
                manual_snapshot,
                control_token=job.control_token,
            )
        )
        experiments_mod._degrade_receipt(trim_view, trim_rung)
        answer_view = experiments_mod.finalize_receipt(
            experiments_mod.render_receipt_snapshot(
                manual_snapshot,
                control_token=job.control_token,
                analysis_answer_channel=True,
            )
        )
        experiments_mod._degrade_receipt(answer_view, answer_rung)
        trim_size = response_budget.estimate_tokens(trim_view)
        answer_size = response_budget.estimate_tokens(answer_view)
        assert answer_size < trim_size
        # The handler's real response carries envelope text this manual view
        # does not (replay observation, progress-augmented hint), so aim the
        # budget a third of the rung gap above the measured answer size —
        # still below trim — instead of exactly at it.
        budget = (
            answer_size
            + (trim_size - answer_size) // 3
            + experiments_mod._RUN_BUDGET_NOTES.reserve
        )
        assert trim_size > budget - experiments_mod._RUN_BUDGET_NOTES.reserve
        answer = _assert_schema(
            await handle_run_experiments(
                request.model_copy(update={"budget": budget}),
                state_with_sim,
            )
        )
        assert job.analysis.result is snapshot_identity
        assert job.analysis.result == pristine_snapshot
        assert experiment_store.serialize_job(job) == pristine_job
        note = next(
            item for item in answer["observations"] if item.get("code") == "budget_truncated"
        )
        assert "(answer)" in note["detail"]
        answer_result = answer["analysis"]["result"]
        entry = answer_result["results"]["vout"]
        values = entry["values"]
        columns = entry.get("values_columns")
        if columns is not None:
            values = [dict(zip(columns, row, strict=True)) for row in values]
        assert values == expected
        assert answer_result["outcome"] == expected_result["outcome"]
        assert answer_result["coverage"] == expected_result["coverage"]
        assert answer_result["next"] is None
        assert answer_result["cursor"] is None
        assert "signals_available" not in answer_result
        assert "signals_available" in job.analysis.result["top"]
        restored = _assert_schema(await handle_run_experiments(request, state_with_sim))
        assert restored["analysis"]["result"]["signals_available"]

    async def test_v1_public_result_survives_restart_replay_and_foreign_status(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "state"))
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-v1.cir")
        request = _args(
            deck,
            "attached-v1",
            analyze={
                "recipes": [{"key": "summary", "metric": "summary"}],
                "include": {"signals_available": True},
            },
        )
        first = _assert_schema(await handle_run_experiments(request, state_with_sim))
        legacy_result = first["analysis"]["result"]
        await state_with_sim.job_registry.drain_pending()

        record = experiment_store.record_path(first["job_id"], work_dir)
        stored = json.loads(record.read_text())
        stored["schema_version"] = 1
        stored["analysis"]["result"] = legacy_result
        record.write_text(json.dumps(stored))
        recent.touch(deck)

        restarted = SessionState.create(
            state_with_sim.config,
            available=state_with_sim.available_simulators,
        )
        foreign = SessionState.create(
            state_with_sim.config,
            available=state_with_sim.available_simulators,
        )
        assert restarted.job_registry.preload_recent() == 1
        assert foreign.job_registry.preload_recent() == 1

        replay = _assert_schema(await handle_run_experiments(request, restarted))
        assert replay["analysis"]["result"] == legacy_result
        assert any(
            item["code"] == "legacy_analysis_result" for item in replay["analysis"]["observations"]
        )

        status_call = await handle_jobs(
            JobsInput.model_validate({"action": "status", "job_id": first["job_id"]}),
            foreign,
        )
        status = status_call.structuredContent
        assert status is not None
        assert status["analysis"]["result"] == legacy_result
        assert any(
            item["code"] == "legacy_analysis_result" for item in status["analysis"]["observations"]
        )

        assert request.analyze is not None and request.analyze.include is not None
        available_projection = await handle_run_experiments(
            request.model_copy(
                update={
                    "analyze": request.analyze.model_copy(
                        update={
                            "include": request.analyze.include.model_copy(
                                update={"fields": ["case_id"]}
                            )
                        }
                    )
                }
            ),
            restarted,
        )
        available_data = _assert_schema(available_projection)
        assert not available_projection.isError
        assert available_data["analysis"]["result"] == legacy_result

        projected = await handle_run_experiments(
            request.model_copy(
                update={
                    "analyze": request.analyze.model_copy(
                        update={
                            "include": request.analyze.include.model_copy(
                                update={"fields": ["value"]}
                            )
                        }
                    )
                }
            ),
            restarted,
        )
        projected_data = _assert_schema(projected)
        assert projected.isError
        assert "legacy rendered result" in projected_data["error"]["message"]

        intact = _assert_schema(await handle_run_experiments(request, restarted))
        assert intact["analysis"]["result"] == legacy_result

    async def test_successful_analysis_does_not_report_partial(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-complete.cir")

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "attached-complete", **_VARIED_ANALYSIS),
                state_with_sim,
            )
        )

        assert data["completeness"]["produced"] == data["completeness"]["expanded"] == 2
        assert data["outcome"] == "complete"
        assert data["status"] == "completed"

    async def test_failed_analysis_keeps_a_run_complete_outcome(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-bad.cir")
        # A malformed block is refused before submission now, so a STAGE-time
        # failure needs the engine itself to fail — patched through
        # tools.analyze, the seam the callback deliberately resolves late.
        analyze = {
            "recipes": [{"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}],
        }

        async def exploding_engine(args, state):
            raise ValueError("analysis engine failure injected by test")

        monkeypatch.setattr(analyze_mod, "capture_attached_analysis", exploding_engine)
        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "attached-bad-request", analyze=analyze),
                state_with_sim,
            )
        )

        # The runs are what `outcome` describes; the analysis failure has its
        # own field, its own observation, and a hint that points at both.
        assert data["outcome"] == "complete"
        assert data["completeness"]["produced"] == 1
        assert data["completeness"]["failed"] == 0
        assert data["analysis"]["status"] == "failed"
        assert "analysis engine failure injected by test" in data["analysis"]["error"]
        assert [item["code"] for item in data["analysis"]["observations"]] == ["analysis_failed"]
        assert "attached analysis failed" in data["hint"]
        # The coordinator still records the stage failure on the job status.
        assert data["status"] == "completed_with_failures"

    async def test_wait_for_runs_returns_before_analysis_finishes(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        recorded_fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-wait.cir")
        released = asyncio.Event()
        engine = analyze_mod.capture_attached_analysis

        async def gated(args, state):
            await released.wait()
            return await engine(args, state)

        monkeypatch.setattr(analyze_mod, "capture_attached_analysis", gated)

        receipt = _assert_schema(
            await handle_run_experiments(
                _args(deck, "attached-wait", wait_s=0.0, **_VARIED_ANALYSIS),
                state_with_sim,
            )
        )
        job_id = receipt["job_id"]

        runs_done = await _jobs_wait(state_with_sim, job_id, "runs", 2.0)
        assert runs_done["timed_out"] is False
        assert runs_done["status"] == "analyzing"

        still_analyzing = await _jobs_wait(state_with_sim, job_id, "all", 0.05)
        assert still_analyzing["timed_out"] is True

        released.set()
        finished = await _jobs_wait(state_with_sim, job_id, "all", 2.0)
        assert finished["timed_out"] is False
        assert finished["status"] == "completed"
        assert finished["analysis_status"] == "completed"


_CORE_INC = ".subckt core in out\nR1 in mid 1k\nC1 mid out 1n\n.ends\n"
_FACTORED_ROOT = '.include "core.inc"\nV1 in 0 1\nX1 in out core\n.op\n.end\n'


def _recording_simulator(monkeypatch: pytest.MonkeyPatch, submitted: list[Path]) -> None:
    """Instant simulator that records the case deck it was handed."""

    def submit(self, netlist: Path, run_filename: str, callback):
        submitted.append(Path(netlist))
        raw = self.output_folder / f"{Path(run_filename).stem}.raw"
        log = self.output_folder / f"{Path(run_filename).stem}.log"
        raw.write_bytes(b"Title: mock")
        log.write_text("ok")
        outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)


def _include_targets(netlist: Path) -> list[Path]:
    """Resolve a deck's include references without the production resolver."""
    targets: list[Path] = []
    for line in netlist.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].casefold() in {".include", ".inc"}:
            raw = Path(parts[1].strip('"'))
            targets.append(raw if raw.is_absolute() else netlist.parent / raw)
    return targets


def _factored_deck(work_dir: Path, *, core: str = _CORE_INC, root: str = _FACTORED_ROOT) -> Path:
    (work_dir / "core.inc").write_text(core)
    return _deck(work_dir / "factored.cir", root)


@pytest.mark.asyncio
class TestVariationsReachIntoIncludes:
    """A component only reachable through .include is still a sweep target.

    Factoring a circuit into a reusable core used to cost the ability to vary
    anything in it, which pushed authors into promoting every value they might
    later sweep to a top-level .param before they knew which ones those were.
    """

    async def test_included_component_is_targetable(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-assign",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["2k"]}}],
                ),
                state_with_sim,
            )
        )

        assert data["outcome"] == "complete"
        assert data["completeness"]["produced"] == 1
        staged_core = _include_targets(submitted[0])[0]
        assert "R1 in mid 2k" in staged_core.read_text()

    async def test_each_case_edits_its_own_copy_of_the_include(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-isolation",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["2k", "3k"]}}],
                ),
                state_with_sim,
            )
        )

        assert data["completeness"]["produced"] == 2
        cores = [_include_targets(path)[0] for path in sorted(submitted, key=lambda p: p.name)]
        assert cores[0] != cores[1]
        assert "R1 in mid 2k" in cores[0].read_text()
        assert "R1 in mid 3k" in cores[1].read_text()

    async def test_authored_include_is_never_written(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-readonly",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["2k", "3k"]}}],
                ),
                state_with_sim,
            )
        )

        # Both cases really did edit the include, so an untouched original is
        # evidence of staging and not of a run that never got that far.
        assert data["completeness"]["produced"] == 2
        assert all("core.inc" in str(_include_targets(path)[0]) for path in submitted)
        assert (work_dir / "core.inc").read_text() == _CORE_INC
        assert not (work_dir / "case-0000__core.inc").is_file()
        assert not (work_dir / "case-0001__core.inc").is_file()

    async def test_root_deck_declaration_wins_over_an_include(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(
            work_dir,
            root='.include "core.inc"\nV1 in 0 1\nR1 in out 1k\nX1 out 0 core\n.op\n.end\n',
        )

        _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-root-wins",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["2k"]}}],
                ),
                state_with_sim,
            )
        )

        assert "R1 in out 2k" in submitted[0].read_text()
        staged_core = _include_targets(submitted[0])[0]
        assert "R1 in mid 1k" in staged_core.read_text()

    async def test_two_includes_declaring_the_target_name_both_files(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        (work_dir / "left.inc").write_text(".subckt left in out\nR1 in out 1k\n.ends\n")
        (work_dir / "right.inc").write_text(".subckt right in out\nR1 in out 2k\n.ends\n")
        deck = _deck(
            work_dir / "two-cores.cir",
            '.include "left.inc"\n.include "right.inc"\nV1 in 0 1\n'
            "X1 in mid left\nX2 mid 0 right\n.op\n.end\n",
        )

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-ambiguous",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["2k"]}}],
                ),
                state_with_sim,
            )
        )

        assert submissions == []
        failure = data["failures"][0]
        assert failure["code"] == "ambiguous_target"
        assert "left.inc" in failure["message"]
        assert "right.inc" in failure["message"]

    async def test_random_rule_reaches_an_included_component(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        deck = _factored_deck(work_dir)

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-random",
                    lint="off",
                    variations=[
                        {
                            "kind": "random",
                            "runs": 2,
                            "seed": 11,
                            "rules": [{"rule": "component", "target": "R1", "tolerance": 0.1}],
                        }
                    ],
                ),
                state_with_sim,
            )
        )

        assert data["completeness"]["produced"] == 2
        values = [
            _include_targets(path)[0].read_text().split("R1 in mid ")[1].split()[0]
            for path in sorted(submitted, key=lambda p: p.name)
        ]
        assert len(set(values)) == 2
        assert all(float(value) != 1000.0 for value in values)

    async def test_two_level_include_chain_resolves_and_is_rewired(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submitted: list[Path] = []
        _recording_simulator(monkeypatch, submitted)
        (work_dir / "core.inc").write_text(_CORE_INC)
        (work_dir / "mid.inc").write_text('.include "core.inc"\n')
        deck = _deck(
            work_dir / "nested.cir",
            '.include "mid.inc"\nV1 in 0 1\nX1 in out core\n.op\n.end\n',
        )

        data = _assert_schema(
            await handle_run_experiments(
                _args(
                    deck,
                    "include-nested",
                    lint="off",
                    variations=[{"kind": "assign", "assign": {"R1": ["7k"]}}],
                ),
                state_with_sim,
            )
        )

        assert data["outcome"] == "complete"
        staged_mid = _include_targets(submitted[0])[0]
        assert staged_mid.name == "case-0000__mid.inc"
        staged_core = _include_targets(staged_mid)[0]
        assert staged_core.name == "case-0000__core.inc"
        assert "R1 in mid 7k" in staged_core.read_text()


class TestReceiptWeight:
    """A receipt carries what the caller acts on; provenance is opt-in.

    Provenance was measured at 29% of an experiment receipt's bytes on a real
    fleet run — absolute paths repeated four ways and a digest per staged file,
    none of which a caller opens, because the analysis tools address runs by
    job_id.
    """

    async def test_provenance_is_absent_by_default_and_returned_on_request(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "weight.cir")

        lean = _assert_schema(await handle_run_experiments(_args(deck, "lean-1"), state_with_sim))
        source = lean["source"][0]
        assert "sha256" not in source
        assert "staged_deck" not in source
        assert "manifest" not in source
        assert "linter_version" not in source
        # The count survives, so "how many files were staged" is still answerable
        # without naming every one of them.
        assert source["staged_files"] >= 1
        # What the caller acts on is untouched.
        assert source["circuit"] == "dut"
        assert source["simulator"]

        full = _assert_schema(
            await handle_run_experiments(_args(deck, "full-1", provenance=True), state_with_sim)
        )
        full_source = full["source"][0]
        assert full_source["sha256"]
        assert full_source["staged_deck"]
        assert full_source["manifest"]
        assert full_source["linter_version"]

    async def test_dropping_provenance_actually_shrinks_the_payload(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "shrink.cir")

        lean = _assert_schema(
            await handle_run_experiments(_args(deck, "shrink-lean"), state_with_sim)
        )
        full = _assert_schema(
            await handle_run_experiments(
                _args(deck, "shrink-full", provenance=True), state_with_sim
            )
        )
        assert len(json.dumps(lean)) < len(json.dumps(full))

    async def test_run_fields_projects_rows_and_preserves_nesting(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_simulator(monkeypatch)
        deck = _deck(work_dir / "proj.cir")

        args = _args(
            deck,
            "proj-1",
            variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
            run_fields=["case_id", "assignments"],
        )
        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        rows = data["runs"]["items"]
        assert rows, "expected the sweep to produce runs"
        for row in rows:
            assert set(row) == {"case_id", "assignments"}
            # Nesting is preserved, not flattened.
            assert isinstance(row["assignments"], dict)

    async def test_asking_for_provenance_replays_rather_than_conflicting(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Verbosity chooses how a receipt reads, not what runs.

        If it reached the idempotency fingerprint, re-reading an existing
        receipt with the audit trail turned on would be rejected as a changed
        request — refusing the caller exactly when they want to see more.
        """
        submissions: list[str] = []
        fake_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "verbosity.cir")

        lean = _assert_schema(
            await handle_run_experiments(_args(deck, "verbosity-1"), state_with_sim)
        )
        again = _assert_schema(
            await handle_run_experiments(
                _args(deck, "verbosity-1", provenance=True, run_fields=["case_id"]),
                state_with_sim,
            )
        )

        assert again.get("error") is None, "a verbosity change is not a conflict"
        assert again["job_id"] == lean["job_id"], "expected a replay of the same job"
        assert len(submissions) == 1, "the replay must not re-run anything"
        assert again["source"][0]["sha256"], "the replay honours the new verbosity"

    async def test_a_live_include_is_reported_even_without_provenance(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An unprovable input is a disclosure, not provenance trivia.

        Stripping the manifest wholesale would turn "this job cannot prove what
        it ran against" into silence, which is the one thing the lean receipt
        must not do.
        """
        fake_simulator(monkeypatch)
        outside = work_dir.parent / "outside_core.inc"
        outside.write_text("R9 in 0 1k\n")
        # A stageable include alongside the live one: without both, "keep only
        # the notable entries" is indistinguishable from "keep every entry".
        inside = work_dir / "inside_core.inc"
        inside.write_text("R8 in 0 2k\n")
        deck = _deck(
            work_dir / "live.cir",
            f".include {inside}\n.include {outside}\nV1 in 0 1\n.op\n.end\n",
        )

        data = _assert_schema(
            await handle_run_experiments(
                _args(deck, "live-1", allow_live_includes=True), state_with_sim
            )
        )
        entries = [e for src in data["source"] for e in src.get("manifest", [])]
        assert entries, "the live include must still be disclosed"
        assert all(e["live"] or e["reason"] for e in entries), (
            "a lean receipt keeps only manifest entries that say something"
        )
        assert not any(str(inside) == e["path"] for e in entries), (
            "the ordinary staged include is bulk, not a disclosure"
        )

    def test_the_fingerprint_covers_exactly_the_execution_arguments(self):
        """A new presentation field must not silently invalidate stored receipts.

        A canonicalizer bump belongs only to a changed representation of a
        previously valid request. Pinning the covered key set makes a real
        execution-field change fail here instead of silently changing replay.
        """
        from ltspice_mcp.lib.experiment_runner import canonical_fingerprint

        model = RunExperimentsInput.model_validate(
            {"request_id": "r", "circuits": [{"path": "/tmp/a.cir", "id": "d"}]}
        )
        covered = model.model_dump(
            mode="json",
            exclude_unset=False,
            exclude=RunExperimentsInput.PRESENTATION_FIELDS,
        )
        assert set(covered) == {
            "request_id",
            "circuits",
            "variations",
            "execution",
            "analyze",
            "lint",
            "suppress",
            "allow_live_includes",
        }
        loud = RunExperimentsInput.model_validate(
            {
                "request_id": "r",
                "circuits": [{"path": "/tmp/a.cir", "id": "d"}],
                "provenance": True,
                "run_fields": ["case_id"],
            }
        )
        assert canonical_fingerprint(model) == canonical_fingerprint(loud)

    def test_day_one_presentation_fields_leave_old_canonical_bytes_unchanged(self):
        assert experiment_store.CANONICALIZER_VERSION == 3
        model = RunExperimentsInput.model_validate(
            {
                "request_id": "stable-bytes",
                "circuits": [{"path": "/tmp/a.cir", "id": "d"}],
                "analyze": {
                    "recipes": [{"key": "summary", "metric": "summary"}],
                    "include": {"per_run": {"limit": 3}},
                },
            }
        )
        current = model.model_dump(
            mode="json",
            exclude_unset=False,
            exclude=RunExperimentsInput.PRESENTATION_FIELDS,
        )
        legacy = model.model_dump(mode="json", exclude_unset=False)
        legacy.pop("budget")
        legacy["analyze"]["include"].pop("fields")
        legacy["execution"].pop("wait_s")
        legacy.pop("provenance")
        legacy.pop("run_fields")

        def encode(value: Any) -> bytes:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()

        assert encode(current) == encode(legacy)

    def test_budget_and_attached_fields_do_not_change_the_fingerprint(self):
        from ltspice_mcp.lib.experiment_runner import canonical_fingerprint

        base = {
            "request_id": "presentation-only",
            "circuits": [{"path": "/tmp/a.cir", "id": "d"}],
            "analyze": {
                "recipes": [{"key": "summary", "metric": "summary"}],
                "include": {"per_run": {"limit": 3}},
            },
        }
        lean = RunExperimentsInput.model_validate({**base, "budget": 500})
        roomy = RunExperimentsInput.model_validate({**base, "budget": 5000})
        wide = RunExperimentsInput.model_validate(
            {
                **base,
                "budget": 5000,
                "analyze": {
                    **base["analyze"],
                    "include": {
                        "per_run": {"limit": 3},
                        "fields": ["value"],
                    },
                },
            }
        )
        assert canonical_fingerprint(lean) == canonical_fingerprint(roomy)
        assert canonical_fingerprint(lean) == canonical_fingerprint(wide)

        without_include = RunExperimentsInput.model_validate(
            {
                "request_id": "presentation-only-container",
                "circuits": [{"path": "/tmp/a.cir", "id": "d"}],
                "analyze": {"recipes": [{"key": "summary", "metric": "summary"}]},
            }
        )
        fields_only = RunExperimentsInput.model_validate(
            {
                "request_id": "presentation-only-container",
                "circuits": [{"path": "/tmp/a.cir", "id": "d"}],
                "analyze": {
                    "recipes": [{"key": "summary", "metric": "summary"}],
                    "include": {"fields": ["value"]},
                },
            }
        )
        assert canonical_fingerprint(without_include) == canonical_fingerprint(fields_only)

    def test_the_fingerprint_ignores_the_dwell_but_not_the_rest_of_execution(self):
        """execution.wait_s bounds only the response (the job is durable either
        way), so the same experiment asked at a different dwell must REPLAY —
        the CLI on-ramp submits with wait_s=0 and its receipt must stay
        replayable by an explicit run-experiments call at the default dwell.
        The rest of execution changes what runs and must keep conflicting."""
        from ltspice_mcp.lib.experiment_runner import canonical_fingerprint

        base = {"request_id": "r", "circuits": [{"path": "/tmp/a.cir", "id": "d"}]}
        quick = RunExperimentsInput.model_validate({**base, "execution": {"wait_s": 0}})
        patient = RunExperimentsInput.model_validate({**base, "execution": {"wait_s": 60}})
        other_engine = RunExperimentsInput.model_validate(
            {**base, "execution": {"wait_s": 0, "simulator": "ngspice"}}
        )
        assert canonical_fingerprint(quick) == canonical_fingerprint(patient)
        assert canonical_fingerprint(quick) != canonical_fingerprint(other_engine)

    def test_a_legacy_job_source_omits_provenance_it_never_had(self):
        """Empty-string digests and an empty manifest say nothing, at a cost.

        A non-experiment job stages nothing, so it has no digest and no
        manifest. Absence states that; placeholders spend bytes to state it
        while looking like real provenance.
        """
        job = make_sim_job(netlist=Path("/tmp/legacy.cir"), simulator="ngspice")
        payload = experiments_mod._legacy_source(job, dialect="ngspice")

        assert set(payload) == {"circuit", "path", "simulator", "dialect"}

    def test_the_lean_manifest_filter_fails_closed(self):
        """An entry that is neither staged, live, nor explained is an anomaly.

        The store rebuilds `staged` with a False default, so this state is
        reachable from a record written by an older or partial writer. A filter
        listing known-bad states would hide it; one that keeps everything except
        an ordinary staged reference cannot.
        """
        from ltspice_mcp.lib.experiment_types import ManifestEntry, SourceRecord

        ordinary = ManifestEntry(path=Path("a.inc"), sha256="x", staged=True, live=False)
        anomalous = ManifestEntry(path=Path("b.inc"), sha256="", staged=False, live=False)
        record = SourceRecord(
            circuit="dut",
            path=Path("dut.cir"),
            sha256="x",
            staged_deck=Path("staged/dut.cir"),
            manifest=[ordinary, anomalous],
        )

        lean = experiments_mod._source_payload(record, provenance=False)

        kept = [entry["path"] for entry in lean.get("manifest", [])]
        assert str(anomalous.path) in kept, "an unexplained un-staged entry must survive"
        assert str(ordinary.path) not in kept, "an ordinary staged entry is bulk"
        assert lean["staged_files"] == 2, "the count still covers every entry"
