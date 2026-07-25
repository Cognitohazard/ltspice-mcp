"""run_experiments receipt, lint, idempotency, and accounting contract."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock

import jsonschema
import pytest
from pydantic import ValidationError

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import experiments as experiments_mod
from ltspice_mcp.tools._base import _build_input_schema
from ltspice_mcp.tools.experiments import (
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    AnalysisPerRun,
    JobsInput,
    RunExperimentsInput,
    handle_jobs,
    handle_run_experiments,
)
from tests.conftest import FIXTURES_DIR


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


def test_variation_input_schema_is_inlined_and_discriminated():
    schema = _build_input_schema(RunExperimentsInput)
    variations = schema["properties"]["variations"]["items"]

    assert "$defs" not in schema
    assert '"$ref"' not in json.dumps(schema)
    assert variations["discriminator"]["propertyName"] == "kind"
    assert len(variations["oneOf"]) == 2
    branches = {branch["properties"]["kind"]["const"]: branch for branch in variations["oneOf"]}
    assert branches["assign"]["additionalProperties"] is False
    assert branches["assign"]["properties"]["combine"]["default"] == "grid"
    random_rules = branches["random"]["properties"]["rules"]["items"]
    assert random_rules["discriminator"]["propertyName"] == "rule"
    assert all(branch["additionalProperties"] is False for branch in random_rules["oneOf"])


def _deck(path: Path, body: str | None = None) -> Path:
    path.write_text(body or "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    return path


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


def _instant_simulator(
    monkeypatch: pytest.MonkeyPatch,
    submissions: list[str],
) -> None:
    def submit(self, _netlist: Path, run_filename: str, callback):
        submissions.append(run_filename)
        raw = self.output_folder / f"{Path(run_filename).stem}.raw"
        log = self.output_folder / f"{Path(run_filename).stem}.log"
        raw.write_bytes(b"Title: mock")
        log.write_text("ok")
        outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)


def _fixture_simulator(
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str = "ltspice_tran_rc",
) -> None:
    """Instant simulator that hands back a recorded raw the analyzers can read."""

    def submit(self, _netlist: Path, run_filename: str, callback):
        stem = Path(run_filename).stem
        raw = self.output_folder / f"{stem}.raw"
        log = self.output_folder / f"{stem}.log"
        shutil.copy(FIXTURES_DIR / f"{fixture_name}.raw", raw)
        shutil.copy(FIXTURES_DIR / f"{fixture_name}.log", log)
        outcome = RunOutcome(str(raw), str(log), raw.stat().st_size, None)
        self.loop.call_soon_threadsafe(callback, outcome)
        return object()

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)


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
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "quick.cir")

        result = await handle_run_experiments(
            _args(deck, "quick-complete"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert data["outcome"] == "complete"
        assert data["status"] == "completed"
        assert data["control_token"]
        assert data["completeness"]["produced"] == 1
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
        _instant_simulator(monkeypatch, submissions)

        def exploding_payload(*_args, **_kwargs):
            raise ValueError("payload exploded")

        monkeypatch.setattr(experiments_mod, "_job_payload", exploding_payload)
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


@pytest.mark.asyncio
class TestIdempotency:
    async def test_matching_replay_returns_token_and_observation(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "replay.cir")
        args = _args(deck, "same-payload")

        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        replay = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert replay["job_id"] == first["job_id"]
        assert replay["control_token"] == first["control_token"]
        assert any(item["code"] == "idempotent_replay" for item in replay["observations"])
        assert len(submissions) == 1

    async def test_different_payload_replay_conflicts(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "conflict.cir")
        await handle_run_experiments(
            _args(deck, "conflicting-payload"),
            state_with_sim,
        )

        result = await handle_run_experiments(
            _args(deck, "conflicting-payload", lint="off"),
            state_with_sim,
        )
        data = _assert_schema(result)

        assert result.isError
        assert data["error"]["code"] == "idempotency_conflict"
        assert "control_token" not in data
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
        _instant_simulator(monkeypatch, submissions)
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
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "removed.cir")
        args = _args(deck, "removed-deck")
        await handle_run_experiments(args, state_with_sim)
        deck.unlink()

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert data["error"]["code"] == "idempotency_conflict"
        assert "no longer readable" in data["error"]["message"]
        assert str(deck) in data["error"]["message"]
        assert len(submissions) == 1

    async def test_record_without_source_digests_conflicts(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A record predating recorded digests must not replay as a match.

        Treating an absent digest as agreement would leave the fix inert for
        exactly the jobs already sitting on disk.
        """
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "digestless.cir")
        args = _args(deck, "digestless-record")
        first = _assert_schema(await handle_run_experiments(args, state_with_sim))
        await state_with_sim.job_registry.drain_pending()

        record = experiment_store.record_path(first["job_id"], work_dir)
        stored = json.loads(record.read_text())
        for source in stored["sources"]:
            source["sha256"] = ""
            for entry in source["manifest"]:
                entry["sha256"] = ""
        record.write_text(json.dumps(stored))
        del state_with_sim.experiment_jobs[first["job_id"]]

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert data["error"]["code"] == "idempotency_conflict"
        assert "digest" in data["error"]["message"]
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
        _instant_simulator(monkeypatch, submissions)
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
        _instant_simulator(monkeypatch, submissions)
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

    async def test_off_skips_linter(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
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
        assert data["lint"] == [{"circuit": "dut", "findings": []}]


@pytest.mark.asyncio
class TestPerCircuitFailuresAndAccounting:
    async def test_multi_circuit_failure_is_isolated_and_ordered(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
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
        _instant_simulator(monkeypatch, submissions)
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
        _instant_simulator(monkeypatch, submissions)
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
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "recent.cir")
        note = AsyncMock()
        monkeypatch.setattr(state_with_sim, "note_recent_circuit", note)

        result = await handle_run_experiments(
            _args(deck, "recent-note"),
            state_with_sim,
        )
        _assert_schema(result)

        note.assert_awaited_once_with(deck.resolve())


@pytest.mark.asyncio
class TestAttachedAnalysis:
    """The analyze block runs on the real recipe engine, not a stub."""

    async def test_recipes_and_group_by_run_end_to_end(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _fixture_simulator(monkeypatch)
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

    async def test_successful_analysis_does_not_report_partial(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _fixture_simulator(monkeypatch)
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
        _fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-bad.cir")
        # run_experiments takes the analyze block as free-form JSON, so a
        # request analyze_results rejects (here: a repeated group_by dimension)
        # only fails once the stage runs. The run accounting must not move.
        analyze = {
            "recipes": [{"key": "vout", "metric": "value", "expr": "V(out)", "at": "900u"}],
            "group_by": ["R1", "R1"],
        }

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
        assert "not a valid analyze_results request" in data["analysis"]["error"]
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
        _fixture_simulator(monkeypatch)
        deck = _deck(work_dir / "attached-wait.cir")
        released = asyncio.Event()
        engine = analyze_mod.handle_analyze_results

        async def gated(args, state):
            await released.wait()
            return await engine(args, state)

        monkeypatch.setattr(analyze_mod, "handle_analyze_results", gated)

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
        _instant_simulator(monkeypatch, submissions)
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
