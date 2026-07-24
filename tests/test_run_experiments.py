"""run_experiments receipt, lint, idempotency, and accounting contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import jsonschema
import pytest

from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import _build_input_schema
from ltspice_mcp.tools.experiments import (
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    RunExperimentsInput,
    handle_run_experiments,
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

    async def test_attached_analysis_request_is_persisted_and_stubbed(
        self,
        state_with_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        submissions: list[str] = []
        _instant_simulator(monkeypatch, submissions)
        deck = _deck(work_dir / "analysis.cir")
        args = _args(
            deck,
            "analysis-stub",
            analyze={"recipes": [{"kind": "summary", "key": "summary"}]},
        )

        data = _assert_schema(await handle_run_experiments(args, state_with_sim))

        assert data["analysis"]["request"]["recipes"][0]["kind"] == "summary"
        assert data["analysis"]["status"] == "failed"
        assert "not yet available" in data["analysis"]["error"]
