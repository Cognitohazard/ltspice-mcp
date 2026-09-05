"""The error codes a client sees are a typed contract, not message matching."""

from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError
from ltspice_mcp.lib import result_store, services
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests.conftest import FIXTURES_DIR


def _copy_raw(work_dir: Path, name: str) -> Path:
    """Copy a recorded raw under ``name`` so the path text is under test control."""
    target = work_dir / name
    target.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes())
    return target


async def _analyze(
    state: SessionState, raw: Path, recipes: list[dict[str, Any]]
) -> dict[str, Any]:
    args = AnalyzeResultsInput.model_validate(
        {"sources": [{"raw_path": str(raw), "label": "dut"}], "recipes": recipes}
    )
    result = await handle_analyze_results(args, state)
    assert result.structuredContent is not None
    return result.structuredContent


def _out_of_quota(path: Path) -> str:
    raise OSError(errno.EDQUOT, os.strerror(errno.EDQUOT), str(path))


def _failure_codes(data: dict[str, Any]) -> set[str]:
    return {
        str(failure["code"]) for failure in data.get("failures", []) if isinstance(failure, dict)
    }


class TestAnalysisDeadlineIsTyped:
    """Only a real deadline is reported as ``analysis_deadline``."""

    async def test_case_cap_message_is_not_a_deadline(self):
        # A cap error says "exceeded" and is not a deadline. Classifying by
        # message text cannot tell the two apart; classifying by type can.
        cap = ResultError("Variation expansion exceeded the configured maximum of 64 cases")
        assert not isinstance(cap, AnalysisDeadlineExceeded)
        assert isinstance(AnalysisDeadlineExceeded("out of time"), ResultError)

    async def test_unreadable_source_named_deadline_is_source_drift(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The file name lands in the OSError text, so a source named after a
        # deadline study used to be reported as a deadline of our own.
        missing = work_dir / "deadline_probe.raw"
        failures = await analyze_mod._verify_direct_sources(
            [
                {
                    "manifest_id": "m1",
                    "raw_path": str(missing),
                    "log_path": None,
                    "composite_sha256": "0" * 64,
                }
            ],
            {"m1"},
            asyncio.get_running_loop().time() + 60.0,
            {},
        )
        assert failures["m1"].split(":", 1)[0] == "source_drift"

    async def test_real_deadline_still_reports_analysis_deadline(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _copy_raw(work_dir, "probe.raw")
        loop = asyncio.get_running_loop()
        source = services.resolve_analysis_source(
            SimpleNamespace(raw_file=str(raw), job_id=None, run_index=0), state_no_sim
        )
        with services.analysis_source_context(source, deadline=loop.time() - 1.0):
            failures = await analyze_mod._verify_direct_sources(
                [
                    {
                        "manifest_id": "m1",
                        "raw_path": str(raw),
                        "log_path": None,
                        "composite_sha256": "0" * 64,
                    }
                ],
                {"m1"},
                loop.time() + 60.0,
                {},
            )
        assert failures["m1"].split(":", 1)[0] == "analysis_deadline"

    async def test_manifest_digest_out_of_quota_is_source_unavailable(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # A full filesystem reports "Disk quota exceeded" (EDQUOT). That is a
        # storage fault, not a deadline, and the manifest must say so.
        raw = _copy_raw(work_dir, "probe.raw")
        monkeypatch.setattr(result_store, "sha256_file", _out_of_quota)
        data = await _analyze(state_no_sim, raw, [{"key": "summary", "metric": "summary"}])
        assert _failure_codes(data) == {"source_unavailable"}

    async def test_signal_named_deadline_is_recipe_failed(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The requested signal name is echoed into the not-found message, so a
        # node named after a timing study used to read as a deadline of ours.
        raw = _copy_raw(work_dir, "probe.raw")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "stats", "metric": "signal_stats", "signal": "V(deadline)"}],
        )
        assert _failure_codes(data) == {"recipe_failed"}

    async def test_artifact_publish_out_of_quota_is_publish_failure(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        raw = _copy_raw(work_dir, "probe.raw")
        real = result_store.sha256_file

        def _quota_on_artifacts(path: Path) -> str:
            # The sources hash fine; only publishing the CSV runs out of room.
            if path.suffix in {".raw", ".log"}:
                return real(path)
            return _out_of_quota(path)

        monkeypatch.setattr(result_store, "sha256_file", _quota_on_artifacts)
        data = await _analyze(
            state_no_sim,
            raw,
            [
                {
                    "key": "wave",
                    "metric": "waveform",
                    "signals": ["V(out)"],
                    "format": "csv",
                }
            ],
        )
        assert _failure_codes(data) == {"artifact_publish_failed"}
