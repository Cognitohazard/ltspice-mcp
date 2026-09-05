"""The error codes a client sees are a typed contract, not message matching."""

from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    NetlistError,
    NoAxisError,
    ResultError,
    SymbolResolutionError,
    raise_site_code,
)
from ltspice_mcp.lib import result_store, services
from ltspice_mcp.lib.deck_staging import DeckStagingError
from ltspice_mcp.lib.experiment_runner import CancelNotAuthorized, ExperimentCancellationError
from ltspice_mcp.lib.netlist_graph import PortArityMismatch, flatten_graph, parse_netlist_graph
from ltspice_mcp.lib.raw_parser import query_point_value
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.circuit import _make_editor
from ltspice_mcp.tools.experiments import _jobs_error_details
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


class TestCancellationCodeIsTyped:
    """A refusal to cancel is told by its type, not by the words it used."""

    def test_wording_alone_does_not_make_it_an_authorization_failure(self):
        # A coordinator that is gone is a different failure from a caller who
        # holds no authority, however the sentence is phrased.
        exc = ExperimentCancellationError(
            "Experiment job j1 is not owned by a live coordinator in this "
            "process, so it is not authorized to stop its cases"
        )
        assert _jobs_error_details(exc)[0] == "cancel_failed"

    def test_authorization_refusal_keeps_its_code_when_reworded(self):
        exc = CancelNotAuthorized("cancellation refused: the control token does not match")
        assert _jobs_error_details(exc) == ("cancel_not_authorized", "cancellation", False)


class TestStageOutranksTheClassDefault:
    """A class code is the default; the stage that failed may still name it."""

    def test_class_default_is_not_a_raise_site_code(self):
        # ResultError.code exists, but nothing chose it for this failure, so
        # a handler reporting its own stage keeps that name (a result read
        # that fails while staging a deck is a submission failure).
        assert ResultError.code == "result_unreadable"
        assert raise_site_code(ResultError("unreadable")) is None

    def test_a_code_named_at_the_raise_site_wins(self):
        assert raise_site_code(DeckStagingError("include_missing", "no such include")) == (
            "include_missing"
        )


class TestSchematicDependencyIsTyped:
    """A missing schematic and a missing dependency are told apart structurally."""

    def test_missing_schematic_whose_name_mentions_asy_is_file_not_found(
        self, work_dir: Path, asc_symbols: Path
    ):
        missing = work_dir / "opamp.asy.asc"
        with pytest.raises(NetlistError) as exc:
            _make_editor(missing)
        assert not isinstance(exc.value, SymbolResolutionError)
        assert "File not found" in str(exc.value)

    def test_missing_sub_sheet_is_a_dependency_failure(self, work_dir: Path, asc_symbols: Path):
        # A hierarchical block whose sheet is gone: the editor names the
        # missing .asc, so a ".asy" match blamed the schematic that opened fine.
        (work_dir / "myblock.asy").write_text(
            "Version 4\nSymbolType BLOCK\nPIN 0 0 LEFT 8\nPINATTR PinName A\n",
            encoding="utf-8",
        )
        sheet = work_dir / "top.asc"
        sheet.write_text(
            "Version 4\nSHEET 1 880 680\nSYMBOL myblock 0 0 R0\nSYMATTR InstName X1\n",
            encoding="utf-8",
        )
        with pytest.raises(SymbolResolutionError) as exc:
            _make_editor(sheet)
        assert "myblock.asc" in str(exc.value)


class TestNoAxisIsTyped:
    """An operating-point raw has no axis; that is a type, not a sentence."""

    def test_query_at_a_point_on_an_op_raw_raises_no_axis(self):
        from spicelib.raw.raw_read import RawRead

        raw = RawRead(str(FIXTURES_DIR / "op_extreme_node.raw"))
        with pytest.raises(NoAxisError):
            query_point_value(raw, raw.get_trace_names()[0], 0.0)


class TestPortArityIsTyped:
    """Recovering from an arity mismatch keys on the type, not the message."""

    def test_flatten_raises_the_arity_type(self):
        graph = parse_netlist_graph(
            "Vin in 0 5\nXU1 in out THREEPORT\n.subckt THREEPORT a b c\n"
            "R1 a b 1k\n.ends THREEPORT\n.end\n"
        )
        with pytest.raises(PortArityMismatch):
            flatten_graph(graph)
