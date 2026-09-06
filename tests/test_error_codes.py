"""The error codes a client sees are a typed contract, not message matching."""

from __future__ import annotations

import ast
import asyncio
import errno
import importlib
import os
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import ltspice_mcp
from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    LTSpiceMCPError,
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
from ltspice_mcp.lib.schematic_ops import make_editor
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.jobs import _jobs_error_details
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
    assert result.structured_content is not None
    return result.structured_content


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
        assert failures["m1"].code == "source_drift"

    async def test_real_deadline_still_reports_analysis_deadline(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _copy_raw(work_dir, "probe.raw")
        loop = asyncio.get_running_loop()
        with services.analysis_deadline(loop.time() - 1.0):
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
        assert failures["m1"].code == "analysis_deadline"

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
            make_editor(missing)
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
            make_editor(sheet)
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


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

# Constructors whose FIRST positional argument is the wire code, so the scan
# below reads codes out of them. The list names constructors, never codes: the
# codes themselves are always read from the source.
_CODE_FIRST_CONSTRUCTORS = frozenset(
    {
        "SourceFault",
        "_JobsActionError",
        "_QueryError",
        "VariationError",
        "DeckStagingError",
        "MismatchPlanError",
    }
)


def _string_literals(node: ast.AST) -> set[str]:
    """Every string a value expression can evaluate to, where that is decidable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _string_literals(node.body) | _string_literals(node.orelse)
    if isinstance(node, ast.BoolOp):
        return set().union(*(_string_literals(value) for value in node.values))
    return set()


def _classifier_returns(tree: ast.AST) -> set[ast.Return]:
    """Tuple returns of functions whose first parameter is an exception.

    ``_circuit_error(exc, ...)`` and ``_jobs_error_details(exc)`` answer with
    ``(code, stage, ...)``. Keying on the parameter name keeps the scan off the
    many other functions in these modules that return a tuple whose first member
    is a plain string (a field name, a unit).
    """
    out: set[ast.Return] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args.posonlyargs + node.args.args
        if not args or args[0].arg != "exc":
            continue
        out.update(child for child in ast.walk(node) if isinstance(child, ast.Return))
    return out


def _codes_in_sources() -> dict[str, set[str]]:
    """Read the emitted codes out of the whole package rather than listing them.

    Every module, not a hand-kept list of directories: a code's membership in
    the frozen vocabulary has to follow where it is EMITTED, not which file it
    happened to be typed in. Scanning only ``tools/`` let seven client-visible
    codes land in ``api/`` and ``lib/`` with no gate at all.
    """
    found: dict[str, set[str]] = {}
    package_dir = Path(ltspice_mcp.__file__).parent
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classifier_returns = _classifier_returns(tree)
        for node in ast.walk(tree):
            values: set[str] = set()
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "code":
                        values |= _string_literals(value)
            elif isinstance(node, ast.keyword) and node.arg == "code":
                values |= _string_literals(node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        target.id == "code" or target.id.endswith("_code")
                    ):
                        values |= _string_literals(node.value)
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name in _CODE_FIRST_CONSTRUCTORS and node.args:
                    values |= _string_literals(node.args[0])
            elif (
                isinstance(node, ast.Return)
                and node in classifier_returns
                and isinstance(node.value, ast.Tuple)
                and node.value.elts
            ):
                values |= _string_literals(node.value.elts[0])
            for value in values:
                found.setdefault(value, set()).add(str(path.relative_to(package_dir)))
    return found


def _error_classes() -> list[type[LTSpiceMCPError]]:
    """Every error class in the package, with every module imported first."""
    for module in pkgutil.walk_packages(ltspice_mcp.__path__, "ltspice_mcp."):
        if module.name.endswith("__main__"):
            continue
        importlib.import_module(module.name)

    def descend(cls: type[LTSpiceMCPError]):
        for sub in cls.__subclasses__():
            yield sub
            yield from descend(sub)

    return [LTSpiceMCPError, *descend(LTSpiceMCPError)]


# Every error code the server can emit: the `code` on each error class, plus the
# codes built into the failure and observation records anywhere in the package.
# The scan follows the whole tree rather than a list of directories, so a code's
# membership follows where it is emitted — the tools, the analysis metrics, the
# experiment coordinator, the variation and staging engines, and the Python
# API's detached hand-off all reach a client through one of these.
#
# ADDING a code is fine — add it here in the same commit and the test passes.
# RENAMING or REMOVING one is a public, client-visible change: a caller that
# branches on `error.code` or reads an observation code has no way to notice,
# so it needs a CHANGELOG entry saying which code changed and what replaced it.
FROZEN_ERROR_CODES = (
    "ambiguous_inner_device",
    "ambiguous_instance_ref",
    "ambiguous_scale",
    "ambiguous_subckt",
    "ambiguous_target",
    "analysis_deadline",
    "analysis_failed",
    "artifact_publish_failed",
    "artifact_too_large",
    "asc_export_unavailable",
    "asymptotic_reading",
    "batch_job_error",
    "budget_not_met",
    "budget_truncated",
    "cancel_failed",
    "cancel_not_authorized",
    "cancel_unavailable",
    "cancelled",
    "case_cap",
    "case_cap_exceeded",
    "case_not_found",
    "circuits_empty",
    "clone_include_unsupported",
    "commit_failed",
    "completeness_mismatch",
    "constant_window",
    "control_write_injected",
    "dangling_request_index_replaced",
    "detached_owner",
    "device_op_points_absent",
    "downsampled",
    "duplicate_assignment_target",
    "duplicate_circuit_id",
    "error",
    "execution_failed",
    "expected_sha256_required",
    "experiment_index_invalid",
    "experiment_index_write_failed",
    "external_cancellation_requested",
    "extreme_value",
    "failures_truncated",
    "geometry_not_literal",
    "idempotency_conflict",
    "idempotent_replay",
    "include_unstaged",
    "inner_device_not_found",
    "inner_model_unresolved",
    "instance_not_found",
    "internal_error",
    "invalid_assignment_value",
    "invalid_at",
    "invalid_circuit_id",
    "invalid_cursor",
    "invalid_instance_param_target",
    "invalid_instance_target",
    "invalid_prefix",
    "invalid_query",
    "job_deadline",
    "job_not_found",
    "job_not_terminal",
    "jobs_failed",
    "kill_attempt_failed",
    "kill_unconfirmed",
    "kill_unconfirmed_capacity",
    "late_simulator_exit",
    "lf_integrator",
    "library_error",
    "lint_blocked",
    "live_include",
    "log_error",
    "log_unread",
    "max_points_not_applied",
    "meas_batch_abort",
    "meas_parse_error",
    "merged_corners",
    "missing_circuit_id",
    "missing_completion",
    "missing_required_raw",
    "model_missing",
    "multiple_random_variations",
    "nested_fet_unsupported",
    "netlist_invalid",
    "ngspice_lib_section",
    "no_axis",
    "non_bsim_inner_device",
    "non_finite",
    "non_minimum_phase",
    "not_found",
    "op_failed",
    "open_failed",
    "open_skipped",
    "order_disagreement",
    "overlapping_mismatch_rules",
    "owner_liveness_unknown",
    "param_namespace_collision",
    "parse_error",
    "path_denied",
    "phase_unwrapped",
    "plot_written",
    "post_commit_failed",
    "preexisting_mismatch_param",
    "process_owned_job",
    "random_nominal_unavailable",
    "raster_unavailable",
    "rational_fit",
    "raw_not_produced",
    "raw_path_without_deck_provenance",
    "read_error",
    "receipt_failed",
    "recipe_failed",
    "recipe_invalid",
    "request_gate_busy",
    "result_unreadable",
    "review_against_plot",
    "revision_conflict",
    "run_not_found",
    "search_error",
    "server_restarted",
    "server_shutdown",
    "simulation_failed",
    "solve_failure",
    "source_drift",
    "source_modified_after_staging",
    "source_not_found",
    "source_unavailable",
    "source_unavailable_after_staging",
    "sparse_sweep",
    "step_axis_unioned",
    "step_value_unavailable",
    "subckt_unresolved",
    "submission_committed",
    "submission_failed",
    "symbol_not_found",
    "symbol_unresolved",
    "transport_delay",
    "unencodable_device_ref",
    "unmet_request",
    "unpersisted_runs_recovered",
    "unplanned_instance",
    "unsupported_file",
    "unsupported_variant",
    "unwrap_warning",
    "widget_delivered",
    "widget_unavailable",
    "window_empty_steps",
    "windows_native_storage_unavailable",
)


class TestErrorCodeVocabulary:
    def test_every_error_class_declares_its_own_code(self):
        missing = [cls.__name__ for cls in _error_classes() if "code" not in vars(cls)]
        assert not missing, (
            "These error classes inherit their wire code instead of naming one: "
            f'{sorted(missing)}. Declare `code = "..."` on each, and add it to '
            "FROZEN_ERROR_CODES."
        )

    def test_error_class_codes_are_distinct(self):
        by_code: dict[str, list[str]] = {}
        for cls in _error_classes():
            by_code.setdefault(vars(cls)["code"], []).append(cls.__name__)
        collisions = {code: names for code, names in by_code.items() if len(names) > 1}
        assert not collisions, f"One code, two meanings: {collisions}"

    def test_vocabulary_matches_the_frozen_list(self):
        emitted = set(_codes_in_sources()) | {vars(cls)["code"] for cls in _error_classes()}
        frozen = set(FROZEN_ERROR_CODES)
        added = sorted(emitted - frozen)
        gone = sorted(frozen - emitted)
        assert not added, (
            f"New error codes are not in FROZEN_ERROR_CODES: {added}. Adding a code "
            "is fine — list it there (sorted) in the same commit."
        )
        assert not gone, (
            f"These error codes are no longer emitted: {gone}. Renaming or removing "
            "a code is a client-visible change: give it a CHANGELOG entry naming the "
            "old code and its replacement, then update FROZEN_ERROR_CODES."
        )

    def test_frozen_list_is_sorted_and_unique(self):
        assert list(FROZEN_ERROR_CODES) == sorted(set(FROZEN_ERROR_CODES))
