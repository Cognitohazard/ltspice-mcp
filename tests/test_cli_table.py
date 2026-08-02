"""Human-readable CLI tables over the supported consolidated envelopes."""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import pytest
from mcp import types

from ltspice_mcp import cli


def _analysis(results: dict[str, Any], **updates: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "outcome": "complete",
        "coverage": {
            "runs_requested": 0,
            "runs_analyzed": 0,
            "missing_cases": {
                "items": [],
                "total": 0,
                "returned": 0,
                "truncated": False,
                "next_cursor": None,
            },
        },
        "results": results,
        "observations": [],
        "failures": [],
        "source_hashes": [],
        "result_set_id": "rs_table",
        "cursor": None,
        "next": None,
    }
    data.update(updates)
    return data


def _emit_table(capsys: pytest.CaptureFixture[str], command: str, data: dict[str, Any]) -> str:
    original = copy.deepcopy(data)
    namespace = argparse.Namespace(command=command, as_json=False, as_table=True)
    result = types.CallToolResult(content=[], structuredContent=data)

    cli.emit(namespace, result, cli.EXIT_OK)

    assert data == original, "table rendering must not rewrite the handler payload"
    return capsys.readouterr().out


def test_empty_analysis_table_keeps_the_deterministic_columns(capsys):
    out = _emit_table(capsys, "analyze-results", _analysis({}))

    assert all(column in out for column in ("recipe key", "group", "stat", "value", "attribution"))
    assert "(no rows)" in out


def test_partial_analysis_keeps_failure_fact_channels_below_the_table(capsys):
    data = _analysis(
        {
            "gain": {
                "metric": "value",
                "reduced": [],
                "warnings": ["measurement window was clamped"],
            }
        },
        outcome="partial",
        observations=[{"code": "coverage_gap", "detail": "one run was absent"}],
        failures=[
            {
                "code": "recipe_failed",
                "stage": "analyze",
                "where": "run-2",
                "message": "raw file is unavailable",
            }
        ],
        hint="Continue with the returned cursor.",
    )

    out = _emit_table(capsys, "analyze-results", data)

    assert "Facts:" in out
    assert "recipe_failed" in out
    assert "coverage_gap" in out
    assert "measurement window was clamped" in out
    assert "Continue with the returned cursor." in out


def test_multiple_analysis_recipes_flatten_groups_values_and_attribution(capsys):
    data = _analysis(
        {
            "gain": {
                "metric": "value",
                "reduced": [
                    {
                        "field": "magnitude_db",
                        "stat": "mean",
                        "value": 12.5,
                        "case_id": None,
                        "run_index": None,
                        "step_index": None,
                        "step_values": {},
                        "assignments": {},
                    }
                ],
                "groups": [
                    {
                        "by": {"corner": "fast"},
                        "reduced": [
                            {
                                "field": "magnitude_db",
                                "stat": "max",
                                "value": 13.1,
                                "case_id": "case-fast",
                                "run_index": 1,
                                "step_index": None,
                                "step_values": {},
                                "assignments": {"corner": "fast"},
                            }
                        ],
                        "count": 1,
                    }
                ],
                "warnings": [],
            },
            "settling": {
                "metric": "transient_response",
                "reduced": [],
                "values": [
                    {
                        "source": "dut",
                        "case_id": "case-0",
                        "run_index": 0,
                        "step_index": None,
                        "step_values": {},
                        "assignments": {"corner": "typ"},
                        "value": {"final_value": 1.02, "settling_time": 0.0004},
                    }
                ],
                "warnings": [],
            },
        }
    )

    out = _emit_table(capsys, "analyze-results", data)

    assert "gain" in out and "settling" in out
    assert "magnitude_db / mean" in out
    assert '"corner":"fast"' in out
    assert '"final_value":1.02' in out
    assert "case-0" in out and "case-fast" in out


def test_budget_columnar_analysis_rows_are_reexpanded_for_display(capsys):
    data = _analysis(
        {
            "gain": {
                "metric": "value",
                "reduced_columns": [
                    "field",
                    "stat",
                    "value",
                    "case_id",
                    "run_index",
                    "step_index",
                    "step_values",
                    "assignments",
                ],
                "reduced": [
                    ["magnitude_db", "min", 9.5, "case-low", 0, None, {}, {"vdd": 1.7}],
                    ["magnitude_db", "max", 11.2, "case-high", 1, None, {}, {"vdd": 1.9}],
                ],
                "warnings": [],
            }
        }
    )

    out = _emit_table(capsys, "analyze-results", data)

    assert "magnitude_db / min" in out and "magnitude_db / max" in out
    assert "9.5" in out and "11.2" in out
    assert "case-low" in out and "case-high" in out


def test_nested_on_ramp_analysis_uses_the_same_table_and_keeps_receipt_facts(capsys):
    result = _analysis(
        {
            "measurements": {
                "metric": "measurements",
                "reduced": [],
                "values": [
                    {
                        "source": "dut",
                        "case_id": "dut-0000",
                        "run_index": 0,
                        "value": {"vfinal": 0.9998},
                    }
                ],
                "warnings": [],
            }
        }
    )
    receipt = {
        "job_id": "exp_table",
        "request_id": "quick-check",
        "status": "completed",
        "outcome": "complete",
        "failures": [],
        "observations": [],
        "warnings": ["receipt warning"],
        "hint": "Receipt hint.",
        "analysis": {
            "status": "completed",
            "result": result,
            "error": None,
            "observations": [{"code": "analysis_note", "detail": "attached stage"}],
        },
    }

    out = _emit_table(capsys, "run", receipt)

    assert "measurements" in out and '"vfinal":0.9998' in out
    assert "exp_table" in out and "quick-check" in out
    assert "receipt warning" in out and "analysis_note" in out and "Receipt hint." in out


def test_jobs_list_table_uses_path_activity_status_columns_and_truncates_paths(capsys):
    long_path = "/workspace/" + "nested/" * 10 + "amplifier.cir"
    data = {
        "action": "list",
        "outcome": "complete",
        "items": [
            {
                "path": long_path,
                "exists": True,
                "last_activity": "2026-08-01T12:00:00Z",
                "status_counts": {"completed": 3},
                "interrupted_job_ids": [],
            }
        ],
        "total": 1,
        "returned": 1,
        "truncated": False,
        "next_cursor": None,
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": "Recent circuit groups are ordered by activity.",
    }

    out = _emit_table(capsys, "jobs", data)

    assert all(field in out for field in ("path", "last_activity", "status_counts"))
    assert "/workspace/" in out and "amplifier.cir" in out and "..." in out
    assert long_path not in out
    assert '"completed":3' in out


def test_empty_jobs_error_table_keeps_the_error_footer(capsys):
    data = {
        "action": "list",
        "outcome": "failed",
        "items": [],
        "total": 0,
        "returned": 0,
        "truncated": False,
        "next_cursor": None,
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": "Use a path inside the configured sandbox.",
        "error": {
            "code": "path_denied",
            "message": "The circuit path is outside the sandbox",
            "stage": "resolution",
            "retryable": False,
            "commit_state": "not_started",
        },
    }

    out = _emit_table(capsys, "jobs", data)

    assert "(no rows)" in out
    assert "Facts:" in out and "path_denied" in out
    assert "Use a path inside the configured sandbox." in out
    assert "Table view is unavailable" not in out


def test_jobs_runs_table_uses_run_record_columns_and_columnar_items(capsys):
    data = {
        "action": "runs",
        "outcome": "partial",
        "job_id": "exp_runs",
        "request_id": "sweep",
        "status": "completed_with_failures",
        "dialect": "ltspice",
        "items_columns": [
            "case_id",
            "run_index",
            "circuit",
            "assignments",
            "status",
            "raw",
            "log",
        ],
        "items": [
            ["case-0", 0, "dut", {"vdd": 1.8}, "produced", "/runs/case-0.raw", None],
            ["case-1", 1, "dut", {"vdd": 2.0}, "failed", None, "/runs/case-1.log"],
        ],
        "total": 2,
        "returned": 2,
        "truncated": False,
        "next_cursor": None,
        "observations": [],
        "warnings": [],
        "failures": [{"case_id": "case-1", "code": "run_failed", "message": "failed"}],
        "hint": "Returned all recorded runs.",
    }

    out = _emit_table(capsys, "jobs", data)

    assert all(field in out for field in ("case_id", "run_index", "assignments", "raw", "log"))
    assert "case-0" in out and "case-1" in out
    assert '"vdd":1.8' in out and "/runs/case-1.log" in out
    assert "run_failed" in out


def test_unsupported_table_envelope_falls_back_to_pretty_json_with_notice(capsys):
    data = {"outcome": "complete", "results": [{"kind": "capabilities"}]}

    out = _emit_table(capsys, "inspect", data)

    assert "Table view is unavailable for this response; showing JSON." in out
    assert json.dumps(data, ensure_ascii=False, indent=1) in out


@pytest.mark.parametrize("argv", [["--table", "jobs"], ["jobs", "--table"]])
def test_table_flag_parses_on_either_side_of_the_subcommand(argv):
    namespace = cli.parse_args(argv)

    assert namespace.as_table is True
    assert namespace.as_json is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "jobs", "--table"],
        ["--table", "jobs", "--json"],
    ],
)
def test_json_and_table_cross_boundary_combinations_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(argv)

    assert raised.value.code == cli.EXIT_REFUSED
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "--json and --table cannot be used together" in payload["error"]["message"]
    assert "spice-mcp jobs --action list --json" in captured.err
