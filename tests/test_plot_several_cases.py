"""plot_waveform draws several cases of one job in one chart.

A point is one simulated circuit: a case, and inside it a step where the deck
has ``.step``. Every trace of a chart of several cases names its point, in its
legend label and in its summary, and the reply names each selected case it did
not draw. ``analyze_results`` reads the same cases for the same selection.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import now, services
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests import _source_scan
from tests.conftest import stage_recorded_fixture


@dataclass
class _Case:
    fixture: str | None
    assignments: dict[str, Any] = field(default_factory=dict)
    status: str = "produced"


async def _job(state: SessionState, work_dir: Path, job_id: str, cases: list[_Case]) -> ExperimentJob:
    """A finished experiment job whose cases read the recorded fixtures named,
    each staged in a folder of its own; a case with no fixture produced no raw."""
    deck = state.working_dir / f"{job_id}.cir"
    deck.write_text(".tran 1m\n.end\n", encoding="utf-8")
    built: list[ExperimentCase] = []
    for index, spec in enumerate(cases):
        raw: Path | None = None
        if spec.fixture is not None:
            folder = work_dir / f"{job_id}-{index}"
            folder.mkdir()
            raw = stage_recorded_fixture(folder, spec.fixture)
        built.append(
            ExperimentCase(
                case_id=f"case-{index:04d}",
                run_index=index,
                circuit="dut",
                circuit_path=deck,
                staged_deck=deck,
                deck_sha256="a" * 64,
                assignments=dict(spec.assignments),
                status=spec.status,  # type: ignore[arg-type]
                raw_file=raw,
                log_file=raw.with_suffix(".log") if raw is not None else None,
                error=None if raw is not None else "the simulator stopped",
                failure_code=None if raw is not None else "simulator_error",
            )
        )
    count = len(built)
    produced = sum(1 for case in built if case.status == "produced")
    job = ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="fingerprint",
        canonicalizer_version=1,
        control_token="control-token",
        store_path=state.working_dir / f"{job_id}.json",
        cases=built,
        sources=[
            SourceRecord(
                circuit="dut", path=deck, sha256="a" * 64, staged_deck=deck, simulator="ltspice"
            )
        ],
        simulator="LTspice",
        completeness=Completeness(
            declared=count, expanded=count, submitted=count, produced=produced
        ),
        status="completed" if produced == count else "partial",  # type: ignore[arg-type]
        completed_at=now(),
    )
    # How each case's results are stepped, recorded as its run's end records
    # it (ExperimentRunner._end_of_run).
    for case in job.cases:
        case.steps = await services.read_steps(job, case, state)
    state.add_experiment_job(job, already_persisted=True)
    return job


async def _plot(state: SessionState, **kwargs: Any) -> dict:
    kwargs.setdefault("open", False)
    result = await handle_plot_waveform(PlotWaveformInput(**kwargs), state)
    assert result.structured_content is not None
    return result.structured_content


def _labels(data: dict) -> list[str]:
    html = Path(data["path"]).read_text(encoding="utf-8")
    start = html.index('type="application/json">') + len('type="application/json">')
    blob = json.loads(html[start : html.index("</script>", start)].replace("<\\/", "</"))
    return [series["label"] for panel in blob["panels"] for series in panel["series"]]


async def _three_rc(state: SessionState, work_dir: Path, job_id: str = "sweep") -> ExperimentJob:
    return await _job(
        state,
        work_dir,
        job_id,
        [
            _Case("ltspice_tran_rc", {"R1": 1000, "C1": 1e-7}),
            _Case("ltspice_tran_rc", {"R1": 2000, "C1": 1e-7}),
            _Case("ltspice_tran_rc", {"R1": 3000, "C1": 1e-7}),
        ],
    )


@pytest.mark.asyncio
class TestSeveralCases:
    async def test_three_cases_are_three_series_labelled_by_what_differs(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _three_rc(state_no_sim, work_dir)
        data = await _plot(state_no_sim, job_id="sweep", runs="all", signals=["V(out)"])

        assert data["series_count"] == 3
        # C1 is the same in every case drawn, so only R1 tells them apart.
        assert _labels(data) == ["V(out) [R1=1000]", "V(out) [R1=2000]", "V(out) [R1=3000]"]
        assert [(t["case_id"], t["run_index"]) for t in data["traces"]] == [
            ("case-0000", 0),
            ("case-0001", 1),
            ("case-0002", 2),
        ]
        assert [t["label"] for t in data["traces"]] == _labels(data)
        assert data["cases"] == [
            {
                "case_id": f"case-{i:04d}",
                "run_index": i,
                "assignments": {"R1": r, "C1": 1e-7},
                "n_steps": 1,
            }
            for i, r in enumerate((1000, 2000, 3000))
        ]

    async def test_a_list_of_run_indices_and_case_ids_select_the_same_cases(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _three_rc(state_no_sim, work_dir)
        by_index = await _plot(state_no_sim, job_id="sweep", runs=[0, 2], signals=["V(out)"])
        by_id = await _plot(
            state_no_sim,
            job_id="sweep",
            runs={"case_ids": ["case-0000", "case-0002"]},
            signals=["V(out)"],
        )
        for data in (by_index, by_id):
            assert _labels(data) == ["V(out) [R1=1000]", "V(out) [R1=3000]"]

    async def test_a_stepped_case_is_labelled_by_its_assignment_and_its_step(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "mixed",
            [
                _Case("ltspice_tran_rc", {"C1": 1e-7}),
                _Case("ltspice_step_tran", {"C1": 2e-7}),
            ],
        )
        data = await _plot(state_no_sim, job_id="mixed", runs="all", signals=["V(out)"])

        assert _labels(data) == [
            "V(out) [C1=1e-07]",
            "V(out) [C1=2e-07;r=1]",
            "V(out) [C1=2e-07;r=22]",
            "V(out) [C1=2e-07;r=680]",
        ]
        assert [(t["case_id"], t.get("step")) for t in data["traces"]] == [
            ("case-0000", None),
            ("case-0001", 0),
            ("case-0001", 1),
            ("case-0001", 2),
        ]
        assert [case["n_steps"] for case in data["cases"]] == [1, 3]

    async def test_cases_with_the_same_assignments_are_labelled_by_their_case_id(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "same",
            [_Case("ltspice_tran_rc", {"R1": 1}), _Case("ltspice_tran_rc", {"R1": 1})],
        )
        data = await _plot(state_no_sim, job_id="same", runs="all", signals=["V(out)"])
        assert _labels(data) == ["V(out) [case-0000]", "V(out) [case-0001]"]

    async def test_a_step_applies_to_every_case_and_a_case_without_it_is_named(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "steps",
            [
                _Case("ltspice_step_tran", {"C1": 1e-7}),
                _Case("ltspice_step_tran", {"C1": 2e-7}),
                _Case("ltspice_tran_rc", {"C1": 3e-7}),
            ],
        )
        data = await _plot(
            state_no_sim,
            job_id="steps",
            runs="all",
            signals=["V(out)"],
            step={"axis": "r", "value": "22"},
        )
        assert _labels(data) == ["V(out) [C1=1e-07;r=22]", "V(out) [C1=2e-07;r=22]"]
        (left_out,) = data["cases_not_drawn"]
        assert left_out["case_id"] == "case-0002"
        assert left_out["code"] == "step_not_found"

    async def test_a_failed_case_is_named_and_the_others_are_drawn(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "partial",
            [
                _Case("ltspice_tran_rc", {"R1": 1000}),
                _Case(None, {"R1": 2000}, status="failed"),
                _Case("ltspice_tran_rc", {"R1": 3000}),
            ],
        )
        data = await _plot(state_no_sim, job_id="partial", runs="all", signals=["V(out)"])

        assert _labels(data) == ["V(out) [R1=1000]", "V(out) [R1=3000]"]
        (failed,) = data["cases_not_drawn"]
        assert failed["case_id"] == "case-0001"
        assert failed["run_index"] == 1
        assert failed["status"] == "failed"
        assert failed["code"] == "raw_not_produced"

    async def test_a_case_not_yet_run_is_named_and_the_finished_are_drawn(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        job = await _job(
            state_no_sim,
            work_dir,
            "running",
            [
                _Case("ltspice_tran_rc", {"R1": 1000}),
                _Case(None, {"R1": 2000}, status="queued"),
                _Case("ltspice_tran_rc", {"R1": 3000}),
            ],
        )
        job.status = "running"  # type: ignore[assignment]
        data = await _plot(state_no_sim, job_id="running", runs="all", signals=["V(out)"])

        assert _labels(data) == ["V(out) [R1=1000]", "V(out) [R1=3000]"]
        (waiting,) = data["cases_not_drawn"]
        assert (waiting["case_id"], waiting["status"], waiting["code"]) == (
            "case-0001",
            "queued",
            "run_not_finished",
        )

    async def test_a_raw_that_cannot_be_read_costs_only_its_case(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        job = await _three_rc(state_no_sim, work_dir, "broken")
        broken = job.cases[1].raw_file
        assert broken is not None
        whole = broken.read_bytes()
        broken.write_bytes(whole[: len(whole) // 3])

        data = await _plot(state_no_sim, job_id="broken", runs="all", signals=["V(out)"])

        assert _labels(data) == ["V(out) [R1=1000]", "V(out) [R1=3000]"]
        (unread,) = data["cases_not_drawn"]
        assert unread["case_id"] == "case-0001"
        assert unread["code"] == "result_unreadable"
        assert "truncated" in unread["detail"]
        assert unread["detail"]

    async def test_a_runs_list_far_longer_than_the_job_gives_a_bounded_reply(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A list of run indices has no bound of its own: the runs the job does
        not hold are counted in one row, and the reply never repeats the list."""
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_TRACE_SUMMARY_MAX", 1)
        await _three_rc(state_no_sim, work_dir, "short")
        data = await _plot(
            state_no_sim, job_id="short", runs=list(range(3000)), signals=["V(out)"]
        )
        (missing,) = data["cases_not_drawn"]
        assert missing["code"] == "run_not_found"
        assert missing["count"] == 2997
        assert missing["run_indices"] == [3, 4, 5, 6, 7]
        assert "3000 run indices" in data["hint"]
        assert len(json.dumps(data)) < 6_000

    async def test_many_cases_not_drawn_are_counted_past_the_rows_listed(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        cases = [_Case("ltspice_tran_rc", {"R1": 1})] + [
            _Case(None, {"R1": i}, status="failed") for i in range(2, 42)
        ]
        await _job(state_no_sim, work_dir, "wide", cases)
        data = await _plot(state_no_sim, job_id="wide", runs=list(range(5000)), signals=["V(out)"])
        assert len(data["cases_not_drawn"]) == 20
        # 40 failed cases, and one row for the 4959 runs the job does not hold.
        assert data["cases_not_drawn_truncated"] == 41
        assert "coverage.missing_cases" in data["hint"]
        assert len(json.dumps(data)) < 12_000

        await _job(
            state_no_sim, work_dir, "dead_wide", [_Case(None, status="failed") for _ in range(40)]
        )
        with pytest.raises(ResultError) as refused:
            await _plot(
                state_no_sim, job_id="dead_wide", runs=list(range(5000)), signals=["V(out)"]
            )
        assert len(str(refused.value)) < 2_000
        assert "5000 run indices" in str(refused.value)

    async def test_a_selection_with_nothing_to_draw_is_refused(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "dead",
            [_Case(None, status="failed"), _Case(None, status="failed")],
        )
        with pytest.raises(ResultError, match=r"case-0000.*case-0001"):
            await _plot(state_no_sim, job_id="dead", runs="all", signals=["V(out)"])

    async def test_a_refusal_of_many_cases_names_a_few_and_counts_the_rest(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(state_no_sim, work_dir, "many", [_Case(None, status="failed") for _ in range(9)])
        with pytest.raises(ResultError) as refused:
            await _plot(state_no_sim, job_id="many", runs="all", signals=["V(out)"])
        message = str(refused.value)
        assert "case-0004" in message and "case-0005" not in message
        assert "showing 5 of 9 rows" in message
        assert 'runs="all"' in message and "coverage.missing_cases" in message

    async def test_cases_of_different_analyses_are_refused_by_name(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "kinds",
            [_Case("ltspice_tran_rc", {"R1": 1}), _Case("ltspice_ac_rc", {"R1": 2})],
        )
        with pytest.raises(ResultError) as refused:
            await _plot(state_no_sim, job_id="kinds", runs="all", signals=["V(out)"])
        message = str(refused.value)
        assert "case-0000" in message and "case-0001" in message
        assert "separate" in message

    async def test_the_summary_limit_counts_series_across_cases(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import ltspice_mcp.tools.analysis as mod

        monkeypatch.setattr(mod, "_TRACE_SUMMARY_MAX", 2)
        await _three_rc(state_no_sim, work_dir)
        data = await _plot(state_no_sim, job_id="sweep", runs="all", signals=["V(out)"])

        assert len(data["traces"]) == 2
        assert data["traces_truncated"] == 3
        # The case whose trace is not summarized is counted, never passed over.
        assert data["cases_unsummarized"] == 1
        # The hint passes the selection on as it was given, never every case id.
        assert 'runs="all"' in data["hint"]
        assert "case-0000" not in data["hint"]

    async def test_the_cell_limit_counts_series_across_cases(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import ltspice_mcp.tools.analysis as mod

        await _three_rc(state_no_sim, work_dir)
        one = await _plot(state_no_sim, job_id="sweep", run_index=0, signals=["V(out)"])
        points = one["points_per_series"][0]
        # One case draws 2 rows (axis and one series); three cases draw 4.
        monkeypatch.setattr(mod, "_PLOT_MAX_CELLS", 3 * points)
        await _plot(state_no_sim, job_id="sweep", run_index=0, signals=["V(out)"])
        with pytest.raises(ResultError, match="cells"):
            await _plot(state_no_sim, job_id="sweep", runs="all", signals=["V(out)"])

    async def test_in_ltspice_with_two_cases_is_refused(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _three_rc(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="one case"):
            await _plot(
                state_no_sim, job_id="sweep", runs=[0, 1], signals=["V(out)"], in_ltspice=True
            )

    async def test_runs_beside_run_index_or_case_id_or_path_is_refused(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        with pytest.raises(ValidationError, match=r"runs.*run_index"):
            PlotWaveformInput(job_id="sweep", runs="all", run_index=1)
        with pytest.raises(ValidationError, match=r"runs.*case_id"):
            PlotWaveformInput(job_id="sweep", runs="all", case_id="case-0000")
        with pytest.raises(ValidationError, match=r"runs.*job_id"):
            PlotWaveformInput(path="x.raw", runs=[0])

    async def test_one_case_selected_by_runs_reads_as_a_plot_of_that_case(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _three_rc(state_no_sim, work_dir)
        alone = await _plot(state_no_sim, job_id="sweep", run_index=1, signals=["V(out)"])
        selected = await _plot(state_no_sim, job_id="sweep", runs=[1], signals=["V(out)"])
        for data in (alone, selected):
            data.pop("path")
        assert selected == alone

    async def test_cases_a_label_would_write_alike_are_labelled_by_case_id(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """1000 and 1000.0 differ as values but a label writes both as 1000;
        a case that assigns none of the names that differ has nothing to show."""
        await _job(
            state_no_sim,
            work_dir,
            "alike",
            [
                _Case("ltspice_tran_rc", {"R1": 1000}),
                _Case("ltspice_tran_rc", {"R1": 1000.0}),
                _Case("ltspice_tran_rc", {"R1": 2000}),
                _Case("ltspice_tran_rc", {}),
            ],
        )
        data = await _plot(state_no_sim, job_id="alike", runs="all", signals=["V(out)"])
        assert _labels(data) == [
            "V(out) [case-0000]",
            "V(out) [case-0001]",
            "V(out) [R1=2000]",
            "V(out) [case-0003]",
        ]

    async def test_a_case_the_window_leaves_empty_is_named_not_drawn(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "window",
            [_Case("ltspice_tran_rc", {"C1": 1}), _Case("ltspice_step_tran", {"C1": 2})],
        )
        whole = await _plot(state_no_sim, job_id="window", runs=[0], signals=["V(out)"])
        end = whole["window_used"][1]
        data = await _plot(
            state_no_sim,
            job_id="window",
            runs="all",
            signals=["V(out)"],
            t_start=f"{end * 1.5}",
        )
        assert [case["case_id"] for case in data["cases"]] == ["case-0001"]
        (empty,) = data["cases_not_drawn"]
        assert (empty["case_id"], empty["code"]) == ("case-0000", "window_empty")
        assert all(t["case_id"] == "case-0001" for t in data["traces"])

    async def test_a_signal_some_cases_lack_is_drawn_where_it_exists(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "lack",
            [_Case("ltspice_tran_rc", {"C1": 1}), _Case("ltspice_step_tran", {"C1": 2})],
        )
        data = await _plot(state_no_sim, job_id="lack", runs="all", signals="all")
        lacking = {item["signal"]: item["case_ids"] for item in data["signals_not_in_cases"]}
        assert lacking
        assert all(case_ids == ["case-0000"] for case_ids in lacking.values())
        drawn = {(t["signal"], t["case_id"]) for t in data["traces"]}
        for signal in lacking:
            assert (signal, "case-0000") not in drawn

    async def test_ac_cases_draw_magnitude_and_phase_for_each_case(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        await _job(
            state_no_sim,
            work_dir,
            "bode",
            [_Case("ltspice_ac_rc", {"C1": 1}), _Case("ltspice_ac_rc", {"C1": 2})],
        )
        data = await _plot(state_no_sim, job_id="bode", runs="all", signals=["V(out)"])
        assert data["analysis_type"] == "ac"
        assert data["panels"] == 2
        assert _labels(data) == ["V(out) [C1=1]", "V(out) [C1=2]"] * 2
        assert all(t["unit"] == "dB" and "phase_final_deg" in t for t in data["traces"])


async def _analyzed_cases(state: SessionState, job_id: str, runs: Any) -> tuple[set, set]:
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "runs": runs}],
                "recipes": [{"key": "r", "metric": "signal_stats", "signal": "V(out)"}],
            }
        ),
        state,
    )
    data = result.structured_content
    assert data is not None
    read = {(row["case_id"], row["run_index"]) for row in data["results"]["r"]["values"]}
    missing = {
        (item.get("case_id"), item.get("run_index"), item["code"])
        for item in data.get("coverage", {}).get("missing_cases", {}).get("items", [])
    }
    return read, missing


@pytest.mark.asyncio
class TestOneSelection:
    """plot_waveform and analyze_results resolve one ``runs`` value to the
    same cases of a job, and name the same cases they could not read."""

    @pytest.mark.parametrize(
        "runs",
        [
            "all",
            [0, 1, 3],
            [2, 7],
            {"case_ids": ["case-0003", "case-0001", "case-0009"]},
        ],
    )
    async def test_the_two_tools_resolve_one_selection_to_the_same_cases(
        self, state_no_sim: SessionState, work_dir: Path, runs: Any
    ):
        await _job(
            state_no_sim,
            work_dir,
            "both",
            [
                _Case("ltspice_tran_rc", {"R1": 1}),
                _Case(None, {"R1": 2}, status="failed"),
                _Case("ltspice_tran_rc", {"R1": 3}),
                _Case("ltspice_tran_rc", {"R1": 4}),
            ],
        )
        read, missing = await _analyzed_cases(state_no_sim, "both", runs)
        data = await _plot(state_no_sim, job_id="both", runs=runs, signals=["V(out)"])

        drawn = (
            {(c["case_id"], c["run_index"]) for c in data["cases"]}
            if "cases" in data
            else {(t["case_id"], t["run_index"]) for t in data["traces"]}
        )
        # Runs the job does not hold are one row per code, naming the first few.
        not_drawn: set[tuple[Any, Any, str]] = set()
        for item in data.get("cases_not_drawn", []):
            if "count" in item:
                assert item["count"] == len(item.get("run_indices") or item["case_ids"])
                not_drawn |= {(None, index, item["code"]) for index in item.get("run_indices", [])}
                not_drawn |= {
                    (case_id, None, item["code"]) for case_id in item.get("case_ids", [])
                }
            else:
                not_drawn.add((item.get("case_id"), item.get("run_index"), item["code"]))
        assert drawn == read
        assert not_drawn == missing


def _formats_a_point(node: ast.JoinedStr) -> bool:
    """An f-string that writes ``name=value`` (a step value or an assignment)
    or ``[step N]`` into text."""
    parts = node.values
    for before, between, after in zip(parts, parts[1:], parts[2:], strict=False):
        if (
            isinstance(before, ast.FormattedValue)
            and isinstance(between, ast.Constant)
            and str(between.value).endswith("=")
            and isinstance(after, ast.FormattedValue)
        ):
            return True
    return any(isinstance(part, ast.Constant) and "[step " in str(part.value) for part in parts)


def test_one_function_writes_a_points_label():
    """A trace's legend label names its point (the assignments that differ,
    then the step values, or the step's index): ``lib/points.point_label``
    writes it, and nothing in the plot tool formats one of its own."""
    tree = _source_scan.parsed(_source_scan.PACKAGE / "tools" / "analysis.py")
    offenders = [
        name
        for name, nodes in _source_scan.scopes(tree)
        if any(isinstance(node, ast.JoinedStr) and _formats_a_point(node) for node in nodes)
    ]
    assert offenders == []


@pytest.mark.asyncio
async def test_a_case_with_a_log_and_no_raw_is_read_by_analysis_and_not_drawn(
    state_no_sim: SessionState, work_dir: Path
):
    """services.read_selected_case answers both tools; they differ only in
    require_raw. analyze_results reads a produced case's log without a raw
    (a log recipe needs none), and a chart, which draws a raw, names it."""
    job = await _three_rc(state_no_sim, work_dir, "noraw")
    job.cases[1].raw_file = None
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": "noraw", "runs": [0, 1]}],
                "recipes": [{"key": "r", "metric": "measurements"}],
            }
        ),
        state_no_sim,
    )
    data = result.structured_content
    assert data is not None
    missing = data.get("coverage", {}).get("missing_cases", {}).get("items", [])
    assert all(item.get("case_id") != "case-0001" for item in missing)

    plotted = await _plot(state_no_sim, job_id="noraw", runs=[0, 1], signals=["V(out)"])
    (left_out,) = plotted["cases_not_drawn"]
    assert (left_out["case_id"], left_out["code"]) == ("case-0001", "raw_not_produced")
