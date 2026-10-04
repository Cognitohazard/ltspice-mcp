"""Plot selection and descriptor facts at the existing public consumers."""

import asyncio
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ltspice_mcp.api import ApiValidationError
from ltspice_mcp.api._primitives import RawResult
from ltspice_mcp.errors import PathSecurityError, ResultError
from ltspice_mcp.lib import result_store, services
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.tools import analysis, analyze, experiments, inspect_tools
from tests.conftest import SyncApi, stage_recorded_fixture
from tests.test_decoded_raw import header
from tests.test_log_consumer_migration import forbid_parent_reads


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (analyze.AnalyzeSourceInput, {"raw_path": "results.raw", "label": "selected"}),
        (analysis.PlotWaveformInput, {"raw_file": "results.raw", "open": False}),
        (experiments.AttachedAnalysis, {"recipes": [{"key": "summary", "metric": "summary"}]}),
    ],
)
def test_existing_inputs_accept_explicit_plot_and_dialect(model, payload):
    selected = model.model_validate({**payload, "plot_index": 1, "dialect": "ngspice"})
    assert selected.plot_index == 1
    assert selected.dialect == "ngspice"


def test_inspect_results_is_an_existing_query_branch():
    query = inspect_tools._validate_query(
        {"kind": "results", "view": "signals", "path": "results.raw", "plot_index": 1}
    )
    assert isinstance(query, inspect_tools.ResultsQuery)
    assert query.plot_index == 1


def test_raw_result_reports_native_table_without_inventing_axis_or_units():
    plot = DecodedPlot(
        header(
            "Transfer Function", [("v(out)/vin", "voltage"), ("v(#input_impedance)", "voltage")], 1
        ),
        [np.array([2 / 3]), np.array([1500.0])],
        snapshot_id="table-example",
    )
    result = RawResult(
        DecodedRaw([plot]), source=Path("table.raw"), dialect="ngspice", step_count=1, steps=[{}]
    )
    assert result.analysis_type == "tf"
    assert result.plot_index == 0
    assert result.descriptor["axis"] is None
    assert len(result.plots) == 1
    assert result.table() == [
        {"signal": "v(out)/vin", "step_index": 0, "sample_index": 0, "value": 2 / 3, "unit": None},
        {
            "signal": "v(#input_impedance)",
            "step_index": 0,
            "sample_index": 0,
            "value": 1500.0,
            "unit": "Ω",
        },
    ]


def test_csv_axis_label_uses_selected_descriptor():
    plot = DecodedPlot(
        header(
            "Sensitivity Analysis",
            [("frequency", "frequency"), ("v(out)", "voltage")],
            2,
            flags=("complex",),
        ),
        [np.array([10 + 0j, 100 + 0j]), np.array([1 + 2j, 3 + 4j])],
        snapshot_id="sensitivity-example",
    )
    assert analysis._csv_x_header(DecodedRaw([plot]), "sens_ac") == "frequency_Hz"


@pytest.mark.parametrize("plot_index", [-1, True, 1.0, "1"])
@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (analyze.AnalyzeSourceInput, {"raw_path": "results.raw", "label": "selected"}),
        (analysis.PlotWaveformInput, {"raw_file": "results.raw", "open": False}),
        (experiments.AttachedAnalysis, {"recipes": [{"key": "summary", "metric": "summary"}]}),
        (inspect_tools.ResultsQuery, {"kind": "results", "path": "results.raw"}),
    ],
)
def test_plot_index_is_strict_on_every_input(model, payload, plot_index):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "plot_index": plot_index})


def test_api_validates_selection_before_loading(state_no_sim):
    api = SyncApi(state_no_sim)
    invalid_indices: list[Any] = [-1, True, "1", 1.0]
    for index in invalid_indices:
        with pytest.raises(ApiValidationError):
            api.load_raw("missing.raw", plot_index=index)
    invalid_dialect: Any = "unknown"
    with pytest.raises(ApiValidationError):
        api.load_raw("missing.raw", dialect=invalid_dialect)


def test_csv_native_complex_table_keeps_first_quantity(tmp_path):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Pole-Zero Analysis",
                    [("pole(1)", "frequency"), ("zero(1)", "frequency")],
                    1,
                    flags=("complex",),
                ),
                [np.array([-2 + 3j]), np.array([-4 + 5j])],
                snapshot_id="pz-example",
            )
        ]
    )
    out = tmp_path / "pz.csv"
    facts = analysis.build_waveform_csv(
        raw,
        tmp_path / "unread.raw",
        [services.Signal(name, name) for name in raw.get_trace_names()],
        1,
        raw.descriptor.analysis,
        None,
        None,
        "mag_phase",
        out,
    )
    with out.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["pole(1)_re", "pole(1)_im", "zero(1)_re", "zero(1)_im"],
        ["-2.0", "3.0", "-4.0", "5.0"],
    ]
    assert facts["window_used"] == []


def test_csv_step_values_are_captured_plain_rows(tmp_path):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Transient Analysis",
                    [("time", "time"), ("V(out)", "voltage")],
                    4,
                    flags=("real", "stepped"),
                ),
                [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])],
                snapshot_id="steps-example",
                steps=[{"corner": "slow"}, {"corner": "fast"}],
                step_offsets=[0, 2],
            )
        ]
    )
    path = tmp_path / "unused.raw"
    path.with_suffix(".log").write_text(".step corner=wrong\n", encoding="utf-8")
    out = tmp_path / "steps.csv"
    analysis.build_waveform_csv(
        raw,
        path,
        [services.Signal("V(out)", "V(out)")],
        2,
        "transient",
        None,
        None,
        "mag_phase",
        out,
        step_values=[dict(step.parameters) for step in raw.descriptor.steps],
    )
    with out.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[1:] == [
        ["0", "corner=slow", "0.0", "1.0"],
        ["0", "corner=slow", "1.0", "2.0"],
        ["1", "corner=fast", "0.0", "3.0"],
        ["1", "corner=fast", "1.0", "4.0"],
    ]


def test_plot_non_ac_complex_preserves_both_components():
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Sensitivity Analysis",
                    [("frequency", "frequency"), ("v(out)", "voltage")],
                    2,
                    flags=("complex",),
                ),
                [np.array([10 + 0j, 100 + 0j]), np.array([1 + 2j, 3 + 4j])],
                snapshot_id="sensitivity-example",
            )
        ]
    )
    plan = analysis.plan_plot(
        raw,
        [[services.Signal("v(out)", "v(out)")]],
        split_by_unit=True,
        netlist=None,
        steps=[0],
        step_dicts=[{}],
        analysis_type=raw.descriptor.analysis,
        x_is_log=True,
        ts=None,
        te=None,
    )
    plot = analysis.extract_plot(raw, plan)
    traces = plot.groups[0][1]
    assert {trace.label for trace in traces} == {"v(out) (real)", "v(out) (imag)"}
    np.testing.assert_array_equal(
        next(t.ys[0] for t in traces if t.label.endswith("(imag)")), [2, 4]
    )
    assert plan.units["v(out)"] is None


def test_attached_analysis_serializes_selection():
    request = experiments.AttachedAnalysis.model_validate(
        {
            "plot_index": 2,
            "dialect": "ngspice",
            "recipes": [{"key": "summary", "metric": "summary"}],
        }
    )
    payload = experiments._attached_analysis_payload(
        "job-placeholder", request.model_dump(mode="json")
    )
    assert payload["sources"][0]["plot_index"] == 2
    assert payload["sources"][0]["dialect"] == "ngspice"


def test_native_table_wrapper_detaches_inventory_and_keeps_complex_values():
    plot = DecodedPlot(
        header("Pole-Zero Analysis", [("pole(1)", "frequency")], 1, flags=("complex",)),
        [np.array([-2 + 3j])],
        snapshot_id="pz-example",
    )
    result = RawResult(
        DecodedRaw([plot]), source=Path("pz.raw"), dialect="ngspice", step_count=1, steps=[{}]
    )
    result.descriptor["traces"][0]["name"] = "changed"
    assert result.plots[0]["traces"][0]["name"] == "pole(1)"
    assert result.table()[0]["value"] == {"real": -2.0, "imag": 3.0}
    with pytest.raises(ResultError):
        result.axis()


def test_table_pager_materializes_only_a_slice():
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Operating Point", [("v(first)", "voltage"), ("v(last)", "voltage")], 100_000
                ),
                [np.arange(100_000, dtype=float), np.arange(100_000, dtype=float) * 2],
                snapshot_id="table-page-example",
            )
        ]
    )
    rows = inspect_tools._TableRows(raw, raw.descriptor.traces)
    assert len(rows) == 200_000
    assert len(rows._blocks) == 2
    assert rows[99_999:100_001] == [
        {
            "signal": "v(first)",
            "step_index": 0,
            "sample_index": 99_999,
            "value": 99_999.0,
            "unit": "V",
        },
        {"signal": "v(last)", "step_index": 0, "sample_index": 0, "value": 0.0, "unit": "V"},
    ]


def test_recorded_api_selects_later_plot_without_changing_first(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    api = SyncApi(state_no_sim)
    first = api.load_raw(path, plot_index=0, dialect="ngspice")
    later = api.load_raw(path, plot_index=1, dialect="ngspice")
    assert first.plot_index == 0
    assert first.descriptor["axis"]["quantity"] == "frequency"
    assert later.descriptor["axis"] is None
    assert len(first.plots) == len(later.plots) == 2
    assert later.signals == ["v(onoise_total)", "v(inoise_total)"]
    assert later.table()[0]["signal"] == "v(onoise_total)"
    assert later.table()[0]["value"] > 0
    np.testing.assert_array_equal(first.axis(), api.load_raw(path).axis())


@pytest.mark.asyncio
async def test_recorded_inspect_pages_table_and_binds_cursor(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    request = {
        "kind": "results",
        "view": "table",
        "path": str(path),
        "plot_index": 1,
        "dialect": "ngspice",
        "limit": 1,
    }

    async def call(query):
        reply = await inspect_tools.handle_inspect(
            inspect_tools.InspectInput.model_validate({"queries": [query]}), state_no_sim
        )
        assert reply.structured_content is not None
        return reply.structured_content["results"][0]

    first = await call(request)
    assert first["ok"]
    assert first["data"]["table"][0]["signal"] == "v(onoise_total)"
    second = await call({**request, "cursor": first["next_cursor"]})
    assert second["ok"]
    assert second["data"]["table"][0]["signal"] == "v(inoise_total)"
    assert second["next_cursor"] is None
    changed_view = await call({**request, "view": "signals", "cursor": first["next_cursor"]})
    assert not changed_view["ok"]
    changed_plot = await call(
        {**request, "view": "signals", "plot_index": 0, "cursor": first["next_cursor"]}
    )
    assert not changed_plot["ok"]


@pytest.mark.asyncio
async def test_recorded_manifests_bind_plot_and_serialized_source(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    manifests = []
    serialized = []
    for plot_index in (0, 1):
        source = services.source_for_raw_path(
            path, state_no_sim, plot_index=plot_index, dialect="ngspice"
        )
        run = analyze._ResolvedRun("source:0", "noise", source, None)
        manifests.append(
            analyze._manifest_for(
                run, await services.load_artifacts(source, state_no_sim, require_raw=False)
            )
        )
        serialized.append(analyze._serialize_run(run))
    assert manifests[0]["composite_sha256"] == manifests[1]["composite_sha256"]
    assert manifests[0]["selection_sha256"] != manifests[1]["selection_sha256"]
    assert [row["plot_index"] for row in serialized] == [0, 1]
    assert all(row["explicit_dialect"] == "ngspice" for row in serialized)
    item = result_store.create(
        working_dir=work_dir,
        inputs={"resolved_runs": serialized},
        work=[],
        source_manifests=manifests,
        source_jobs={},
        ttl_hours=1,
    )
    restored = analyze._deserialize_runs(item, state_no_sim)
    assert [run.source.plot_index for run in restored] == [0, 1]
    assert all(run.source.explicit_dialect == "ngspice" for run in restored)


@pytest.mark.asyncio
async def test_recorded_analysis_exports_selected_native_table(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [
                {
                    "raw_path": str(path),
                    "label": "integrated",
                    "plot_index": 1,
                    "dialect": "ngspice",
                }
            ],
            "recipes": [
                {
                    "key": "quantities",
                    "metric": "waveform",
                    "format": "csv",
                    "signals": ["v(onoise_total)", "v(inoise_total)"],
                }
            ],
        }
    )
    reply = await analyze.handle_analyze_results(request, state_no_sim)
    data = reply.structured_content
    assert data is not None
    assert data["outcome"] == "complete", data.get("failures")
    row = data["results"]["quantities"]["values"][0]
    assert row["plot_index"] == 1
    assert row["dialect"] == "ngspice"
    assert row["snapshot_id"] == data["source_hashes"][0]["snapshot_id"]

    def read_csv():
        with Path(row["value"]["artifact"]["path"]).open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))

    rows = await asyncio.to_thread(read_csv)
    assert rows[0] == ["v(onoise_total)", "v(inoise_total)"]
    assert float(rows[1][0]) == pytest.approx(1.646346154870315e-05)
    assert float(rows[1][1]) == pytest.approx(1.648388775530897e-05)


def test_analysis_cursor_request_identity_includes_selection():
    payload = {
        "sources": [
            {"raw_path": "noise.raw", "label": "noise", "plot_index": 0, "dialect": "ngspice"}
        ],
        "recipes": [{"key": "summary", "metric": "summary"}],
    }
    first = analyze.AnalyzeResultsInput.model_validate(payload)
    changed_plot = analyze.AnalyzeResultsInput.model_validate(
        {**payload, "sources": [{**payload["sources"][0], "plot_index": 1}]}
    )
    changed_dialect = analyze.AnalyzeResultsInput.model_validate(
        {**payload, "sources": [{**payload["sources"][0], "dialect": "ltspice"}]}
    )
    assert analyze._request_hash(first) != analyze._request_hash(changed_plot)
    assert analyze._request_hash(first) != analyze._request_hash(changed_dialect)


@pytest.mark.asyncio
async def test_manifest_never_hashes_unapproved_companion(state_no_sim, work_dir, monkeypatch):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    outside = work_dir.parent / "unapproved-companion.log"
    outside.write_text("unapproved example bytes", encoding="utf-8")
    path.with_suffix(".log").unlink()
    try:
        path.with_suffix(".log").symlink_to(outside)
    except OSError:
        pytest.skip("Creating file symlinks requires platform privileges")
    source = services.source_for_raw_path(path, state_no_sim)
    forbid_parent_reads(monkeypatch, path)
    with pytest.raises(PathSecurityError):
        analyze._manifest_for(
            analyze._ResolvedRun("source:0", "source", source, None),
            await services.load_artifacts(source, state_no_sim, require_raw=False),
        )
