"""Recorded log facts stay usable independently of optional RAW payloads."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import PathSecurityError, ResultError
from ltspice_mcp.lib import result_store, services
from ltspice_mcp.tools import analyze, experiments, inspect_tools
from tests.conftest import (
    LTSPICE_TRAN_RC_VFINAL,
    SyncApi,
    make_experiment_job,
    stage_recorded_fixture,
)
from tests.test_log_consumer_migration import forbid_parent_reads


def recorded_log(work_dir: Path) -> Path:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.unlink()
    return raw.with_suffix(".log")


def test_api_imports_measurements_with_no_raw(state_no_sim, work_dir, monkeypatch):
    log = recorded_log(work_dir)
    forbid_parent_reads(monkeypatch, log.with_suffix(".raw"))
    api = SyncApi(state_no_sim)
    value = api.measurements(log)
    assert value["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]
    value["measurements"].clear()
    assert api.measurements(log_path=log)["measurements"]


@pytest.mark.parametrize(
    "case",
    [
        "tf_table",
        "pz_table",
        "sens_dc_table",
        "sens_ac_table",
        "disto_harm_table",
        "disto_two_table",
    ],
)
def test_existing_inspect_exposes_detached_native_rows(state_no_sim, work_dir, case):
    log = work_dir / f"{case}.log"
    shutil.copyfile(Path(__file__).parent / "fixtures" / "native_log_tables" / log.name, log)
    api = SyncApi(state_no_sim)
    reply = api.inspect(queries=[{"kind": "results", "view": "native_tables", "path": str(log)}])
    result = reply["results"][0]
    assert result["ok"], result
    data = result["data"]
    assert data["capture_facts"]["absent"] == ["raw", "console"]
    assert data["section"]["status"] == "parsed"
    rows = data["native_tables"]
    assert rows and all("plot_id" not in row and "step_index" not in row for row in rows)
    assert all(row["analysis_extent"] == "unknown" for row in rows)
    if case == "pz_table":
        assert rows[0]["entry"]["label"] == "all"
        assert rows[0]["entry"]["real"] == -1000.0
        assert rows[0]["entry"]["unit"] is None
    rows[0].clear()
    fresh = api.inspect(queries=[{"kind": "results", "view": "native_tables", "path": str(log)}])
    assert fresh["results"][0]["data"]["native_tables"][0]


def test_analysis_measurements_no_raw_and_null_identity(state_no_sim, work_dir, monkeypatch):
    log = recorded_log(work_dir)
    forbid_parent_reads(monkeypatch, log.with_suffix(".raw"))
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"log_path": str(log), "label": "log", "dialect": "ltspice"}],
        recipes=[{"metric": "measurements", "key": "meas"}],
        include={
            "per_run": True,
            "provenance": True,
            "signals_available": True,
            "fields": ["source", "plot_index", "step_index", "step_values", "dialect", "value"],
        },
    )
    row = reply["results"]["meas"]["per_run"]["items"][0]
    assert row["step_index"] is None and row["plot_index"] is None
    assert row["step_values"] == {}
    assert row["dialect"] is None
    assert reply["signals_available"] == {"log:0": []}
    manifest = reply["source_hashes"][0]
    assert manifest["raw_path"] is None and manifest["raw_sha256"] is None
    assert manifest["explicit_dialect"] == "ltspice"
    assert manifest["producing_dialect"] is None
    assert manifest["raw_present"] is False
    assert manifest["log_present"] is True and manifest["console_present"] is False
    assert any(
        observation["code"] == "log_path_without_deck_provenance"
        for observation in reply["observations"]
    )
    stored = result_store.load(reply["result_set_id"], working_dir=work_dir)
    assert stored.inputs["resolved_runs"][0]["raw"] is None
    assert analyze._deserialize_runs(stored, state_no_sim)[0].source.raw is None


def test_malformed_raw_does_not_suppress_measurements(state_no_sim, work_dir):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.write_bytes(b"malformed RAW")
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"raw_path": str(raw), "label": "mixed"}],
        recipes=[{"metric": "summary", "key": "raw"}, {"metric": "measurements", "key": "meas"}],
        include={"per_run": True},
    )
    assert reply["outcome"] == "partial"
    assert reply["results"]["meas"]["per_run"]["items"]
    assert "raw" not in reply["results"]
    assert any(failure["stage"] == "analyze" for failure in reply["failures"])


@pytest.mark.parametrize("selector", [{"all_steps": True}, {"step": {"axis": "R", "value": 1}}])
def test_whole_log_measurements_refuse_raw_step_selection(state_no_sim, work_dir, selector):
    log = recorded_log(work_dir)
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"log_path": str(log), "label": "log"}],
        recipes=[{"metric": "measurements", "key": "meas"}],
        **selector,
    )
    assert reply["outcome"] == "failed"
    assert "whole-log" in reply["failures"][0]["message"]


def test_optional_selection_and_attached_serializer():
    attached = experiments.AttachedAnalysis.model_validate(
        {"recipes": [{"metric": "measurements", "key": "m"}]}
    )
    assert attached.plot_index is None
    payload = experiments._attached_analysis_payload("job", attached.model_dump(mode="json"))
    assert payload["sources"][0]["plot_index"] is None
    for model, request in [
        (analyze.AnalyzeSourceInput, {"log_path": "result.log", "label": "log"}),
        (
            inspect_tools.ResultsQuery,
            {"kind": "results", "view": "native_tables", "path": "result.log"},
        ),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({**request, "plot_index": 0})


def test_explicit_log_ignores_even_malformed_raw_sibling(state_no_sim, work_dir):
    log = recorded_log(work_dir)
    log.with_suffix(".raw").write_bytes(b"malformed RAW sibling")
    api = SyncApi(state_no_sim)
    query = {"kind": "results", "view": "measurements", "path": str(log)}
    reply = api.inspect(queries=[query])
    data = reply["results"][0]["data"]
    assert data["capture_facts"]["absent"] == ["raw", "console"]
    assert data["measurements"][0]["value"] == LTSPICE_TRAN_RC_VFINAL
    with pytest.raises(ResultError):
        api.load_raw(log.with_suffix(".raw"))


def test_already_produced_record_can_read_measurements_without_raw(state_no_sim, work_dir):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = make_experiment_job(state_no_sim, job_id="recorded-log-only", raw=raw)
    job.cases[0].raw_file = None
    raw.unlink()
    api = SyncApi(state_no_sim)
    assert api.measurements(job_id=job.job_id)["measurements"]
    reply = api.analyze_results(
        sources=[{"job_id": job.job_id, "label": "case"}],
        recipes=[{"metric": "measurements", "key": "m"}],
        include={"per_run": True},
    )
    assert reply["coverage"]["runs_analyzed"] == 1
    assert reply["results"]["m"]["per_run"]["items"][0].get("plot_index") is None
    with pytest.raises(ResultError):
        api.load_raw(job_id=job.job_id)


def test_log_paging_binds_view_prefix_and_console_absence(state_no_sim, work_dir):
    log = work_dir / "tf_table.log"
    shutil.copyfile(Path(__file__).parent / "fixtures" / "native_log_tables" / log.name, log)
    api = SyncApi(state_no_sim)
    query = {"kind": "results", "view": "native_tables", "path": str(log), "limit": 1}
    first = api.inspect(raw_page=True, queries=[query])["results"][0]
    assert first["ok"] and first["next_cursor"] is not None
    next_query = {**query, "cursor": first["next_cursor"]}
    second = api.inspect(raw_page=True, queries=[next_query])["results"][0]
    assert second["data"]["native_tables"][0]["entry"]["label"] == "output_impedance_at_v(out)"
    for change in [{"view": "measurements"}, {"prefix": "transfer"}]:
        assert not api.inspect(raw_page=True, queries=[{**next_query, **change}])["results"][0][
            "ok"
        ]
    log.with_suffix(".exe.log").write_bytes(b"")
    stale = api.inspect(raw_page=True, queries=[next_query])["results"][0]
    assert not stale["ok"]
    fresh = api.inspect(raw_page=True, queries=[query])["results"][0]
    console = next(
        file for file in fresh["data"]["capture_facts"]["files"] if file["role"] == "console"
    )
    assert console["size_bytes"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["log", "console", "raw"])
async def test_manifests_reject_content_and_absence_drift(state_no_sim, work_dir, change):
    log = recorded_log(work_dir)
    source = services.resolve_analysis_source(state_no_sim, log_file=str(log))
    if change == "raw":
        # A RAW import records absent log companions, unlike a direct log import.
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        raw.with_suffix(".log").unlink()
        source = services.source_for_raw_path(raw, state_no_sim)
    run = analyze._ResolvedRun("log:0", "log", source, None)
    manifest = analyze._manifest_for(
        run, await services.load_artifacts(source, state_no_sim, require_raw=False)
    )
    if change == "console":
        log.with_suffix(".exe.log").write_bytes(b"")
    elif change == "raw":
        assert source.log is not None
        source.log.write_text("new log companion\n", encoding="utf-8")
    else:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    failures = await analyze._verify_direct_sources(
        [manifest],
        {"log:0"},
        time.monotonic() + 30,
        state=state_no_sim,
    )
    assert failures["log:0"].code == "source_drift"


def test_native_truncation_is_query_error_not_prefix_rows(state_no_sim, work_dir):
    log = work_dir / "tf_table.log"
    body = (Path(__file__).parent / "fixtures" / "native_log_tables" / log.name).read_text()
    log.write_text(body.rsplit("ngspice-42 done", 1)[0], encoding="utf-8")
    reply = SyncApi(state_no_sim).inspect(
        queries=[{"kind": "results", "view": "native_tables", "path": str(log)}]
    )
    assert not reply["results"][0]["ok"]
    assert "native_tables" in reply["results"][0]["error"]["message"]


def test_log_console_companion_is_authorized_before_capture(state_no_sim, work_dir, monkeypatch):
    log = recorded_log(work_dir)
    outside = work_dir.parent / "unapproved-console.log"
    outside.write_text("private example bytes", encoding="utf-8")
    try:
        log.with_suffix(".exe.log").symlink_to(outside)
    except OSError:
        pytest.skip("Creating file symlinks requires platform privileges")
    forbid_parent_reads(monkeypatch, log.with_suffix(".raw"))
    with pytest.raises(PathSecurityError):
        SyncApi(state_no_sim).measurements(log)


def test_raw_analysis_retains_decoded_dialect_without_import_hint(state_no_sim, work_dir):
    raw = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    reply = SyncApi(state_no_sim).analyze_results(
        sources=[{"raw_path": str(raw), "label": "noise", "plot_index": 1}],
        recipes=[
            {
                "metric": "waveform",
                "key": "table",
                "format": "csv",
                "signals": ["v(onoise_total)", "v(inoise_total)"],
            }
        ],
    )
    row = reply["results"]["table"]["values"][0]
    assert row["plot_index"] == 1 and row["dialect"] == "ngspice"
    assert row["snapshot_id"] == reply["source_hashes"][0]["snapshot_id"]
    # Capture-only manifest records caller/producer hints, not inferred evidence.
    assert reply["source_hashes"][0]["dialect"] is None
