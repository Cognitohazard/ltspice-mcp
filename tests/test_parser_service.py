"""Source-based loading uses the real finite parser, never a simulator."""

import asyncio
import gc
import os
import struct
import threading
import time
import traceback
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded, PathSecurityError, ResultError
from ltspice_mcp.lib import parser_process, parser_service, services
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.file_stamp import file_stamp
from ltspice_mcp.lib.result_cache import ResultCache
from tests.conftest import FIXTURES_DIR, LIVENESS_S, await_until, stage_recorded_fixture


def write_raw(path: Path, value=1.0):
    path.write_bytes(
        (
            "Title: cache input\nDate: recorded\nCommand: ngspice-42\n"
            "Plotname: Operating Point\nFlags: real\nNo. Variables: 1\nNo. Points: 1\n"
            "Variables:\n\t0\tv(out)\tvoltage\nBinary:\n"
        ).encode("ascii")
        + struct.pack("<d", value)
    )
    return path


def source(path, state, **kwargs):
    return services.source_for_raw_path(path, state, **kwargs)


def parser_requests(monkeypatch):
    """Every request the service sends a parser process, in order."""
    requests = []
    run = parser_service.run_parser_sync

    def record(request, **kwargs):
        requests.append(request)
        return run(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", record)
    return requests


def test_state_results_uses_resident_content_cache(state_no_sim):
    assert isinstance(state_no_sim.results, ResultCache)


def test_sync_capture_reap_and_cache_survive_source_deletion(state_no_sim, work_dir):
    path = write_raw(work_dir / "test.raw")
    raw = services.load_raw_sync(source(path, state_no_sim), state_no_sim)
    assert isinstance(raw, DecodedRaw)
    assert raw.get_wave("v(out)")[0] == 1
    assert not raw.get_wave(0).flags.writeable
    assert state_no_sim.results.entry_count == 1
    assert not list((state_no_sim.store.root / "parsing").glob("*"))
    path.unlink()
    assert raw.get_wave(0)[0] == 1


@pytest.mark.parametrize("stamps", ["settled_stamps", "unsettled_stamps"])
def test_same_stamp_rewrite_and_companion_presence_bind_content(
    state_no_sim, work_dir, monkeypatch, request, stamps
):
    request.getfixturevalue(stamps)
    requests = parser_requests(monkeypatch)
    path = write_raw(work_dir / "changed.raw")
    selected = source(path, state_no_sim)
    first = services.load_raw_sync(selected, state_no_sim)
    repeated = services.load_raw_sync(selected, state_no_sim)
    assert repeated is first
    assert len(requests) == (1 if stamps == "settled_stamps" else 2)
    stamp = path.stat()
    write_raw(path, 2.0)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = services.load_raw_sync(selected, state_no_sim)
    assert second.get_wave(0)[0] == 2
    assert second.descriptor.snapshot_id != first.descriptor.snapshot_id
    path.with_suffix(".log").write_text("complete\n", encoding="ascii")
    third = services.load_raw_sync(selected, state_no_sim)
    assert third.descriptor.snapshot_id != second.descriptor.snapshot_id
    path.with_suffix(".exe.log").write_text("console\n", encoding="ascii")
    fourth = services.load_raw_sync(selected, state_no_sim)
    assert fourth.descriptor.snapshot_id != third.descriptor.snapshot_id
    path.with_suffix(".log").unlink()
    assert (
        services.load_raw_sync(selected, state_no_sim).descriptor.snapshot_id
        != fourth.descriptor.snapshot_id
    )


def test_all_plots_are_cached_before_selection_and_wrong_dialect_refuses(state_no_sim, work_dir):
    path = work_dir / "noise.raw"
    path.write_bytes((FIXTURES_DIR / "ngspice_noise_2plot.raw").read_bytes())
    first = services.load_raw_sync(source(path, state_no_sim), state_no_sim)
    second = services.load_raw_sync(source(path, state_no_sim, plot_index=1), state_no_sim)
    assert first.plots is second.plots
    assert first.get_nr_plots() == 2
    assert first.descriptor.plot_index == 0
    assert second.descriptor.plot_index == 1
    assert second.descriptor.axis is None
    assert state_no_sim.results.entry_count == 1
    with pytest.raises(ResultError, match=r"dialect|Dialect|contradict"):
        services.load_raw_sync(source(path, state_no_sim, dialect="ltspice"), state_no_sim)


def test_imported_companions_are_authorized_before_worker_admission(
    state_no_sim, work_dir, tmp_path
):
    path = write_raw(work_dir / "admitted.raw")
    outside = work_dir.parent / "outside-parser-companion.log"
    imported = replace(source(path, state_no_sim), log=outside)
    with pytest.raises(PathSecurityError):
        services.load_raw_sync(imported, state_no_sim)
    assert state_no_sim.results.entry_count == 0


def test_absolute_analysis_deadline_refuses_before_creating_worker_files(state_no_sim, work_dir):
    path = write_raw(work_dir / "deadline.raw")
    with services.analysis_deadline(time.monotonic() - 1), pytest.raises(AnalysisDeadlineExceeded):
        services.load_raw_sync(source(path, state_no_sim), state_no_sim)
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_continuation_bound_snapshot_refuses_changed_companion(state_no_sim, work_dir):
    path = write_raw(work_dir / "bound.raw")
    selected = source(path, state_no_sim)
    first = services.load_raw_sync(selected, state_no_sim)
    bound = replace(selected, identity={"snapshot_id": first.descriptor.snapshot_id})
    assert services.load_raw_sync(bound, state_no_sim) is first
    path.with_suffix(".log").write_text("new companion\n", encoding="ascii")
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_raw_sync(bound, state_no_sim)
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_continuation_binding_is_checked_on_cache_hits(state_no_sim, work_dir):
    path = write_raw(work_dir / "cached-bound.raw")
    selected = source(path, state_no_sim)
    services.load_raw_sync(selected, state_no_sim)
    bound = replace(selected, identity={"snapshot_id": "0" * 64})
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_raw_sync(bound, state_no_sim)


def test_worker_cache_snapshot_survives_concurrent_clear(state_no_sim, work_dir, monkeypatch):
    path = write_raw(work_dir / "snapshot.raw")
    selected = source(path, state_no_sim)
    first = services.load_raw_sync(selected, state_no_sim)
    run = parser_service.run_parser_sync

    def clear_before_capture(request, **kwargs):
        assert request["existing_cache_keys"]
        state_no_sim.results.clear()
        return run(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", clear_before_capture)
    assert services.load_raw_sync(selected, state_no_sim) is first
    assert state_no_sim.results.entry_count == 0
    assert first.get_wave(0)[0] == 1


def test_capture_key_binds_explicit_dialect_and_input_budget(state_no_sim, work_dir):
    path = write_raw(work_dir / "budget.raw")
    selected = source(path, state_no_sim)
    first = services.load_raw_sync(selected, state_no_sim)
    explicit = services.load_raw_sync(source(path, state_no_sim, dialect="ngspice"), state_no_sim)
    state_no_sim.config.max_raw_mb += 1
    changed = services.load_raw_sync(selected, state_no_sim)
    assert len({raw.descriptor.snapshot_id for raw in (first, explicit, changed)}) == 3


async def test_async_loader_keeps_event_loop_responsive(state_no_sim, work_dir, monkeypatch):
    """The parse runs off the loop: held mid-call, the loop still turns."""
    path = work_dir / "recorded.raw"
    path.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes())
    entered = threading.Event()
    release = threading.Event()
    run = parser_service.run_parser_sync

    def held(request, **kwargs):
        entered.set()
        assert release.wait(LIVENESS_S)
        return run(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", held)
    load = asyncio.create_task(services.load_raw(source(path, state_no_sim), state_no_sim))
    try:
        # Polled from the loop, so it returns only if the loop keeps turning.
        await await_until(entered.is_set, what="the parse to start")
        assert not load.done()
    finally:
        release.set()
    raw = await load
    assert raw.descriptor.analysis == "transient"
    np.testing.assert_array_equal(raw.get_wave("V(out)"), raw.plots[0].get_wave("V(out)"))


async def test_public_artifacts_loader_returns_shared_snapshot_and_captures(
    state_no_sim, work_dir
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    artifacts = await services.load_artifacts(selected, state_no_sim, require_raw=True)
    assert artifacts.raw is not None
    assert artifacts.snapshot_id == artifacts.raw.descriptor.snapshot_id
    assert artifacts.raw.logs is artifacts.logs
    assert {item.role for item in artifacts.logs.captured.files} == {"raw", "log"}
    assert "console" in artifacts.logs.captured.absent
    assert await services.load_raw(selected, state_no_sim) is artifacts.raw
    assert await services.load_logs(selected, state_no_sim) is artifacts.logs


def test_unconfirmed_tree_exit_retains_session_admission(state_no_sim, work_dir, monkeypatch):
    path = write_raw(work_dir / "unconfirmed.raw")
    cleanup = parser_process._cleanup
    cleaned_pids = []

    def unconfirmed(process, *args):
        assert cleanup(process, *args)
        cleaned_pids.append(process.pid)
        # Simulate loss of the cleanup receipt after real owned-tree cleanup.
        return False

    monkeypatch.setattr(parser_process, "_cleanup", unconfirmed)
    selected = source(path, state_no_sim)
    with pytest.raises(parser_service.ParserCleanupError) as first:
        services.load_raw_sync(selected, state_no_sim)
    assert first.value.directory.is_dir()
    state_no_sim.results.clear()
    started = time.monotonic()
    with pytest.raises(parser_service.ParserCleanupError) as repeated:
        services.load_raw_sync(selected, state_no_sim)
    assert repeated.value.directory == first.value.directory
    assert time.monotonic() - started < 1
    assert len(cleaned_pids) == 1


def test_retained_admission_errors_release_sources_and_have_fresh_tracebacks(
    state_no_sim, work_dir, monkeypatch
):
    path = write_raw(work_dir / "retained_failure.raw")
    cleanup = parser_process._cleanup
    cleaned_pids = []
    source_refs = []

    def unconfirmed(process, *args):
        assert cleanup(process, *args)
        cleaned_pids.append(process.pid)
        return False

    def failed_call():
        selected = source(path, state_no_sim)
        source_refs.append(weakref.ref(selected))
        try:
            services.load_raw_sync(selected, state_no_sim)
        except parser_service.ParserCleanupError as failure:
            return failure
        pytest.fail("Unconfirmed cleanup must keep parser admission closed")

    monkeypatch.setattr(parser_process, "_cleanup", unconfirmed)
    initial = failed_call()
    directory, worker_pid = initial.directory, initial.worker_pid
    assert directory.is_dir()
    assert worker_pid is not None
    blocked = [failed_call(), failed_call()]
    state_no_sim.results.clear()
    blocked.append(failed_call())
    assert len(cleaned_pids) == 1
    assert all(item.directory == directory and item.worker_pid == worker_pid for item in blocked)
    fresh = len({id(item) for item in [initial, *blocked]}) == 4
    frames = [len(traceback.extract_tb(item.__traceback__)) for item in blocked]
    no_causes = all(item.__cause__ is None for item in blocked)
    blocked.clear()
    del initial
    gc.collect()
    assert all(ref() is None for ref in source_refs)
    assert fresh
    assert len(set(frames)) == 1
    assert no_causes
    assert state_no_sim.results.entry_count == state_no_sim.results.byte_count == 0


def test_scratch_removal_failure_after_exit_releases_admission(
    state_no_sim, work_dir, monkeypatch
):
    path = write_raw(work_dir / "scratch.raw")
    remove = parser_service.shutil.rmtree
    retained = []

    def fail_once(directory):
        if not retained:
            retained.append(directory)
            raise OSError("scratch removal denied")
        remove(directory)

    monkeypatch.setattr(parser_service.shutil, "rmtree", fail_once)
    with pytest.raises(parser_service.ParserCleanupError):
        services.load_raw_sync(source(path, state_no_sim), state_no_sim)
    assert retained[0].is_dir()
    assert services.load_raw_sync(source(path, state_no_sim), state_no_sim).get_wave(0)[0] == 1


def test_raw_and_logs_share_captured_entry_and_plot_selection(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    raw = services.load_raw_sync(selected, state_no_sim)
    logs = services.load_logs_sync(selected, state_no_sim)
    assert raw.logs is logs
    assert raw.select_plot(0).logs is logs
    assert logs.section("measurements")["value"]["measurements"]["vfinal"]["values"] == [
        0.999876166042
    ]
    entry = next(iter(state_no_sim.results.snapshot().values()))
    assert entry.raw is raw and entry.logs is logs
    assert entry.snapshot_id == raw.descriptor.snapshot_id
    assert entry.snapshot_id != logs.snapshot_id
    assert state_no_sim.results.entry_count == 1
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_logs_only_cache_cannot_claim_resident_raw(
    state_no_sim, work_dir, monkeypatch, settled_stamps
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    requests = parser_requests(monkeypatch)
    logs = services.load_logs_sync(selected, state_no_sim)
    before = next(iter(state_no_sim.results.snapshot().values()))
    assert before.raw is None
    raw = services.load_raw_sync(selected, state_no_sim)
    assert requests[0]["op"] == "load_logs" and requests[1]["op"] == "load_raw"
    assert requests[1]["existing_cache_keys"] == []
    assert requests[0]["limits"] == requests[1]["limits"]
    assert before.snapshot_id == raw.descriptor.snapshot_id
    assert raw.logs is not None
    assert raw.logs.as_dict() == logs.as_dict()
    assert services.load_logs_sync(selected, state_no_sim) is raw.logs
    assert len(requests) == 2
    # The same bytes under a new stamp reach a worker, which finds them retained.
    stamp = path.stat()
    path.write_bytes(path.read_bytes())
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10**9))
    assert services.load_logs_sync(selected, state_no_sim) is raw.logs
    assert len(requests) == 3
    assert requests[-1]["existing_cache_keys"] == [before.snapshot_id]
    assert state_no_sim.results.entry_count == 1


def test_unchanged_source_is_answered_from_its_stamp_without_a_worker(
    state_no_sim, work_dir, monkeypatch, settled_stamps
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    requests = parser_requests(monkeypatch)
    raw = services.load_raw_sync(selected, state_no_sim)
    assert services.load_raw_sync(selected, state_no_sim) is raw
    assert services.load_logs_sync(selected, state_no_sim) is raw.logs
    bound = replace(selected, identity={"snapshot_id": "0" * 64})
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_raw_sync(bound, state_no_sim)
    assert [request["op"] for request in requests] == ["load_raw"]
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_stamp_answers_only_paths_the_sandbox_still_admits(
    state_no_sim, work_dir, monkeypatch, settled_stamps
):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    requests = parser_requests(monkeypatch)
    services.load_raw_sync(selected, state_no_sim)
    state_no_sim.sandbox_pinned = True
    monkeypatch.setattr(state_no_sim.config, "allowed_paths", [work_dir / "elsewhere"])
    with pytest.raises(PathSecurityError):
        services.load_raw_sync(selected, state_no_sim)
    assert len(requests) == 1


def test_retained_parser_failure_refuses_a_read_its_stamp_could_answer(
    state_no_sim, work_dir, monkeypatch, settled_stamps
):
    path = write_raw(work_dir / "answered.raw")
    selected = source(path, state_no_sim)
    services.load_raw_sync(selected, state_no_sim)
    cleanup = parser_process._cleanup

    def unconfirmed(process, *args):
        assert cleanup(process, *args)
        return False

    monkeypatch.setattr(parser_process, "_cleanup", unconfirmed)
    other = write_raw(work_dir / "unconfirmed.raw", 2.0)
    with pytest.raises(parser_service.ParserCleanupError) as failure:
        services.load_raw_sync(source(other, state_no_sim), state_no_sim)
    with pytest.raises(parser_service.ParserCleanupError) as refused:
        services.load_raw_sync(selected, state_no_sim)
    assert refused.value.directory == failure.value.directory


_DAY_NS = 86_400 * 10**9


@pytest.mark.parametrize(
    ("mtime", "after_change_ns", "served"),
    [
        ("as_written", parser_service._SETTLE_NS // 2, False),
        ("restored", parser_service._SETTLE_NS // 2, False),
        ("restored", parser_service._SETTLE_NS * 5, True),
        ("as_written", parser_service._SETTLE_NS * 5, True),
    ],
)
def test_a_stamp_is_recorded_only_once_its_times_are_settled(
    state_no_sim, work_dir, monkeypatch, mtime, after_change_ns, served
):
    """A write in the same timestamp tick as the read could keep every stamp
    field, so a stamp whose modification or change time is that recent is not
    recorded. A restored modification time does not settle a recent change."""
    path = write_raw(work_dir / "settling.raw")
    if mtime == "restored":
        written_at = path.stat().st_mtime_ns
        restored = (written_at // 10**9) * 10**9 - _DAY_NS + 123_400
        os.utime(path, ns=(restored, restored))
    stamp = file_stamp(str(path))
    assert isinstance(stamp, tuple)
    monkeypatch.setattr(parser_service, "_now_ns", lambda: stamp[4] + after_change_ns)
    requests = parser_requests(monkeypatch)
    selected = source(path, state_no_sim)
    first = services.load_raw_sync(selected, state_no_sim)
    assert services.load_raw_sync(selected, state_no_sim) is first
    assert len(requests) == (1 if served else 2)


@pytest.mark.parametrize("field", [3, 4])
@pytest.mark.parametrize(
    ("moment", "age", "settled"),
    [
        (10**18 + 1, parser_service._SETTLE_NS, True),
        (10**18 + 1, parser_service._SETTLE_NS - 1, False),
        (10**18, parser_service._SETTLE_NS * 5, False),
        (10**18, parser_service._COARSE_SETTLE_NS, True),
    ],
)
def test_whole_second_times_wait_out_a_coarse_filesystem_tick(field, moment, age, settled):
    """A whole-second time marks a filesystem that stamps writes coarsely (FAT
    keeps two seconds), so it settles only after the coarse margin."""
    stamp: list[int] = [1, 2, 3, 10**17 + 1, 10**17 + 1]
    stamp[field] = moment
    stamps = (tuple(stamp), None, "absent")
    assert parser_service._settled(stamps, moment + age) is settled


@pytest.mark.parametrize("raw_present", [False, True])
def test_log_diagnostics_ignore_absent_or_malformed_raw(state_no_sim, work_dir, raw_present):
    path = work_dir / "diagnostics.raw"
    if raw_present:
        path.write_bytes(b"malformed raw payload")
    path.with_suffix(".log").write_bytes((FIXTURES_DIR / "ltspice_tran_rc.log").read_bytes())
    selected = source(path, state_no_sim)
    logs = services.load_logs_sync(selected, state_no_sim)
    assert logs.section("measurements")["status"] == "parsed"
    assert ("raw" in logs.captured.absent) is not raw_present
    with pytest.raises(ResultError):
        services.load_raw_sync(selected, state_no_sim)
    assert next(iter(state_no_sim.results.snapshot().values())).raw is None
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_absent_log_sections_preserved_but_no_inputs_refuses(state_no_sim, work_dir):
    path = write_raw(work_dir / "no-logs.raw")
    selected = source(path, state_no_sim)
    raw = services.load_raw_sync(selected, state_no_sim)
    logs = services.load_logs_sync(selected, state_no_sim)
    assert raw.logs is logs
    assert logs.section("diagnostics")["status"] == "absent"
    assert logs.section("diagnostics")["value"] == {
        "warnings": [],
        "errors": [],
        "meas_errors": [],
    }
    assert logs.section("measurements")["status"] == "absent"
    assert logs.section("measurements")["value"] is None
    path.unlink()
    with pytest.raises(ResultError, match="No parser input files"):
        services.load_logs_sync(selected, state_no_sim)


def test_logs_continuation_binds_shared_snapshot_on_hits_and_changes(state_no_sim, work_dir):
    path = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    selected = source(path, state_no_sim)
    logs = services.load_logs_sync(selected, state_no_sim)
    entry = next(iter(state_no_sim.results.snapshot().values()))
    bound = replace(selected, identity={"snapshot_id": entry.snapshot_id})
    assert services.load_logs_sync(bound, state_no_sim) is logs
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_logs_sync(
            replace(bound, identity={"snapshot_id": logs.snapshot_id}), state_no_sim
        )
    path.with_suffix(".exe.log").write_text("new console\n", encoding="ascii")
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_logs_sync(bound, state_no_sim)


def test_log_only_capture_ignores_sibling_raw_and_binds_log_console(state_no_sim, work_dir):
    log = work_dir / "log_only.log"
    log.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.log").read_bytes())
    sibling_raw = log.with_suffix(".raw")
    sibling_raw.write_bytes(b"unrelated RAW bytes")
    selected = services.resolve_analysis_source(state_no_sim, log_file=str(log))
    logs = services.load_logs_sync(selected, state_no_sim)
    assert "raw" in logs.captured.absent
    assert logs.value("measurements")["measurements"]["vfinal"]["values"] == [0.999876166042]
    entry = next(iter(state_no_sim.results.snapshot().values()))
    assert entry.raw is None
    bound = replace(selected, identity={"snapshot_id": entry.snapshot_id})
    sibling_raw.write_bytes(b"changed unrelated bytes")
    assert services.load_logs_sync(bound, state_no_sim) is logs
    log.with_suffix(".exe.log").write_text("new console\n", encoding="ascii")
    with pytest.raises(ResultError, match=r"snapshot.*changed"):
        services.load_logs_sync(bound, state_no_sim)
    assert not list((state_no_sim.store.root / "parsing").glob("*"))


def test_log_only_missing_inputs_error_names_log_source(state_no_sim, work_dir):
    log = work_dir / "absent.log"
    selected = services.resolve_analysis_source(state_no_sim, log_file=str(log))
    with pytest.raises(ResultError, match="No parser input files") as failure:
        services.load_logs_sync(selected, state_no_sim)
    assert str(log) in str(failure.value)
    assert not list((state_no_sim.store.root / "parsing").glob("*"))
