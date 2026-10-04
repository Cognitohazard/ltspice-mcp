"""Byte admission, resident backing ownership, and concurrent cache snapshots."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.lib.log_decode import LogLimits, decode_logs
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_capture import CapturedInputs
from ltspice_mcp.lib.result_cache import ResultCache, resident_size
from tests.test_decoded_raw import header


def raw(values, *, offset="0"):
    wave = np.array(values, dtype="<f8")
    h = header("Transient Analysis", [("time", "time")], len(wave), offset=offset)
    return DecodedRaw([DecodedPlot(h, [wave], snapshot_id="cache-example")])


@pytest.fixture
def empty_logs(tmp_path):
    captured = CapturedInputs((), ("raw", "log", "console"))
    return DecodedLog(
        decode_logs(captured, tmp_path, limits=LogLimits(4096, 4096, 100, 1000, 4096))
    )


def parsed(raw, key, logs):
    return ParsedArtifacts(snapshot_id=key, raw=raw, logs=logs)


def test_lru_is_bounded_by_bytes_and_entries(empty_logs):
    first, second, third = (parsed(raw([0.0, 1.0]), key * 64, empty_logs) for key in "abc")
    charge = resident_size(first)
    cache = ResultCache(max_bytes=2 * charge, max_entries=2)
    cache.put("a" * 64, first)
    cache.put("b" * 64, second)
    assert cache.get("a" * 64) is first
    cache.put("c" * 64, third)
    assert cache.get("b" * 64) is None
    assert cache.entry_count == 2
    assert cache.byte_count <= cache.max_bytes
    assert set(cache.snapshot()) == {"a" * 64, "c" * 64}


def test_oversized_result_is_returnable_but_not_retained(empty_logs):
    value = parsed(raw(np.arange(100)), "a" * 64, empty_logs)
    cache = ResultCache(max_bytes=resident_size(value) - 1, max_entries=2)
    assert cache.put("a" * 64, value) is value
    assert cache.entry_count == cache.byte_count == 0
    assert cache.get("a" * 64) is None


def test_snapshot_keeps_evicted_and_cleared_arrays_alive(empty_logs):
    value = parsed(raw([0, -1], offset="2"), "a" * 64, empty_logs)
    cache = ResultCache(max_bytes=1_000_000, max_entries=1)
    cache.put("a" * 64, value)
    snapshot = cache.snapshot()
    cache.put("b" * 64, parsed(raw([3, 4]), "b" * 64, empty_logs))
    cache.clear()
    assert cache.entry_count == cache.byte_count == 0
    retained = snapshot["a" * 64].raw
    assert retained is not None
    np.testing.assert_array_equal(retained.get_axis(), [2, 3])


def test_shared_array_backing_is_counted_once_and_corrected_axis_is_counted(empty_logs):
    backing = np.linspace(0, 1, 2000).copy()
    h = header("Operating Point", [("V(a)", "voltage"), ("V(b)", "voltage")], 2000)
    shared = DecodedRaw([DecodedPlot(h, [backing, backing.view()], snapshot_id="cache-example")])
    separate = DecodedRaw(
        [DecodedPlot(h, [backing.copy(), backing.copy()], snapshot_id="cache-example")]
    )
    assert (
        resident_size(parsed(separate, "a" * 64, empty_logs))
        - resident_size(parsed(shared, "a" * 64, empty_logs))
        >= backing.nbytes
    )
    original = raw(backing)
    corrected = raw(backing, offset="1")
    assert (
        resident_size(parsed(corrected, "a" * 64, empty_logs))
        - resident_size(parsed(original, "a" * 64, empty_logs))
        >= backing.nbytes
    )
    assert corrected.select_plot(0).plots is corrected.plots


def test_concurrent_mutations_keep_accounting_consistent(empty_logs):
    cache = ResultCache(max_bytes=1_000_000, max_entries=4)
    values = [parsed(raw([0, index]), f"{index:064x}", empty_logs) for index in range(8)]

    def update(index):
        for _ in range(20):
            cache.put(f"{index:064x}", values[index])
            cache.get(f"{index:064x}")
            cache.snapshot()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(update, range(8)))
    snapshot = cache.snapshot()
    assert cache.entry_count == len(snapshot) <= 4
    assert cache.byte_count == sum(resident_size(value) for value in snapshot.values())


def test_waiting_parser_admission_can_cancel_without_touching_active_call():
    cache = ResultCache(max_bytes=1_000_000, max_entries=4)
    entered = threading.Event()
    release = threading.Event()

    def active():
        with cache.parse_slot(deadline=time.monotonic() + 5):
            entered.set()
            release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(active)
        assert entered.wait(2)
        cancel = threading.Event()
        cancel.set()
        try:
            with (
                pytest.raises(InterruptedError, match="cancelled"),
                cache.parse_slot(deadline=time.monotonic() + 1, cancel=cancel),
            ):
                pytest.fail("Cancelled queued calls must not enter")
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=2)
    with cache.parse_slot(deadline=time.monotonic() + 1):
        pass


def test_invalid_limits_refuse():
    with pytest.raises(ValueError, match="positive"):
        ResultCache(max_bytes=0, max_entries=1)
    with pytest.raises(ValueError, match="32"):
        ResultCache(max_bytes=1, max_entries=33)


def test_log_backing_metadata_is_charged_and_oversized_logs_are_not_retained(tmp_path):
    from tests.test_log_decode import LIMITS, capture

    captured, directory = capture(tmp_path, text="")
    metadata = decode_logs(captured, directory, limits=LIMITS)
    small = parsed(None, "a" * 64, DecodedLog(metadata))
    metadata["fourier"] = {
        "status": "error",
        "value": None,
        "error": {"type": "ValueError", "message": "x" * 50_000},
        "nonfinite_count": 0,
    }
    large = parsed(None, "b" * 64, DecodedLog(metadata))
    assert resident_size(large) >= sys.getsizeof(metadata["fourier"]["error"]["message"])
    cache = ResultCache(max_bytes=resident_size(small), max_entries=2)
    assert cache.put("b" * 64, large) is large
    assert cache.entry_count == cache.byte_count == 0


def test_capability_snapshot_excludes_logs_only_entries(empty_logs):
    cache = ResultCache()
    logs_only = parsed(None, "a" * 64, empty_logs)
    with_raw = parsed(raw([0, 1]), "b" * 64, empty_logs)
    cache.put("a" * 64, logs_only)
    cache.put("b" * 64, with_raw)
    assert cache.snapshot() == {"a" * 64: logs_only, "b" * 64: with_raw}
    assert cache.snapshot(require_raw=True) == {"b" * 64: with_raw}
