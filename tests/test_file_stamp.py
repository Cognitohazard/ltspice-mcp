"""The stat stamp a parsed source is remembered by."""

import os
import sys
import time

from ltspice_mcp.lib.file_stamp import ABSENT, file_stamp

_DAY_NS = 86_400 * 10**9


def test_a_missing_file_stamps_as_absent(tmp_path):
    assert file_stamp(str(tmp_path / "missing.raw")) == ABSENT


def test_only_a_regular_file_is_stamped(tmp_path):
    assert file_stamp(str(tmp_path)) is None


def test_stamp_carries_identity_size_and_both_times_in_unix_nanoseconds(tmp_path):
    path = tmp_path / "result.raw"
    path.write_bytes(b"payload")
    stamp = file_stamp(str(path))
    assert isinstance(stamp, tuple)
    info = os.stat(path)
    assert stamp[:4] == (info.st_dev, info.st_ino, 7, info.st_mtime_ns)
    # The change time shares the clock the settle rule reads, on every platform
    # (Windows counts it from 1601 in 100-nanosecond ticks).
    assert abs(stamp[4] - time.time_ns()) < _DAY_NS
    if sys.platform != "win32":
        assert stamp[4] == info.st_ctime_ns
