"""Parser revision identity comes from complete captured bytes and companions."""

import hashlib
from types import SimpleNamespace

import pytest

from ltspice_mcp.lib import parser_capture
from ltspice_mcp.lib.parser_capture import CaptureError, SourceFiles, capture_inputs
from ltspice_mcp.lib.store import Store


@pytest.fixture
def capture_files(tmp_path):
    raw = tmp_path / "circuit.raw"
    raw.write_bytes(b"RAW\r\n\x00\x1a\xff\n")
    log = raw.with_suffix(".log")
    log.write_bytes(b"solver output\r\n")
    sources = SourceFiles(raw, log, raw.with_suffix(".exe.log"))
    store = Store(tmp_path)
    directory = store.parser_dir("parse_1")
    directory.mkdir(parents=True)
    return sources, store, directory


def test_capture_preserves_binary_bytes_and_companion_absence(capture_files):
    sources, _store, directory = capture_files
    result = capture_inputs(sources, directory, input_bytes=100, log_bytes=50)

    assert result.absent == ("console",)
    assert [item.role for item in result.files] == ["raw", "log"]
    for item in result.files:
        source = getattr(sources, item.role)
        expected = source.read_bytes()
        assert (directory / item.name).read_bytes() == expected
        assert item.size_bytes == len(expected)
        assert item.sha256 == hashlib.sha256(expected).hexdigest()


def test_revision_uses_bytes_not_path_or_timestamp(capture_files, tmp_path):
    sources, store, directory = capture_files
    first = capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
    copy = tmp_path / "different.raw"
    copy.write_bytes(sources.raw.read_bytes())
    copied_log = copy.with_suffix(".log")
    copied_log.write_bytes(sources.log.read_bytes())
    second_dir = store.parser_dir("parse_2")
    second_dir.mkdir()
    second = capture_inputs(
        SourceFiles(copy, copied_log), second_dir, input_bytes=100, log_bytes=50
    )
    options = {"dialect": None, "producing_dialect": "ngspice", "revision": "1"}
    assert first.cache_key(**options) == second.cache_key(**options)
    for key, value in (("dialect", "ngspice"), ("producing_dialect", None), ("revision", "2")):
        assert first.cache_key(**{**options, key: value}) != first.cache_key(**options)


@pytest.mark.parametrize("change", ["raw", "log", "console"])
def test_revision_changes_with_content_or_new_companion(capture_files, change):
    sources, store, directory = capture_files
    first = capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
    getattr(sources, change).write_bytes(b"changed bytes")
    second_dir = store.parser_dir("parse_2")
    second_dir.mkdir()
    second = capture_inputs(sources, second_dir, input_bytes=100, log_bytes=50)
    options = {"dialect": None, "producing_dialect": None, "revision": "1"}
    assert first.cache_key(**options) != second.cache_key(**options)


@pytest.mark.parametrize("limit", ["input_bytes", "log_bytes"])
def test_capture_limits_refuse_before_copy(capture_files, limit):
    sources, _store, directory = capture_files
    limits = {"input_bytes": 100, "log_bytes": 50, limit: 1}
    with pytest.raises(CaptureError, match="byte limit"):
        capture_inputs(
            sources, directory, input_bytes=limits["input_bytes"], log_bytes=limits["log_bytes"]
        )
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("change", ["raw", "log", "console"])
def test_capture_refuses_change_before_all_companions_finish(capture_files, monkeypatch, change):
    sources, _store, directory = capture_files
    copy = parser_capture._copy_file

    def changed_after_copy(*args):
        result = copy(*args)
        getattr(sources, change).write_bytes(b"a different source generation")
        return result

    monkeypatch.setattr(parser_capture, "_copy_file", changed_after_copy)
    with pytest.raises(CaptureError, match="changed"):
        capture_inputs(sources, directory, input_bytes=100, log_bytes=50)


def test_capture_never_overwrites_an_existing_snapshot(capture_files):
    sources, _store, directory = capture_files
    capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
    before = (directory / "input.raw").read_bytes()
    with pytest.raises(FileExistsError):
        capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
    assert (directory / "input.raw").read_bytes() == before


def test_missing_raw_is_not_an_absent_optional_companion(capture_files):
    sources, _store, directory = capture_files
    sources.raw.unlink()
    with pytest.raises(FileNotFoundError):
        capture_inputs(sources, directory, input_bytes=100, log_bytes=50)


def test_capture_accepts_log_only_inputs(capture_files):
    sources, _store, directory = capture_files
    result = capture_inputs(SourceFiles(log=sources.log), directory, input_bytes=100, log_bytes=50)
    assert result.absent == ("raw", "console")
    assert [item.role for item in result.files] == ["log"]


def test_log_capture_can_record_a_named_missing_raw(capture_files):
    sources, _store, directory = capture_files
    sources.raw.unlink()
    result = capture_inputs(sources, directory, input_bytes=100, log_bytes=50, require_raw=False)
    assert result.absent == ("raw", "console")
    assert [item.role for item in result.files] == ["log"]


def test_nonregular_source_is_refused(capture_files):
    sources, _store, directory = capture_files
    with pytest.raises(CaptureError, match="regular"):
        capture_inputs(
            SourceFiles(raw=sources.raw.parent), directory, input_bytes=100, log_bytes=50
        )


@pytest.mark.parametrize("changes_while_open", [False, True])
def test_handle_change_time_is_compared_with_the_same_handle_view(
    capture_files, monkeypatch, changes_while_open
):
    sources, _store, directory = capture_files
    real_fstat = parser_capture.os.fstat
    calls = 0

    def handle_stat(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        # On Windows path stat can report creation time while fstat reports
        # change time. Each view still detects changes within that view.
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=info.st_ctime_ns + 1000 + (calls if changes_while_open else 0),
        )

    monkeypatch.setattr(parser_capture.os, "fstat", handle_stat)
    if changes_while_open:
        with pytest.raises(CaptureError, match="during capture"):
            capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
    else:
        result = capture_inputs(sources, directory, input_bytes=100, log_bytes=50)
        assert (directory / "input.raw").read_bytes() == sources.raw.read_bytes()
        assert len(result.files) == 2
