"""Captured completion excerpts retain the shared renderer and strict facts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import log_parser
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.log_decode import LogDecodeError, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from tests.test_log_decode import FIXTURES, LIMITS, capture


@pytest.mark.parametrize("name", sorted(path.name for path in FIXTURES.glob("*.log")))
def test_recorded_error_context_matches_existing_rendering(tmp_path, name):
    source = FIXTURES / name
    expected = log_parser.extract_error_context(source, max_lines=20)
    captured, directory = capture(tmp_path, source=source)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    assert metadata["error_context"] == {
        "status": "parsed",
        "value": expected,
        "error": None,
        "nonfinite_count": 0,
    }
    assert DecodedLog(metadata).value("error_context") == expected


@pytest.mark.parametrize(
    "body",
    [
        "",
        "only one line\n",
        "\n".join(f"progress line {i}" for i in range(50)),
        "Fatal Error: head marker\n" + "progress\n" * 40 + "Fatal Error: tail marker\n",
    ],
)
def test_empty_last_lines_and_first_last_error_windows_preserve_parity(tmp_path, body):
    captured, directory = capture(tmp_path, text=body)
    expected = log_parser.extract_error_context(tmp_path / "source.log", max_lines=20)
    result = DecodedLog(decode_logs(captured, directory, limits=LIMITS))
    assert result.value("error_context") == expected
    if not body:
        assert result.value("error_context") == "(Empty log file)"
    if "head marker" in body:
        assert "head marker" in expected and "tail marker" in expected


def test_source_deletion_and_console_changes_do_not_change_captured_excerpt(tmp_path):
    captured, directory = capture(
        tmp_path, text="Fatal Error: captured primary\n", console="Error: captured console\n"
    )
    expected = log_parser.extract_error_context(tmp_path / "source.log", max_lines=20)
    (tmp_path / "source.log").unlink()
    (tmp_path / "source.exe.log").write_text("Fatal Error: changed console\n", encoding="utf-8")
    log = DecodedLog(decode_logs(captured, directory, limits=LIMITS))
    assert log.value("error_context") == expected
    assert "console" not in log.value("error_context")
    assert log.value("diagnostics")["errors"] == [
        "Fatal Error: captured primary",
        "Error: captured console",
    ]


def test_console_only_has_diagnostics_without_primary_excerpt(tmp_path):
    captured, directory = capture(tmp_path, console="Error: console only\n")
    log = DecodedLog(decode_logs(captured, directory, limits=LIMITS))
    assert log.section("error_context") == {
        "status": "absent",
        "value": None,
        "error": None,
        "nonfinite_count": 0,
    }
    assert log.value("error_context") is None
    assert log.value("diagnostics")["errors"] == ["Error: console only"]


def test_display_clipping_does_not_clip_full_capture_or_diagnostics(tmp_path):
    limits = replace(LIMITS, log_bytes=16 * 1024 * 1024, lines=100_000)
    filler = ("x" * 127 + "\n") * (36 * 1024)
    half = 4 * 1024 * 1024
    head = "\nFatal Error: head marker\n"
    middle = "Fatal Error: middle marker\n"
    tail = "\nFatal Error: tail marker\n"
    body = (
        filler[: half - len(head)]
        + head
        + middle
        + filler[: 1024 * 1024 - len(middle)]
        + tail
        + filler[: half - len(tail)]
    )
    source = tmp_path / "source.log"
    source.write_text(body, encoding="utf-8", newline="")
    directory = tmp_path / "worker"
    directory.mkdir()
    captured = capture_inputs(
        SourceFiles(log=source),
        directory,
        input_bytes=limits.log_bytes,
        log_bytes=limits.log_bytes,
        require_raw=False,
    )
    expected = log_parser.extract_error_context(source, max_lines=20)
    source.unlink()
    log = DecodedLog(decode_logs(captured, directory, limits=limits))
    assert log.value("error_context") == expected
    assert "head marker" in expected and "tail marker" in expected
    assert "middle marker" not in expected
    assert f"(log truncated for excerpt: {len(body.encode('utf-8'))} bytes total)" in expected
    assert "Fatal Error: middle marker" in log.value("diagnostics")["errors"]
    assert log.scan == {"complete": True, "capturedbytes": len(body.encode("utf-8"))}
    assert captured.files[0].size_bytes == (directory / "input.log").stat().st_size


def test_renderer_read_failure_is_section_error_not_a_parsed_excerpt(tmp_path, monkeypatch):
    captured, directory = capture(tmp_path, text="Circuit: recorded\n")
    read_bytes = Path.read_bytes

    def refuse(self):
        if self == directory / "input.log":
            raise OSError("recorded read refusal")
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    assert metadata["error_context"]["status"] == "error"
    assert metadata["error_context"]["value"] is None
    assert metadata["error_context"]["nonfinite_count"] == 0
    with pytest.raises(ResultError, match=r"error_context.*recorded read refusal"):
        DecodedLog(metadata).value("error_context")


def test_capture_validation_precedes_excerpt_read(tmp_path, monkeypatch):
    captured, directory = capture(tmp_path, text="Circuit: recorded\n")
    (directory / "input.log").write_text("modified capture\n", encoding="utf-8")
    monkeypatch.setattr(
        log_parser,
        "extract_error_context",
        lambda *_args, **_kwargs: pytest.fail("Excerpt read before validation"),
    )
    with pytest.raises(LogDecodeError, match=r"size|digest"):
        decode_logs(captured, directory, limits=LIMITS)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "absent"),
        ("value", None),
        ("value", 7),
        ("nonfinite_count", 1),
        ("error", {"type": "OSError", "message": "invalid success"}),
        ("extra", True),
    ],
)
def test_parent_rejects_bad_excerpt_shape_and_presence(tmp_path, key, value):
    captured, directory = capture(tmp_path, text="Circuit: recorded\n")
    metadata = decode_logs(captured, directory, limits=LIMITS)
    metadata["error_context"][key] = value
    with pytest.raises(ValueError, match="error_context"):
        DecodedLog(metadata)


def test_parent_requires_excerpt_field_and_rejects_console_only_text(tmp_path):
    captured, directory = capture(tmp_path, console="Error: console only\n")
    metadata = decode_logs(captured, directory, limits=LIMITS)
    missing = dict(metadata)
    missing.pop("error_context")
    with pytest.raises(ValueError, match="error_context"):
        DecodedLog(missing)
    metadata["error_context"]["value"] = "invented primary excerpt"
    with pytest.raises(ValueError, match="error_context"):
        DecodedLog(metadata)
