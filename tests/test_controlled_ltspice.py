"""Frozen LTspice settings remain separate from each launch's writable profile."""

import hashlib
import os
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest

from ltspice_mcp.lib.pdk_native import ArtifactDigest, NativeLaunchPolicy
from ltspice_mcp.lib.recovery_records import ExecutionRecord, RecoveryError, StartupPolicy
from ltspice_mcp.lib.simulator_build import SimulatorExecutable

_BUILD = "a94eb1789084db9f46375cce05110e03578f9cdb931867a0faaca5b200793f06"
_QUERY = 1_700_000_000


@pytest.fixture
def adapter_module(monkeypatch):
    from ltspice_mcp.lib import controlled_ltspice as module

    # Filesystem/argv tests run everywhere; native profile parsing is tested
    # separately without replacing the Windows API.
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.time, "time", lambda: _QUERY)
    if os.name != "nt":
        monkeypatch.setattr(
            module,
            "_read_profile",
            lambda path: {
                ("options", "uuid"): "12345678901234567890",
                ("options", "captureanalytics"): "false",
                ("options", "lastwebupdatequery"): str(_QUERY),
            },
        )
    return module


def _template(tmp_path):
    path = tmp_path / "startup" / "template.ini"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        (
            f"[Options]\r\nUUID=12345678901234567890\r\nCaptureAnalytics=false\r\nLastWebUpdateQuery={_QUERY}\r\n"
        ).encode("utf-16")
    )
    return ArtifactDigest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _execution(template):
    return ExecutionRecord(
        10,
        "request",
        2,
        None,
        5,
        ("LTspice.exe",),
        SimulatorExecutable("LTspice.exe", _BUILD, 1, "2026-10-03"),
        None,
        "win32",
        StartupPolicy("ltspice-established-ini-v1", False, ini_template=template),
    )


def test_copy_is_exclusive_and_retains_vendor_writeback(adapter_module, tmp_path):
    template = _template(tmp_path)
    initial = template.path.read_bytes()
    path = tmp_path / "startup" / "attempt-1" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    adapter_module.verify_attempt_ini(template, path, tmp_path)
    path.write_bytes(initial + b"vendor writeback")
    adapter_module.verify_ini_template(template, tmp_path)
    assert template.path.read_bytes() == initial
    with pytest.raises(RecoveryError, match="changed"):
        adapter_module.verify_attempt_ini(template, path, tmp_path)
    with pytest.raises(RecoveryError, match="overwrite"):
        adapter_module.prepare_attempt_ini(template, path, tmp_path)
    assert path.read_bytes() == initial + b"vendor writeback"


def test_capture_preserves_exact_bytes(adapter_module, tmp_path):
    source = tmp_path / "source.ini"
    content = b"\xff\xfe[\x00O\x00p\x00t\x00i\x00o\x00n\x00s\x00]\x00\r\x00\n\x00"
    # This mixed-encoding fixture is used only with the parser seam on Linux.
    # The native test below exercises actual effective-value interpretation.
    if os.name == "nt":
        content = _template(tmp_path / "source-root").path.read_bytes()
    source.write_bytes(content)
    root = tmp_path / "lineage"
    target = root / "startup" / "template.ini"
    artifact = adapter_module.capture_ini_template(source, target, root)
    assert target.read_bytes() == source.read_bytes() == content
    assert artifact == ArtifactDigest(target, hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize("relative", ["outside.ini", "startup/../bad.ini", "startup/a/other.ini"])
def test_attempt_path_refused_before_write(adapter_module, tmp_path, relative):
    template = _template(tmp_path)
    path = tmp_path / relative
    with pytest.raises(RecoveryError):
        adapter_module.prepare_attempt_ini(template, path, tmp_path)
    assert not path.exists()


def test_template_drift_prevents_copy(adapter_module, tmp_path):
    template = _template(tmp_path)
    template.path.write_bytes(b"changed")
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    with pytest.raises(RecoveryError, match="changed"):
        adapter_module.prepare_attempt_ini(template, path, tmp_path)
    assert not path.exists()


@pytest.mark.parametrize(
    ("age", "accepted"), [(0, True), (1_296_000, True), (1_296_001, False), (-1, False)]
)
def test_exact_freshness_boundary(age, accepted):
    from ltspice_mcp.lib.controlled_ltspice import _validate_profile

    profile = {
        ("options", "uuid"): "12345678901234567890",
        ("options", "captureanalytics"): "false",
        ("options", "lastwebupdatequery"): str(_QUERY),
    }
    if accepted:
        _validate_profile(profile, _QUERY + age)
    else:
        with pytest.raises(RecoveryError):
            _validate_profile(profile, _QUERY + age)


@pytest.mark.parametrize(
    "query",
    [
        "",
        "0",
        "-1",
        "+1700000000",
        "1700000000x",
        "01700000000",
        "2147483648",
        "\u0661\u0667" + "\u0660" * 8,
    ],
)
def test_malformed_query_is_refused(query):
    from ltspice_mcp.lib.controlled_ltspice import _validate_profile

    profile = {
        ("options", "uuid"): "12345678901234567890",
        ("options", "captureanalytics"): "false",
        ("options", "lastwebupdatequery"): query,
    }
    with pytest.raises(RecoveryError):
        _validate_profile(profile, _QUERY)


@pytest.mark.parametrize("identifier", ["0", "000", "0" * 20, "1" * 19, "1x", "1" * 21])
def test_unestablished_identifier_is_refused(identifier):
    from ltspice_mcp.lib.controlled_ltspice import _validate_profile

    profile = {
        ("options", "uuid"): identifier,
        ("options", "captureanalytics"): "false",
        ("options", "lastwebupdatequery"): str(_QUERY),
    }
    with pytest.raises(RecoveryError):
        _validate_profile(profile, _QUERY)


def test_fixed_width_identifier_preserves_leading_zero():
    from ltspice_mcp.lib.controlled_ltspice import _validate_profile

    profile = {
        ("options", "uuid"): "01234567890123456789",
        ("options", "captureanalytics"): "false",
        ("options", "lastwebupdatequery"): str(_QUERY),
    }
    _validate_profile(profile, _QUERY)
    assert profile[("options", "uuid")] == "01234567890123456789"


def test_native_profile_api_preserves_mixed_encoding_capture(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("Effective INI parsing requires native Windows")
    from ltspice_mcp.lib import controlled_ltspice as module

    monkeypatch.setattr(module.time, "time", lambda: _QUERY)
    source = tmp_path / "source.ini"
    # The duplicated tail reproduces a partial ANSI write after a Unicode
    # profile. Windows selects the ANSI section for this mixed fixture;
    # capture must preserve that actual choice, including its query value.
    source.write_bytes(
        (
            f"[Options]\r\nUUID=12345678901234567890\r\nCaptureAnalytics=false\r\nLastWebUpdateQuery={_QUERY}\r\n"
            "[Unrelated]\r\nKeep=value\r\n"
        ).encode("utf-16")
        + f"[Options]\r\nUUID=45678901234567890123\r\nCaptureAnalytics=true\r\nLastWebUpdateQuery={_QUERY}\r\n".encode(
            "ascii"
        )
    )
    before = source.read_bytes()
    root = tmp_path / "lineage"
    artifact = module.capture_ini_template(source, root / "startup" / "template.ini", root)
    assert source.read_bytes() == artifact.path.read_bytes() == before
    assert module._read_profile(source) == module._read_profile(artifact.path)
    assert module._read_profile(source)[("options", "uuid")] == "45678901234567890123"


def test_attempt_effective_values_must_match_template(adapter_module, tmp_path, monkeypatch):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    original = adapter_module._read_profile

    def path_sensitive_profile(profile_path):
        values = original(profile_path).copy()
        if profile_path == path:
            values[("options", "captureanalytics")] = "true"
        return values

    monkeypatch.setattr(adapter_module, "_read_profile", path_sensitive_profile)
    with pytest.raises(RecoveryError, match="effective"):
        adapter_module.verify_attempt_ini(template, path, tmp_path)


def test_native_mixed_profile_without_effective_query_is_refused(tmp_path):
    if os.name != "nt":
        pytest.skip("Effective INI parsing requires native Windows")
    from ltspice_mcp.lib import controlled_ltspice as module

    source = tmp_path / "source.ini"
    source.write_bytes(
        (
            f"[Options]\r\nUUID=12345678901234567890\r\nCaptureAnalytics=false\r\nLastWebUpdateQuery={_QUERY}\r\n"
        ).encode("utf-16")
        + b"[Options]\r\nUUID=45678901234567890123\r\nCaptureAnalytics=true\r\n"
    )
    target = tmp_path / "lineage" / "startup" / "template.ini"
    assert ("options", "lastwebupdatequery") not in module._read_profile(source)
    with pytest.raises(RecoveryError) as error:
        module.capture_ini_template(source, target, tmp_path / "lineage")
    assert error.value.code == "recovery_startup_unsupported"
    assert not target.exists()


def test_missing_attempt_has_typed_error(adapter_module, tmp_path):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "absent" / "LTspice.ini"
    with pytest.raises(RecoveryError) as error:
        adapter_module.verify_attempt_ini(template, path, tmp_path)
    assert error.value.code == "recovery_startup_io"


def test_launch_uses_documented_flags_and_child_environment(adapter_module, tmp_path, monkeypatch):
    template = _template(tmp_path)
    execution = _execution(template)
    # Tokens themselves use the Store-safe alphabet; spaces elsewhere remain
    # one argv operand without shell quoting.
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    monkeypatch.setenv("APPDATA", "ambient profile")
    before = dict(os.environ)
    checked = []
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    adapter = adapter_module.controlled_ltspice(execution, path, lambda: checked.append(True))
    deck = tmp_path / "divider with spaces.cir"
    assert adapter.run(deck, timeout=10, cwd=tmp_path, exe_log=True) == 0
    assert checked == [True]
    assert run.call_args.args[0] == ["LTspice.exe", "-Run", "-b", str(deck), "-ini", str(path)]
    assert run.call_args.kwargs["env"]["APPDATA"] == str(path.parent)
    assert run.call_args.kwargs["timeout"] == 10
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert "- ini" not in run.call_args.args[0]
    assert dict(os.environ) == before
    assert not deck.with_suffix(".exe.log").exists()


@pytest.mark.parametrize("exe_log", [False, True])
def test_launch_does_not_inherit_or_redirect_stream_handles(
    adapter_module, tmp_path, monkeypatch, exe_log
):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    adapter = adapter_module.controlled_ltspice(_execution(template), path, lambda: None)
    deck = tmp_path / "divider.cir"
    assert adapter.run(deck, exe_log=exe_log) == 0
    options = run.call_args.kwargs
    assert all(key not in options for key in ("stdin", "stdout", "stderr", "creationflags"))
    assert options["close_fds"] is True
    assert not deck.with_suffix(".exe.log").exists()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_stream_override_refused_before_spawn(adapter_module, tmp_path, monkeypatch, stream):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    adapter = adapter_module.controlled_ltspice(_execution(template), path, lambda: None)
    with pytest.raises(RecoveryError):
        adapter.run(tmp_path / "divider.cir", **{stream: object()})
    run.assert_not_called()


def test_drift_after_callback_refuses_before_spawn(adapter_module, tmp_path, monkeypatch):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    adapter = adapter_module.controlled_ltspice(
        _execution(template), path, lambda: path.write_bytes(b"changed")
    )
    with pytest.raises(RecoveryError, match="changed"):
        adapter.run(tmp_path / "divider.cir")
    run.assert_not_called()


@pytest.mark.parametrize("switches", [["-alt"], ["-ini", "other.ini"], ["-sync"], "-b"])
def test_unrecorded_switches_refused(adapter_module, tmp_path, monkeypatch, switches):
    template = _template(tmp_path)
    path = tmp_path / "startup" / "attempt" / "LTspice.ini"
    adapter_module.prepare_attempt_ini(template, path, tmp_path)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    adapter = adapter_module.controlled_ltspice(_execution(template), path, lambda: None)
    with pytest.raises(RecoveryError):
        adapter.run(tmp_path / "divider.cir", switches)
    run.assert_not_called()


@pytest.mark.parametrize("change", ["build", "platform", "seed", "native", "argv", "startup"])
def test_unsupported_execution_refused(adapter_module, tmp_path, change):
    execution = _execution(_template(tmp_path))
    edits = {
        "build": {"executable": replace(execution.executable, sha256="b" * 64)},
        "platform": {"platform": "linux"},
        "seed": {"simulator_seed": 1},
        "native": {"native_policy": NativeLaunchPolicy()},
        "argv": {"simulator_argv": ("wine", "LTspice.exe")},
        "startup": {"startup": replace(execution.startup, user_init_disabled=True)},
    }
    with pytest.raises(RecoveryError):
        adapter_module.verify_ltspice_execution(replace(execution, **edits[change]))
