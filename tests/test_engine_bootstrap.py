"""Behavior tests for the shared server and library engine bootstrap."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from spicelib import AscEditor

import ltspice_mcp.engine as engine
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now, recent, result_store
from ltspice_mcp.server import server, server_lifespan
from ltspice_mcp.state import SessionState
from tests.conftest import write_legacy_sidecar


def _detect_without_simulators(config: ServerConfig, diagnostics: list[str]) -> dict[str, type]:
    del config
    diagnostics.append("detector diagnostic")
    return {}


def _stage_expired_result_set(working_dir: Path) -> Path:
    item = result_store.create(
        working_dir=working_dir,
        inputs={"working_dir": str(working_dir)},
        work=[],
        source_manifests=[],
        source_jobs={},
        ttl_hours=24,
    )
    path = result_store.result_path(item.result_set_id, working_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expires_at"] = (now() - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


async def _stage_persisted_job(working_dir: Path, circuit: Path) -> None:
    seed = SessionState.create(
        ServerConfig(
            working_dir=working_dir,
            allowed_paths=[working_dir],
            persist_jobs=True,
        ),
        available={},
    )
    write_legacy_sidecar(circuit, "sim_bootstrap_preload")
    await seed.job_registry.drain_pending()
    await seed.shutdown()


def _startup_snapshot(state: SessionState, expired_path: Path) -> dict[str, object]:
    return {
        "symbol_paths": list(AscEditor.custom_lib_paths),
        "allowed_paths": state.config.allowed_paths,
        "expired_result_removed": not expired_path.exists(),
        "jobs": sorted(state.all_jobs),
        "diagnostics": state.diagnostics,
    }


@pytest.mark.asyncio
async def test_server_and_library_bootstrap_have_matching_startup_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "project"
    symbol_dir = working_dir / "symbols"
    symbol_dir.mkdir(parents=True)
    circuit = working_dir / "saved.cir"
    circuit.write_text(".end\n", encoding="utf-8")
    (working_dir / "ltspice-mcp.toml").write_text(
        f'[schematic]\nsymbol_paths = ["{symbol_dir}"]\n'
        "[state]\npersist_jobs = true\npreload_recent_count = 10\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(working_dir)
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "state-home-server"))
    monkeypatch.setattr(engine, "detect_simulators", _detect_without_simulators)
    monkeypatch.setattr(AscEditor, "custom_lib_paths", [])
    await _stage_persisted_job(working_dir, circuit)
    recent.touch(circuit)

    server_expired = _stage_expired_result_set(working_dir)
    with patch("ltspice_mcp.server.logging.basicConfig") as basic_config:
        async with server_lifespan(server) as context:
            server_snapshot = _startup_snapshot(context["state"], server_expired)
    assert basic_config.call_count == 1
    assert basic_config.call_args.kwargs["force"] is True

    # A second home, seeded the same way, so the two bootstraps read equal but
    # independent indexes. The preload prunes and rewrites the index it reads,
    # which would otherwise make the first bootstrap an input to the second.
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "state-home-library"))
    recent.touch(circuit)

    library_expired = _stage_expired_result_set(working_dir)
    boot = await engine.bootstrap_library_engine(working_dir=working_dir)
    try:
        library_snapshot = _startup_snapshot(boot.state, library_expired)
        assert boot.preloaded_circuits == 1
    finally:
        await boot.state.shutdown()

    assert (
        server_snapshot
        == library_snapshot
        == {
            "symbol_paths": [str(symbol_dir)],
            "allowed_paths": [working_dir],
            "expired_result_removed": True,
            "jobs": ["sim_bootstrap_preload"],
            "diagnostics": ["detector diagnostic"],
        }
    )


@pytest.mark.asyncio
async def test_library_working_dir_selects_its_toml_and_default_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_dir = tmp_path / "process"
    working_dir = tmp_path / "selected"
    process_dir.mkdir()
    working_dir.mkdir()
    (process_dir / "ltspice-mcp.toml").write_text(
        "[simulation]\ntimeout = 91\n",
        encoding="utf-8",
    )
    (working_dir / "ltspice-mcp.toml").write_text(
        "[simulation]\ntimeout = 17\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(process_dir)
    monkeypatch.setenv("LTSPICE_MCP_CONFIG", str(process_dir / "ltspice-mcp.toml"))
    monkeypatch.delenv("LTSPICE_MCP_WORKING_DIR", raising=False)
    monkeypatch.delenv("LTSPICE_MCP_ALLOWED_PATHS", raising=False)
    monkeypatch.setattr(engine, "detect_simulators", _detect_without_simulators)
    monkeypatch.setattr("ltspice_mcp.lib.wsl.is_wsl", lambda: False)

    boot = await engine.bootstrap_library_engine(
        working_dir=working_dir,
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        assert boot.state.config.default_timeout == 17.0
        assert boot.state.config.config_path == working_dir / "ltspice-mcp.toml"
        assert boot.state.working_dir == working_dir
        assert boot.state.config.allowed_paths == [working_dir]
    finally:
        await boot.state.shutdown()


@pytest.mark.asyncio
async def test_library_overrides_take_precedence_over_environment_toml_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "project"
    working_dir.mkdir()
    toml_sandbox = tmp_path / "toml-sandbox"
    env_sandbox = tmp_path / "env-sandbox"
    explicit_sandbox = tmp_path / "explicit-sandbox"
    (working_dir / "ltspice-mcp.toml").write_text(
        f'[simulation]\ntimeout = 10\n[security]\nallowed_paths = ["{toml_sandbox}"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("LTSPICE_MCP_TIMEOUT", "20")
    monkeypatch.setenv("LTSPICE_MCP_ALLOWED_PATHS", str(env_sandbox))
    monkeypatch.setattr(engine, "detect_simulators", _detect_without_simulators)
    monkeypatch.setattr("ltspice_mcp.lib.wsl.is_wsl", lambda: False)

    boot = await engine.bootstrap_library_engine(
        working_dir=working_dir,
        default_timeout=30,
        allowed_paths=[explicit_sandbox],
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        config = boot.state.config
        assert config.default_timeout == 30
        assert config.allowed_paths == [explicit_sandbox]
        assert config.max_points_returned == 10_000
    finally:
        await boot.state.shutdown()


@pytest.mark.asyncio
async def test_library_bootstrap_rejects_unknown_and_irrelevant_overrides(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="unknown_setting"):
        await engine.bootstrap_library_engine(working_dir=tmp_path, unknown_setting=True)
    with pytest.raises(TypeError, match="tool_profile"):
        await engine.bootstrap_library_engine(working_dir=tmp_path, tool_profile="agentic")


@pytest.mark.asyncio
async def test_library_bootstrap_does_not_mutate_root_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol_dir = tmp_path / "symbols"
    symbol_dir.mkdir()
    monkeypatch.setattr(engine, "detect_simulators", _detect_without_simulators)
    monkeypatch.setattr(AscEditor, "custom_lib_paths", [])
    root = logging.getLogger()
    before = (tuple(root.handlers), root.level, tuple(root.filters), root.disabled)

    boot = await engine.bootstrap_library_engine(
        working_dir=tmp_path,
        symbol_paths=[symbol_dir],
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        after = (tuple(root.handlers), root.level, tuple(root.filters), root.disabled)
        assert after == before
    finally:
        await boot.state.shutdown()


def test_every_library_override_name_is_a_config_field() -> None:
    """A renamed config field must not strand an override name behind it.

    ``_LIBRARY_OVERRIDE_NAMES`` is the allowlist ``Api(**overrides)`` validates
    against; a name in it that no longer exists on ``ServerConfig`` would be
    accepted from the caller and then silently dropped by ``ServerConfig.load``.
    """
    config_fields = {field.name for field in dataclasses.fields(ServerConfig)}
    stranded = sorted(engine._LIBRARY_OVERRIDE_NAMES - config_fields)
    assert not stranded, f"override name(s) with no ServerConfig field: {', '.join(stranded)}"
