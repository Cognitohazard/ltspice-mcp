"""Shared engine startup for MCP server and in-process callers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import result_store
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState

ConfigLoadedHook = Callable[[ServerConfig], None]

logger = logging.getLogger(__name__)

_LIBRARY_OVERRIDE_NAMES = frozenset(
    {
        "simulator",
        "enabled_simulators",
        "simulator_exe",
        "ngbehavior",
        "allowed_paths",
        "max_parallel_sims",
        "max_experiment_cases",
        "default_timeout",
        "max_estimated_points",
        "max_raw_mb",
        "max_points_returned",
        "analysis_budget_s",
        "result_set_ttl_hours",
        "symbol_paths",
        "persist_jobs",
        "preload_recent_count",
    }
)
_PATH_OVERRIDE_NAMES = frozenset({"simulator_exe"})
_PATH_LIST_OVERRIDE_NAMES = frozenset({"allowed_paths", "symbol_paths"})


@dataclass(frozen=True)
class BootstrapResult:
    """State created by the shared bootstrap and its startup preload count."""

    state: SessionState
    preloaded_circuits: int


def _expanded_path(value: object, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a path-like value")
    return Path(value).expanduser()


def _path_list(value: object, name: str) -> list[Path]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of path-like values")
    return [_expanded_path(item, name) for item in value]


def _library_config(
    working_dir: str | os.PathLike[str] | None,
    config_path: str | os.PathLike[str] | None,
    overrides: Mapping[str, object],
) -> ServerConfig:
    unsupported = sorted(set(overrides) - _LIBRARY_OVERRIDE_NAMES)
    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(f"Unsupported library configuration override(s): {names}")

    normalized = dict(overrides)
    for name in _PATH_OVERRIDE_NAMES & normalized.keys():
        value = normalized[name]
        normalized[name] = None if value is None else _expanded_path(value, name)
    for name in _PATH_LIST_OVERRIDE_NAMES & normalized.keys():
        normalized[name] = _path_list(normalized[name], name)
    if "enabled_simulators" in normalized:
        value = normalized["enabled_simulators"]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("enabled_simulators must be a sequence of strings")
        if not all(isinstance(item, str) for item in value):
            raise TypeError("enabled_simulators must be a sequence of strings")
        normalized["enabled_simulators"] = [item.strip().lower() for item in value]

    resolved_working_dir: Path | None = None
    if working_dir is not None:
        resolved_working_dir = _expanded_path(working_dir, "working_dir").resolve()
        normalized["working_dir"] = resolved_working_dir

    if config_path is not None:
        selected_config = _expanded_path(config_path, "config_path").resolve()
    elif resolved_working_dir is not None:
        selected_config = resolved_working_dir / "ltspice-mcp.toml"
    else:
        selected_config = None
    return ServerConfig.load(
        selected_config,
        overrides=normalized,
    )


def configure_asc_editor(
    config: ServerConfig,
    available: dict,
    *,
    target_logger: logging.Logger | None = None,
) -> None:
    """Configure AscEditor library paths for .asc schematic support.

    Schematic editing only needs the ``.asy`` symbol library, not a working
    simulator binary, so symbol resolution is independent of simulator
    detection.

    Resolution order:
    1. Explicit config.symbol_paths / LTSPICE_MCP_SYMBOL_PATHS override
    2. WSL paths resolved through Windows ``%LOCALAPPDATA%``
    3. spicelib preparation for a detected LTspice class
    4. No schematic symbol support
    """
    from spicelib.editor.asc_editor import AscEditor

    log = target_logger or logger

    if config.symbol_paths:
        valid = [str(path) for path in config.symbol_paths if path.is_dir()]
        if valid:
            AscEditor.custom_lib_paths = valid
            log.info(f"AscEditor symbol paths from config: {valid}")
            return
        log.warning(f"Configured symbol_paths do not exist: {config.symbol_paths}")

    from ltspice_mcp.lib.wsl import get_ltspice_lib_paths, is_wsl

    if is_wsl():
        lib_paths = get_ltspice_lib_paths()
        if lib_paths:
            AscEditor.custom_lib_paths = lib_paths
            log.info(f"AscEditor WSL library paths: {lib_paths}")
            return
        log.info(
            ".asc schematic graphics editing unavailable on WSL (no LTspice symbol "
            "library found); SPICE simulation and netlist editing are unaffected. "
            "To enable it, set [schematic] symbol_paths in ltspice-mcp.toml or "
            "LTSPICE_MCP_SYMBOL_PATHS env var."
        )
        return

    ltspice_cls = available.get("ltspice")
    if ltspice_cls is None:
        log.info(
            ".asc schematic graphics editing unavailable (no LTspice symbol library "
            "found); SPICE simulation and netlist editing are unaffected"
        )
        return

    try:
        AscEditor.prepare_for_simulator(ltspice_cls)
        if AscEditor.simulator_lib_paths or AscEditor.custom_lib_paths:
            log.info("AscEditor configured via prepare_for_simulator()")
            return
        log.warning("prepare_for_simulator() found no library paths")
    except Exception as exc:
        log.warning(f"AscEditor prepare_for_simulator failed: {exc}")


async def _bootstrap(
    config: ServerConfig,
    target_logger: logging.Logger | None,
) -> BootstrapResult:
    """The initialization every host shares, once its config is resolved."""
    diagnostics: list[str] = []
    available = detect_simulators(config, diagnostics)
    state = SessionState.create(config, available, diagnostics)
    configure_asc_editor(config, available, target_logger=target_logger)
    await asyncio.to_thread(result_store.cleanup, state.working_dir)

    preloaded = 0
    if config.persist_jobs and config.preload_recent_count > 0:
        preloaded = state.job_registry.preload_recent(max_circuits=config.preload_recent_count)
    return BootstrapResult(state=state, preloaded_circuits=preloaded)


async def bootstrap_server_engine(
    *,
    on_config_loaded: ConfigLoadedHook,
    logger: logging.Logger,
) -> BootstrapResult:
    """Create the engine session behind the MCP server's lifespan.

    Configuration comes from the process environment and TOML only — there is
    no override channel here. ``on_config_loaded`` keeps process-wide logging
    setup in the MCP startup path, where it belongs.
    """
    config = ServerConfig.load()
    on_config_loaded(config)
    return await _bootstrap(config, logger)


async def bootstrap_library_engine(
    *,
    working_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    **overrides: object,
) -> BootstrapResult:
    """Create the engine session behind an in-process :class:`Api`.

    Overrides are applied after environment and TOML values, and every name
    must be a ``ServerConfig`` field the library exposes. Supplying
    ``working_dir`` makes the directory the default sandbox root and selects
    its TOML unless ``config_path`` is explicit.
    """
    config = _library_config(working_dir, config_path, overrides)
    return await _bootstrap(config, None)
