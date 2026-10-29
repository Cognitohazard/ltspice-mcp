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
from ltspice_mcp.lib.simulator import (
    detect_named_simulators,
    detect_simulators,
    symbol_library_for,
)
from ltspice_mcp.lib.symbol_library import SymbolLibrary
from ltspice_mcp.state import SessionState

ConfigLoadedHook = Callable[[ServerConfig], None]

logger = logging.getLogger(__name__)

_LIBRARY_OVERRIDE_NAMES = frozenset(
    {
        "simulator",
        "enabled_simulators",
        "simulator_exe",
        "simulator_executables",
        "ltspice_ini",
        "hidden_desktop",
        "ngbehavior",
        "allowed_paths",
        "max_parallel_sims",
        "max_experiment_cases",
        "default_timeout",
        "run_timeout",
        "max_estimated_points",
        "max_raw_mb",
        "max_points_returned",
        "analysis_budget_s",
        "result_set_ttl_hours",
        "symbol_paths",
        "sync_open_window",
        "start_ltspice",
        "persist_jobs",
        "preload_recent_count",
    }
)
_PATH_OVERRIDE_NAMES = frozenset({"simulator_exe", "ltspice_ini"})
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
    if "simulator_executables" in normalized:
        value = normalized["simulator_executables"]
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise TypeError("simulator_executables must be a mapping of name to path")
        normalized["simulator_executables"] = {
            key.strip().lower(): _expanded_path(path, "simulator_executables")
            for key, path in value.items()
        }
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


def session_symbol_library(config: ServerConfig, available: Mapping[str, type]) -> SymbolLibrary:
    """The symbol library a session with ``config`` and these simulators reads.

    Schematic editing needs the ``.asy`` symbol library, not a working
    simulator, so the library is found whether or not a simulator was:

    1. the folders ``[schematic] symbol_paths`` (``LTSPICE_MCP_SYMBOL_PATHS``)
       names that exist;
    2. then the ``lib/sym`` folder of the LTspice the session runs, or of
       an installed LTspice detection did not pick (spicelib's ``LTspice``,
       every build), as spicelib finds it, and through ``%LOCALAPPDATA%`` on
       WSL.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    return symbol_library_for(config.symbol_paths, available.get("ltspice") or LTspice)


def local_symbol_library() -> SymbolLibrary:
    """The symbol library a session started here would read, for a tool that runs
    without one (the render command, the benchmark script): the configuration
    in the working directory and the environment, and the LTspice detection
    finds."""
    config = ServerConfig.load()
    return session_symbol_library(config, detect_simulators(config, []))


def configure_asc_editor(
    config: ServerConfig,
    available: dict,
    *,
    target_logger: logging.Logger | None = None,
) -> SymbolLibrary:
    """The session's symbol library (``session_symbol_library``), also given to
    spicelib's schematic editor.

    spicelib's editor still opens sheets, and keeps its search paths as class
    attributes; this is the one place that sets them, from the library, so it
    searches what every reader here searches.
    """
    from spicelib.editor.asc_editor import AscEditor

    log = target_logger or logger

    missing = [path for path in config.symbol_paths if not path.is_dir()]
    if missing:
        log.warning(f"Configured symbol_paths do not exist: {missing}")
    library = session_symbol_library(config, available)

    # spicelib appends ``sym`` (symbols) and ``sub`` (sub-sheets and models) to
    # each of its simulator paths and takes custom paths as given.
    AscEditor.custom_lib_paths = [str(path) for path in library.configured]
    AscEditor.simulator_lib_paths = [str(path) for path in library.build_roots]
    if library.roots:
        log.info(f"Schematic symbol libraries: {[str(root) for root in library.roots]}")
    else:
        log.info(
            ".asc schematic graphics editing unavailable (no LTspice symbol library "
            "found); SPICE simulation and netlist editing are unaffected. To enable "
            "it, set [schematic] symbol_paths in ltspice-mcp.toml or the "
            "LTSPICE_MCP_SYMBOL_PATHS env var."
        )
    return library


async def _bootstrap(
    config: ServerConfig,
    target_logger: logging.Logger | None,
    *,
    sandbox_pinned: bool = False,
) -> BootstrapResult:
    """The initialization every host shares, once its config is resolved."""
    diagnostics: list[str] = []
    available = detect_simulators(config, diagnostics)
    named = detect_named_simulators(config, diagnostics)
    symbols = configure_asc_editor(config, available, target_logger=target_logger)
    state = SessionState.create(
        config,
        available,
        diagnostics,
        named=named,
        sandbox_pinned=sandbox_pinned,
        symbols=symbols,
    )
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
    ``working_dir`` makes it the directory relative paths are taken from and
    selects its TOML unless ``config_path`` is explicit.

    An explicit ``allowed_paths`` pins the sandbox for the session: it outranks
    the TOML here, so an edit to the TOML later must not replace it.
    """
    config = _library_config(working_dir, config_path, overrides)
    return await _bootstrap(config, None, sandbox_pinned="allowed_paths" in overrides)
